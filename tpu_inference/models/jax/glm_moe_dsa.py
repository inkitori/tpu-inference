# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GLM 5.2 (GlmMoeDsa) JAX model — Phase 1a foundations.

This module provides the JAX RoPE primitives used by later tasks:

  * ``build_rope_cos_sin_np`` — host-side numpy fp32 cos/sin table builder
    (V4 TPU lesson: compute outside any device mesh to eliminate pow-divergence
    and avoid uninit-HBM-on-reshard; only the apply runs on device).
  * ``apply_rope_interleaved_jax`` — MLA interleaved RoPE (matches HF
    ``apply_rotary_pos_emb_interleave``; even/odd dim split, first-half cos/sin).
  * ``apply_rope_rotate_half_jax`` — indexer rotate-half RoPE (matches HF
    ``apply_rotary_pos_emb``; full-width cos/sin, rotate-half convention).

Bit-for-bit parity with the HF oracle is verified at 1e-6 (fp32 bit-for-bit
class), including near-1M positions with rope_theta=8_000_000.
"""
from __future__ import annotations

from typing import Dict, Tuple

import jax
import numpy as np
from jax import lax as jax_lax
from jax import numpy as jnp


# ---------------------------------------------------------------------------
# Host-side cos/sin table builder (numpy fp32, no JAX device ops)
# ---------------------------------------------------------------------------

def build_rope_cos_sin_np(
    positions: np.ndarray,
    rope_theta: float,
    head_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build RoPE cos/sin tables on the host using numpy fp32.

    Reproduces ``GlmMoeDsaRotaryEmbedding.forward()`` exactly:
      inv_freq = 1 / (theta ** (arange(0,d,2,int64→fp32) / d))  # shape [d/2]
      freqs[t,k] = position[t] * inv_freq[k]                      # [T, d/2]
      emb = cat(freqs, freqs, axis=-1)                            # [T, d]
      cos = cos(emb),  sin = sin(emb)   (attention_scaling=1 for default rope)

    Args:
        positions: 1-D integer array of token positions, shape [T].
        rope_theta: RoPE base frequency (e.g. 10000.0 or 8_000_000.0).
        head_dim: RoPE head dimension (= ``config.qk_rope_head_dim`` = 64 for
            GLM 5.2; inv_freq has shape [head_dim//2]).

    Returns:
        (cos, sin): each numpy float32 array of shape [T, head_dim].
        Both are host arrays — call ``jnp.array(...)`` to put on device.

    Note:
        MUST be called OUTSIDE any live JAX device mesh (V4 lesson).  The
        returned arrays are passed into ``apply_rope_interleaved_jax`` or
        ``apply_rope_rotate_half_jax`` which run on device.

    Implementation note:
        inv_freq is computed via torch (matching HF's exact float32 scalar-power
        path) rather than pure numpy.  A pure-numpy ``theta ** float32_array``
        uses numpy's fp32 pow which can differ by 1 ULP from torch's scalar-pow
        for the same exponent.  At near-1M positions that 1-ULP inv_freq error
        accumulates to ~3e-2 in angle, breaking the 1e-6 table gate.  Using
        torch keeps the inv_freq bit-identical to ``GlmMoeDsaRotaryEmbedding``.
    """
    import torch as _torch
    # HF: arange(0, dim, 2, dtype=int64) → float32 / dim → scalar-pow in torch
    # Using torch matches GlmMoeDsaRotaryEmbedding.compute_default_rope_parameters
    # exactly (1-ULP-safe even at rope_theta=8_000_000 and near-1M positions).
    inv_freq = (1.0 / (
        rope_theta ** (
            _torch.arange(0, head_dim, 2, dtype=_torch.int64).to(dtype=_torch.float32)
            / head_dim
        )
    )).numpy()  # [d/2], fp32
    # freqs[t, i] = positions[t] * inv_freq[i]  →  [T, d/2]
    positions_f32 = np.asarray(positions, dtype=np.float32)
    freqs = positions_f32[:, None] * inv_freq[None, :]    # [T, d/2]
    # HF: emb = cat(freqs, freqs, dim=-1)  →  [T, d]
    emb = np.concatenate([freqs, freqs], axis=-1).astype(np.float32)
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    return cos, sin   # [T, head_dim], [T, head_dim]


# ---------------------------------------------------------------------------
# MLA interleaved RoPE apply  (matches HF apply_rotary_pos_emb_interleave)
# ---------------------------------------------------------------------------

def apply_rope_interleaved_jax(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply MLA interleaved RoPE to query and key tensors.

    Matches HF ``apply_rotary_pos_emb_interleave`` exactly:
      - Slices cos/sin to first half: ``cos = cos[..., :d//2]``
        (because the full ``emb = cat(freqs, freqs)`` duplicates; first half
        holds the per-pair angle for even/odd dim pairs 0/1, 2/3, …).
      - Even/odd split: ``q1, q2 = q[..., 0::2], q[..., 1::2]``
      - Rotation: ``q_embed = cat([q1*cos - q2*sin, q2*cos + q1*sin], -1)``

    Args:
        q: Query tensor, shape ``[B, n_heads, T, head_dim]`` (fp32).
        k: Key tensor,   shape ``[B, n_heads, T, head_dim]`` (fp32).
        cos: Full-width cos table, shape ``[T, head_dim]`` or
             ``[B, T, head_dim]`` (both work via broadcast).
        sin: Full-width sin table, same shape as cos.

    Returns:
        (q_embed, k_embed): rotated tensors with the same shape as q, k.

    Consumed by: Task 3 (MLA attention module).
    """
    # Slice to first half to match HF: cos[..., :d//2]
    d = cos.shape[-1]
    cos_half = cos[..., : d // 2]   # [T, d/2]  or  [B, T, d/2]
    sin_half = sin[..., : d // 2]

    # Broadcast over batch and heads: insert dims so shape is [1, 1, T, d/2]
    # (q is [B, n_heads, T, head_dim])
    if cos_half.ndim == 2:
        # [T, d/2] → [1, 1, T, d/2]
        cos_half = cos_half[None, None, :, :]
        sin_half = sin_half[None, None, :, :]
    elif cos_half.ndim == 3:
        # [B, T, d/2] → [B, 1, T, d/2]
        cos_half = cos_half[:, None, :, :]
        sin_half = sin_half[:, None, :, :]

    # Even/odd split along last dim
    q1 = q[..., 0::2]    # [B, n_heads, T, d/2]
    q2 = q[..., 1::2]
    k1 = k[..., 0::2]
    k2 = k[..., 1::2]

    # HF: cat([q1*cos - q2*sin, q2*cos + q1*sin], dim=-1)
    q_embed = jnp.concatenate([q1 * cos_half - q2 * sin_half,
                                q2 * cos_half + q1 * sin_half], axis=-1)
    k_embed = jnp.concatenate([k1 * cos_half - k2 * sin_half,
                                k2 * cos_half + k1 * sin_half], axis=-1)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Indexer rotate-half RoPE apply  (matches HF apply_rotary_pos_emb)
# ---------------------------------------------------------------------------

def _rotate_half_jax(x: jnp.ndarray) -> jnp.ndarray:
    """Rotate half the hidden dims: cat([-x2, x1]).

    Matches HF ``rotate_half``:
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        return cat((-x2, x1), dim=-1)
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope_rotate_half_jax(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply indexer rotate-half RoPE to query and key tensors.

    Matches HF ``apply_rotary_pos_emb`` exactly:
      - Uses full-width cos/sin (no slicing; ``emb = cat(freqs,freqs)`` already
        full-width, which is what ``rotate_half`` expects).
      - ``q_embed = q * cos + rotate_half(q) * sin``

    Args:
        q: Query tensor, shape ``[B, n_heads, T, head_dim]`` (fp32).
        k: Key tensor,   shape ``[B, n_heads, T, head_dim]`` (fp32).
        cos: Full-width cos table, shape ``[T, head_dim]`` or
             ``[B, T, head_dim]`` (both work via broadcast).
        sin: Full-width sin table, same shape as cos.

    Returns:
        (q_embed, k_embed): rotated tensors with the same shape as q, k.

    Consumed by: Task 5 (indexer RoPE, Phase 2 prep).
    """
    # Broadcast over batch and heads: shape → [1, 1, T, head_dim] or [B, 1, T, d]
    if cos.ndim == 2:
        # [T, d] → [1, 1, T, d]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif cos.ndim == 3:
        # [B, T, d] → [B, 1, T, d]
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]

    q_embed = q * cos + _rotate_half_jax(q) * sin
    k_embed = k * cos + _rotate_half_jax(k) * sin
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# RMSNorm (pure-jnp fp32) — matches GlmMoeDsaRMSNorm
# ---------------------------------------------------------------------------

def _rms_norm_jax(x: jnp.ndarray, weight: jnp.ndarray, eps: float) -> jnp.ndarray:
    """RMSNorm: x / sqrt(mean(x²) + eps) * weight  (no mean-subtract, no bias).

    Matches ``GlmMoeDsaRMSNorm.forward`` (modeling_glm_moe_dsa.py:57-62), which
    upcasts to fp32, computes the variance as ``x.pow(2).mean(-1)``, scales by
    ``rsqrt(var + eps)``, then multiplies by ``weight``.  All arithmetic here is
    fp32.

    Args:
        x: input, shape ``[..., dim]`` (fp32).
        weight: per-feature scale, shape ``[dim]`` (fp32).
        eps: variance epsilon (1e-6 for q_a/kv_a layernorms; see class docstring).
    """
    x = x.astype(jnp.float32)
    variance = jnp.mean(x * x, axis=-1, keepdims=True)
    x = x * jax_lax.rsqrt(variance + eps)
    return weight.astype(jnp.float32) * x


# Canonical MLA submodule param names (bare, relative to the layer prefix).
_MLA_PARAM_NAMES = (
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
)


# ---------------------------------------------------------------------------
# Non-absorbed pure-jnp fp32 MLA reference  (the math answer key — Task 3)
# ---------------------------------------------------------------------------

class GlmMoeDsaAttentionRef:
    """Non-absorbed, explicit-`kv_b`-split, all-fp32 MLA forward — the math oracle.

    This is the pure-``jnp`` reference (NOT the shipped absorbed kernel path).
    It reproduces ``GlmMoeDsaAttention.forward`` (modeling_glm_moe_dsa.py:409)
    exactly in fp32, with the indexer omitted: at ``seq < index_topk`` the DSA
    indexer selects all causal keys, so a single MLA block reduces to dense
    causal MLA (spec §A5).  Later tasks consume this as the answer key:
    Task 6 (full-model math oracle) and Task 7 (kernel-algebra gate compares the
    absorbed kernel against THIS).

    Forward (all fp32, NOPE-first concat):

        q_resid = q_a_layernorm(q_a_proj(x))            # RMSNorm eps=1e-6
        q       = q_b_proj(q_resid) -> [B,T,N,256]
        q_nope, q_rope = split(q, [192, 64])

        compressed = kv_a_proj_with_mqa(x)              # [B,T,512+64]
        latent, k_rope = split(compressed, [512, 64])
        kv      = kv_b_proj(kv_a_layernorm(latent))     # RMSNorm eps=1e-6 on latent ONLY
                  -> [B,T,N,448]
        k_nope, v = split(kv, [192, 256])

        q_rope, k_rope = interleaved-RoPE(q_rope, k_rope)   # k_rope shared across heads
        q = concat([q_nope, q_rope]); k = concat([k_nope, k_rope])   # NOPE-first
        attn = softmax(q·kᵀ * sm_scale + causal_mask) ; sm_scale = 256**-0.5
        out  = (attn·v) -> reshape -> o_proj

    Critical correctness points (one wrong value = a silent gate failure):
      * q_a/kv_a layernorm eps = **1e-6** (class default; NOT rms_norm_eps=1e-5).
      * ``kv_a_layernorm`` applies to the 512-latent ONLY, not the rope part.
      * NOPE-first concat for both q and k.
      * ``sm_scale = qk_head_dim**-0.5 = 256**-0.5`` (default rope: mscale=1, no YaRN).
      * NON-ABSORBED: explicit per-forward ``kv_b`` split ``[qk_nope|v_head_dim]``;
        ``kv_b`` is NOT absorbed (absorption is Task 7).

    Weights (loaded via :meth:`load_weights`, keyed by bare HF MLA names, already
    run through ``t2j_weights`` so linear kernels are ``[in, out]``):
        ``self_attn.q_a_proj.weight``           [hidden, q_lora_rank]
        ``self_attn.q_a_layernorm.weight``      [q_lora_rank]
        ``self_attn.q_b_proj.weight``           [q_lora_rank, N*qk_head_dim]
        ``self_attn.kv_a_proj_with_mqa.weight`` [hidden, kv_lora_rank+qk_rope]
        ``self_attn.kv_a_layernorm.weight``     [kv_lora_rank]
        ``self_attn.kv_b_proj.weight``          [kv_lora_rank, N*(qk_nope+v)]
        ``self_attn.o_proj.weight``             [N*v_head_dim, hidden]
    ``attention_bias=False`` for GLM, so no projection has a bias term.
    """

    # eps for q_a/kv_a layernorms is the GlmMoeDsaRMSNorm class default, NOT
    # config.rms_norm_eps (=1e-5).  Copying rms_norm_eps here is a silent bug.
    NORM_EPS = 1e-6

    def __init__(self, config):
        self.config = config
        # Explicit MLA dims — NEVER read config.head_dim (overwritten to 64 in
        # __post_init__); use the explicit fields.
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim   # 192
        self.qk_rope_head_dim = config.qk_rope_head_dim   # 64
        self.v_head_dim = config.v_head_dim               # 256
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 256
        # Default rope (mscale=1, no YaRN): sm_scale = qk_head_dim**-0.5.
        self.sm_scale = self.qk_head_dim ** (-0.5)
        self._w: Dict[str, jnp.ndarray] = {}

    # -- weight management ---------------------------------------------------
    def load_weights(self, weights: Dict[str, jnp.ndarray]) -> None:
        """Load JAX MLA weights (bare HF names, ``t2j_weights``-converted, fp32).

        Stores fp32 copies of every weight in :data:`_MLA_PARAM_NAMES`.  The
        kernels are ``[in, out]`` (already transposed by ``t2j_weights``); norm
        weights are 1-D.  Missing keys raise ``KeyError``.
        """
        self._w = {name: jnp.asarray(weights[name]).astype(jnp.float32)
                   for name in _MLA_PARAM_NAMES}

    def export_weights(self) -> Dict[str, jnp.ndarray]:
        """Return the loaded weight map (for ``assert_identical_weights``)."""
        return dict(self._w)

    # -- forward -------------------------------------------------------------
    def __call__(self, hidden_states: jnp.ndarray,
                 cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
        """Run the non-absorbed fp32 MLA forward (causal). Returns ``[B, T, hidden]``.

        Args:
            hidden_states: ``[B, T, hidden]`` (cast to fp32 internally).
            cos, sin: RoPE tables for the rope slice, ``[T, qk_rope_head_dim]``
                (or batched ``[B, T, qk_rope_head_dim]``).  Same convention as
                :func:`apply_rope_interleaved_jax`.

        Every matmul/einsum runs under ``default_matmul_precision("highest")``:
            this is the *answer key*, so it must be true-fp32.  On TPU, JAX's
            default fp32 matmul uses the bf16x3/"high" pass which diverges from
            torch's genuine fp32 by ~5e-3 — enough to blow the 1e-3 math gate.
            HIGHEST forces the full-fp32 pass (matches torch to ~1e-6).
        """
        with jax.default_matmul_precision("highest"):
            return self._forward(hidden_states, cos, sin)

    def _forward(self, hidden_states, cos, sin):
        w = self._w
        x = hidden_states.astype(jnp.float32)
        B, T, _ = x.shape
        N = self.num_heads

        # --- query path: q_a -> q_a_layernorm(eps=1e-6) -> q_b -> split ------
        q_resid = x @ w["self_attn.q_a_proj.weight"]            # [B,T,q_lora]
        q_resid = _rms_norm_jax(q_resid, w["self_attn.q_a_layernorm.weight"],
                                self.NORM_EPS)
        q = q_resid @ w["self_attn.q_b_proj.weight"]            # [B,T,N*256]
        q = q.reshape(B, T, N, self.qk_head_dim).transpose(0, 2, 1, 3)  # [B,N,T,256]
        q_nope = q[..., : self.qk_nope_head_dim]                # [B,N,T,192]
        q_rope = q[..., self.qk_nope_head_dim:]                 # [B,N,T,64]

        # --- kv path: kv_a_proj_with_mqa -> split -> kv_a_layernorm(latent) --
        compressed = x @ w["self_attn.kv_a_proj_with_mqa.weight"]  # [B,T,512+64]
        latent = compressed[..., : self.kv_lora_rank]             # [B,T,512]
        k_rope = compressed[..., self.kv_lora_rank:]              # [B,T,64]
        # kv_a_layernorm (eps=1e-6) on the 512-latent ONLY, NOT the rope part.
        latent = _rms_norm_jax(latent, w["self_attn.kv_a_layernorm.weight"],
                               self.NORM_EPS)
        kv = latent @ w["self_attn.kv_b_proj.weight"]            # [B,T,N*448]
        kv = kv.reshape(B, T, N, self.qk_nope_head_dim + self.v_head_dim)
        kv = kv.transpose(0, 2, 1, 3)                            # [B,N,T,448]
        k_nope = kv[..., : self.qk_nope_head_dim]                # [B,N,T,192]
        value = kv[..., self.qk_nope_head_dim:]                  # [B,N,T,256]

        # --- RoPE on the rope slices (k_rope is shared across heads) ---------
        # k_rope: [B,T,64] -> [B,1,T,64] (single shared head, like HF k_rot).
        k_rope = k_rope.reshape(B, 1, T, self.qk_rope_head_dim)
        q_rope, k_rope = apply_rope_interleaved_jax(q_rope, k_rope, cos, sin)
        # broadcast the shared k_rope across all heads
        k_rope = jnp.broadcast_to(k_rope, (B, N, T, self.qk_rope_head_dim))

        # --- NOPE-first concat ---------------------------------------------
        query_states = jnp.concatenate([q_nope, q_rope], axis=-1)  # [B,N,T,256]
        key_states = jnp.concatenate([k_nope, k_rope], axis=-1)    # [B,N,T,256]

        # --- scaled-dot-product attention, causal --------------------------
        scores = jnp.einsum("bnqd,bnkd->bnqk", query_states, key_states)
        scores = scores * self.sm_scale
        # additive causal mask: -inf above the diagonal
        neg = jnp.finfo(jnp.float32).min
        causal = jnp.triu(jnp.full((T, T), neg, dtype=jnp.float32), k=1)
        scores = scores + causal[None, None, :, :]
        # softmax in fp32 (matches eager_attention_forward dtype=float32)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bnqk,bnkd->bnqd", attn, value)          # [B,N,T,256]

        # --- merge heads -> o_proj -----------------------------------------
        out = out.transpose(0, 2, 1, 3).reshape(B, T, N * self.v_head_dim)
        out = out @ w["self_attn.o_proj.weight"]                  # [B,T,hidden]
        return out
