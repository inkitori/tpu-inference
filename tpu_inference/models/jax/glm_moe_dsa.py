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

from typing import Dict, List, Optional, Tuple

import jax
import numpy as np
from flax import nnx
from jax import lax as jax_lax
from jax import numpy as jnp
from jax.sharding import Mesh


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

    def __init__(self, config, *, norm_eps: float = NORM_EPS, rngs=None):
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
        # norm_eps injection knob (test affordance; default=1e-6 = NORM_EPS).
        # Used by the eps-teeth test to prove the gate catches a wrong eps.
        # Production code never passes this — the default is the only safe value.
        self._norm_eps = norm_eps
        self._w: Dict[str, jnp.ndarray] = {}
        # When constructed inside the scaffold an ``rngs`` is supplied so the
        # block is forward-able right after init (like every other NNX
        # submodule). Real weights are then overwritten via ``load_weights``.
        # The standalone math-gate tests construct WITHOUT rngs and call
        # ``load_weights`` themselves.
        if rngs is not None:
            self._init_random_weights(rngs)

    def _init_random_weights(self, rngs) -> None:
        """Fill ``_w`` with small random fp32 weights of the correct shapes."""
        key = rngs.params() if hasattr(rngs, "params") else jax.random.PRNGKey(0)
        N = self.num_heads
        shapes = {
            "self_attn.q_a_proj.weight": (self.hidden_size, self.q_lora_rank),
            "self_attn.q_a_layernorm.weight": (self.q_lora_rank,),
            "self_attn.q_b_proj.weight": (self.q_lora_rank, N * self.qk_head_dim),
            "self_attn.kv_a_proj_with_mqa.weight":
                (self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim),
            "self_attn.kv_a_layernorm.weight": (self.kv_lora_rank,),
            "self_attn.kv_b_proj.weight":
                (self.kv_lora_rank,
                 N * (self.qk_nope_head_dim + self.v_head_dim)),
            "self_attn.o_proj.weight": (N * self.v_head_dim, self.hidden_size),
        }
        w = {}
        for name, shape in shapes.items():
            key, sub = jax.random.split(key)
            if name.endswith("layernorm.weight"):
                w[name] = jnp.ones(shape, dtype=jnp.float32)
            else:
                w[name] = (jax.random.normal(sub, shape, dtype=jnp.float32)
                           * 0.02)
        self._w = w

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
                                self._norm_eps)
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
                               self._norm_eps)
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


# ===========================================================================
# Phase 1a Task 7 — absorbed mla/v2 kernel MLA (the SHIPPED attention path).
#
# GlmMoeDsaMLA computes the SAME math as the non-absorbed GlmMoeDsaAttentionRef
# via the W_UK/W_UV absorption identity (attention in latent space), but routes
# the score/AV matmuls through the real mla/v2 Mosaic kernel
# (mla_ragged_paged_attention, via mla_attention). It shares the exact pre-
# attention math (RMSNorm eps=1e-6, interleaved RoPE, NOPE-first) and weight
# layout with the jnp-ref, so the fp32 kernel-algebra parity gate is a single
# delta isolating (absorption split + kernel) vs the ref math.
#
# Absorption (latent space, lkv_dim = kv_lora_rank = 512):
#   q_nope[T,N,192]  --k_up_proj("TNH,ANH->NTA")-->  q_NTA[N,T,512]
#   kernel scores = q_NTA . kv_c[S,512]^T  +  q_pe[T,N,64] . k_pe[S,64]^T
#   kernel out    = P . kv_c[S,512]        ->  out_NTA[N,T,512]
#   out_NTA --v_up_proj("NTA,ANH->TNH")--> out_TNH[T,N,256] --reshape--> o_proj
#
# The standalone no-quant kv_b split reshapes kv_b_proj [A, N*(192+256)] ->
# [A,N,448] -> jnp.split at qk_nope_head_dim=192 -> the two up-proj einsum
# kernels. NO quantize/dequantize (MLAEinsum.load_weights asserts a quant
# config; we deliberately do NOT reuse it).
#
# The kernel hard-asserts r_dim%128==0 / lkv_dim%128==0; lkv_dim=512 is already
# aligned, and qk_rope_head_dim=64 is padded 64->128 INSIDE the kernel
# (prepare_q_inputs / prepare_kv_inputs), so the caller passes the actual r_dim
# arrays. sm_scale = qk_head_dim**-0.5 = 256**-0.5 (default rope, mscale=1).
# ===========================================================================


class GlmMoeDsaMLA:
    """Absorbed MLA via the mla/v2 Mosaic kernel — the shipped GLM attention path.

    Drop-in for ``GlmMoeDsaAttentionRef`` as the decoder's pluggable ``self_attn``
    slot (``__call__(hidden, cos, sin)``), but additionally exposes
    :meth:`forward` which threads the per-request KV cache + attention metadata
    and the kernel precision knobs (``s_dtype``, ``p_same_dtype_as_v``,
    ``two_step_flash_attention``) straight into ``mla_attention``.

    Pre-attention math is byte-for-byte the jnp-ref's (shared ``_rms_norm_jax`` +
    ``apply_rope_interleaved_jax`` + NOPE-first split) so any parity delta vs the
    ref is the absorption + kernel, nothing else.
    """

    NORM_EPS = 1e-6   # q_a/kv_a layernorm eps (class default, NOT rms_norm_eps)

    def __init__(self, config, *, mesh: Mesh,
                 dtype: jnp.dtype = jnp.bfloat16,
                 s_dtype: jnp.dtype = jnp.bfloat16,
                 p_same_dtype_as_v: bool = True,
                 two_step_flash_attention: bool = True,
                 norm_eps: float = NORM_EPS,
                 prefix: str = "self_attn"):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.prefix = prefix
        # Kernel precision knobs (shipped defaults = bf16; the fp32 algebra gate
        # overrides to s_dtype=fp32 / p_same_dtype_as_v=False).
        self.s_dtype = s_dtype
        self.p_same_dtype_as_v = p_same_dtype_as_v
        self.two_step_flash_attention = two_step_flash_attention
        self._norm_eps = norm_eps

        # Explicit MLA dims — NEVER read config.head_dim (overwritten to 64).
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank          # 512 (lkv_dim, 128-aligned)
        self.qk_nope_head_dim = config.qk_nope_head_dim  # 192
        self.qk_rope_head_dim = config.qk_rope_head_dim  # 64 (r_dim, padded->128 by kernel)
        self.v_head_dim = config.v_head_dim              # 256
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 256
        # Default rope (mscale=1, no YaRN): sm_scale = qk_head_dim**-0.5.
        self.sm_scale = self.qk_head_dim ** (-0.5)

        # Sharding specs for the mla_attention shard_map (single-device safe;
        # MLP_TENSOR/ATTN_DATA collapse to replicated on make_glm_mesh(1)).
        from tpu_inference.layers.common.sharding import ShardingAxisName
        from jax.sharding import PartitionSpec as P
        self._query_nth = P(None, ShardingAxisName.MLP_TENSOR, None)
        self._query_tnh = P(ShardingAxisName.ATTN_DATA, None, None)
        self._keyvalue_skh = P(ShardingAxisName.ATTN_DATA, None)
        self._attn_o_nth = P(None, ShardingAxisName.MLP_TENSOR, None)

        # Linear weights (bare-HF-name -> fp32/dtype array), set by load_weights.
        self._w: Dict[str, jnp.ndarray] = {}
        # Absorbed up-projection einsum kernels (built in load_weights).
        self.k_up_proj = None   # weight (A, N, qk_nope_head_dim)
        self.v_up_proj = None   # weight (A, N, v_head_dim)

        # Decoder-pluggable forward (`__call__(hidden, cos, sin)`) reads these;
        # set them via `bind_cache(...)` before invoking the layer in a stack.
        self._kv_cache = None
        self._md = None

    # -- weight management ---------------------------------------------------
    def load_weights(self, weights: Dict[str, jnp.ndarray]) -> None:
        """Load bare-HF-name, t2j-converted MLA weights and build the no-quant split.

        Stores the linear/norm weights (cast to ``self.dtype``) and builds the
        standalone, unquantized ``k_up_proj`` / ``v_up_proj`` einsum kernels from
        ``kv_b_proj`` (reshape [A, N*448] -> [A,N,448] -> split at 192). NO
        quantize/dequantize — ``MLAEinsum.load_weights`` is deliberately NOT
        reused (it asserts a quant config).
        """
        from tpu_inference.layers.jax.linear import JaxEinsum

        self._w = {name: jnp.asarray(weights[name]).astype(self.dtype)
                   for name in _MLA_PARAM_NAMES}

        A = self.kv_lora_rank
        N = self.num_heads
        qk_nope = self.qk_nope_head_dim
        v_head = self.v_head_dim
        kv_b = self._w["self_attn.kv_b_proj.weight"]    # [A, N*(qk_nope+v_head)]
        if kv_b.shape != (A, N * (qk_nope + v_head)):
            raise ValueError(
                f"kv_b_proj has shape {kv_b.shape}, expected "
                f"{(A, N * (qk_nope + v_head))}")
        kv_b = kv_b.reshape(A, N, qk_nope + v_head)     # [A, N, 448]
        k_ANH, v_ANH = jnp.split(kv_b, [qk_nope], axis=-1)  # [A,N,192], [A,N,256]

        # Standalone NO-QUANT JaxEinsums; set kernels directly (no quant path).
        self.k_up_proj = JaxEinsum(
            einsum_str="TNH,ANH->NTA",
            kernel_shape=(A, N, qk_nope),
            rngs=nnx.Rngs(0),
            prefix=self.prefix + ".k_up_proj",
        )
        self.v_up_proj = JaxEinsum(
            einsum_str="NTA,ANH->TNH",
            kernel_shape=(A, N, v_head),
            rngs=nnx.Rngs(0),
            prefix=self.prefix + ".v_up_proj",
        )
        self.k_up_proj.weight.value = k_ANH
        self.v_up_proj.weight.value = v_ANH

    def export_weights(self) -> Dict[str, jnp.ndarray]:
        return dict(self._w)

    # -- decoder-pluggable hooks ---------------------------------------------
    def bind_cache(self, kv_cache, md):
        """Bind the per-request KV cache + metadata for ``__call__`` (decoder use)."""
        self._kv_cache = kv_cache
        self._md = md

    def __call__(self, hidden, cos, sin):
        """Decoder slot: run the absorbed kernel using the bound cache/metadata.

        Returns ONLY the [B, T, hidden] attention-block output (the decoder adds
        the residual). The updated KV cache is exposed via :attr:`last_kv_cache`.
        """
        if self._kv_cache is None or self._md is None:
            raise RuntimeError(
                "GlmMoeDsaMLA.__call__ needs a bound KV cache + metadata; call "
                "bind_cache(kv_cache, md) first (or use forward(...) directly).")
        x = hidden
        squeeze = False
        if x.ndim == 3:
            assert x.shape[0] == 1, "absorbed kernel path runs one sequence"
            x = x[0]
            squeeze = True
        new_cache, out_TD = self.forward(x, cos, sin, self._kv_cache, self._md)
        self.last_kv_cache = new_cache
        self._kv_cache = new_cache
        return out_TD[None, :, :] if squeeze else out_TD

    # -- forward -------------------------------------------------------------
    def forward(self, hidden_TD, cos, sin, kv_cache, md):
        """Absorbed MLA forward through the mla/v2 kernel.

        Args:
            hidden_TD: [T, hidden] input (cast to self.dtype).
            cos, sin:  RoPE tables [T, qk_rope_head_dim] (interleaved convention,
                       same as the jnp-ref / apply_rope_interleaved_jax).
            kv_cache:  paged MLA KV cache (see get_kv_cache_shape).
            md:        AttentionMetadata (seq_lens, block_tables, query_start_loc,
                       request_distribution, input_positions).

        Returns:
            (new_kv_cache, out_TD) where out_TD is [T, hidden].
        """
        from tpu_inference.layers.common.attention_interface import \
            mla_attention

        # fp32 algebra gate: genuine-fp32 matmuls on BOTH sides. On TPU the
        # default fp32 matmul is the bf16x3 "high" pass (~5e-3 off true fp32),
        # which alone blows the 1e-3 gate; "highest" forces the full-fp32 pass.
        # (For bf16 this is a no-op on the bf16 operands.)
        if self.dtype == jnp.float32:
            ctx = jax.default_matmul_precision("highest")
        else:
            import contextlib
            ctx = contextlib.nullcontext()
        with ctx:
            return self._forward(hidden_TD, cos, sin, kv_cache, md,
                                 mla_attention)

    def _forward(self, hidden_TD, cos, sin, kv_cache, md, mla_attention):
        x = jnp.asarray(hidden_TD, self.dtype)
        T = x.shape[0]
        N = self.num_heads
        w = self._w

        # --- query path: q_a -> q_a_layernorm(eps=1e-6) -> q_b -> split ------
        q_resid = x @ w["self_attn.q_a_proj.weight"]            # [T, q_lora]
        q_resid = _rms_norm_jax(q_resid, w["self_attn.q_a_layernorm.weight"],
                                self._norm_eps).astype(self.dtype)
        q = q_resid @ w["self_attn.q_b_proj.weight"]            # [T, N*256]
        q = q.reshape(T, N, self.qk_head_dim)                   # [T, N, 256]
        q_nope_TNH = q[..., :self.qk_nope_head_dim]             # [T, N, 192]
        q_rope_TNH = q[..., self.qk_nope_head_dim:]             # [T, N, 64]

        # --- kv path: kv_a_proj_with_mqa -> split -> kv_a_layernorm(latent) --
        compressed = x @ w["self_attn.kv_a_proj_with_mqa.weight"]  # [T, 512+64]
        latent = compressed[..., :self.kv_lora_rank]              # [T, 512]
        k_rope_SH = compressed[..., self.kv_lora_rank:]          # [T, 64]
        latent = _rms_norm_jax(latent, w["self_attn.kv_a_layernorm.weight"],
                               self._norm_eps).astype(self.dtype)  # kv_c (latent)

        # --- interleaved RoPE on the rope slices (shared k_rope) -------------
        # apply_rope_interleaved_jax expects [B, n_heads, T, head_dim].
        q_rope_b = q_rope_TNH.transpose(1, 0, 2)[None]          # [1, N, T, 64]
        k_rope_b = k_rope_SH[None, None]                        # [1, 1, T, 64]
        q_rope_b, k_rope_b = apply_rope_interleaved_jax(
            q_rope_b, k_rope_b, cos, sin)
        # back to kernel layouts: q_rope [T, N, 64]; k_rope [T, 64] (shared head).
        q_rope_TNH = q_rope_b[0].transpose(1, 0, 2)            # [T, N, 64]
        k_rope_SH = k_rope_b[0, 0]                              # [T, 64]

        # --- absorb q_nope into latent space: q_NTA = k_up_proj(q_nope) ------
        # einsum "TNH,ANH->NTA": [T,N,192] x [A,N,192] -> [N,T,512] (head-major).
        ql_nope_NTA = self.k_up_proj(q_nope_TNH)               # [N, T, 512]

        # fp32 kernel-output path is gated on q dtype; cast q + the latent KV to
        # self.dtype so all kernel operands agree (fp32 for the algebra gate).
        ql_nope_NTA = ql_nope_NTA.astype(self.dtype)
        q_rope_TNH = q_rope_TNH.astype(self.dtype)
        kv_c_SA = latent.astype(self.dtype)                    # [T, 512]
        k_rope_SH = k_rope_SH.astype(self.dtype)

        # --- attention in latent space via the mla/v2 kernel ----------------
        new_kv_cache, out_NTA = mla_attention(
            ql_nope_NTA,
            q_rope_TNH,
            kv_c_SA,
            k_rope_SH,
            kv_cache,
            md,
            self.mesh,
            self.num_heads,
            self.qk_nope_head_dim,
            query_nth_sharding=self._query_nth,
            query_tnh_sharding=self._query_tnh,
            keyvalue_skh_sharding=self._keyvalue_skh,
            attn_o_nth_sharding=self._attn_o_nth,
            sm_scale=self.sm_scale,
            s_dtype=self.s_dtype,
            p_same_dtype_as_v=self.p_same_dtype_as_v,
            two_step_flash_attention=self.two_step_flash_attention,
        )

        # --- project latent attention output back: v_up_proj then o_proj ----
        # out_NTA is head-major [N, T, 512]; v_up_proj "NTA,ANH->TNH".
        out_TNH = self.v_up_proj(out_NTA)                      # [T, N, 256]
        out_TR = out_TNH.reshape(T, N * self.v_head_dim)       # [T, N*256]
        out_TD = out_TR.astype(self.dtype) @ w["self_attn.o_proj.weight"]
        return new_kv_cache, out_TD


# ===========================================================================
# Phase 1a Task 4 — GLM model scaffold (config-driven from hf_config).
#
# Mirrors the DeepSeek NNX structure (deepseek_v3.py) but reads ALL dims from
# ``vllm_config.model_config.hf_config`` (the GlmMoeDsaConfig) and applies the
# GLM router/MoE deltas (sigmoid, n_group=1, *2.5 on routed output, UNSCALED
# shared, /(sum+1e-20) renorm). NEVER reads ``config.head_dim`` (overwritten to
# qk_rope_head_dim=64 in __post_init__) — uses the explicit qk_nope/qk_rope/
# v_head_dim fields.
#
# The decoder layer's attention slot is PLUGGABLE: Task 4 wires in the Task-3
# ``GlmMoeDsaAttentionRef`` (pure-jnp fp32, non-absorbed) so the model
# constructs + forwards; Task 7 swaps in the absorbed Mosaic kernel path. The
# decoder calls attention with host-precomputed (cos, sin) — the V4 TPU lesson
# (RoPE freqs built on host numpy, outside any live device mesh).
#
# Phase-1a forward gates run at seq < index_topk so the DSA indexer is the
# dense identity (spec §A5) and the model reduces to dense causal MLA + MoE.
# ===========================================================================

# Lazy imports inside __init__ keep this module importable from the cheap-helper
# / RoPE / MLA-ref tests without dragging in the full vLLM + DeepSeek stack.


def _glm_config_dims(hf_config) -> Dict[str, object]:
    """Read every GLM model dim from the HF config (NEVER ``head_dim``).

    Returns a plain dict so the scaffold never reaches back into module globals
    (the DeepSeek template reads its dims from globals; GLM must not).
    """
    rope_params = getattr(hf_config, "rope_parameters", None) or {}
    rope_theta = float(rope_params.get("rope_theta",
                                       getattr(hf_config, "rope_theta", 10000.0)))
    return dict(
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_hidden_layers=hf_config.num_hidden_layers,
        vocab_size=hf_config.vocab_size,
        q_lora_rank=hf_config.q_lora_rank,
        kv_lora_rank=hf_config.kv_lora_rank,
        qk_nope_head_dim=hf_config.qk_nope_head_dim,   # 192 — explicit, not head_dim
        qk_rope_head_dim=hf_config.qk_rope_head_dim,   # 64
        v_head_dim=hf_config.v_head_dim,               # 256
        qk_head_dim=hf_config.qk_nope_head_dim + hf_config.qk_rope_head_dim,  # 256
        rms_norm_eps=hf_config.rms_norm_eps,           # 1e-5 (input/post/final)
        intermediate_size=hf_config.intermediate_size,  # dense FFN
        moe_intermediate_size=hf_config.moe_intermediate_size,
        n_routed_experts=hf_config.n_routed_experts,
        num_experts_per_tok=hf_config.num_experts_per_tok,
        n_shared_experts=hf_config.n_shared_experts,
        routed_scaling_factor=hf_config.routed_scaling_factor,  # 2.5
        n_group=hf_config.n_group,                     # 1 (grouping OFF)
        topk_group=hf_config.topk_group,               # 1
        norm_topk_prob=hf_config.norm_topk_prob,       # True
        first_k_dense_replace=hf_config.first_k_dense_replace,  # 3
        hidden_act=hf_config.hidden_act,               # silu
        scoring_func=getattr(hf_config, "scoring_func", "sigmoid"),
        rope_theta=rope_theta,
    )


class GlmMoeDsaDecoderLayer(nnx.Module):
    """One GLM decoder block: pre-norm attention + pre-norm MLP, plain residuals.

    Matches ``GlmMoeDsaDecoderLayer.forward`` (modeling_glm_moe_dsa.py:640-670):
    plain residual adds — NO layer-0 ``.clone()`` and NO fp16-overflow rescale
    (HF GLM-DSA dropped DeepSeek-V3's). The attention module is pluggable:
    Task 4 holds ``GlmMoeDsaAttentionRef``; Task 7 swaps the kernel path.
    """

    def __init__(self, *, input_layernorm, post_attention_layernorm, self_attn,
                 mlp):
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.self_attn = self_attn
        self.mlp = mlp

    def __call__(self, x, cos, sin):
        # --- pre-norm self-attention + residual ---------------------------
        # Attention operates on [B, T, hidden]; the FFN/MoE operate on the
        # token-flattened [B*T, hidden] (they are token-wise / batch-agnostic),
        # so flatten before the MLP and reshape back after.
        residual = x
        hidden = self.input_layernorm(x)
        attn_out = self.self_attn(hidden, cos, sin)   # [B, T, hidden]
        hidden = residual + attn_out

        # --- pre-norm MLP/MoE + residual ----------------------------------
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        B, T, D = hidden.shape
        mlp_out = self.mlp(hidden.reshape(B * T, D))
        expert_indices = None
        if isinstance(mlp_out, tuple):
            mlp_out, expert_indices = mlp_out
        hidden = residual + mlp_out.reshape(B, T, D)
        return hidden, expert_indices


class GlmMoeDsa(nnx.Module):
    """Inner GLM stack: embed -> decoder layers (dense/MoE schedule) -> norm.

    Reads dims from ``vllm_config.model_config.hf_config``. Builds the dense vs
    sparse layer schedule from ``first_k_dense_replace`` (layers < k are dense
    ``DeepseekV3MLP`` SwiGLU; layers >= k are sparse param-ized ``DeepseekV2Moe``
    with the GLM router/MoE deltas). RoPE cos/sin are precomputed on the HOST
    (numpy fp32, V4 lesson) at forward time and threaded to each layer.
    """

    def __init__(self, vllm_config, rng: nnx.Rngs, mesh: Mesh, *,
                 prefix: str = "model"):
        from tpu_inference.layers.common.moe import MoEBackend
        from tpu_inference.layers.jax.embed import JaxEmbed
        from tpu_inference.layers.jax.norm import JaxRmsNorm
        from tpu_inference.models.jax.deepseek_v3 import (DeepseekV2Moe,
                                                          DeepseekV3MLP)

        self.vllm_config = vllm_config
        self.mesh = mesh
        hf_config = vllm_config.model_config.hf_config
        d = _glm_config_dims(hf_config)
        self.dims = d
        # fp32 here: Phase-1a parity is an fp32 math surface. The shipped bf16
        # path (Task 6+) flips this to model_config.dtype.
        dtype = jnp.float32

        self.rope_theta = d["rope_theta"]
        self.qk_rope_head_dim = d["qk_rope_head_dim"]

        # --- embedding (untied; HF embed_tokens is a plain lookup table) ---
        self.embed_tokens = JaxEmbed(
            num_embeddings=d["vocab_size"],
            features=d["hidden_size"],
            param_dtype=dtype,
            dtype=dtype,
            rngs=rng,
            prefix=prefix + ".embed_tokens",
        )

        # --- per-layer dense/sparse schedule -------------------------------
        first_k = d["first_k_dense_replace"]

        def make_norm():
            return JaxRmsNorm(d["hidden_size"], epsilon=d["rms_norm_eps"],
                              dtype=dtype, param_dtype=dtype, rngs=rng)

        def make_self_attn(i):
            # Task-4 attention slot: the Task-3 pure-jnp fp32 MLA reference.
            # (Task 7 replaces this with the absorbed Mosaic kernel path.)
            # Pass rngs so the block self-inits forward-able random weights;
            # load_weights overwrites them with the real HF MLA weights.
            return GlmMoeDsaAttentionRef(hf_config, rngs=rng)

        def make_dense_mlp(i):
            return DeepseekV3MLP(
                dtype=dtype,
                hidden_act=d["hidden_act"],
                hidden_size=d["hidden_size"],
                intermediate_size=d["intermediate_size"],
                rngs=rng,
            )

        def make_sparse_moe(i):
            from tpu_inference.layers.jax.quantization.unquantized import \
                UnquantizedConfig
            return DeepseekV2Moe(
                mesh=mesh,
                dtype=dtype,
                num_expert_parallelism=1,
                moe_backend=MoEBackend.DENSE_MAT,
                quant_config=UnquantizedConfig({}),
                scoring_func=d["scoring_func"],
                rng=rng,
                prefix=f"{prefix}.layers.{i}.mlp",
                num_local_experts=d["n_routed_experts"],
                hidden_size=d["hidden_size"],
                moe_intermediate_size=d["moe_intermediate_size"],
                num_experts_per_tok=d["num_experts_per_tok"],
                n_group=d["n_group"],
                topk_groups=d["topk_group"],
                norm_topk_prob=d["norm_topk_prob"],
                routed_scaling_factor=d["routed_scaling_factor"],
                num_shared_experts=d["n_shared_experts"],
                hidden_act=d["hidden_act"],
            )

        def get_decoder_layer(i: int):
            mlp = make_dense_mlp(i) if i < first_k else make_sparse_moe(i)
            return GlmMoeDsaDecoderLayer(
                input_layernorm=make_norm(),
                post_attention_layernorm=make_norm(),
                self_attn=make_self_attn(i),
                mlp=mlp,
            )

        # Phase 1a is single-device: build the full layer stack directly (no
        # pipeline-parallel split / PP-group global state). Task 1b wires the
        # PP-aware make_layers path if/when multi-host PP is needed.
        n_layers = d["num_hidden_layers"]
        self.start_layer, self.end_layer = 0, n_layers
        self.layers = nnx.List(
            [get_decoder_layer(i) for i in range(n_layers)])

        # --- final norm (eps = rms_norm_eps = 1e-5) ------------------------
        self.norm = JaxRmsNorm(d["hidden_size"], epsilon=d["rms_norm_eps"],
                               dtype=dtype, param_dtype=dtype, rngs=rng)

    def _build_cos_sin(self, positions):
        """Host-side (numpy fp32) RoPE cos/sin for the rope slice (V4 lesson).

        ``positions`` may be a jax / numpy array of shape [T] or [B, T]; cos/sin
        are returned as device arrays of shape [T, qk_rope_head_dim] (the
        attention ref broadcasts over batch + heads).
        """
        pos = np.asarray(jax.device_get(positions)).reshape(-1)
        cos, sin = build_rope_cos_sin_np(pos, self.rope_theta,
                                         self.qk_rope_head_dim)
        return jnp.asarray(cos), jnp.asarray(sin)

    def __call__(self, input_ids, positions, inputs_embeds=None):
        """Run the inner stack. Returns ``(hidden, stacked_expert_indices)``.

        ``input_ids``: [B, T] (or [T]); ``positions``: [T] (or [B, T]).
        ``inputs_embeds`` (optional) bypasses the embed lookup.
        """
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)
        if x.ndim == 2:
            x = x[None, :, :]   # [T, D] -> [1, T, D]

        cos, sin = self._build_cos_sin(positions)

        all_expert_ids: List[jax.Array] = []
        for layer in self.layers:
            x, expert_ids = layer(x, cos, sin)
            if expert_ids is not None:
                all_expert_ids.append(expert_ids)
        x = self.norm(x)

        stacked = (jnp.stack(all_expert_ids, axis=0)
                   if all_expert_ids else None)
        return x, stacked

    # --- weight loading -----------------------------------------------------
    def load_weights(self, jax_weights: Dict[str, jnp.ndarray]) -> set:
        """Load a converted (``t2j_weights``) HF weight map into the stack.

        Maps the dense-FFN, MoE (router/shared/experts), per-layer norms, MLA
        attention, embed and final norm.  Enforces the CONSUME-OR-SKIP contract:
        every key handed to this function must be either consumed (loaded into a
        parameter) or in the explicit expected-skip set (indexer params on "full"
        layers, MTP/nextn layers).  Any key that is neither consumed nor in the
        expected-skip set RAISES ``ValueError`` — no silent drops.

        The expected-skip set is computed from the same gating oracle
        (``_is_expected_skip_key``) that ``convert_hf_weights`` uses, so the two
        callers can never drift apart.

        Expects keys prefixed ``model.layers.{i}.`` (and ``model.embed_tokens``,
        ``model.norm``). Linear kernels must already be ``[in, out]`` (i.e. run
        through ``t2j_weights``). Returns the set of loaded keys.
        """
        loaded: set = set()
        d = self.dims
        first_k = d["first_k_dense_replace"]

        def take(name):
            loaded.add(name)
            return jnp.asarray(jax_weights[name])

        # embed (untransposed lookup) + final norm
        self.embed_tokens.weight.value = take("model.embed_tokens.weight")
        self.norm.weight.value = take("model.norm.weight")

        layers = list(self.layers)
        for i in range(d["num_hidden_layers"]):
            p = f"model.layers.{i}."
            layer = layers[i]
            # norms
            layer.input_layernorm.weight.value = take(p + "input_layernorm.weight")
            layer.post_attention_layernorm.weight.value = take(
                p + "post_attention_layernorm.weight")
            # attention (pure-jnp ref consumes bare HF MLA names)
            mla_w = {bare: take(p + bare) for bare in _MLA_PARAM_NAMES}
            layer.self_attn.load_weights(mla_w)
            # mlp
            if i < first_k:
                layer.mlp.gate_proj.weight.value = take(p + "mlp.gate_proj.weight")
                layer.mlp.up_proj.weight.value = take(p + "mlp.up_proj.weight")
                layer.mlp.down_proj.weight.value = take(p + "mlp.down_proj.weight")
            else:
                moe = layer.mlp
                # router gate kernel ([in,out]) + selection bias
                moe.gate.weight.value = take(p + "mlp.gate.weight")
                moe.gate.e_score_correction_bias.value = take(
                    p + "mlp.gate.e_score_correction_bias")
                # shared expert ([in,out] linears)
                moe.shared_experts.gate_proj.weight.value = take(
                    p + "mlp.shared_experts.gate_proj.weight")
                moe.shared_experts.up_proj.weight.value = take(
                    p + "mlp.shared_experts.up_proj.weight")
                moe.shared_experts.down_proj.weight.value = take(
                    p + "mlp.shared_experts.down_proj.weight")
                # routed experts: t2j_weights split gate_up -> gate/up but left
                # the 3-D tensors in HF [E,F,D] / [E,D,F] order. The DENSE_MAT
                # experts want gate/up = [E,D,F], down = [E,F,D] => swap last 2.
                g = take(p + "mlp.experts.gate_proj")          # [E,F,D]
                u = take(p + "mlp.experts.up_proj")            # [E,F,D]
                dn = take(p + "mlp.experts.down_proj")         # [E,D,F]
                moe.experts.kernel_gating_EDF.value = jnp.swapaxes(g, -2, -1)
                moe.experts.kernel_up_proj_EDF.value = jnp.swapaxes(u, -2, -1)
                moe.experts.kernel_down_proj_EFD.value = jnp.swapaxes(dn, -2, -1)

        # --- consume-or-skip contract enforcement ----------------------------
        # Compute the expected-skip set: keys in jax_weights that were NOT
        # consumed (i.e. not in `loaded`) but are EXPECTED to be skipped.
        # Uses the same _is_expected_skip_key oracle as convert_hf_weights.
        hf_config = self.vllm_config.model_config.hf_config
        indexer_types = list(getattr(hf_config, "indexer_types", []) or [])
        num_hidden_layers = int(hf_config.num_hidden_layers)

        unconsumed = set(jax_weights) - loaded
        unexpected = {
            k for k in unconsumed
            if not _is_expected_skip_key(k, indexer_types, num_hidden_layers)
        }
        if unexpected:
            raise ValueError(
                f"load_weights received {len(unexpected)} unrecognized key(s) "
                f"that were neither consumed nor in the expected-skip set "
                f"(indexer params + MTP layers). Unrecognized keys:\n"
                + "\n".join(f"  {k!r}" for k in sorted(unexpected))
                + "\nThis usually means a typo in a mapped key name, a new "
                f"parameter without a load rule, or a key from the wrong model.")

        return loaded


class GlmMoeDsaForCausalLM(nnx.Module):
    """GLM 5.2 (GlmMoeDsa) causal-LM scaffold — Phase 1a.

    Signature mirrors ``DeepseekV3ForCausalLM``: ``__init__(vllm_config,
    rng_key, mesh)``, ``__call__`` returning ``(kv_caches, hidden, [],
    expert_indices)``, plus ``compute_logits`` and ``load_weights``. The
    ``lm_head`` is UNTIED from ``embed_tokens`` (separate weight; HF
    ``tie_word_embeddings=False``).
    """

    def __init__(self, vllm_config, rng_key, mesh: Mesh):
        from tpu_inference.layers.jax.linear import JaxLmHead

        self.vllm_config = vllm_config
        self.mesh = mesh
        rng = nnx.Rngs(rng_key)
        hf_config = vllm_config.model_config.hf_config
        d = _glm_config_dims(hf_config)
        self.dims = d
        dtype = jnp.float32

        self.model = GlmMoeDsa(vllm_config, rng, mesh, prefix="model")

        self.lm_head = JaxLmHead(
            hidden_size=d["hidden_size"],
            vocab_size=d["vocab_size"],
            param_dtype=dtype,
            dtype=dtype,
            rngs=rng,
            prefix="lm_head",
        )

    def __call__(self, kv_caches, input_ids, attention_metadata,
                 inputs_embeds=None, *args, **kwargs):
        """Forward the model.

        ``attention_metadata`` may be a real ``AttentionMetadata`` (its
        ``.input_positions`` are used) or a plain positions array [T]/[B,T].
        Returns ``(kv_caches, hidden, [], expert_indices)`` to match the
        DeepSeek signature. (No KV cache is mutated in the Phase-1a dense-
        equivalent ref path; the cache list is returned unchanged.)
        """
        positions = attention_metadata
        if hasattr(attention_metadata, "input_positions"):
            positions = attention_metadata.input_positions
        if positions is None:
            T = input_ids.shape[-1]
            positions = jnp.arange(T, dtype=jnp.int32)

        hidden, expert_indices = self.model(input_ids, positions,
                                            inputs_embeds=inputs_embeds)
        return kv_caches, hidden, [], expert_indices

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        # lm_head einsum is 'TD,DV->TV' (2-D). The Phase-1a ref path carries a
        # leading batch dim ([B, T, hidden]); flatten leading dims to feed the
        # head and restore them on the output.
        if hidden_states.ndim > 2:
            lead = hidden_states.shape[:-1]
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = self.lm_head(flat)
            return logits.reshape(*lead, logits.shape[-1])
        return self.lm_head(hidden_states)

    def load_weights(self, jax_weights: Dict[str, jnp.ndarray]) -> set:
        """Load a ``t2j_weights``-converted HF map (incl. ``lm_head.weight``).

        Splits the weight dict by namespace before delegating:
          * ``model.*`` keys go to ``GlmMoeDsa.load_weights``, which enforces
            the consume-or-skip contract for the inner stack.
          * ``lm_head.weight`` is consumed here.
          * Any remaining key (in neither namespace) triggers a ``ValueError``.

        This split-before-dispatch ensures neither loader sees keys that belong
        to the other scope — so neither can silently drop them.
        """
        # Partition by namespace.
        model_weights = {k: v for k, v in jax_weights.items()
                         if k.startswith("model.")}
        lm_head_weight = jax_weights.get("lm_head.weight")

        # All keys that are neither model.* nor lm_head.weight are unexpected.
        accounted = set(model_weights)
        if lm_head_weight is not None:
            accounted.add("lm_head.weight")
        top_level_unexpected = set(jax_weights) - accounted
        if top_level_unexpected:
            raise ValueError(
                f"GlmMoeDsaForCausalLM.load_weights: {len(top_level_unexpected)} "
                f"key(s) in jax_weights are outside the 'model.*' and "
                f"'lm_head.weight' namespaces:\n"
                + "\n".join(f"  {k!r}" for k in sorted(top_level_unexpected)))

        # Load inner stack (enforces its own consume-or-skip contract).
        loaded = self.model.load_weights(model_weights)

        # Load lm_head.
        if lm_head_weight is not None:
            self.lm_head.weight.value = jnp.asarray(lm_head_weight)
            loaded.add("lm_head.weight")
        return loaded


# ---------------------------------------------------------------------------
# HF state_dict -> JAX weight converter (the Task-5 weight-map contract).
#
# Builds on the shared ``t2j`` (utils.py) — NEVER shadows it — and applies the
# GLM transpose/split rules (identical to the harness ``t2j_weights`` on every
# MAPPED key), PLUS the two explicit-skip rules Phase 1a needs:
#   * indexer params (``self_attn.indexer.*``) are RECOGNIZED and routed to the
#     ``skipped`` set, gated on ``indexer_types[i] != "shared"`` (present on
#     "full" layers, absent on "shared"). Phase 1a has no indexer module yet
#     (that is Phase 2), so they must NOT silently fall through unconsumed.
#   * MTP / nextn layers (``model.layers.{i}`` with ``i >= num_hidden_layers``,
#     e.g. the real checkpoint's ``layers.78``) are dropped to ``skipped``,
#     mirroring HF ``_keys_to_ignore_on_load_unexpected=[r"model\.layers\.78.*"]``
#     and the DeepSeek ``skip_substrs`` MTP pattern (deepseek_v3.py:1485-1495).
# ---------------------------------------------------------------------------

# Relative indexer param names a "full" layer contributes (GlmMoeDsaIndexer:
# wq_b/wk/weights_proj Linears + a k_norm LayerNorm with weight AND bias).
_INDEXER_SUBSTR = ".self_attn.indexer."


def _layer_index_of(name: str) -> Optional[int]:
    """Return the decoder layer index in ``model.layers.{i}.`` (or None)."""
    parts = name.split(".")
    for j in range(len(parts) - 1):
        if parts[j] == "layers" and parts[j + 1].isdigit():
            return int(parts[j + 1])
    return None


def _is_expected_skip_key(name: str, indexer_types: list,
                          num_hidden_layers: int) -> bool:
    """Return True if ``name`` is EXPECTED to be silently skipped by load_weights.

    This is the SINGLE gating oracle shared by ``convert_hf_weights`` and
    ``GlmMoeDsa.load_weights`` so the two callers can never drift apart:

    * MTP / nextn layers — any ``model.layers.{i}.*`` with ``i >= num_hidden_layers``.
    * Indexer params — keys containing ``_INDEXER_SUBSTR`` on "full" layers
      (gated on ``indexer_types[i] != "shared"``).  Phase 1a has no indexer
      module (Phase 2), so they are recognized-but-unloadable.

    A key that passes this predicate is in the EXPECTED-SKIP set; it must NOT
    produce an error even though it was not consumed by the loader.  Any key
    that fails this predicate AND was not consumed IS an error (unrecognized key
    → unknown param → silent random-init).
    """
    layer_idx = _layer_index_of(name)

    # MTP / nextn: decoder layer beyond the built stack.
    if layer_idx is not None and layer_idx >= num_hidden_layers:
        return True

    # Indexer param on a "full" layer (Phase 1a: no indexer module yet).
    if _INDEXER_SUBSTR in name:
        kind = (indexer_types[layer_idx]
                if layer_idx is not None and layer_idx < len(indexer_types)
                else None)
        # "shared" layers carry no indexer params on disk — the key must not
        # appear at all.  Here we just return False so the caller will raise.
        # "full" layers DO carry indexer params → skip.
        return kind != "shared"

    return False


def convert_hf_weights(state_dict, hf_config):
    """Convert an HF ``GlmMoeDsa`` ``state_dict`` to JAX weights + a skip set.

    Returns ``(jax_weights, skipped)``:
      * ``jax_weights`` — ``{name: jnp.ndarray}`` for every MAPPED param, with
        the documented transpose/split rules: every 2-D linear transposed
        ``[out,in]->[in,out]`` (``x @ kernel`` wants ``[in,out]``) EXCEPT
        ``embed_tokens`` (a lookup table); ``lm_head`` transposes like any
        linear (UNTIED); the fused experts ``gate_up_proj [E,2F,D]`` is split
        along the doubled (``-2``) axis into ``gate_proj`` (first half ``[:F]``)
        / ``up_proj`` (second half ``[F:]``); 1-D params (norms, biases) pass
        through. These names + values are IDENTICAL to the harness
        ``t2j_weights`` on every mapped key (both reuse the same ``t2j``).
      * ``skipped`` — the set of HF (pre-conversion) keys EXPLICITLY dropped:
        indexer params on "full" layers (gated on ``indexer_types[i]``) and any
        MTP/nextn layer (``layers.{i}`` with ``i >= num_hidden_layers``).

    Raises ``ValueError`` if an indexer param is found on a "shared" layer
    (would mean the ``indexer_types`` gate disagrees with the checkpoint).
    """
    from tpu_inference.utils import t2j

    indexer_types = list(getattr(hf_config, "indexer_types", []) or [])
    num_hidden_layers = int(hf_config.num_hidden_layers)

    jax_weights: Dict[str, jnp.ndarray] = {}
    skipped: set = set()

    for name, tensor in state_dict.items():
        layer_idx = _layer_index_of(name)

        # --- MTP / nextn drop + indexer drop (delegate to shared oracle) -----
        if _is_expected_skip_key(name, indexer_types, num_hidden_layers):
            # Sanity-check: an indexer param on a "shared" layer must NOT
            # appear on disk at all.  _is_expected_skip_key returns False for
            # that case (forcing the caller to raise), but guard here too.
            if _INDEXER_SUBSTR in name:
                kind = (indexer_types[layer_idx]
                        if layer_idx is not None and layer_idx < len(indexer_types)
                        else None)
                if kind == "shared":
                    raise ValueError(
                        f"indexer param {name!r} found on a 'shared' layer "
                        f"(indexer_types[{layer_idx}]=='shared'); the indexer-type "
                        f"gate disagrees with the checkpoint")
            skipped.add(name)
            continue

        # --- mapped param: transpose / fused-split rules (reuse t2j) ---------
        arr = t2j(tensor)
        if name.endswith("gate_up_proj") or name.endswith("gate_up_proj.weight"):
            axis = -2 if arr.ndim == 3 else 0
            gate, up = jnp.split(arr, 2, axis=axis)
            jax_weights[name.replace("gate_up_proj", "gate_proj")] = gate
            jax_weights[name.replace("gate_up_proj", "up_proj")] = up
        elif arr.ndim == 2 and "embed_tokens" not in name:
            jax_weights[name] = arr.T
        else:
            jax_weights[name] = arr

    return jax_weights, skipped


# ---------------------------------------------------------------------------
# Test affordance: a minimal VllmConfig whose model_config.hf_config is a real
# GlmMoeDsaConfig. Lets the scaffold tests build the model without standing up
# the full vLLM engine. (Production load goes through the real VllmConfig.)
# ---------------------------------------------------------------------------

def build_glm_vllm_config(hf_config, *, mesh=None):
    """Build a duck-typed VllmConfig exposing ``model_config.hf_config``.

    Only the fields the GLM scaffold reads are populated. ``hf_config`` is a
    real ``GlmMoeDsaConfig`` (e.g. from ``tiny_glm_moe_dsa_config()``).
    """

    class _ModelConfig:
        def __init__(self, hf_config):
            self.hf_config = hf_config
            self.dtype = jnp.float32
            self.use_mla = True

        def get_vocab_size(self):
            return self.hf_config.vocab_size

    class _VllmConfig:
        def __init__(self, hf_config):
            self.model_config = _ModelConfig(hf_config)
            self.quant_config = None

    return _VllmConfig(hf_config)
