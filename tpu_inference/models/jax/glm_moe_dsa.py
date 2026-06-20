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

from typing import Tuple

import numpy as np
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
    """
    positions = np.asarray(positions, dtype=np.float32)  # [T]
    # HF: arange(0, dim, 2, dtype=int64) → float32 / dim
    k = np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32)
    inv_freq = (1.0 / (rope_theta ** (k / head_dim))).astype(np.float32)  # [d/2]
    # freqs[t, i] = positions[t] * inv_freq[i]  →  [T, d/2]
    freqs = positions[:, None] * inv_freq[None, :]    # [T, d/2]
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
