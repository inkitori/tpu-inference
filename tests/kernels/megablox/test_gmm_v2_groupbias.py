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
"""Tests gmm_v2 per-group additive bias (rhs_groupbias) for MLX affine quant.

MLX affine quant is ``w = scale * q + bias`` where ``scale`` and ``bias`` are
per quant-group, per output-channel. gmm_v2 already applies the per-group
``rhs_scale``; this exercises the parallel per-group ``rhs_groupbias`` term:

    out[t, n] = sum_k lhs[t, k] * (q[k, n] * scale[g(k), n] + bias[g(k), n])
              = [scale path] + sum_g bias[g, n] * (sum_{k in g} lhs[t, k])

Calling convention (verified against gmm_v2 + existing tests/kernels/gmm_test.py):
  * ``rhs`` (the int4 codes) is passed AS a jnp.int4 array directly; gmm_v2
    packs it to uint32 in-kernel via ``weight.bitcast``. No manual packing.
    Codes are signed int4 in [-8, 8): the Mosaic TPU dot_general interprets all
    integer matmul inputs as signed (uint4 is rejected), and the reference uses
    the same signed values, so the full affine dequant (q*scale + bias) is
    exercised faithfully.
  * ``rhs_scale`` / ``rhs_groupbias`` are ``[G, num_blocks, 1, N]`` float32,
    matching the kernel's quant-block index map.

Both k-loop application sites are covered:
  * Test A  -> unquantized-lhs path (maybe_quantize_lhs=False). f32 accumulator,
    multiple quant blocks; bit-exact affine equality vs a numpy reference.
  * Test B  -> quantized-lhs path (maybe_quantize_lhs=True). lhs is quantized
    in-kernel (lossy), so we validate the additive-bias DELTA: the output with
    rhs_groupbias minus the output without it equals the analytic bias term
    ``sum_g bias[g] * sum_{k in g} lhs`` (the bias uses full-precision lhs and
    is added after the matmul, so the delta is clean). A single quant block is
    used so each k-loop iteration maps to exactly one rhs quant group (the
    kernel collapses to b_id=0 when the lhs quant block spans multiple rhs quant
    blocks -- a pre-existing property shared with the rhs_scale path).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.megablox.gmm_v2 import (calculate_tiling, gmm_v2,
                                                   make_gmm_configs)


def _affine_reference(lhs, q, scale, gbias, group_sizes, gs):
    """Dequantize affine (w = q*scale + bias) then grouped matmul, in f32."""
    G, K, N = q.shape
    M = lhs.shape[0]
    ref = np.zeros((M, N), np.float32)
    row = 0
    for g in range(G):
        n = int(group_sizes[g])
        s = np.repeat(scale[g, :, 0, :], gs, axis=0)  # [K, N]
        b = np.repeat(gbias[g, :, 0, :], gs, axis=0)  # [K, N]
        w = q[g].astype(np.float32) * s + b  # [K, N]
        ref[row:row + n] = lhs[row:row + n].astype(np.float32) @ w
        row += n
    return ref


def _bias_contribution(lhs, gbias, group_sizes, gs):
    """Analytic per-group additive-bias term: sum_g bias[g] * sum_{k in g} lhs."""
    G, num_blocks, _, N = gbias.shape
    M = lhs.shape[0]
    out = np.zeros((M, N), np.float32)
    row = 0
    for g in range(G):
        n = int(group_sizes[g])
        c = np.zeros((n, N), np.float32)
        for b in range(num_blocks):
            k_start = b * gs
            k_end = k_start + gs
            lhs_sum = lhs[row:row + n, k_start:k_end].sum(axis=1, keepdims=True)
            c += lhs_sum * gbias[g, b, 0, :][None, :]
        out[row:row + n] = c
        row += n
    return out


@pytest.mark.skipif(not any(d.platform == "tpu" for d in jax.devices()),
                    reason="requires TPU")
def test_gmm_v2_groupbias_unquantized_lhs_matches_affine_reference():
    """Unquantized-lhs k-loop site: bit-exact affine equality, multi-block."""
    G, M, K, N, gs = 2, 128, 512, 256, 256
    num_blocks = K // gs
    rng = np.random.default_rng(0)

    lhs = rng.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    q = rng.integers(-8, 8, size=(G, K, N)).astype(np.int32)
    scale = (rng.uniform(-1.0, 1.0, size=(G, num_blocks, 1, N)) *
             0.05).astype(np.float32)
    gbias = (rng.uniform(-1.0, 1.0, size=(G, num_blocks, 1, N)) *
             0.5).astype(np.float32)
    group_sizes = jnp.array([M // 2, M - M // 2], dtype=jnp.int32)

    ref = _affine_reference(lhs, q, scale, gbias, np.asarray(group_sizes), gs)

    out = gmm_v2(
        jnp.asarray(lhs, dtype=jnp.float32),
        jnp.asarray(q, dtype=jnp.int4),
        group_sizes,
        rhs_scale=jnp.asarray(scale),
        rhs_groupbias=jnp.asarray(gbias),
        maybe_quantize_lhs=False,
    )

    np.testing.assert_allclose(np.asarray(out, dtype=np.float32),
                               ref,
                               atol=1e-1,
                               rtol=1e-1)


@pytest.mark.skipif(not any(d.platform == "tpu" for d in jax.devices()),
                    reason="requires TPU")
def test_gmm_v2_groupbias_w2_shaped_matches_affine_reference():
    """w2 (down_proj) layout: contraction K = moe_intermediate, output N =
    hidden, with many quant blocks (gs=64 -> 24 blocks for K=1536). Proves the
    per-group ``rhs_groupbias`` affine fold holds for the w2-shaped GMM too
    (Stage-2 keeps w2 int4), not just the w13-shaped case above. Unquantized-lhs
    k-loop site: bit-exact affine equality vs a numpy reference.

    K=1536/gs=64 mirrors Hy3's down_proj contraction (moe_intermediate=1536,
    24 quant blocks); N=2048 is a representative hidden out_dim."""
    G, M, K, N, gs = 2, 64, 1536, 2048, 64
    num_blocks = K // gs  # 24
    rng = np.random.default_rng(3)

    lhs = rng.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    q = rng.integers(-8, 8, size=(G, K, N)).astype(np.int32)
    scale = (rng.uniform(-1.0, 1.0, size=(G, num_blocks, 1, N)) *
             0.05).astype(np.float32)
    gbias = (rng.uniform(-1.0, 1.0, size=(G, num_blocks, 1, N)) *
             0.5).astype(np.float32)
    group_sizes = jnp.array([M // 2, M - M // 2], dtype=jnp.int32)

    ref = _affine_reference(lhs, q, scale, gbias, np.asarray(group_sizes), gs)

    out = gmm_v2(
        jnp.asarray(lhs, dtype=jnp.float32),
        jnp.asarray(q, dtype=jnp.int4),
        group_sizes,
        rhs_scale=jnp.asarray(scale),
        rhs_groupbias=jnp.asarray(gbias),
        maybe_quantize_lhs=False,
    )

    np.testing.assert_allclose(np.asarray(out, dtype=np.float32),
                               ref,
                               atol=1e-1,
                               rtol=1e-1)


@pytest.mark.skipif(not any(d.platform == "tpu" for d in jax.devices()),
                    reason="requires TPU")
def test_gmm_v2_non_lane_multiple_size_k_does_not_overread_quant_blocks():
    """Deterministic OOB-trip guard for non-lane-multiple ``size_k``.

    The runtime symptom (NaN) only fires when the out-of-bounds per-group
    scale/groupbias read lands on a NaN bit pattern in adjacent HBM, which is
    not deterministic at the kernel level. The *root cause* IS deterministic:
    the per-quant-block DMA count ``num_quant_blocks_per_tile_k`` must never
    exceed the number of real quant blocks on the scale/groupbias axis
    (``rhs_scale.shape[1]``), otherwise the BlockSpec DMAs (and the inner loop
    indexes) one block past the end.

    For w2 (down_proj) at tp=8: per-shard ``size_k = 1536/8 = 192`` (NOT a
    multiple of 128). ``calculate_tiling`` over-aligns ``tile_k`` to 256, so
    ``num_quant_blocks_per_tile_k = cdiv(256, 64) = 4`` while the scale axis has
    only ``cdiv(192, 64) = 3`` blocks. Pre-fix this assertion FAILS (4 > 3),
    proving the over-read; post-fix the DMA/index count is clamped to the real
    remaining blocks so the invariant holds."""
    G, M, K, N, gs = 2, 16, 192, 256, 64
    num_blocks = K // gs  # 3
    rng = np.random.default_rng(7)

    lhs = jnp.asarray(rng.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32))
    q = jnp.asarray(rng.integers(-8, 8, size=(G, K, N)).astype(np.int32),
                    dtype=jnp.int4)
    scale = jnp.asarray((rng.uniform(-1.0, 1.0, size=(G, num_blocks, 1, N)) *
                         0.05).astype(np.float32))
    gbias = jnp.asarray((rng.uniform(-1.0, 1.0, size=(G, num_blocks, 1, N)) *
                         0.5).astype(np.float32))
    group_sizes = jnp.array([M // 2, M - M // 2], dtype=jnp.int32)
    group_offset = jnp.array([0], dtype=jnp.int32)

    cfgs = make_gmm_configs(
        lhs, q, scale, gbias, None, group_sizes, group_offset,
        tile_info=calculate_tiling, vmem_limit_bytes=128 * 1024 * 1024,
        out_dtype=None, acc_dtype=None, maybe_quantize_lhs=False,
        zero_initialize=True, fuse_act=None)

    real_blocks = scale.shape[1]  # 3
    # The unclamped tile-block count over-aligns (tile_k=256 -> 4 blocks) and is
    # still used only for index-map STRIDE math. The count that actually bounds
    # the BlockSpec DMA and the inner-loop scale/groupbias index is
    # num_quant_blocks_per_tile_k_read; it MUST not exceed the real blocks.
    assert cfgs.num_quant_blocks_per_tile_k_read <= real_blocks, (
        "scale/groupbias over-read: num_quant_blocks_per_tile_k_read="
        f"{cfgs.num_quant_blocks_per_tile_k_read} > real quant blocks="
        f"{real_blocks} (tile_k={cfgs.tiles.tile_k}, size_k={cfgs.dims.size_k}, "
        f"quant_block_size={cfgs.rhs_cfgs.quant_block_size}). The per-quant-"
        "block DMA/index count must be clamped to the real remaining blocks.")


@pytest.mark.skipif(not any(d.platform == "tpu" for d in jax.devices()),
                    reason="requires TPU")
def test_gmm_v2_groupbias_non_lane_multiple_size_k_is_finite_and_correct():
    """Regression: size_k NOT a multiple of num_lanes (128) must not read the
    per-group scale/groupbias out of bounds.

    This mirrors the real failure at tensor_parallel_size=8: the MoE w2
    (down_proj) per-shard contraction is moe_intermediate/tp = 1536/8 = 192,
    which is NOT a multiple of 128. ``calculate_tiling`` over-aligns the tile to
    ``tile_k = align_to(192, 128) = 256`` and (for a tiny rhs) the tile-shrink
    loops never fire, so ``num_quant_blocks_per_tile_k = cdiv(256, 64) = 4``
    while the per-shard scale/groupbias axis only has ``cdiv(192, 64) = 3`` real
    quant blocks. The scale/groupbias BlockSpec then DMAs block range [0:4] from
    a 3-long axis (disable_bounds_checks=True) and the inner loop reads the OOB
    block 3 -> NaN/garbage scale & groupbias inject sparse NaNs into specific
    (token, n) accumulator entries.

    Asserts the gmm_v2 output is (a) finite everywhere and (b) close to a
    dequant-then-grouped-matmul reference. Pre-fix this FAILS (NaN / large diff);
    post-fix the over-aligned tail block must contribute exactly zero (the k-tail
    mask already zeros its matmul value and lhs sum)."""
    # size_k=192 is NOT a multiple of num_lanes(128) -> over-alignment trip.
    # gs=64 -> 3 real quant blocks; tile_k=256 -> 4 blocks read (1 OOB).
    G, M, K, N, gs = 2, 16, 192, 256, 64
    num_blocks = K // gs  # 3
    rng = np.random.default_rng(7)

    lhs = rng.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    q = rng.integers(-8, 8, size=(G, K, N)).astype(np.int32)
    scale = (rng.uniform(-1.0, 1.0, size=(G, num_blocks, 1, N)) *
             0.05).astype(np.float32)
    gbias = (rng.uniform(-1.0, 1.0, size=(G, num_blocks, 1, N)) *
             0.5).astype(np.float32)
    group_sizes = jnp.array([M // 2, M - M // 2], dtype=jnp.int32)

    ref = _affine_reference(lhs, q, scale, gbias, np.asarray(group_sizes), gs)

    out = gmm_v2(
        jnp.asarray(lhs, dtype=jnp.float32),
        jnp.asarray(q, dtype=jnp.int4),
        group_sizes,
        rhs_scale=jnp.asarray(scale),
        rhs_groupbias=jnp.asarray(gbias),
        maybe_quantize_lhs=False,
    )
    out_np = np.asarray(out, dtype=np.float32)

    assert np.isfinite(out_np).all(), (
        "gmm_v2 produced non-finite values (NaN/Inf) for non-lane-multiple "
        f"size_k={K}: {np.count_nonzero(~np.isfinite(out_np))} bad entries "
        "(out-of-bounds per-group scale/groupbias read).")
    np.testing.assert_allclose(out_np, ref, atol=1e-1, rtol=1e-1)


@pytest.mark.skipif(not any(d.platform == "tpu" for d in jax.devices()),
                    reason="requires TPU")
def test_gmm_v2_groupbias_quantized_lhs_delta_matches_analytic_bias():
    """Quantized-lhs k-loop site: with-minus-without equals the analytic bias."""
    G, M, K, N = 2, 128, 512, 256
    gs = K  # single quant block -> b_id maps cleanly on the quantized path
    num_blocks = K // gs
    rng = np.random.default_rng(5)

    lhs = rng.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    q = rng.integers(-8, 8, size=(G, K, N)).astype(np.int32)
    scale = (rng.uniform(-1.0, 1.0, size=(G, num_blocks, 1, N)) *
             0.05).astype(np.float32)
    gbias = (rng.uniform(-1.0, 1.0, size=(G, num_blocks, 1, N)) *
             0.3).astype(np.float32)
    group_sizes = jnp.array([M // 2, M - M // 2], dtype=jnp.int32)

    lhs_j = jnp.asarray(lhs, dtype=jnp.float32)
    q_j = jnp.asarray(q, dtype=jnp.int4)

    out_no_bias = gmm_v2(
        lhs_j,
        q_j,
        group_sizes,
        rhs_scale=jnp.asarray(scale),
        maybe_quantize_lhs=True,
    )
    out_bias = gmm_v2(
        lhs_j,
        q_j,
        group_sizes,
        rhs_scale=jnp.asarray(scale),
        rhs_groupbias=jnp.asarray(gbias),
        maybe_quantize_lhs=True,
    )

    delta = np.asarray(out_bias, np.float32) - np.asarray(out_no_bias,
                                                          np.float32)
    expected_bias = _bias_contribution(lhs, gbias, np.asarray(group_sizes), gs)

    np.testing.assert_allclose(delta, expected_bias, atol=1e-1, rtol=1e-1)
