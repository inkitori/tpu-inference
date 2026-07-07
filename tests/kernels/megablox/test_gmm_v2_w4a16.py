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
"""Tests the gmm_v2 W4A16 path (unquantized lhs, per-group rhs scale).

The unquantized-lhs path folds the per-quant-block rhs scale into the
dequantized rhs so each n-chunk runs one full-tile_k matmul:

    out[t, n] = sum_k lhs[t, k] * q[k, n] * scale[g(k), n]
              = lhs @ (q * scale_broadcast)

Calling convention (matches tests/kernels/gmm_test.py): ``rhs`` (int4 codes)
is passed as a jnp.int4 array directly; gmm_v2 packs it in-kernel. Codes are
signed int4 in [-8, 8).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.megablox.gmm_v2 import (calculate_tiling, gmm_v2,
                                                   make_gmm_configs)

requires_tpu = pytest.mark.skipif(
    not any(d.platform == "tpu" for d in jax.devices()), reason="requires TPU")


def _dequant_reference(lhs, q, scale, group_sizes, gs):
    """Dequantize (w = q*scale) then grouped matmul, in f32."""
    G, K, N = q.shape
    M = lhs.shape[0]
    ref = np.zeros((M, N), np.float32)
    row = 0
    for g in range(G):
        n = int(group_sizes[g])
        s = np.repeat(scale[g, :, 0, :], gs, axis=0)  # [K, N]
        w = q[g].astype(np.float32) * s  # [K, N]
        ref[row:row + n] = lhs[row:row + n].astype(np.float32) @ w
        row += n
    return ref


def _make_case(G, M, K, N, gs, seed):
    num_blocks = K // gs
    rng = np.random.default_rng(seed)
    lhs = rng.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    q = rng.integers(-8, 8, size=(G, K, N)).astype(np.int32)
    scale = (rng.uniform(-1.0, 1.0, size=(G, num_blocks, 1, N)) *
             0.05).astype(np.float32)
    group_sizes = np.array([M // 2, M - M // 2], dtype=np.int32)
    return lhs, q, scale, group_sizes


@requires_tpu
@pytest.mark.parametrize(
    "gs,K,N",
    [
        (64, 512, 256),  # fine-grained blocks (gs < mxu_size)
        (256, 512, 256),  # coarse blocks (gs >= mxu_size on v6e)
        (64, 1536, 2048),  # down_proj-like shape, 24 blocks
    ])
def test_w4a16_scale_fold_matches_dequant_reference(gs, K, N):
    G, M = 2, 128
    lhs, q, scale, group_sizes = _make_case(G, M, K, N, gs, seed=0)
    ref = _dequant_reference(lhs, q, scale, group_sizes, gs)

    out = gmm_v2(
        jnp.asarray(lhs, dtype=jnp.bfloat16),
        jnp.asarray(q, dtype=jnp.int4),
        jnp.asarray(group_sizes),
        rhs_scale=jnp.asarray(scale),
        maybe_quantize_lhs=False,
    )

    # bf16 lhs + bf16 dequant product, fp32 accumulation.
    np.testing.assert_allclose(np.asarray(out, dtype=np.float32),
                               ref,
                               atol=2e-1,
                               rtol=2e-1)
    rel_l2 = (np.linalg.norm(np.asarray(out, np.float32) - ref) /
              np.linalg.norm(ref))
    assert rel_l2 < 2e-2, f"rel L2 {rel_l2} too large"


@requires_tpu
def test_non_lane_multiple_size_k_does_not_overread_quant_blocks():
    """size_k NOT a multiple of num_lanes (128) must not read the per-group
    scale out of bounds.

    E.g. a down_proj at tp=8: per-shard size_k = 1536/8 = 192.
    calculate_tiling over-aligns tile_k to 256, so
    num_quant_blocks_per_tile_k = cdiv(256, 64) = 4 while the scale axis has
    only cdiv(192, 64) = 3 blocks. The scale BlockSpec would DMA block range
    [0:4] from a 3-long axis (disable_bounds_checks=True) and the kernel would
    read the OOB block. num_quant_blocks_per_tile_k_read clamps the DMA/index
    count to the real remaining blocks.
    """
    G, M, K, N, gs = 2, 16, 192, 256, 64
    lhs, q, scale, group_sizes = _make_case(G, M, K, N, gs, seed=7)
    group_offset = jnp.array([0], dtype=jnp.int32)

    cfgs = make_gmm_configs(jnp.asarray(lhs),
                            jnp.asarray(q, dtype=jnp.int4),
                            jnp.asarray(scale),
                            None,
                            None,
                            jnp.asarray(group_sizes),
                            group_offset,
                            tile_info=calculate_tiling,
                            vmem_limit_bytes=128 * 1024 * 1024,
                            out_dtype=None,
                            acc_dtype=None,
                            maybe_quantize_lhs=False,
                            zero_initialize=True,
                            fuse_act=None)

    real_blocks = scale.shape[1]  # 3
    assert cfgs.num_quant_blocks_per_tile_k_read <= real_blocks

    # And end-to-end: output must be finite and match the reference.
    ref = _dequant_reference(lhs, q, scale, group_sizes, gs)
    out = np.asarray(gmm_v2(
        jnp.asarray(lhs, dtype=jnp.float32),
        jnp.asarray(q, dtype=jnp.int4),
        jnp.asarray(group_sizes),
        rhs_scale=jnp.asarray(scale),
        maybe_quantize_lhs=False,
    ),
                     dtype=np.float32)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, atol=1e-1, rtol=1e-1)
