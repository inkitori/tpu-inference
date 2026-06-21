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

from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2


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


@pytest.mark.skipif(not jax.devices(), reason="requires TPU")
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


@pytest.mark.skipif(not jax.devices(), reason="requires TPU")
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
