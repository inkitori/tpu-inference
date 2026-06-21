import jax.numpy as jnp
import numpy as np
import pytest

from tests.utils.mlx_synthetic import _quantize_affine
from tpu_inference.layers.common.quantization import mlx_unpack, mlx_dequantize


@pytest.mark.parametrize("force_neg", [False, True])
def test_mlx_dequantize_matches_affine_golden(force_neg):
    rng = np.random.default_rng(0)
    out_features, in_features, group_size = 16, 128, 64
    w = rng.standard_normal((out_features, in_features)).astype(np.float32)
    packed, scales, biases, golden = _quantize_affine(w, group_size, force_neg)

    got = np.asarray(mlx_dequantize(
        jnp.asarray(packed), jnp.asarray(scales), jnp.asarray(biases),
        group_size=group_size, bits=4)).astype(np.float32)

    # Exact reconstruction of the *quantized* (golden) weights, modulo bf16 rounding.
    np.testing.assert_allclose(got, golden.astype(np.float32), atol=1e-2, rtol=1e-2)


def _pack_u4(vals_row):
    # vals_row: list[int] length multiple of 8 -> list[uint32], element 0 = low nibble
    words = []
    for i in range(0, len(vals_row), 8):
        w = 0
        for k in range(8):
            w |= (vals_row[i + k] & 0xF) << (4 * k)
        words.append(w)
    return words


def test_unpack_low_nibble_first():
    # word with nibbles 0..7 in order -> 0x76543210
    packed = jnp.asarray([[0x76543210]], dtype=jnp.uint32)
    out = mlx_unpack(packed, bits=4)
    np.testing.assert_array_equal(np.asarray(out), [[0, 1, 2, 3, 4, 5, 6, 7]])


def test_dequantize_affine_with_negative_scale_and_bias():
    # one output row, in=64 (one group). q = i % 16.
    q = [(i % 16) for i in range(64)]
    packed = jnp.asarray([_pack_u4(q)], dtype=jnp.uint32)           # [1, 8]
    scales = jnp.asarray([[-0.5]], dtype=jnp.bfloat16)              # [1, 1]
    biases = jnp.asarray([[3.25]], dtype=jnp.bfloat16)             # [1, 1]
    w = mlx_dequantize(packed, scales, biases, group_size=64, bits=4)
    expected = (np.asarray(q, dtype=np.float32) * -0.5) + 3.25
    np.testing.assert_allclose(np.asarray(w, dtype=np.float32), expected[None, :], atol=0.05)


def test_dequantize_two_groups_distinct_scales():
    q = [(i % 16) for i in range(128)]                              # in=128 -> 2 groups
    packed = jnp.asarray([_pack_u4(q)], dtype=jnp.uint32)           # [1, 16]
    scales = jnp.asarray([[0.5, 2.0]], dtype=jnp.bfloat16)          # [1, 2]
    biases = jnp.asarray([[0.0, 1.0]], dtype=jnp.bfloat16)         # [1, 2]
    w = mlx_dequantize(packed, scales, biases, group_size=64, bits=4)
    q_np = np.asarray(q, dtype=np.float32)
    expected = np.concatenate([q_np[:64] * 0.5 + 0.0, q_np[64:] * 2.0 + 1.0])
    np.testing.assert_allclose(np.asarray(w, dtype=np.float32), expected[None, :], atol=0.05)


def test_dequantize_stacked_experts_leading_dim():
    q = [(i % 16) for i in range(64)]
    packed = jnp.asarray([[_pack_u4(q)], [_pack_u4(q)]], dtype=jnp.uint32)  # [E=2, out=1, 8]
    scales = jnp.asarray([[[1.0]], [[-1.0]]], dtype=jnp.bfloat16)           # [2, 1, 1]
    biases = jnp.asarray([[[0.0]], [[5.0]]], dtype=jnp.bfloat16)            # [2, 1, 1]
    w = mlx_dequantize(packed, scales, biases, group_size=64, bits=4)
    assert w.shape == (2, 1, 64)
    q_np = np.asarray(q, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(w[0], dtype=np.float32), q_np[None, :], atol=0.05)
    np.testing.assert_allclose(np.asarray(w[1], dtype=np.float32), (q_np * -1.0 + 5.0)[None, :], atol=0.05)


def test_matches_mlx_core_if_available():
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 128)).astype(np.float32)
    wq, s, b = mx.quantize(mx.array(w), group_size=64, bits=4)
    ref = np.asarray(mx.dequantize(wq, s, b, group_size=64, bits=4))
    ours = np.asarray(mlx_dequantize(
        jnp.asarray(np.asarray(wq).view(np.uint32)),
        jnp.asarray(np.asarray(s)), jnp.asarray(np.asarray(b)),
        group_size=64, bits=4), dtype=np.float32)
    np.testing.assert_allclose(ours, ref, atol=0.05)
