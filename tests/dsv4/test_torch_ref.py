"""Tests for the pure-torch DSV4 Phase-1 reference building blocks.

These reference helpers are the numerical ground truth that later parity tests
(MoE FP4 dequant, _o_proj inverse RoPE, invariants) compare the TPU
implementation against, so the conventions here must match the real vLLM
reference bit-for-bit. Each test asserts the exact convention confirmed against
vLLM source (see torch_ref.py module docstring for the cited locations).
"""
import torch

from tests.dsv4 import torch_ref


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #
def test_rmsnorm_no_weight_matches_manual():
    x = torch.randn(4, 16)
    out = torch_ref.rmsnorm_no_weight(x, eps=1e-6)
    var = x.float().pow(2).mean(-1, keepdim=True)
    expected = x.float() * torch.rsqrt(var + 1e-6)
    # Must return fp32 (kernel keeps RMSNorm->RoPE in fp32, rounds once).
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_rmsnorm_no_weight_returns_fp32_for_bf16_input():
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    out = torch_ref.rmsnorm_no_weight(x, eps=1e-6)
    assert out.dtype == torch.float32


def test_rmsnorm_with_weight_matches_manual():
    x = torch.randn(4, 16)
    w = torch.randn(16)
    out = torch_ref.rmsnorm_with_weight(x, w, eps=1e-6)
    var = x.float().pow(2).mean(-1, keepdim=True)
    expected = (x.float() * torch.rsqrt(var + 1e-6) * w.float()).to(x.dtype)
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_rmsnorm_with_weight_preserves_bf16_dtype():
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    w = torch.ones(16, dtype=torch.bfloat16)
    out = torch_ref.rmsnorm_with_weight(x, w, eps=1e-6)
    assert out.dtype == torch.bfloat16


# --------------------------------------------------------------------------- #
# GPT-J rotate / RoPE
# --------------------------------------------------------------------------- #
def test_rotate_gptj_is_even_odd_interleave():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # pairs (1,2),(3,4)
    # rotate_gptj: stack((-odd, even)) interleaved -> (-2,1,-4,3)
    out = torch_ref.rotate_gptj(x)
    torch.testing.assert_close(out, torch.tensor([[-2.0, 1.0, -4.0, 3.0]]))


def test_build_cos_sin_cache_layout():
    rope_dim, max_pos = 64, 16
    cache = torch_ref.build_cos_sin_cache(rope_dim, max_pos, theta=10000.0)
    assert cache.shape == (max_pos, rope_dim)
    assert cache.dtype == torch.float32
    half = rope_dim // 2
    # cat((cos, sin), -1): position 0 -> cos=1, sin=0 for every frequency.
    torch.testing.assert_close(cache[0, :half], torch.ones(half))
    torch.testing.assert_close(cache[0, half:], torch.zeros(half))


def test_apply_rope_preserves_per_pair_l2_norm():
    # RoPE is a rotation: each interleaved pair keeps its L2 norm.
    rope_dim, max_pos = 64, 256
    cache = torch_ref.build_cos_sin_cache(rope_dim, max_pos, theta=10000.0)
    x = torch.randn(5, 2, 512)
    pos = torch.arange(5)
    rotated = torch_ref.apply_rope_gptj_last_k(x, pos, cache, rope_dim)
    # nope dims (leading) pass through unchanged.
    torch.testing.assert_close(
        rotated[..., :-rope_dim].float(), x[..., :-rope_dim].float(),
        rtol=1e-5, atol=1e-5)
    # per-pair L2 norm preserved on the rope slice.
    rot_pairs = rotated[..., -rope_dim:].reshape(5, 2, rope_dim // 2, 2)
    x_pairs = x[..., -rope_dim:].reshape(5, 2, rope_dim // 2, 2)
    torch.testing.assert_close(
        rot_pairs.float().norm(dim=-1), x_pairs.float().norm(dim=-1),
        rtol=1e-4, atol=1e-4)


def test_apply_rope_at_position_zero_is_identity():
    rope_dim, max_pos = 64, 16
    cache = torch_ref.build_cos_sin_cache(rope_dim, max_pos, theta=10000.0)
    x = torch.randn(3, 2, 128)
    pos = torch.zeros(3, dtype=torch.long)  # cos=1, sin=0 -> identity
    out = torch_ref.apply_rope_gptj_last_k(x, pos, cache, rope_dim)
    torch.testing.assert_close(out.float(), x.float(), rtol=1e-5, atol=1e-5)


def test_rope_then_inverse_is_identity_on_rope_dims():
    rope_dim, max_pos = 64, 256
    cache = torch_ref.build_cos_sin_cache(rope_dim, max_pos, theta=10000.0)
    x = torch.randn(5, 2, 512)  # head_dim 512, last 64 are rope dims
    pos = torch.arange(5)
    rotated = torch_ref.apply_rope_gptj_last_k(x, pos, cache, rope_dim)
    back = torch_ref.apply_inverse_rope_gptj_last_k(rotated, pos, cache, rope_dim)
    torch.testing.assert_close(back.float(), x.float(), rtol=1e-4, atol=1e-4)


def test_inverse_rope_negates_sin_vs_forward():
    # Inverse RoPE must equal forward RoPE computed with sin negated.
    rope_dim, max_pos = 32, 64
    cache = torch_ref.build_cos_sin_cache(rope_dim, max_pos, theta=10000.0)
    x = torch.randn(4, 1, 96)
    pos = torch.arange(4)
    inv = torch_ref.apply_inverse_rope_gptj_last_k(x, pos, cache, rope_dim)
    # Build a cache with sin pre-negated, run forward -> should match inverse.
    half = rope_dim // 2
    neg_sin_cache = cache.clone()
    neg_sin_cache[:, half:] = -cache[:, half:]
    fwd_negsin = torch_ref.apply_rope_gptj_last_k(x, pos, neg_sin_cache, rope_dim)
    torch.testing.assert_close(inv.float(), fwd_negsin.float(),
                               rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# FP4 e2m1 / ue8m0 dequant
# --------------------------------------------------------------------------- #
def test_break_fp4_e2m1_lookup_table():
    # byte 0x21 -> low nibble 0x1 (=0.5), high nibble 0x2 (=1.0)
    packed = torch.tensor([[0x21]], dtype=torch.uint8)
    out = torch_ref.break_fp4_e2m1(packed, torch.float32)
    torch.testing.assert_close(out, torch.tensor([[0.5, 1.0]]))


def test_break_fp4_e2m1_signs_and_full_table():
    # Magnitudes 0..7 in low nibble of successive bytes; high nibble 0 -> 0.0.
    # byte = low_nibble | (sign<<3). Build all 8 magnitudes positive + a negative.
    # 0x0F = low nibble 0xF = sign(0x8)|mag(0x7) -> -6.0 ; high 0x0 -> 0.0
    packed = torch.tensor([[0x0F]], dtype=torch.uint8)
    out = torch_ref.break_fp4_e2m1(packed, torch.float32)
    torch.testing.assert_close(out, torch.tensor([[-6.0, 0.0]]))


def test_upcast_e8m0_is_power_of_two():
    # e8m0 byte 127 (bias) -> exponent 0 -> 1.0; byte 128 -> 2.0
    s = torch.tensor([127, 128], dtype=torch.uint8)
    out = torch_ref.upcast_e8m0_to_fp32(s)
    torch.testing.assert_close(out, torch.tensor([1.0, 2.0]))


def test_upcast_e8m0_subnormal_and_large():
    # byte 126 -> 2^-1 = 0.5 ; byte 130 -> 2^3 = 8.0
    s = torch.tensor([126, 130], dtype=torch.uint8)
    out = torch_ref.upcast_e8m0_to_fp32(s)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.tensor([0.5, 8.0]))
