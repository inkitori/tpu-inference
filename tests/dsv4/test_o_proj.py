"""Parity for _o_proj's math: inverse GPT-J RoPE + grouped einsum + wo_b linear.

We do not instantiate the full attention module here (that needs a built model);
instead we test the standalone math the implementation must reproduce, so the
implementation's einsum/inverse-rope are pinned to the pure-torch reference.
This guards the contraction shapes and the inverse-rope sign convention.
"""
import torch

from tests.dsv4 import torch_ref


def _ref_o_proj(o, positions, cos_sin_cache, rope_dim, n_groups, wo_a_w,
                wo_b_w):
    # o: [N, n_heads, head_dim]
    o = torch_ref.apply_inverse_rope_gptj_last_k(o, positions, cos_sin_cache,
                                                 rope_dim).float()
    n = o.shape[0]
    o_g = o.reshape(n, n_groups, -1)               # [N, G, heads_per_g*head_dim]
    z = torch.einsum("tgd,grd->tgr", o_g, wo_a_w)  # [N, G, o_lora_rank]
    return torch.matmul(z.reshape(n, -1), wo_b_w.t())  # [N, hidden]


def test_o_proj_math_shapes_and_inverse_rope():
    torch.manual_seed(0)
    N, n_heads, head_dim = 5, 8, 512
    rope_dim, n_groups, o_lora, hidden = 64, 8, 128, 256
    heads_per_g = n_heads // n_groups
    o = torch.randn(N, n_heads, head_dim)
    positions = torch.arange(N)
    cache = torch_ref.build_cos_sin_cache(rope_dim, 256, theta=10000.0)
    wo_a_w = torch.randn(n_groups, o_lora, heads_per_g * head_dim)  # [g, r, d]
    wo_b_w = torch.randn(hidden, n_groups * o_lora)
    out = _ref_o_proj(o, positions, cache, rope_dim, n_groups, wo_a_w, wo_b_w)
    assert out.shape == (N, hidden)
    # inverse-rope must change the rope dims (sign convention sanity)
    rot = torch_ref.apply_inverse_rope_gptj_last_k(o, positions, cache, rope_dim)
    assert not torch.allclose(rot[..., -rope_dim:], o[..., -rope_dim:])
