import torch
import torch.nn.functional as F

from tests.dsv4 import moe_ref


def test_router_selects_topk_and_scales():
    torch.manual_seed(0)
    n_tok, n_exp, topk = 4, 16, 6
    gating = torch.randn(n_tok, n_exp)
    bias = torch.randn(n_exp)
    w, idx = moe_ref.topk_softplus_sqrt(
        gating, bias, topk=topk, renormalize=True, routed_scaling_factor=2.5)
    assert idx.shape == (n_tok, topk)
    assert w.shape == (n_tok, topk)
    # exactly topk distinct experts per token
    for r in range(n_tok):
        assert len(set(idx[r].tolist())) == topk
    # selection uses biased scores; weights come from UNBIASED scores
    scores = torch.sqrt(F.softplus(gating.float()))
    _, expected_idx = torch.topk(scores + bias.float(), k=topk, dim=-1)
    torch.testing.assert_close(idx, expected_idx)
    # renormalized then scaled by routed_scaling_factor
    raw = scores.gather(1, expected_idx)
    expected_w = raw / raw.sum(-1, keepdim=True).clamp(min=1e-20) * 2.5
    torch.testing.assert_close(w, expected_w, rtol=1e-5, atol=1e-5)


def test_fp4_expert_dequant_block_scaled():
    # one block (block_size=2 here for test): bytes -> e2m1 values * e8m0 scale
    packed = torch.tensor([[0x21]], dtype=torch.uint8)   # -> [0.5, 1.0]
    scale = torch.tensor([[128]], dtype=torch.uint8)     # e8m0 128 -> 2.0
    out = moe_ref.dequant_fp4_expert(packed, scale, block_size=2,
                                     out_dtype=torch.float32)
    torch.testing.assert_close(out, torch.tensor([[1.0, 2.0]]))
