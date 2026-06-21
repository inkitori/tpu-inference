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
"""Task 10: forward orchestration + wo_a BMM-layout dequant.

Two layers of coverage:

* Static guards (cheap regression tripwires): ``forward`` is no longer the
  pass-through stub and ``process_weights_after_loading`` is no longer a bare
  ``pass``.
* Real numerical parity tests (CLAUDE.md hard requirement -- synthetic weights
  in the real quant formats, compared against the pure-torch reference in
  ``tests/dsv4/torch_ref``):
    - the q-only fused norm/RoPE helper vs ``rmsnorm_no_weight`` +
      ``apply_rope_gptj_last_k`` (FORWARD GPT-J RoPE, ``+sin``);
    - the FP8 + ue8m0 ``wo_a`` dequant (``_expand_wo_a_block_scales`` and
      ``process_weights_after_loading``) vs a hand-computed reference, using a
      synthetic ``SimpleNamespace`` ``wo_a`` stub (``float8_e4m3fn`` weight +
      ue8m0 ``weight_scale_inv``).

We do NOT instantiate the full attention module (it needs a built model -- that
end-to-end path is the Task-12 gate). The numerical tests drive the standalone
math methods on lightweight stubs.
"""
import inspect
from types import SimpleNamespace

import torch

from tests.dsv4 import torch_ref
from tpu_inference.layers.vllm.custom_ops.deepseek_v4_attention import \
    VllmDeepseekV4MLAAttention


# --------------------------------------------------------------------------- #
# Static guards (brief Step 1)
# --------------------------------------------------------------------------- #
def test_forward_is_not_passthrough_stub():
    src = inspect.getsource(VllmDeepseekV4MLAAttention.forward)
    assert "just a pass-through" not in src, "forward is still the stub"
    assert "return hidden_states" not in src.split("def forward")[1][:400], \
        "forward must not return hidden_states unchanged"
    # must call the two TPU ops
    assert "forward_mqa" in src
    assert "_o_proj" in src


def test_pwal_no_longer_bare_pass():
    src = inspect.getsource(
        VllmDeepseekV4MLAAttention.process_weights_after_loading)
    body = src.split(":", 1)[1].strip()
    assert body != "pass", "process_weights_after_loading is still a no-op"


# --------------------------------------------------------------------------- #
# Numerical test 1: q-only fused norm/RoPE helper
# --------------------------------------------------------------------------- #
def test_fused_qnorm_rope_q_only_matches_reference():
    """q-side weight-free RMSNorm + FORWARD GPT-J RoPE on q AND kv."""
    torch.manual_seed(0)
    N, n_heads, head_dim, rope_dim = 4, 6, 32, 8
    eps = 1e-6
    max_pos = 16

    q = torch.randn(N, n_heads, head_dim, dtype=torch.bfloat16)
    kv = torch.randn(N, head_dim, dtype=torch.bfloat16)
    positions = torch.arange(N, dtype=torch.long)
    cos_sin_cache = torch_ref.build_cos_sin_cache(rope_dim, max_pos,
                                                  theta=10000.0)

    # Drive the real method via a lightweight stub carrying only the attributes
    # it reads (rope_head_dim, eps, rotary_emb.cos_sin_cache).
    stub = SimpleNamespace(
        rope_head_dim=rope_dim,
        eps=eps,
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin_cache),
    )
    q_out, kv_out = VllmDeepseekV4MLAAttention._fused_qnorm_rope_kv_insert_q_only(
        stub, q, kv, positions)

    # Reference: weight-free per-head RMSNorm (fp32, then cast happens via the
    # RoPE helper's final cast), then forward GPT-J RoPE on the last rope_dim.
    q_norm_ref = torch_ref.rmsnorm_no_weight(q, eps)  # fp32 [N, n_heads, head_dim]
    q_ref = torch_ref.apply_rope_gptj_last_k(
        q_norm_ref.to(q.dtype), positions, cos_sin_cache, rope_dim)
    # kv has no head axis; add one for the [T, H, D] helper, then drop it.
    kv_ref = torch_ref.apply_rope_gptj_last_k(
        kv.unsqueeze(1), positions, cos_sin_cache, rope_dim).squeeze(1)

    assert q_out.dtype == q.dtype and kv_out.dtype == kv.dtype
    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(kv_out, kv_ref, atol=2e-2, rtol=2e-2)

    # NoPE dims (before -rope_dim) must be the normalized-but-unrotated q.
    nope = head_dim - rope_dim
    torch.testing.assert_close(
        q_out[..., :nope].float(),
        q_norm_ref[..., :nope].to(q.dtype).float(),
        atol=2e-2, rtol=2e-2)
    # kv NoPE dims must pass through the norm-free path unchanged (rope only).
    torch.testing.assert_close(kv_out[..., :nope], kv[..., :nope])


def test_weighted_rmsnorm_matches_reference():
    """Inline weighted RMSNorm == torch_ref.rmsnorm_with_weight (no 1+w)."""
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    w = torch.randn(32, dtype=torch.bfloat16)
    got = VllmDeepseekV4MLAAttention._weighted_rmsnorm(x, w, eps)
    ref = torch_ref.rmsnorm_with_weight(x, w, eps)
    assert got.dtype == x.dtype
    torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------- #
# Numerical test 2: FP8 + ue8m0 wo_a dequant
# --------------------------------------------------------------------------- #
def test_wo_a_fp8_ue8m0_dequant_matches_reference():
    """process_weights_after_loading dequant of the FP8-block wo_a weight.

    Synthetic wo_a stub in the REAL quant formats, in the TPU-transposed
    (in, out) = (d, g*r) layout the linear-method PWAL produces: float8_e4m3fn
    2D weight (d, g*r) with a ue8m0 (uint8) block-scale weight_scale_inv shaped
    (d_blocks, g*r_blocks). process_weights_after_loading must .t() both back to
    the GPU (g*r, d) / (g*r_blocks, d_blocks) orientation before the (g, r, d)
    view + block-scale dequant. Compare against a hand-computed reference.

    NB: on a real TPU FP8 build the FP8 linear PWAL deletes weight_scale_inv and
    re-stores the requantized scale as weight_scale (fp8.py), so this hasattr-
    weight_scale_inv branch is exercised only by hand-built parity stubs like
    this one. The else (plain transposed view+cast) branch is what the live
    synthetic build reaches; see test_wo_a_dequant_else_branch_view_cast.
    """
    torch.manual_seed(0)
    g, r, d = 2, 8, 16
    block = 4  # r and d are exact multiples of the block size
    r_blocks, d_blocks = r // block, d // block

    # FP8 e4m3 weight, 2D TPU layout (d, g*r). Round random bf16 through the fp8
    # dtype so the stored values are exactly representable in fp8 (no double
    # rounding). The GPU-orientation w3 (g, r, d) is recovered by .t().view.
    w3_full = torch.randn(g, r, d) * 0.5            # logical GPU [g, r, d]
    w_gr_d = w3_full.reshape(g * r, d)              # GPU (g*r, d)
    w_tpu = w_gr_d.t().contiguous()                 # TPU (d, g*r)
    w_fp8 = w_tpu.to(torch.float8_e4m3fn)

    # ue8m0 (uint8) block scales: small biased exponents around 127 (== 2**0).
    # Stored 2D in the TPU-transposed (d_blocks, g*r_blocks) orientation; the
    # production code does weight_scale_inv.t().contiguous().view(g, -1, ...).
    scale_gr_d = torch.randint(120, 132, (g * r_blocks, d_blocks),
                               dtype=torch.uint8)    # GPU (g*r_blocks, d_blocks)
    scale_u8 = scale_gr_d.t().contiguous()           # TPU (d_blocks, g*r_blocks)

    wo_a = SimpleNamespace(
        weight=w_fp8,
        weight_scale_inv=scale_u8,
        bmm_batch_size=g,
    )
    # process_weights_after_loading calls self._expand_wo_a_block_scales; bind
    # the real (unbound) method onto the stub so the nested call resolves.
    stub = SimpleNamespace(
        wo_a=wo_a,
        o_lora_rank=r,
        _expand_wo_a_block_scales=(
            lambda scale, rr, dd:
            VllmDeepseekV4MLAAttention._expand_wo_a_block_scales(
                None, scale, rr, dd)),
    )

    VllmDeepseekV4MLAAttention.process_weights_after_loading(stub)
    got = stub.wo_a_bf16

    # Hand-computed reference dequant, from the recovered GPU orientation.
    w3 = w_fp8.t().contiguous().view(g, r, d).to(torch.float32)
    # ue8m0 byte -> fp32 power of two: (byte << 23) reinterpreted as float32,
    # i.e. 2**(byte - 127). Matches torch_ref.upcast_e8m0_to_fp32.
    scale_fp32 = torch_ref.upcast_e8m0_to_fp32(
        scale_u8.t().contiguous().view(g, r_blocks, d_blocks))
    scale_exp = scale_fp32.repeat_interleave(block, dim=1) \
                          .repeat_interleave(block, dim=2)
    ref = (w3 * scale_exp).to(torch.bfloat16)

    assert got.shape == (g, r, d)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)


def test_wo_a_dequant_else_branch_view_cast():
    """Non-quantized (synthetic) build with no weight_scale_inv: transpose + view + cast.

    The TPU linear-method PWAL transposes wo_a.weight to (in, out) = (d, g*r)
    before process_weights_after_loading runs, so the stored weight is (d, g*r)
    -- NOT the GPU (g*r, d) layout. process_weights_after_loading must .t() it
    back to (g*r, d), then view (g, r, d). This orientation is exactly what the
    Task-12 integration crux surfaced; building the GPU layout here hid the bug.
    """
    torch.manual_seed(0)
    g, r, d = 3, 4, 8
    w = torch.randn(d, g * r, dtype=torch.bfloat16)  # TPU (in, out) = (d, g*r)
    wo_a = SimpleNamespace(weight=w, bmm_batch_size=g)
    stub = SimpleNamespace(wo_a=wo_a, o_lora_rank=r)

    VllmDeepseekV4MLAAttention.process_weights_after_loading(stub)
    got = stub.wo_a_bf16

    assert got.shape == (g, r, d)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, w.t().contiguous().view(g, r, d))


def test_expand_wo_a_block_scales_repeats_blocks():
    """_expand_wo_a_block_scales: ue8m0 decode + per-block repeat_interleave."""
    g, r, d = 2, 6, 12
    block = 3
    r_blocks, d_blocks = r // block, d // block
    scale_u8 = torch.randint(120, 132, (g, r_blocks, d_blocks),
                             dtype=torch.uint8)

    out = VllmDeepseekV4MLAAttention._expand_wo_a_block_scales(
        None, scale_u8, r, d)

    assert out.shape == (g, r, d)
    assert out.dtype == torch.float32
    ref = torch_ref.upcast_e8m0_to_fp32(scale_u8) \
        .repeat_interleave(block, dim=1).repeat_interleave(block, dim=2)
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)
