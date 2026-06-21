"""Pure-torch MoE router (sqrtsoftplus + noaux_tc) + FP4 expert dequant reference.

Numerical ground truth for DSV4 Phase-1 MoE/router parity (spec §7.4 step 2).
Later end-to-end tests (T12/T14) lean on this, so the selection/weighting logic
must match the real vLLM implementation exactly.

Verified against vLLM source (editable install at /home/enyouki/vllm):

* ``topk_softplus_sqrt`` mirrors ``_topk_softplus_sqrt_torch`` at
  ``model_executor/layers/fused_moe/router/fused_topk_bias_router.py:59-102``:
    - ``scores = sqrt(softplus(gating.float()))`` (line 72)
    - bias added for SELECTION ONLY: ``scores_for_choice = scores + bias`` (78)
    - top-k taken on the BIASED scores: ``topk(scores_for_choice, ...)`` (93)
    - weights gathered from the UNBIASED scores: ``scores.gather(...)`` (96)
    - renormalize ``weights / weights.sum().clamp(min=1e-20)`` (99)
    - then ``* routed_scaling_factor`` (101), applied regardless of renormalize.
  (The noaux_tc softplus_sqrt router has no group-limited grouping, so none is
  applied here.)
* ``dequant_fp4_expert`` block-scaled FP4->bf16/fp32 uses Task 5's
  ``break_fp4_e2m1`` (e2m1 table ``[0,.5,1,1.5,2,3,4,6]``, low nibble first) and
  ``upcast_e8m0_to_fp32`` (value = ``2**(byte-127)``) — the same e2m1 decode the
  GMM dequant-in-VMEM branch performs (spec §3).
"""
import torch
import torch.nn.functional as F

from tests.dsv4.torch_ref import break_fp4_e2m1, upcast_e8m0_to_fp32


def topk_softplus_sqrt(gating, e_score_correction_bias, topk, renormalize,
                       routed_scaling_factor):
    """sqrtsoftplus + noaux_tc top-k router reference.

    Returns ``(topk_weights, topk_indices)``. Experts are SELECTED with the
    bias-adjusted scores; combine WEIGHTS come from the UNBIASED scores, then
    (optionally) renormalized and always scaled by ``routed_scaling_factor``.
    """
    scores = torch.sqrt(F.softplus(gating.float()))
    scores_for_choice = scores
    if e_score_correction_bias is not None:
        # Bias is used for expert SELECTION only, not for weight computation.
        scores_for_choice = scores + e_score_correction_bias.float()
    _, indices = torch.topk(scores_for_choice, k=topk, dim=-1)
    weights = scores.gather(1, indices)  # weights from UNBIASED scores
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    weights = weights * routed_scaling_factor
    return weights, indices


def dequant_fp4_expert(packed_u8, e8m0_scale_u8, block_size, out_dtype):
    """Block-scaled FP4 (e2m1) -> ``out_dtype`` dequant.

    ``packed_u8``: ``[rows, cols]`` uint8 with 2 e2m1 values per byte (low
    nibble first), decoding to ``[rows, cols*2]``. ``e8m0_scale_u8``:
    ``[rows, num_blocks]`` ue8m0 per-block scales, one scale per ``block_size``
    consecutive (decoded) values. Mirrors the GMM dequant-in-VMEM math.
    """
    vals = break_fp4_e2m1(packed_u8, torch.float32)  # [rows, cols*2]
    scales = upcast_e8m0_to_fp32(e8m0_scale_u8)      # [rows, num_blocks]
    rows, n = vals.shape
    num_blocks = n // block_size
    vals = vals.reshape(rows, num_blocks, block_size)
    vals = vals * scales.reshape(rows, num_blocks, 1)
    return vals.reshape(rows, n).to(out_dtype)
