# Copyright 2025 Google LLC
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
"""Fast CPU-only method-level tests for ``VllmMLXMoEMethod``.

Covers what the TPU e2e cannot isolate: the stacked-param ``create_weights``
shapes and the per-expert un-stacking/registration path -- synthetic per-expert
4-bit packs are loaded through vLLM's REAL ``FusedMoE.weight_loader`` (gate->w1,
up->w3, down->w2) and the custom MLX bias loader, then dequantized by
``process_weights_after_loading``, and the result is compared to golden experts.

CPU-only: ``JAX_PLATFORMS=cpu`` is forced before any jax import. With one CPU
device the MoE backend resolves to GMM_TP and the whole load + dequant +
``process_unquantized_moe_weights`` / ``shard_moe_weights`` path runs on CPU.
Only the final MoE forward kernel (``apply_monolithic``) is TPU-bound, so it is
NOT called here; correctness of the new MLX logic (dequant + un-stacking) is
proved instead by feeding the SAME golden bf16 experts through the SAME
unquantized processing path and asserting bit-for-bit-close stacked weights.

NOTE (param names): the brief's Step 2 names the stacked params
``w13_qweight``/``w2_qweight``, but the actual implementation in ``mlx.py``
registers ``w13_weight``/``w2_weight`` (plus ``w13_scales``/``w2_scales`` and
``w13_biases``/``w2_biases``); these tests assert the real names. The shapes
match the brief: w13 [E, 2I, H//8], w2 [E, H, I//8], scales/biases at //group.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import types  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchax.interop import jax_view  # noqa: E402
from vllm.model_executor.layers.fused_moe import FusedMoE  # noqa: E402
from vllm.model_executor.layers.fused_moe.layer import \
    MoEActivation  # noqa: E402

from tests.layers.common import utils as test_utils  # noqa: E402
from tests.utils.mlx_synthetic import _quantize_affine  # noqa: E402
from tpu_inference.layers.common.process_weights.moe_weights import \
    shard_moe_weights  # noqa: E402
from tpu_inference.layers.common.quantization.unquantized import \
    process_unquantized_moe_weights  # noqa: E402
from tpu_inference.layers.vllm.quantization.mlx import (  # noqa: E402
    VllmMLXConfig, VllmMLXMoEMethod)

E = 4  # experts
H = 128  # hidden_size (divisible by group_size and pack_factor)
I = 64  # intermediate_size per expert
GROUP_SIZE = 64
BITS = 4
PACK_FACTOR = 32 // BITS  # 8


class _IdentityExpertMap:
    """No expert-parallel: global expert id == local expert id (matches the
    real ExpertMapManager behavior on a single rank)."""

    def map_global_to_local(self, gid):
        return gid


class _FakeFusedMoE(FusedMoE):
    """A faithful FusedMoE for CPU unit tests.

    Subclassing FusedMoE (so the method's ``assert isinstance(layer, FusedMoE)``
    holds) but bypassing its heavy constructor (which needs a model config +
    distributed env). We set only the attributes the create_weights /
    weight_loader / process_weights_after_loading path actually touches, and we
    INHERIT the real vLLM ``weight_loader`` / ``_load_w13`` / ``_load_w2`` /
    ``_map_global_expert_id_to_local_expert_id`` -- so the per-expert un-stacking
    is exercised through production code, not a mock. ``tp_rank``/``tp_size``/
    ``use_ep`` are class attributes that shadow the property-backed originals."""

    tp_rank = 0
    tp_size = 1
    use_ep = False

    def __init__(self, num_experts, hidden_size, intermediate_size):
        torch.nn.Module.__init__(self)
        self.num_experts = num_experts
        self.expert_map_manager = _IdentityExpertMap()
        # Newer vLLM's FusedMoE.weight_loader maps the global->local expert id
        # via ``_map_global_expert_id_to_local_expert_id``, which reads
        # ``self._expert_map`` (None == no expert-parallel, identity mapping).
        # We run single-rank with no EP, so None gives the identity behavior the
        # old ``_IdentityExpertMap`` provided.
        self._expert_map = None
        self.activation = MoEActivation.from_str("silu")
        self.quant_config = None  # FusedMoE.weight_loader reads .quant_config
        # hidden_size / intermediate_size_per_partition are read off moe_config;
        # is_act_and_mul drives the w13 half-split; use_ep drives backend select.
        self.moe_config = types.SimpleNamespace(
            is_act_and_mul=True,
            use_ep=False,
            hidden_dim=hidden_size,
            intermediate_size_per_partition=intermediate_size)


def _bf16_to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr).view(np.uint16)).view(
        torch.bfloat16)


def _quant_stack(rng, shape, negate):
    """Per-expert affine 4-bit packs for a [E, out, in] stack. Returns lists of
    per-expert (packed, scales, biases) plus the stacked golden bf16 [E,out,in]
    (as float32)."""
    w = rng.standard_normal((E, *shape)).astype(np.float32)
    packs = [_quantize_affine(w[e], GROUP_SIZE, negate) for e in range(E)]
    golden = np.stack([p[3].astype(np.float32) for p in packs])
    return ([p[0] for p in packs], [p[1] for p in packs],
            [p[2] for p in packs], golden)


def _load_expert_weights(layer, group_name, shard_id, packed):
    """Drive the real packed-weight weight_loader per expert, exactly as vLLM's
    model loader does for one projection of one expert."""
    pw = getattr(layer, f"{group_name}_weight")
    for e in range(E):
        pw.weight_loader(pw, torch.from_numpy(packed[e].astype(np.uint32)),
                         f"{group_name}_weight", shard_id, e)


def _load_expert(layer, group_name, shard_id, packed, scales, biases):
    """Drive the real (and custom-bias) weight_loaders per expert for weight,
    scales, and biases -- exactly as vLLM's model loader does for one
    projection of one expert."""
    _load_expert_weights(layer, group_name, shard_id, packed)
    ps = getattr(layer, f"{group_name}_scales")
    pb = getattr(layer, f"{group_name}_biases")
    for e in range(E):
        ps.weight_loader(ps, _bf16_to_torch(scales[e]),
                         f"{group_name}_scales", shard_id, e)
        pb.weight_loader(pb, _bf16_to_torch(biases[e]),
                         f"{group_name}_biases", shard_id, e)


def _build_method_and_layer(mesh):
    layer = _FakeFusedMoE(E, H, I)
    method = VllmMLXMoEMethod(VllmMLXConfig(group_size=GROUP_SIZE, bits=BITS),
                              layer, mesh)
    # FusedMoE.weight_loader reads layer.quant_method.__class__.__name__.
    layer.quant_method = method
    method.create_weights(layer,
                          num_experts=E,
                          hidden_size=H,
                          intermediate_size_per_partition=I,
                          params_dtype=torch.bfloat16,
                          weight_loader=layer.weight_loader)
    return method, layer


def test_create_weights_stacked_shapes_and_dtypes():
    """create_weights registers the stacked MLX params with the spec shapes:
    w13_weight uint32 [E, 2I, H//8], w2_weight uint32 [E, H, I//8],
    scales/biases bf16 at //group_size. (Names are w13_weight/w2_weight, not
    the brief's w13_qweight -- see module docstring.)"""
    method, layer = _build_method_and_layer(test_utils.get_spmd_mesh(1))

    assert tuple(layer.w13_weight.shape) == (E, 2 * I, H // PACK_FACTOR)
    assert tuple(layer.w2_weight.shape) == (E, H, I // PACK_FACTOR)
    assert layer.w13_weight.dtype == torch.uint32
    assert layer.w2_weight.dtype == torch.uint32

    assert tuple(layer.w13_scales.shape) == (E, 2 * I, H // GROUP_SIZE)
    assert tuple(layer.w13_biases.shape) == (E, 2 * I, H // GROUP_SIZE)
    assert tuple(layer.w2_scales.shape) == (E, H, I // GROUP_SIZE)
    assert tuple(layer.w2_biases.shape) == (E, H, I // GROUP_SIZE)
    for name in ("w13_scales", "w13_biases", "w2_scales", "w2_biases"):
        assert getattr(layer, name).dtype == torch.bfloat16


def test_weight_loader_unstacks_gate_up_into_w13_halves():
    """The real FusedMoE weight_loader must route gate_proj -> w1 (first I rows
    of w13) and up_proj -> w3 (second I rows), per expert. Load distinct
    sentinel packs and assert the two halves land in the right place and other
    experts are untouched -- proving the un-stacking/registration wiring."""
    method, layer = _build_method_and_layer(test_utils.get_spmd_mesh(1))
    n_words = H // PACK_FACTOR
    gate = [np.full((I, n_words), 11 + e, dtype=np.uint32) for e in range(E)]
    up = [np.full((I, n_words), 91 + e, dtype=np.uint32) for e in range(E)]

    _load_expert_weights(layer, "w13", "w1", gate)
    _load_expert_weights(layer, "w13", "w3", up)

    w13 = layer.w13_weight.cpu().numpy()
    for e in range(E):
        assert (w13[e, :I] == 11 + e).all()  # w1 half = gate
        assert (w13[e, I:] == 91 + e).all()  # w3 half = up


def _reconstruct_w13_from_stored(codes, scale, groupbias):
    """Reconstruct the dequantized w13 from the STORED Stage-2 layer params
    EXACTLY as ``gmm_v2`` does in-kernel: ``w = codes * scale + groupbias``.

    Stored layout (post ``process_weights_after_loading`` / GMM_TP):
      * codes:     int4 [E, size_k, size_n]   (signed [-8, 7])
      * scale:     f32  [E, num_blocks, 1, size_n]
      * groupbias: f32  [E, num_blocks, 1, size_n]
    ``size_k % num_blocks == 0``; each quant block spans ``size_k//num_blocks``
    contracting rows, so broadcast scale/groupbias back over k by repeating each
    block's row ``size_k//num_blocks`` times (the kernel indexes block =
    k // (size_k // num_blocks)). Returns f32 [E, size_k, size_n]."""
    codes = np.asarray(codes).astype(np.float32)
    scale = np.asarray(scale).astype(np.float32)
    groupbias = np.asarray(groupbias).astype(np.float32)
    _, size_k, _ = codes.shape
    num_blocks = scale.shape[1]
    assert size_k % num_blocks == 0
    rows_per_block = size_k // num_blocks
    scale_full = np.repeat(scale[:, :, 0, :], rows_per_block, axis=1)
    groupbias_full = np.repeat(groupbias[:, :, 0, :], rows_per_block, axis=1)
    return codes * scale_full + groupbias_full


def _reconstruct_w2_from_stored(codes, scale, groupbias):
    """Reconstruct the dequantized w2 from the STORED Stage-2 layer params, the
    same ``w = codes * scale + groupbias`` fold the kernel does in-kernel.

    w2 has the SAME stored layout as w13 after ``process_moe_weights`` (GMM_TP):
    codes int4 [E, size_k=I, size_n=H], scale/groupbias f32
    [E, num_blocks=I//gs, 1, size_n=H]. Only the dim sizes differ (contraction
    is the intermediate dim, output is the hidden dim), so the reconstruction is
    identical to ``_reconstruct_w13_from_stored``."""
    return _reconstruct_w13_from_stored(codes, scale, groupbias)


def test_process_weights_dequant_matches_golden_experts():
    """End-to-end (Stage-2 hybrid): load per-expert 4-bit packs through the real
    weight_loaders, run process_weights_after_loading, and assert the result
    matches feeding the SAME golden bf16 experts through the SAME unquantized
    processing path.

    Stage-2 keeps BOTH w13 AND w2 as SIGNED int4 CODES + per-group
    ``{w13,w2}_weight_scale`` + per-group affine ``{w13,w2}_groupbias`` (in-kernel
    dequant via gmm_v2); the codes are NOT bf16, so we cannot compare them to the
    bf16 golden directly. Instead we RECONSTRUCT the dequantized weight from the
    stored params exactly as the kernel does (``codes * scale + groupbias``) and
    compare THAT to the golden, for w13 AND w2. (At tp=1 / single CPU device,
    ``w13_reorder_size == 1`` so the w2-int4 gate is always taken -- w2 is int4
    here.) This isolates the new MLX int4-keep + sign-fold + affine-bias logic
    from the shared (already tested) GMM layout transform, and is a real-behavior
    check (golden = the bf16 the checkpoint ships)."""
    mesh = test_utils.get_spmd_mesh(1)
    method, layer = _build_method_and_layer(mesh)
    assert method.moe_backend.name == "GMM_TP"  # single-device, no EP

    rng = np.random.default_rng(0)
    gate_q, gate_s, gate_b, gate_gold = _quant_stack(rng, (I, H), negate=True)
    up_q, up_s, up_b, up_gold = _quant_stack(rng, (I, H), negate=False)
    down_q, down_s, down_b, down_gold = _quant_stack(rng, (H, I), negate=False)

    _load_expert(layer, "w13", "w1", gate_q, gate_s, gate_b)
    _load_expert(layer, "w13", "w3", up_q, up_s, up_b)
    _load_expert(layer, "w2", "w2", down_q, down_s, down_b)

    method.process_weights_after_loading(layer)

    # Oracle: same unquantized path on the golden bf16 experts. w13 stacks gate
    # (w1) then up (w3) along the output dim, matching create_weights' layout.
    w13_gold = jnp.asarray(np.concatenate([gate_gold, up_gold],
                                          axis=1)).astype(jnp.bfloat16)
    w2_gold = jnp.asarray(down_gold).astype(jnp.bfloat16)
    ref = shard_moe_weights(
        process_unquantized_moe_weights(mesh=mesh,
                                        moe_backend=method.moe_backend,
                                        activation=layer.activation,
                                        w13_weight=w13_gold,
                                        w13_bias=None,
                                        w2_weight=w2_gold,
                                        w2_bias=None), method.moe_backend, mesh)
    ref_w13 = np.asarray(ref.w13_weight).astype(np.float32)
    ref_w2 = np.asarray(ref.w2_weight).astype(np.float32)

    # --- w13: stored as SIGNED int4 codes + scale + affine groupbias ---
    stored_codes = np.asarray(jax_view(layer.w13_weight))
    stored_scale = np.asarray(jax_view(layer.w13_weight_scale))
    stored_gbias = np.asarray(jax_view(layer.w13_groupbias))

    # The codes must be SIGNED int4 (the -8 sign-fold was applied at load).
    assert stored_codes.min() < 0, (
        "w13 codes must be signed int4 ([-8,7]); a missing -8 sign-fold would "
        "leave them unsigned [0,15]")
    assert stored_codes.min() >= -8 and stored_codes.max() <= 7

    # The affine groupbias must be PRESENT and non-zero (not dropped).
    assert stored_gbias.shape == stored_scale.shape
    assert np.abs(stored_gbias).max() > 0, "affine groupbias was dropped"

    recon_w13 = _reconstruct_w13_from_stored(stored_codes, stored_scale,
                                             stored_gbias)
    assert recon_w13.shape == ref_w13.shape
    np.testing.assert_allclose(recon_w13, ref_w13, atol=2e-2, rtol=2e-2)

    # TEETH: a wrong sign-fold (codes left unsigned, i.e. +8 not folded out) and
    # a dropped affine bias must BOTH break the match -- guards against a
    # regression that silently drops the fold or the bias while still "passing".
    recon_wrong_fold = _reconstruct_w13_from_stored(stored_codes + 8,
                                                    stored_scale, stored_gbias)
    assert not np.allclose(recon_wrong_fold, ref_w13, atol=2e-2, rtol=2e-2), (
        "test has no teeth: a wrong (unsigned) sign-fold still matched")
    recon_no_bias = _reconstruct_w13_from_stored(stored_codes, stored_scale,
                                                 np.zeros_like(stored_gbias))
    assert not np.allclose(recon_no_bias, ref_w13, atol=2e-2, rtol=2e-2), (
        "test has no teeth: dropping the affine groupbias still matched")

    # --- w2: ALSO stored as SIGNED int4 codes + scale + affine groupbias ---
    # (At single-device tp=1 the w2-int4 gate is always taken.) Mirror the w13
    # checks: the scale/groupbias params must now exist, the codes must be
    # signed, the groupbias must be present, and the in-kernel fold must
    # reconstruct the golden bf16 w2.
    assert method._w2_int4, (
        "at single-device tp=1 (w13_reorder_size==1) w2 must be kept int4")
    assert hasattr(layer, "w2_weight_scale") and hasattr(layer, "w2_groupbias"), (
        "process_weights must register w2_weight_scale / w2_groupbias when w2 "
        "is kept int4")

    w2_codes = np.asarray(jax_view(layer.w2_weight))
    w2_scale = np.asarray(jax_view(layer.w2_weight_scale))
    w2_gbias = np.asarray(jax_view(layer.w2_groupbias))

    # The w2 codes must be SIGNED int4 (the -8 sign-fold was applied at load).
    assert w2_codes.min() < 0, (
        "w2 codes must be signed int4 ([-8,7]); a missing -8 sign-fold would "
        "leave them unsigned [0,15]")
    assert w2_codes.min() >= -8 and w2_codes.max() <= 7

    # The affine groupbias must be PRESENT and non-zero (not dropped).
    assert w2_gbias.shape == w2_scale.shape
    assert np.abs(w2_gbias).max() > 0, "w2 affine groupbias was dropped"

    recon_w2 = _reconstruct_w2_from_stored(w2_codes, w2_scale, w2_gbias)
    assert recon_w2.shape == ref_w2.shape
    np.testing.assert_allclose(recon_w2, ref_w2, atol=2e-2, rtol=2e-2)

    # TEETH: the same wrong-sign-fold and dropped-bias controls must BOTH break
    # the w2 match -- proving the int4-w2 fold is what makes the test pass.
    recon_w2_wrong_fold = _reconstruct_w2_from_stored(w2_codes + 8, w2_scale,
                                                      w2_gbias)
    assert not np.allclose(recon_w2_wrong_fold, ref_w2, atol=2e-2, rtol=2e-2), (
        "test has no teeth: a wrong (unsigned) sign-fold still matched w2")
    recon_w2_no_bias = _reconstruct_w2_from_stored(w2_codes, w2_scale,
                                                   np.zeros_like(w2_gbias))
    assert not np.allclose(recon_w2_no_bias, ref_w2, atol=2e-2, rtol=2e-2), (
        "test has no teeth: dropping the w2 affine groupbias still matched")
