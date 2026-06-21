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

from typing import Any, Optional

import jax
import jax.numpy as jnp
import torch
from jax.sharding import NamedSharding
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.fused_moe import (FusedMoE,
                                                  FusedMoEMethodBase)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.layer import \
    FusedMoeWeightScaleSupported
from vllm.model_executor.layers.linear import LinearBase, set_weight_attrs
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)

from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.quant_methods import MLX
from tpu_inference.layers.common.quantization import (mlx_dequantize,
                                                      mlx_unpack)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import (
    reorder_concatenated_tensor_for_sharding,
    slice_sharded_tensor_for_concatenation)
from tpu_inference.layers.vllm.interface.moe import (
    MoEBackend, select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.utils import get_mesh_shape_product, t2j


def is_mlx_quantized(hf_config) -> bool:
    """MLX checkpoints carry a quant block with group_size+bits and NO quant_method."""
    for attr in ("quantization_config", "quantization"):
        q = getattr(hf_config, attr, None)
        if isinstance(q, dict) and "group_size" in q and "bits" in q \
                and "quant_method" not in q:
            return True
    return False


@register_quantization_config(MLX)
class VllmMLXConfig(QuantizationConfig, VllmQuantConfig):

    def __init__(self,
                 group_size: int,
                 bits: int,
                 modules_to_not_convert: Optional[list[str]] = None):
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.pack_factor = 32 // bits  # 8 for 4-bit
        self.modules_to_not_convert = modules_to_not_convert or []

    @classmethod
    def get_name(cls) -> str:
        return MLX

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "VllmMLXConfig":
        return cls(group_size=config["group_size"],
                   bits=config["bits"],
                   modules_to_not_convert=config.get("modules_to_not_convert"))

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    def get_quant_method(
            self, layer: torch.nn.Module,
            prefix: str) -> Optional[QuantizeMethodBase]:
        match layer:
            case LinearBase():
                linear_config = self.get_linear_config(layer)
                if is_layer_skipped(prefix, self.modules_to_not_convert):
                    from tpu_inference.layers.vllm.quantization.unquantized import \
                        VllmUnquantizedLinearMethod
                    return VllmUnquantizedLinearMethod(linear_config)
                return VllmMLXLinearMethod(self, linear_config)
            case FusedMoE():
                layer.moe_config = self.get_moe_config(layer)
                return VllmMLXMoEMethod(self, layer, self.mesh)
            case _:
                return None


class VllmMLXLinearMethod(QuantizeMethodBase):
    """MLX 4-bit affine linear method (keep-4bit, dequant-in-XLA at apply time).

    The MLX weight is ``uint32`` packed along the INPUT dim:
      * ``weight``  : ``[out, in // pack_factor]`` (uint32, packed_dim=1)
      * ``scales``  : ``[out, in // group_size]``  (params_dtype, affine scale)
      * ``biases``  : ``[out, in // group_size]``  (params_dtype, affine bias)
    Dequant is ``w = scale * q + bias`` (see ``mlx_dequantize``); the apply math
    contracts the input dim: ``y = einsum("bd,fd->bf", x, dequant_weight)`` with
    the dequantized weight in ``[out, in]`` layout.

    Mirrors ``VllmAWQLinearMethod`` (awq.py) but: MLX packs along input (AWQ along
    output); MLX is affine (scales+biases) not (q - z) * s; weight stays packed
    here and is dequantized at apply time via ``mlx_dequantize``.
    """

    def __init__(self, quant_config: "VllmMLXConfig",
                 linear_config: VllmQuantLinearConfig):
        self.quant_config = quant_config
        self.linear_config = linear_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        gs = self.quant_config.group_size
        pf = self.quant_config.pack_factor
        out = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # Weight is packed along the input dim (packed_dim=1=input_dim);
        # fusion/sharding act on the unpacked output dim 0.
        weight = PackedvLLMParameter(
            data=torch.empty(out,
                             input_size_per_partition // pf,
                             dtype=torch.uint32),
            output_dim=0,
            input_dim=1,
            packed_dim=1,
            packed_factor=pf,
            weight_loader=weight_loader)
        scales = GroupQuantScaleParameter(
            data=torch.empty(out,
                             input_size_per_partition // gs,
                             dtype=params_dtype),
            output_dim=0,
            input_dim=1,
            weight_loader=weight_loader)
        biases = GroupQuantScaleParameter(
            data=torch.empty(out,
                             input_size_per_partition // gs,
                             dtype=params_dtype),
            output_dim=0,
            input_dim=1,
            weight_loader=weight_loader)

        layer.register_parameter("weight", weight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("biases", biases)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Keep the weight uint32-packed (no unpack/dequant here; that happens in
        # apply). Two steps, mirroring AWQ's process_linear_weights +
        # shard_linear_weights, but on the three MLX tensors directly:
        #   1. If this is a fused projection (QKV / merged gate_up), reorder the
        #      output dim 0 from contiguous concat [q | k | v] into
        #      interleaved-by-shard layout so the apply-time
        #      slice_sharded_tensor_for_concatenation recovers each projection.
        #      (No-op when output_sizes has a single entry.)
        #   2. Shard each tensor along the output dim 0 with weight_sharding
        #      (= P(out_axis, None)), which applies directly to the 2D
        #      [out, in//pf] weight and [out, in//gs] scales/biases.
        mesh = self.linear_config.mesh
        wsh = self.linear_config.weight_sharding
        output_sizes = self.linear_config.output_sizes
        n_shards = self.linear_config.n_shards
        # tp=8 RowParallel input-dim sharding correctness (Task 8, Step 3).
        # The same weight_sharding spec is device_put onto all three tensors.
        # For RowParallelLinear it is P(None, ATTN_HEAD), which shards axis 1 --
        # the INPUT dim, which for MLX is packed (weight: in//pf, one uint32 =
        # pf=8 nibbles) AND grouped (scales/biases: in//gs, one affine pair per
        # gs=64 inputs). The split is correct ONLY if each shard owns whole
        # words AND whole groups for the SAME contiguous input range, i.e. both
        # in//pf and in//gs are divisible by the axis-1 shard count. When they
        # are, word and group boundaries align to the same per-shard input
        # slice and per-shard dequant == the input-slice of the full dequant
        # (verified numerically in test_mlx_linear_method.py::
        # test_rowparallel_input_dim_sharding_dequant_consistency; for Qwen3-30B
        # o_proj at tp=8: in=4096 -> in//8=512 (÷8 ✓), in//64=64 (÷8 ✓)).
        # If axis 1 is replicated (None) the shard count is 1 and this is a
        # no-op; we only need the guard when the input dim is actually sharded.
        in_axis = wsh[1] if len(wsh) > 1 else None
        in_shards = get_mesh_shape_product(mesh, in_axis)
        if in_shards > 1:
            pf = self.quant_config.pack_factor
            gs = self.quant_config.group_size
            n_words = layer.weight.shape[1]
            n_groups = layer.scales.shape[1]
            assert n_words % in_shards == 0 and n_groups % in_shards == 0, (
                "MLX RowParallel input-dim sharding would split a uint32 word "
                f"or a quant group across chips: packed dim in//{pf}={n_words} "
                f"and grouped dim in//{gs}={n_groups} must both be divisible by "
                f"the input shard count {in_shards}. Got remainders "
                f"{n_words % in_shards}/{n_groups % in_shards}.")
        # MLX keeps a single packed Parameter per tensor and always uses the
        # fused-style apply (one dequant + einsum + slice). The split path
        # (per-projection ParameterLists, AWQ's _apply_split) is not built here,
        # so a non-fused multi-projection layer would be mis-sliced at apply.
        # Fail loudly instead of silently corrupting QKV/gate_up outputs.
        assert self.linear_config.fuse_matmuls or len(output_sizes) == 1, (
            "VllmMLXLinearMethod only supports fused multi-projection layers; "
            f"got fuse_matmuls=False with output_sizes={output_sizes}.")
        do_reorder = self.linear_config.fuse_matmuls and len(output_sizes) > 1

        def _process(t):
            # Loaded params are plain CPU torch tensors (PackedvLLMParameter /
            # GroupQuantScaleParameter), not torchax-wrapped, so cross into JAX
            # with t2j (the AWQ/FP8/unquantized load-time idiom), not jax_view
            # (which asserts an already-torchax tensor).
            arr = t2j(t, use_dlpack=False)
            if do_reorder:
                arr = reorder_concatenated_tensor_for_sharding(
                    arr, output_sizes, n_shards, dim=0)
            return torch_view(jax.device_put(arr, NamedSharding(mesh, wsh)))

        layer.weight = torch.nn.Parameter(_process(layer.weight),
                                          requires_grad=False)
        layer.scales = torch.nn.Parameter(_process(layer.scales),
                                          requires_grad=False)
        layer.biases = torch.nn.Parameter(_process(layer.biases),
                                          requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        with jax.named_scope(layer._get_name()):
            x_jax = jax_view(x)
            # Dequant in XLA: w = scale * q + bias, weight in [out, in] layout.
            weight = mlx_dequantize(jax_view(layer.weight),
                                    jax_view(layer.scales),
                                    jax_view(layer.biases),
                                    group_size=self.quant_config.group_size,
                                    bits=self.quant_config.bits)
            # Contract the input dim: y[b, f] = sum_d x[b, d] * weight[f, d].
            outs = jnp.einsum("bd,fd->bf", x_jax, weight)

            if bias is not None and not layer.skip_bias_add:
                outs = outs + jax_view(bias)

            # Split a fused output back into its projections (no-op pass-through
            # when there is a single projection), mirroring AWQ's apply.
            outs = slice_sharded_tensor_for_concatenation(
                outs, self.linear_config.output_sizes,
                self.linear_config.n_shards)
            return torch_view(jnp.concatenate(outs, axis=-1))


def _make_mlx_moe_bias_loader(layer: "FusedMoE"):
    """Custom per-param ``weight_loader`` for MLX MoE affine biases.

    vLLM's ``FusedMoE.weight_loader`` routes by substring: names containing
    ``scale``/``zero``/``offset`` hit the group-scale branch, names containing
    ``weight`` hit the model-weight branch, and EVERYTHING ELSE falls through
    and is silently dropped (``return False``). Our per-expert affine biases
    arrive as ``...experts.{e}.{gate,up,down}_proj.biases`` and are mapped by
    ``make_expert_params_mapping`` into params named ``w13_biases``/``w2_biases``
    -- a name that matches none of those substrings, so the stock loader would
    drop them. Dropping the bias would turn dequant into ``scale * q`` (no
    ``+ bias``) and silently corrupt every expert.

    The biases have the SAME shape/attrs as the scales (``[E, 2I, n_groups]`` /
    ``[E, H, n_groups]``, ``is_transposed`` unset, group quant), so we route
    them through the exact same internal helper the group-scale branch uses,
    replicating the ``weight_loader`` prologue (global->local expert id +
    ``is_transposed`` shard-dim flip).
    """
    # gate_proj/up_proj -> w1/w3 (output dim 0); down_proj -> w2 (input dim 1).
    SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

    def _loader(param, loaded_weight, weight_name, shard_id, expert_id,
                return_success=False):
        local = layer._map_global_expert_id_to_local_expert_id(expert_id)
        if local == -1:
            # Not local to this rank; let the model loop try other replicas.
            return False if return_success else None
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)
        # Per-expert tensors are 2D (full_load is False), so index by expert.
        layer._load_model_weight_or_group_weight_scale(
            shard_dim=shard_dim,
            expert_data=param.data[local],
            shard_id=shard_id,
            loaded_weight=loaded_weight,
            tp_rank=layer.tp_rank,
            load_full_w2=getattr(param, "load_full_w2", False),
        )
        return True if return_success else None

    return _loader


class VllmMLXMoEMethod(FusedMoEMethodBase):
    """MLX 4-bit affine fused-MoE method (Stage 2 hybrid: w13 in-kernel 4-bit,
    w2 dequant-at-load -> bf16).

    The MLX checkpoint ships each expert's ``gate/up/down`` projection as
    ``uint32``-packed (along the INPUT dim) ``weight`` plus per-group affine
    ``scales``/``biases``. Task 6's loader transform un-stacks ``switch_mlp``
    into per-expert names so vLLM's ``FusedMoE.weight_loader`` can route them
    into the stacked params registered here:

      * ``w13_weight`` uint32 ``[E, 2I, H // pack_factor]`` (gate->w1 first I
        rows, up->w3 second I rows; packed along H)
      * ``w13_scales``/``w13_biases`` ``[E, 2I, H // group_size]``
      * ``w2_weight``  uint32 ``[E, H, I // pack_factor]`` (down_proj)
      * ``w2_scales``/``w2_biases`` ``[E, H, I // group_size]``

    ``process_weights_after_loading`` keeps **w13** packed int4 + per-group
    scale + per-group affine groupbias, so dequant happens INSIDE ``gmm_v2``
    (true 4-bit in HBM for the dominant expert weight). **w2** is dequantized to
    bf16 at load (its per-group scale/bias cannot shard cleanly at tp=8), so it
    flows through the same ``process_moe_weights`` / ``shard_moe_weights`` path
    with scale/groupbias = None. The two are run as separate ``gmm_v2`` calls, so
    a mixed int4-w13 / bf16-w2 forward is well-defined. The in-kernel w13 dequant
    is numerically identical to the Stage-1 load-time dequant (the synthetic e2e
    gate asserts an exact token match against the bf16 reference).
    """

    def __init__(self,
                 quant_config: "VllmMLXConfig",
                 layer: torch.nn.Module,
                 mesh,
                 ep_axis_name: str = "model"):
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config
        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)
        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name)

    @property
    def is_monolithic(self) -> bool:
        return True

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> None:
        return None

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int,
                       intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        gs = self.quant_config.group_size
        pf = self.quant_config.pack_factor
        E = num_experts
        H = hidden_size
        I = intermediate_size_per_partition

        # The packed weight is uint32 (8 nibbles/word); scales/biases share the
        # same [E, out, n_groups] layout. The output dim (2I for w13, H for w2)
        # is dim 0 of each per-expert slice, matching SHARD_ID_TO_SHARDED_DIM
        # ({w1:0, w3:0, w2:1}) with is_transposed unset.
        weight_attrs = dict(extra_weight_attrs)
        weight_attrs["quant_method"] = FusedMoeWeightScaleSupported.GROUP.value

        def _reg(name, shape, dtype, attrs):
            p = Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, attrs)
            return p

        _reg("w13_weight", (E, 2 * I, H // pf), torch.uint32, weight_attrs)
        _reg("w2_weight", (E, H, I // pf), torch.uint32, weight_attrs)
        _reg("w13_scales", (E, 2 * I, H // gs), params_dtype, weight_attrs)
        _reg("w2_scales", (E, H, I // gs), params_dtype, weight_attrs)

        # Biases reuse the scale layout but the suffix "biases" matches no
        # routing substring in vLLM's FusedMoE.weight_loader, so attach a custom
        # loader that drives the same group-scale helper (see docstring above).
        bias_attrs = dict(extra_weight_attrs)
        bias_attrs["quant_method"] = FusedMoeWeightScaleSupported.GROUP.value
        bias_attrs["weight_loader"] = _make_mlx_moe_bias_loader(layer)
        _reg("w13_biases", (E, 2 * I, H // gs), params_dtype, bias_attrs)
        _reg("w2_biases", (E, H, I // gs), params_dtype, bias_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Stage-2 (hybrid): w13 stays packed int4 (in-kernel dequant via
        ``gmm_v2``); w2 is dequantized to bf16 at load.

        w13 mirrors ``VllmCompressedTensorsW4A8MoEMethod`` (int4 codes +
        per-group scale flow through ``process_moe_weights`` -> GMM), with the
        MLX affine ``+ bias`` term routed into the NEW ``w13_groupbias`` ->
        ``gmm_v2(rhs_groupbias=...)`` so reconstruction is ``w = scale*q +
        groupbias``. Because MLX codes are UNSIGNED [0,15] but the kernel matmul
        is SIGNED int4, we shift codes by -8 and fold the offset back into the
        groupbias: ``(q-8)*scale + (bias + 8*scale) == q*scale + bias``.

        w2 cannot shard its per-group scale/bias cleanly at tp=8
        (num_blocks(w2)=I/64=12 not divisible by 8), so it stays bf16 -- the
        Stage-1 behavior -- with scale/groupbias = None (the gmm chain runs w1
        and w2 as separate gmm calls, so mixed int4-w13 / bf16-w2 is fine).
        """
        assert isinstance(layer, FusedMoE)
        gs = self.quant_config.group_size
        bits = self.quant_config.bits

        # Loaded params are CPU torch tensors; cross into JAX with t2j (the AWQ/
        # FP8/unquantized load-time idiom), not jax_view.
        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w13_scales = t2j(layer.w13_scales, use_dlpack=False)
        w13_biases = t2j(layer.w13_biases, use_dlpack=False)
        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w2_scales = t2j(layer.w2_scales, use_dlpack=False)
        w2_biases = t2j(layer.w2_biases, use_dlpack=False)
        for name in ("w13_weight", "w2_weight", "w13_scales", "w2_scales",
                     "w13_biases", "w2_biases"):
            delattr(layer, name)

        w13_interleave = layer.activation == MoEActivation.SWIGLUOAI
        w13_reorder_size = get_mesh_shape_product(self.mesh,
                                                  ShardingAxisName.MLP_TENSOR)

        @jax.jit
        def _process(w13q, w13s, w13b, w2q, w2s, w2b):
            # --- w13: keep int4, fold sign offset into the groupbias ---
            # MLX uint32-packed [E, 2I, H//pf] -> unsigned codes [E, 2I, H].
            w13_codes_u = mlx_unpack(w13q, bits)  # int32, unsigned [0, 15]
            # Shift to signed int4 [-8, 7] for the signed in-kernel matmul.
            w13_codes = (w13_codes_u - 8).astype(jnp.int4)  # [E, 2I, H]
            w13_scale = w13s.astype(jnp.float32)  # [E, 2I, H//gs]
            # groupbias = MLX affine bias + 8*scale (re-absorbs the -8 shift),
            # same [E, out, n_groups] layout as scale; process_moe_weights then
            # reshapes both to [E, num_blocks, 1, N] identically.
            w13_groupbias = (w13b.astype(jnp.float32) + 8.0 * w13_scale)

            # --- w2: dequant to bf16 at load (Stage-1 behavior) ---
            w2 = mlx_dequantize(w2q, w2s, w2b, group_size=gs, bits=bits)

            weights = FusedMoEWeights(
                w13_weight=w13_codes,
                w13_weight_scale=w13_scale,
                w13_groupbias=w13_groupbias,
                w13_bias=None,
                w2_weight=w2,
                w2_weight_scale=None,
                w2_groupbias=None,
                w2_bias=None,
            )
            return process_moe_weights(
                weights,
                moe_backend=self.moe_backend,
                w13_reorder_size=w13_reorder_size,
                w13_interleave=w13_interleave,
            )

        weights = _process(w13_weight, w13_scales, w13_biases, w2_weight,
                           w2_scales, w2_biases)
        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))

        # Store back: w13 stays packed int4 + per-group scale + groupbias; w2 is
        # plain bf16. The int4 dtype survives the torch Parameter round-trip as a
        # torchax wrapper (reported as int8 but the underlying jax buffer stays
        # int4); apply_monolithic re-casts to int4 defensively -- same as W4A8.
        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(weights.w13_weight_scale,
                                           requires_grad=False)
        layer.w13_groupbias = Parameter(weights.w13_groupbias,
                                        requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

        # Release intermediate buffers before the next layer (mirrors the
        # unquantized path's barrier to avoid cross-layer buffer accumulation).
        jax.effects_barrier()

    def apply_monolithic(self,
                         layer: "FusedMoE",
                         x: torch.Tensor,
                         router_logits: torch.Tensor,
                         input_ids: Optional[torch.Tensor] = None
                         ) -> torch.Tensor:
        # w13: packed int4 + per-group scale + affine groupbias straight through
        # (dequant happens inside gmm_v2). w2: bf16, scale/groupbias None.
        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight).astype(jnp.int4),
            w13_weight_scale=jax_view(layer.w13_weight_scale),
            w13_groupbias=jax_view(layer.w13_groupbias),
            w13_bias=None,
            w2_weight=jax_view(layer.w2_weight),
            w2_weight_scale=None,
            w2_groupbias=None,
            w2_bias=None)
        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits)
