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
from jax.sharding import PartitionSpec as P
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.fused_moe import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.linear import LinearBase, set_weight_attrs
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)

from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.quant_methods import MLX
from tpu_inference.layers.common.quantization import mlx_unpack
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
        group_size = config["group_size"]
        bits = config["bits"]
        # MLX per-module quant overrides appear as dict-valued entries keyed by
        # the full module path (e.g. Hy3's 79 ``model.layers.{1..79}.mlp.router.
        # gate -> {group_size, bits: 8}``). The only override we support is the
        # 8-bit router gate, which is dequantized to bf16 at load (the GateLinear
        # is unquantized in vLLM). Validate fail-fast so an unhandled override
        # (a different module, or a different bit-width) errors loudly instead of
        # silently mis-loading.
        for key, val in config.items():
            if not isinstance(val, dict):
                continue
            if not key.endswith("mlp.router.gate"):
                raise ValueError(
                    f"Unsupported MLX per-module quant override for {key!r}: only "
                    "'*.mlp.router.gate' overrides are supported.")
            ovr_bits = val.get("bits")
            ovr_gs = val.get("group_size")
            if ovr_bits != 8:
                raise ValueError(
                    f"Unsupported MLX router-gate override {key!r}={val!r}: only "
                    "bits=8 is supported (dequantized to bf16 at load).")
            if ovr_gs != group_size:
                raise ValueError(
                    f"MLX router-gate override {key!r} group_size {ovr_gs} must "
                    f"match the model group_size {group_size}.")
        return cls(group_size=group_size,
                   bits=bits,
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


def _mlx_int4_matmul(x, codes, scale, groupbias, group_sizes, mesh, in_axis,
                     out_axis, vmem_limit_bytes=None):
    """Single-group ``gmm_v2`` dense int4 matmul, sharded like the MoE TP path.

    ``gmm_v2`` is a Pallas kernel: GSPMD does not auto-partition it, so (exactly
    like ``tensor_parallel_gmm``) we wrap it in ``shard_map`` over per-shard
    local tensors. ColumnParallel (out sharded, ``in_axis is None``) needs no
    reduction; RowParallel (contraction sharded) ``psum``s the partial sums over
    ``in_axis``. At tp=1 every spec naming a size-1 axis is a replicated no-op.

      * ``x``         ``[M, in]``            -> P(None, in_axis)
      * ``codes``     ``[1, in, out]`` int4  -> P(None, in_axis, out_axis)
      * scale/gbias   ``[1, in//gs, 1, out]``-> P(None, in_axis, None, out_axis)
      * output        ``[M, out]``           -> P(None, out_axis)

    ``vmem_limit_bytes`` is normally None (gmm_v2 picks ``0.9*vmem_capacity``).
    It only needs raising for a large UNSHARDED contraction dim (a dense down/o
    proj run at tp=1, where K never gets split across chips); at real serving TP
    the per-shard K is small and the default tiling fits.
    """

    def _local(lhs, rhs, sc, gb, gs):
        y = gmm_v2(lhs=lhs,
                   rhs=rhs,
                   group_sizes=gs,
                   rhs_scale=sc,
                   rhs_groupbias=gb,
                   maybe_quantize_lhs=False,
                   preferred_element_type=jnp.bfloat16,
                   vmem_limit_bytes=vmem_limit_bytes)
        if in_axis is not None:  # RowParallel: reduce contraction-dim shards.
            y = jax.lax.psum(y, axis_name=in_axis)
        return y

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(None, in_axis), P(None, in_axis, out_axis),
                  P(None, in_axis, None, out_axis),
                  P(None, in_axis, None, out_axis), P()),
        out_specs=P(None, out_axis),
        check_vma=False,
    )(x, codes, scale, groupbias, group_sizes)


class VllmMLXLinearMethod(QuantizeMethodBase):
    """MLX 4-bit affine linear method — in-kernel dequant via ``gmm_v2``.

    The MLX checkpoint ships the weight ``uint32`` packed along the INPUT dim:
      * ``weight``  : ``[out, in // pack_factor]`` (uint32, packed_dim=1)
      * ``scales``  : ``[out, in // group_size]``  (params_dtype, affine scale)
      * ``biases``  : ``[out, in // group_size]``  (params_dtype, affine bias)

    ``process_weights_after_loading`` unpacks ONCE into the ``gmm_v2`` int4
    layout — signed int4 codes ``[1, in, out]`` plus per-group ``scale`` /
    ``groupbias`` ``[1, in//gs, 1, out]`` — so the 4-bit weight is never
    materialized as bf16. ``apply`` runs a single-group ``gmm_v2`` that dequants
    ``w = scale*q + groupbias`` INSIDE the kernel (4-bit stays in HBM), the dense
    analogue of the MoE w13 path. Because MLX codes are unsigned ``[0,15]`` but
    the kernel matmul is signed, codes are shifted ``q -> q-8`` and the offset is
    folded back into the groupbias: ``(q-8)*scale + (bias+8*scale) ==
    q*scale + bias``.
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
        # Unpack ONCE into the gmm_v2 int4 layout (codes [1, in, out] +
        # scale/groupbias [1, in//gs, 1, out]); the 4-bit weight never becomes
        # bf16. weight_sharding is the [out, in] spec: out_axis shards dim 0
        # (ColumnParallel: qkv/gate_up), in_axis shards dim 1 (RowParallel:
        # o/down). The transformed specs put out_axis on the trailing out dim
        # and in_axis on the in/block dim, matching tensor_parallel_gmm's w1/w2.
        mesh = self.linear_config.mesh
        wsh = self.linear_config.weight_sharding
        output_sizes = self.linear_config.output_sizes
        n_shards = self.linear_config.n_shards
        bits = self.quant_config.bits
        out_axis = wsh[0]
        in_axis = wsh[1] if len(wsh) > 1 else None
        self._mesh = mesh
        self._in_axis = in_axis
        self._out_axis = out_axis
        self._output_sizes = output_sizes
        self._n_shards = n_shards

        # RowParallel input-dim sharding must own whole uint32 words AND whole
        # quant groups per shard, else a word/group straddles a chip boundary.
        in_shards = get_mesh_shape_product(mesh, in_axis)
        if in_shards > 1:
            n_words = layer.weight.shape[1]   # in // pack_factor
            n_groups = layer.scales.shape[1]  # in // group_size
            assert n_words % in_shards == 0 and n_groups % in_shards == 0, (
                "MLX RowParallel sharding splits a uint32 word or quant group: "
                f"in//pf={n_words} and in//gs={n_groups} must both be divisible "
                f"by the input shard count {in_shards}.")

        # Single packed Parameter per tensor + fused-style apply (one gmm +
        # slice). Non-fused multi-projection layers are unsupported (would be
        # mis-sliced); fail loudly instead of corrupting QKV/gate_up.
        assert self.linear_config.fuse_matmuls or len(output_sizes) == 1, (
            "VllmMLXLinearMethod only supports fused multi-projection layers; "
            f"got fuse_matmuls=False with output_sizes={output_sizes}.")
        do_reorder = self.linear_config.fuse_matmuls and len(output_sizes) > 1

        @jax.jit
        def _transform(weight, scales, biases):
            # Reorder the fused output (dim 0) into interleaved-by-shard order
            # BEFORE the transpose, so apply's slice/concat recovers each proj.
            if do_reorder:
                weight = reorder_concatenated_tensor_for_sharding(
                    weight, output_sizes, n_shards, dim=0)
                scales = reorder_concatenated_tensor_for_sharding(
                    scales, output_sizes, n_shards, dim=0)
                biases = reorder_concatenated_tensor_for_sharding(
                    biases, output_sizes, n_shards, dim=0)
            # Unpack uint32 -> unsigned codes [out, in], shift to signed int4,
            # fold the -8 offset into groupbias. Transpose in int32, cast last.
            codes = mlx_unpack(weight, bits) - 8                # int32 [out, in]
            scale = scales.astype(jnp.float32)                  # [out, in//gs]
            groupbias = biases.astype(jnp.float32) + 8.0 * scale
            codes = jnp.transpose(codes, (1, 0))[None].astype(jnp.int4)
            scale = jnp.transpose(scale, (1, 0))[None, :, None, :]
            groupbias = jnp.transpose(groupbias, (1, 0))[None, :, None, :]
            return codes, scale, groupbias

        codes, scale, groupbias = _transform(
            t2j(layer.weight, use_dlpack=False),
            t2j(layer.scales, use_dlpack=False),
            t2j(layer.biases, use_dlpack=False))

        codes_sh = NamedSharding(mesh, P(None, in_axis, out_axis))
        sg_sh = NamedSharding(mesh, P(None, in_axis, None, out_axis))
        layer.weight = torch.nn.Parameter(
            torch_view(jax.device_put(codes, codes_sh)), requires_grad=False)
        layer.scales = torch.nn.Parameter(
            torch_view(jax.device_put(scale, sg_sh)), requires_grad=False)
        layer.biases = torch.nn.Parameter(
            torch_view(jax.device_put(groupbias, sg_sh)), requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        with jax.named_scope(layer._get_name()):
            x_jax = jax_view(x)                              # [M, in]
            # int4 survives the torch Parameter round-trip as a torchax wrapper
            # (reported as int8); re-cast defensively, same as W4A8/the MoE path.
            codes = jax_view(layer.weight).astype(jnp.int4)  # [1, in, out]
            scale = jax_view(layer.scales)                   # [1, in//gs, 1, out]
            groupbias = jax_view(layer.biases)
            group_sizes = jnp.array([x_jax.shape[0]], dtype=jnp.int32)
            # Single-group in-kernel int4 matmul (dequant inside gmm_v2).
            outs = _mlx_int4_matmul(x_jax, codes, scale, groupbias, group_sizes,
                                    self._mesh, self._in_axis, self._out_axis)

            if bias is not None and not layer.skip_bias_add:
                outs = outs + jax_view(bias)

            # Split a fused output back into its projections (no-op pass-through
            # when there is a single projection).
            outs = slice_sharded_tensor_for_concatenation(
                outs, self._output_sizes, self._n_shards)
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
    """MLX 4-bit affine fused-MoE method (w13 and w2 both in-kernel 4-bit).

    The MLX checkpoint ships each expert's ``gate/up/down`` projection as
    ``uint32``-packed (along the INPUT dim) ``weight`` plus per-group affine
    ``scales``/``biases``. The loader transform un-stacks ``switch_mlp`` into
    per-expert names so vLLM's ``FusedMoE.weight_loader`` can route them into the
    stacked params registered here:

      * ``w13_weight`` uint32 ``[E, 2I, H // pack_factor]`` (gate->w1 first I
        rows, up->w3 second I rows; packed along H)
      * ``w13_scales``/``w13_biases`` ``[E, 2I, H // group_size]``
      * ``w2_weight``  uint32 ``[E, H, I // pack_factor]`` (down_proj)
      * ``w2_scales``/``w2_biases`` ``[E, H, I // group_size]``

    ``process_weights_after_loading`` keeps both w13 and w2 packed int4 +
    per-group scale + per-group affine groupbias, so dequant happens INSIDE
    ``gmm_v2`` (true 4-bit in HBM) for every expert weight.
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
        """w13 and w2 both stay packed int4 (in-kernel dequant via ``gmm_v2``).

        Each mirrors ``VllmCompressedTensorsW4A8MoEMethod`` (int4 codes +
        per-group scale flow through ``process_moe_weights`` -> GMM), with the
        MLX affine ``+ bias`` term routed into the ``groupbias`` ->
        ``gmm_v2(rhs_groupbias=...)`` so reconstruction is ``w = scale*q +
        groupbias``. Because MLX codes are UNSIGNED [0,15] but the kernel matmul
        is SIGNED int4, we shift codes by -8 and fold the offset back into the
        groupbias: ``(q-8)*scale + (bias + 8*scale) == q*scale + bias``.
        """
        assert isinstance(layer, FusedMoE)
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
            # Both projections: unpack uint32 -> unsigned codes, shift to signed
            # int4 [-8, 7], fold the -8 offset back into groupbias = bias +
            # 8*scale. process_moe_weights reshapes scale/groupbias from
            # [E, out, n_groups] to [E, num_blocks, 1, N] identically.
            def _fold(q, s, b):
                codes = (mlx_unpack(q, bits) - 8).astype(jnp.int4)
                scale = s.astype(jnp.float32)
                groupbias = b.astype(jnp.float32) + 8.0 * scale
                return codes, scale, groupbias

            w13_codes, w13_scale, w13_groupbias = _fold(w13q, w13s, w13b)
            w2_codes, w2_scale, w2_groupbias = _fold(w2q, w2s, w2b)

            weights = FusedMoEWeights(
                w13_weight=w13_codes,
                w13_weight_scale=w13_scale,
                w13_groupbias=w13_groupbias,
                w13_bias=None,
                w2_weight=w2_codes,
                w2_weight_scale=w2_scale,
                w2_groupbias=w2_groupbias,
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

        # Store back: w13 and w2 both stay packed int4 + per-group scale +
        # groupbias. The int4 dtype survives the torch Parameter round-trip as a
        # torchax wrapper (reported as int8 but the underlying jax buffer stays
        # int4); apply_monolithic re-casts to int4 defensively -- same as W4A8.
        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(weights.w13_weight_scale,
                                           requires_grad=False)
        layer.w13_groupbias = Parameter(weights.w13_groupbias,
                                        requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)
        layer.w2_weight_scale = Parameter(weights.w2_weight_scale,
                                          requires_grad=False)
        layer.w2_groupbias = Parameter(weights.w2_groupbias,
                                       requires_grad=False)

        # Release intermediate buffers before the next layer (mirrors the
        # unquantized path's barrier to avoid cross-layer buffer accumulation).
        jax.effects_barrier()

    def apply_monolithic(self,
                         layer: "FusedMoE",
                         x: torch.Tensor,
                         router_logits: torch.Tensor,
                         input_ids: Optional[torch.Tensor] = None
                         ) -> torch.Tensor:
        # w13 and w2: packed int4 + per-group scale + affine groupbias straight
        # through (dequant happens inside gmm_v2).
        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight).astype(jnp.int4),
            w13_weight_scale=jax_view(layer.w13_weight_scale),
            w13_groupbias=jax_view(layer.w13_groupbias),
            w13_bias=None,
            w2_weight=jax_view(layer.w2_weight).astype(jnp.int4),
            w2_weight_scale=jax_view(layer.w2_weight_scale),
            w2_groupbias=jax_view(layer.w2_groupbias),
            w2_bias=None)
        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits)
