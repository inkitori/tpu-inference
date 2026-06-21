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

from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import torch
from jax.sharding import NamedSharding
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)

from tpu_inference.layers.common.quant_methods import MLX
from tpu_inference.layers.common.quantization import mlx_dequantize
from tpu_inference.layers.common.utils import (
    reorder_concatenated_tensor_for_sharding,
    slice_sharded_tensor_for_concatenation)
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)


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
            prefix: str) -> Optional[Union[QuantizeMethodBase]]:
        # MoE method added in Task 7; only the LinearBase branch is wired here.
        match layer:
            case LinearBase():
                linear_config = self.get_linear_config(layer)
                if is_layer_skipped(prefix, self.modules_to_not_convert):
                    from tpu_inference.layers.vllm.quantization.unquantized import \
                        VllmUnquantizedLinearMethod
                    return VllmUnquantizedLinearMethod(linear_config)
                return VllmMLXLinearMethod(self, linear_config)
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
            arr = jax_view(t)
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
