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

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import torch
from compressed_tensors.quantization import (ActivationOrdering,
                                             QuantizationArgs)
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
    CompressedTensorsScheme
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    unpack_quantized_values_into_int32
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter)
from vllm.scalar_type import scalar_types

from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class VllmCompressedTensorsWNA16(CompressedTensorsScheme):
    """Weight-only int4 (WNA16) linear scheme for TPU.

    Supports compressed-tensors pack-quantized checkpoints with group or
    channel scales, symmetric or asymmetric (group zero points). The weights
    are dequantized to the activation dtype at load time and applied as a
    regular dense matmul, mirroring W4A16 semantics exactly.
    """

    _supported_num_bits = (4, )

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        linear_config: VllmQuantLinearConfig,
    ):
        self.weight_quant = weight_quant
        self.num_bits = weight_quant.num_bits
        self.pack_factor = 32 // weight_quant.num_bits
        self.strategy = weight_quant.strategy
        self.symmetric = weight_quant.symmetric
        self.group_size = (-1 if weight_quant.group_size is None else
                           weight_quant.group_size)

        if self.num_bits not in self._supported_num_bits:
            raise ValueError(
                f"Unsupported num_bits = {self.num_bits} for WNA16 on TPU. "
                f"Supported num_bits = {self._supported_num_bits}")
        self.wtype = scalar_types.uint4

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError(
                "Pack-quantized format requires group quantization or "
                "channelwise quantization, but found no group size and "
                "strategy is not channelwise.")

        if weight_quant.actorder == ActivationOrdering.GROUP:
            raise NotImplementedError(
                "Activation reordering (actorder=group) is not supported "
                "for WNA16 on TPU.")

        self.linear_config = linear_config

    @classmethod
    def get_min_capability(cls) -> int:
        # TPU has no CUDA compute capability; return the lowest tier.
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)

        # If group_size is -1, we are in channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = input_size != input_size_per_partition

        partition_scales = self.strategy == "group" and not row_parallel

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )

        weight_scale_args = {
            "data":
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
            "weight_loader":
            weight_loader,
        }

        if partition_scales:
            weight_scale = GroupQuantScaleParameter(output_dim=0,
                                                    input_dim=1,
                                                    **weight_scale_args)
        else:
            weight_scale = ChannelQuantScaleParameter(output_dim=0,
                                                      **weight_scale_args)

        weight_shape = BasevLLMParameter(data=torch.empty(2,
                                                          dtype=torch.int64),
                                         weight_loader=weight_loader)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        if not self.symmetric:
            # Zero points are packed along the output dim.
            zeros_args = {
                "data":
                torch.zeros(
                    output_size_per_partition // self.pack_factor,
                    scales_and_zp_size,
                    dtype=torch.int32,
                ),
                "weight_loader":
                weight_loader,
            }
            if partition_scales:
                weight_zero_point = PackedvLLMParameter(
                    input_dim=1,
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )
            else:
                weight_zero_point = PackedColumnParameter(
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )
            layer.register_parameter("weight_zero_point", weight_zero_point)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Values are unsigned 4-bit integers (0-15).
        unpacked_weights = unpack_quantized_values_into_int32(
            layer.weight_packed, self.wtype, packed_dim=1)
        uint_weight = t2j(unpacked_weights, use_dlpack=False)
        delattr(layer, "weight_packed")

        weight_scale = t2j(layer.weight_scale, use_dlpack=False)
        delattr(layer, "weight_scale")

        if self.symmetric:
            uint_zero_point = None
        else:
            unpacked_zero_point = unpack_quantized_values_into_int32(
                layer.weight_zero_point, self.wtype, packed_dim=0)
            uint_zero_point = t2j(unpacked_zero_point, use_dlpack=False)
            delattr(layer, "weight_zero_point")

        if getattr(layer, "bias",
                   None) is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        @jax.jit
        def dequantize_wna16_linear_weights(
            uint_weight: jax.Array,
            weight_scale: jax.Array,
            uint_zero_point: jax.Array | None,
            bias: jax.Array | None,
        ) -> LinearWeights:
            out_size, in_size = uint_weight.shape
            num_groups = weight_scale.shape[-1]

            # dequant = (w_u - zp_u) * scale, with w_u/zp_u in [0, 15].
            # For symmetric checkpoints the implicit zero point is 8.
            weight = uint_weight.reshape(out_size, num_groups, -1)
            if uint_zero_point is None:
                weight = weight - 8
            else:
                weight = weight - uint_zero_point[:, :, None]
            weight = weight.astype(weight_scale.dtype)
            weight = weight * weight_scale[:, :, None]
            weight = weight.reshape(out_size, in_size)

            # [out, in] -> [in, out], as expected by the matmul below.
            weight = jnp.transpose(weight)

            return process_linear_weights(
                LinearWeights(
                    weight=weight,
                    weight_scale=None,
                    zero_point=None,
                    bias=bias,
                ),
                fused=self.linear_config.fuse_matmuls,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
            )

        weights = dequantize_wna16_linear_weights(uint_weight, weight_scale,
                                                  uint_zero_point, bias)
        weights = torch_view(
            shard_linear_weights(
                weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
            ))

        if self.linear_config.fuse_matmuls:
            layer.weight = Parameter(weights.weight, requires_grad=False)
            if bias is not None:
                layer.bias = Parameter(weights.bias, requires_grad=False)
        else:
            layer.weight = to_parameter_list(weights.weight)
            if bias is not None:
                layer.bias = to_parameter_list(weights.bias)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
                return self._apply_fused(layer, x, bias)
            else:
                return self._apply_split(layer, x, bias)

    def _apply_fused(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        x_jax = jax_view(x)
        weight = jax_view(layer.weight)

        outs = jnp.einsum("bd,df->bf", x_jax, weight)

        if bias is not None and not layer.skip_bias_add:
            outs += jax_view(bias)

        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        return torch_view(jnp.concatenate(outs, axis=-1))

    def _apply_split(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        assert isinstance(layer.weight, torch.nn.ParameterList)

        x_jax = jax_view(x)
        outs = []
        for i, weight in enumerate(layer.weight):
            out = jnp.einsum("bd,df->bf", x_jax, jax_view(weight))

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])

            outs.append(out)
        return torch_view(jnp.concatenate(outs, axis=-1))
