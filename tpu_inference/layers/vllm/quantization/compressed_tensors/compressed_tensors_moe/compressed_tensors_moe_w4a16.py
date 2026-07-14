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

import jax
import jax.numpy as jnp
import torch
from compressed_tensors.quantization import QuantizationArgs
from jax.sharding import NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import torch_view
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, RoutedExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.utils import set_weight_attrs

import tpu_inference.envs as envs
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, _get_expert_shard_axis, process_quantized_moe_weights,
    shard_moe_weights_to_tpu)
from tpu_inference.layers.common.quantization import u32_unpack_i4
from tpu_inference.layers.common.utils import general_device_put
from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a8 import \
    VllmCompressedTensorsW4A8MoEMethod
from tpu_inference.logger import init_logger
from tpu_inference.utils import t2j, to_jax_dtype

logger = init_logger(__name__)


class VllmCompressedTensorsW4A16MoEMethod(VllmCompressedTensorsW4A8MoEMethod):
    """MoE method for asymmetric int4 weights (compressed-tensors WNA16).

    Handles pack-quantized checkpoints with group zero points (e.g. AWQ-style
    asymmetric MSE quantization exported in compressed-tensors format). Zero
    points are folded into the weights at load time and the result is
    requantized to symmetric int4 so the weights stay 4-bit in HBM and flow
    through the same fused-MoE kernels as the symmetric W4A8 path.
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        moe: FusedMoEConfig,
        mesh: jax.sharding.Mesh,
        ep_axis_name: str = "model",
    ):
        assert not weight_quant.symmetric, (
            "Symmetric int4 MoE checkpoints are handled by "
            "VllmCompressedTensorsW4A8MoEMethod")
        # Temporarily flip the symmetric flag so the parent __init__ passes
        # its symmetric-only validation; all shared bookkeeping is identical.
        weight_quant = weight_quant.model_copy(update={"symmetric": True})
        super().__init__(weight_quant, input_quant, moe, mesh, ep_axis_name)
        self.weight_quant = weight_quant.model_copy(
            update={"symmetric": False})

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        super().create_weights(layer, num_experts, hidden_size,
                               intermediate_size_per_partition, params_dtype,
                               **extra_weight_attrs)

        assert (2 * intermediate_size_per_partition) % self.packed_factor == 0
        assert hidden_size % self.packed_factor == 0
        num_groups_w13 = hidden_size // self.group_size
        num_groups_w2 = intermediate_size_per_partition // self.group_size

        # Zero points are packed along the output dim (dim 1 of the per-expert
        # tensors), matching the compressed-tensors pack-quantized layout.
        w13_weight_zero_point = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition // self.packed_factor,
                num_groups_w13,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_zero_point",
                                 w13_weight_zero_point)
        set_weight_attrs(w13_weight_zero_point, extra_weight_attrs)

        w2_weight_zero_point = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size // self.packed_factor,
                num_groups_w2,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_zero_point", w2_weight_zero_point)
        set_weight_attrs(w2_weight_zero_point, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, RoutedExperts)

        # Transfer packed weights directly to TPU, sharded by expert, so the
        # unpack/dequant/requant below runs in parallel across chips without
        # materializing full-size tensors on a single device.
        w13_weight_packed = t2j(layer.w13_weight_packed.view(torch.int32))
        w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)

        w2_weight_packed = t2j(layer.w2_weight_packed.view(torch.int32))
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)

        w13_zp_packed = t2j(layer.w13_weight_zero_point.view(torch.int32))
        w2_zp_packed = t2j(layer.w2_weight_zero_point.view(torch.int32))

        if self.moe.has_bias:
            w13_bias = t2j(layer.w13_bias, use_dlpack=False)
            w2_bias = t2j(layer.w2_bias, use_dlpack=False)
        else:
            w13_bias = w2_bias = None

        weights = FusedMoEWeights(
            w13_weight=w13_weight_packed,
            w13_weight_scale=w13_weight_scale,
            w13_bias=w13_bias,
            w2_weight=w2_weight_packed,
            w2_weight_scale=w2_weight_scale,
            w2_bias=w2_bias,
        )
        weights = shard_moe_weights_to_tpu(weights, self.mesh)

        ep_sharding = NamedSharding(self.mesh,
                                    PartitionSpec(
                                        _get_expert_shard_axis(self.mesh)))
        w13_zp_packed = general_device_put(w13_zp_packed, ep_sharding)
        w2_zp_packed = general_device_put(w2_zp_packed, ep_sharding)

        # Keep the weights 4-bit in HBM: fold the zero points in and
        # requantize to symmetric int4 (unless the requantization envs
        # request a different dtype/block size).
        if envs.MOE_REQUANTIZE_WEIGHT_DTYPE:
            desired_quant_dtype = to_jax_dtype(
                envs.MOE_REQUANTIZE_WEIGHT_DTYPE)
        else:
            desired_quant_dtype = jnp.int4
        if envs.MOE_REQUANTIZE_BLOCK_SIZE:
            requant_block_size = int(envs.MOE_REQUANTIZE_BLOCK_SIZE)
        else:
            requant_block_size = self.group_size

        activation_str = "swigluoai" if layer.activation == MoEActivation.SWIGLUOAI else ""

        @jax.jit(static_argnames=("desired_quant_dtype",
                                  "requant_block_size"))
        def unpack_and_process(
            weights: FusedMoEWeights,
            w13_zp_packed: jax.Array,
            w2_zp_packed: jax.Array,
            desired_quant_dtype: jnp.dtype,
            requant_block_size: int,
        ) -> FusedMoEWeights:
            # Both the weights and the zero points come out of u32_unpack_i4
            # shifted by -8, so the offsets cancel in the subtraction and
            # (w - zp) is exact in int8.
            w13_unpacked = u32_unpack_i4(weights.w13_weight).astype(jnp.int8)
            w2_unpacked = u32_unpack_i4(weights.w2_weight).astype(jnp.int8)
            # Zero points: [E, out // pack_factor, num_groups], packed along
            # the output dim -> [E, out, num_groups] after unpacking.
            w13_zp = u32_unpack_i4(w13_zp_packed, axis=1).astype(jnp.int8)
            w2_zp = u32_unpack_i4(w2_zp_packed, axis=1).astype(jnp.int8)

            def subtract_group_zp(w: jax.Array, zp: jax.Array) -> jax.Array:
                num_experts, out_size, in_size = w.shape
                num_groups = zp.shape[-1]
                w = w.reshape(num_experts, out_size, num_groups, -1)
                w = w - zp[:, :, :, None]
                return w.reshape(num_experts, out_size, in_size)

            weights_unpacked = FusedMoEWeights(
                w13_weight=subtract_group_zp(w13_unpacked, w13_zp),
                w13_weight_scale=weights.w13_weight_scale,
                w13_bias=weights.w13_bias,
                w2_weight=subtract_group_zp(w2_unpacked, w2_zp),
                w2_weight_scale=weights.w2_weight_scale,
                w2_bias=weights.w2_bias,
            )

            return process_quantized_moe_weights(
                weights=weights_unpacked,
                moe_backend=self.moe_backend,
                mesh=self.mesh,
                activation=activation_str,
                weight_block_size=(1, self.group_size),
                desired_quant_dtype=desired_quant_dtype,
                requant_block_size=requant_block_size,
            )

        weights = unpack_and_process(
            weights,
            w13_zp_packed,
            w2_zp_packed,
            desired_quant_dtype,
            requant_block_size,
        )

        weights = torch_view(weights)

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

        layer.w13_weight_scale = Parameter(weights.w13_weight_scale,
                                           requires_grad=False)
        layer.w2_weight_scale = Parameter(weights.w2_weight_scale,
                                          requires_grad=False)

        if self.moe.has_bias:
            layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)

        # Clean up packed parameters and shape metadata
        for name in ("w13_weight_packed", "w2_weight_packed",
                     "w13_weight_zero_point", "w2_weight_zero_point",
                     "w13_weight_shape", "w2_weight_shape"):
            if hasattr(layer, name):
                delattr(layer, name)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
