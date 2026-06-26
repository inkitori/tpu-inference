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
import jax.numpy as jnp
import torch
import torchax
from torchax.interop import jax_view, torch_view
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import (FusedMoEMethodBase,
                                                  RoutedExperts)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

from tpu_inference import envs
from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.moe_weights import \
    FusedMoEWeights
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)


def select_moe_backend_from_fused_moe_config(
        moe: FusedMoEConfig) -> MoEBackend:
    """
    Select the MoE backend based on the FusedMoEConfig.

    NOTE (jacobplatin): we don't currently support DENSE_MAT or MEGABLX_GMM
    backends on the vLLM path for now.

    Args:
        moe: The FusedMoEConfig.

    Returns:
        The selected MoE backend.
    """

    if envs.USE_MOE_EP_KERNEL:
        if moe.use_ep:
            logger.info_once("[MoE]: Using fused MoE EP kernel")
            return MoEBackend.FUSED_MOE
        logger.warning_once(
            "USE_MOE_EP_KERNEL=1 but expert parallelism is not "
            "enabled. Falling back to gmm implementation.")

    if moe.use_ep:
        logger.info_once("[MoE]: Using GMM EP kernel")
        return MoEBackend.GMM_EP

    # Use default implementation.
    logger.info_once("[MoE]: Using GMM TP kernel")
    return MoEBackend.GMM_TP


def vllm_moe_apply(layer: RoutedExperts,
                   weights: FusedMoEWeights,
                   quant_method_instance: FusedMoEMethodBase,
                   x: torch.Tensor,
                   router_logits: torch.Tensor,
                   input_ids: torch.Tensor | None = None) -> torch.Tensor:
    """
    Shared function for applying a FusedMoE layer for the TorchAX/vLLM backend.

    Args:
        layer: The FusedMoE layer.
        weights: The FusedMoE weights.
        quant_method_instance: The quantization method instance.
        x: The input tensor.
        router_logits: The router logits.

    Returns:
        The output tensor from the MoE fowrard pass.
    """
    assert isinstance(layer, RoutedExperts)
    assert isinstance(quant_method_instance, FusedMoEMethodBase)
    assert isinstance(weights, FusedMoEWeights)

    from tpu_inference.models.vllm.vllm_model_wrapper_context import \
        get_vllm_model_wrapper_context
    try:
        context = get_vllm_model_wrapper_context()
        vllm_config = context.vllm_config
    except AssertionError:
        vllm_config = None

    enable_return_routed_experts = vllm_config.model_config.enable_return_routed_experts if vllm_config else False

    if enable_return_routed_experts:
        if isinstance(router_logits, torch.Tensor):
            _, expert_indices = torch.topk(router_logits, layer.top_k, dim=-1)
            try:
                context = get_vllm_model_wrapper_context()
                context.expert_indices_list.append(jax_view(expert_indices))
            except AssertionError:
                pass

    mesh = quant_method_instance.mesh

    # DeepSeek-V3 style routing (e.g. Hy3): the per-expert selection bias and the
    # routed scaling factor live on the FusedMoE layer. Move the bias parameter
    # onto the JAX device (jax_view is a no-copy view of the torchax tensor) and
    # plumb both into the TPU gating recompute. Both default to no-op (None / 1.0)
    # for models without them (e.g. Qwen3 softmax routing).
    e_score_correction_bias = getattr(layer, "e_score_correction_bias", None)
    if e_score_correction_bias is not None:
        # vLLM's FusedMoE stores e_score_correction_bias as a plain attribute
        # alias to the parent module's nn.Parameter (no register_parameter), so
        # it escapes functional_call's reparametrization and arrives here as a
        # raw, un-moved nn.Parameter rather than a torchax tensor. Move it onto
        # the JAX device (the torchax env is active in this forward) before
        # taking a jax view. Guard so an already-torchax tensor is left alone.
        if not isinstance(e_score_correction_bias,
                          (torchax.tensor.Tensor, torchax.tensor.View)):
            e_score_correction_bias = e_score_correction_bias.to(device="jax")
        e_score_correction_bias = jax_view(e_score_correction_bias)
    routed_scaling_factor = getattr(layer, "routed_scaling_factor", None)

    # Number of real (non-padding) rows. The runner pads the token dim up to a
    # static compiled shape (e.g. 1 real decode token padded to 16); padding rows
    # carry garbage hidden states that route to arbitrary experts and inflate the
    # distinct-active-expert count driving the grouped-matmul cost. Plumb the real
    # token count so the MoE routing collapses padding rows onto a single expert.
    # query_start_loc is the per-request cumsum of scheduled tokens, padded with 1
    # past the last real request, so its max is the total actual token count (a
    # dynamic traced scalar -- no recompile, no new dynamic shape). attn_metadata
    # is set on the forward context by the vLLM model wrapper (set_forward_context).
    #
    # Guard to the single attention-DP-rank case: with attention DP > 1, the token
    # tensor seen by the MoE is the per-rank blocks concatenated, each with its own
    # real-prefix + padding, so a single contiguous arange >= n mask is invalid.
    # Fall back to None there (no masking, no regression).
    num_actual_tokens = None
    if get_mesh_shape_product(mesh, ShardingAxisName.ATTN_DATA) == 1:
        fwd_ctx = get_forward_context()
        attn_metadata = getattr(fwd_ctx, "attn_metadata", None)
        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
        if query_start_loc is not None:
            # attn_metadata on the TPU forward context is the JAX
            # AttentionMetadata dataclass, so query_start_loc is a raw jax.Array.
            num_actual_tokens = jnp.max(query_start_loc)

    # Upstream DP scatter + chunking + hash-based routing knobs.
    attn_dp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_DATA)
    dp_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_DATA)
    is_dp = (attn_dp_size // dp_size) > 1

    extra_kwargs = dict(quant_method_instance.extra_backend_kwargs)
    extra_kwargs["scatter_results"] = is_dp
    extra_kwargs["moe_chunk_size"] = envs.VLLM_MOE_CHUNK_SIZE

    if getattr(layer, "hash_indices_table", None) is not None:
        assert input_ids is not None, "input_ids must be provided when hash_indices_table is present in the layer"
        hash_table = layer.hash_indices_table
        hash_based_topk_indices = jax_view(hash_table)[jax_view(input_ids)]
        extra_kwargs["hash_based_topk_indices"] = hash_based_topk_indices
    # NOTE: e_score_correction_bias is passed below as a dedicated moe_apply
    # kwarg (hy3 path, with the nn.Parameter device-move guard), so it is not
    # duplicated into extra_kwargs here.

    return torch_view(
        moe_apply(
            layer=layer,
            x=jax_view(x),
            gating_output=jax_view(router_logits),
            weights=weights,
            moe_backend=quant_method_instance.moe_backend,
            mesh=quant_method_instance.mesh,
            extra_backend_kwargs=extra_kwargs,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            num_actual_tokens=num_actual_tokens,
        ))
