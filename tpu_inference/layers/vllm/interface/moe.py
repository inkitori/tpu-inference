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
import torch
import torchax
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

from tpu_inference import envs
from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.moe_weights import \
    FusedMoEWeights
from tpu_inference.logger import init_logger

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


def vllm_moe_apply(layer: FusedMoE, weights: FusedMoEWeights,
                   quant_method_instance: FusedMoEMethodBase, x: torch.Tensor,
                   router_logits: torch.Tensor) -> torch.Tensor:
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
    assert isinstance(layer, FusedMoE)
    assert isinstance(quant_method_instance, FusedMoEMethodBase)
    assert isinstance(weights, FusedMoEWeights)

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

    return torch_view(
        moe_apply(
            layer=layer,
            x=jax_view(x),
            gating_output=jax_view(router_logits),
            weights=weights,
            moe_backend=quant_method_instance.moe_backend,
            mesh=quant_method_instance.mesh,
            extra_backend_kwargs=quant_method_instance.extra_backend_kwargs,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        ))
