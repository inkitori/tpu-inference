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

import os

import torch
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import \
    SharedExpertsOrder
from vllm.model_executor.layers.linear import RowParallelLinear

from tpu_inference import envs
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from tpu_inference.utils import get_mesh_shape_product

_UNSET = object()


@MoERunner.register_oot
class VllmMoERunner(MoERunner):

    def _shared_psum_merge_target(self):
        """The shared-expert down_proj to run in stacked-partials mode, or None.

        When eligible, the shared-expert down_proj all-reduce is merged into
        the routed-MoE combine all-reduce (psum(r) + psum(s) == psum(r + s)):
        the down_proj returns unreduced per-shard partials which moe_gmm_local
        adds to the routed partial sums before its single combine psum. Saves
        one [T, H] all-reduce per MoE layer per step.

        Eligibility is computed once and cached; every gate failing leaves all
        existing paths byte-identical.
        """
        cached = getattr(self, "_tpu_shared_psum_merge", _UNSET)
        if cached is not _UNSET:
            return cached

        # Import here: quantization/mlx.py imports interface/moe.py which this
        # module must not import at module scope (registration order).
        from tpu_inference.layers.vllm.quantization.mlx import (
            VllmMLXLinearMethod, VllmMLXMoEMethod)

        target = None
        qm = self._quant_method
        se = self._shared_experts
        if (os.environ.get("MERGE_SHARED_EXPERT_PSUM", "1") == "1"
                and se is not None and isinstance(qm, VllmMLXMoEMethod)
                and qm.moe_backend in (MoEBackend.GMM_TP, MoEBackend.GMM_EP)
                and self.routed_input_transform is None
                and self.routed_output_transform is None
                and float(self.routed_scaling_factor) == 1.0
                and not envs.ENABLE_RS_KERNEL):
            down = getattr(se._layer, "down_proj", None)
            lm = getattr(down, "quant_method", None)
            if (isinstance(down, RowParallelLinear)
                    and isinstance(lm, VllmMLXLinearMethod)
                    and getattr(lm, "_in_axis", None) is not None
                    and getattr(lm, "_out_axis", _UNSET) is None):
                mesh = qm.mesh
                combine_axis = (ShardingAxisName.EXPERT
                                if qm.moe_backend == MoEBackend.GMM_EP else
                                ShardingAxisName.MLP_TENSOR)
                # The stacked partials must live on exactly the axis the MoE
                # combine reduces over, and there must be no attention/MLP DP
                # (which would engage scatter paths the merge does not support).
                if (lm._in_axis == combine_axis
                        and get_mesh_shape_product(
                            mesh, ShardingAxisName.ATTN_DATA) == 1
                        and get_mesh_shape_product(
                            mesh, ShardingAxisName.MLP_DATA) == 1):
                    target = down
        self._tpu_shared_psum_merge = target
        return target

    def _apply_quant_method(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        down = self._shared_psum_merge_target()
        if down is None or shared_experts_input is None:
            return super()._apply_quant_method(hidden_states, router_logits,
                                               shared_experts_input, input_ids)

        # Run the shared expert with its down_proj in stacked-partials mode
        # (no psum); the stacked [n_shards, T, H] output rides into the MoE
        # combine below.
        down._tpu_stack_partial_output = True
        try:
            self._maybe_apply_shared_experts(shared_experts_input,
                                             SharedExpertsOrder.NO_OVERLAP)
        finally:
            down._tpu_stack_partial_output = False
        shared_partial_stacked = self._shared_experts.output  # pops the slot

        fused_out = self._quant_method.apply_monolithic(
            layer=self.routed_experts,
            x=hidden_states,
            router_logits=router_logits,
            input_ids=input_ids,
            shared_partial_stacked=shared_partial_stacked)

        # The shared contribution is already inside fused_out. Return zeros so
        # the runner's downstream `shared + fused` add and reduction hooks stay
        # structurally intact; XLA folds the 0-add away under the outer jit.
        return torch.zeros_like(fused_out), fused_out

    def _maybe_reduce_final_output(self, states: torch.Tensor,
                                   trunc_size: int) -> torch.Tensor:
        try:
            context = get_vllm_model_wrapper_context()
            mesh = context.mesh
        except AssertionError:
            mesh = None

        is_dp = False
        if mesh is not None:
            attn_dp_size = get_mesh_shape_product(mesh,
                                                  ShardingAxisName.ATTN_DATA)
            dp_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_DATA)
            is_dp = (attn_dp_size // dp_size) > 1

        if is_dp:
            return states[..., :trunc_size]

        return super()._maybe_reduce_final_output(states, trunc_size)
