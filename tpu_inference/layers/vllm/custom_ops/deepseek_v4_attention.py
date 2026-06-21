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
"""TPU interception for DeepSeek-V4 MLA attention (torchax path).

``DeepseekV4Attention`` is a plain ``nn.Module`` that vLLM instantiates directly
(the AMD decoder does ``self.attn = DeepseekV4ROCMAiterMLAAttention(...)``, a
``DeepseekV4Attention`` subclass, in ``deepseek_v4/amd/model.py``). Unlike the
MHC ops or the attention-impl bases, it is NOT a vLLM ``CustomOp`` and has no
``register_oot`` hook, so there is no registry-based way to swap it. Its
constructor is also CUDA-bound (allocates ``torch.cuda.Event``), so it cannot
run on TPU as-is.

Instead we substitute the class symbol before the model is built. Because
``amd/model.py`` does ``from ...amd.rocm import DeepseekV4ROCMAiterMLAAttention``,
the name is bound into the ``amd.model`` module namespace at import time; patching
it on ``amd.rocm`` alone would not take effect. ``patch_deepseek_v4_mla_cls``
rebinds it on ``amd.model`` directly. It is invoked from
``_maybe_patch_for_deepseek_v4`` in ``vllm_model_wrapper`` while ``is_rocm`` is
forced True and the package has been reloaded onto the AMD implementation.
"""
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from tpu_inference.layers.vllm.backends.flash_attn_mla import \
    PallasMLAttentionBackend
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class VllmDeepseekV4MLAAttention(DeepseekV4Attention):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
    ) -> None:
        # Build the full DeepseekV4Attention parameter manifest (attn_sink,
        # fused_wqa_wkv, q_norm, wq_b, kv_norm, wo_a/wo_b, rotary_emb, plus the
        # per-layer indexer/compressor sub-modules) by running the real base
        # __init__, so the checkpoint loads with the correct FP8-block quant
        # wiring. The only GPU-bound line in the base __init__ is
        # ``self.ln_events = [torch.cuda.Event() ...]`` (CUDA stream ordering,
        # irrelevant on TPU). ``torch.cuda.Stream`` is already neutralized by
        # ``_maybe_patch_for_deepseek_v4`` in the wrapper; here we additionally
        # neutralize ``torch.cuda.Event`` for the duration of construction.
        _orig_event = torch.cuda.Event
        torch.cuda.Event = lambda *args, **kwargs: None
        try:
            super().__init__(
                vllm_config,
                prefix,
                topk_indices_buffer=topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            )
        finally:
            torch.cuda.Event = _orig_event
        # ``get_kv_cache_spec`` reports the cache dtype string straight from
        # cache_config (base stores the resolved ``kv_cache_dtype`` separately).
        self.cache_dtype = vllm_config.cache_config.cache_dtype

    # Abstract platform hooks required to instantiate the DeepseekV4Attention
    # ABC; unused on the TPU pass-through path.
    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def forward_mqa(self, q: torch.Tensor, kv: torch.Tensor,
                    positions: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError

    def _o_proj(self, o: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.cache_dtype,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
            model_version="deepseek_v4",
        )

    def process_weights_after_loading(self, act_order: bool = False) -> None:
        pass

    def get_attn_backend(self) -> type[AttentionBackend]:
        return PallasMLAttentionBackend

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logger.error(
            "VllmDeepseekV4MLAAttention.forward is not implemented, just a pass-through for now"
        )
        return hidden_states


def patch_deepseek_v4_mla_cls() -> None:
    """Rebind ``DeepseekV4ROCMAiterMLAAttention`` to the TPU subclass.

    Must run after ``vllm.models.deepseek_v4.amd.model`` is imported (it holds
    its own ``from ...amd.rocm import DeepseekV4ROCMAiterMLAAttention``
    reference) and before the model is constructed.
    """
    import vllm.models.deepseek_v4.amd.model as ds_v4_amd_model
    ds_v4_amd_model.DeepseekV4ROCMAiterMLAAttention = VllmDeepseekV4MLAAttention
    logger.info(
        "Patched DeepseekV4ROCMAiterMLAAttention -> VllmDeepseekV4MLAAttention for TPU."
    )
