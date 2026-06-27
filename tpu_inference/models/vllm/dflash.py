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
"""Torchax wrapper for the PyTorch DFlash draft model.

Loads the original HuggingFace DFlash model (pure PyTorch) and wraps it
so it can run on TPU via torchax + JAX JIT.  This avoids rewriting the
model in JAX and guarantees numerical equivalence with the GPU reference.
"""

import functools
from typing import Any

import jax
import torch
import torch.nn
import torchax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from transformers import AutoModel

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import \
    shard_model_to_tpu
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class _DFlashRunner(torch.nn.Module):
    """Wrapper that adapts the HF DFlash model for ``functional_call``."""

    def __init__(self, dflash_model: torch.nn.Module):
        super().__init__()
        self.dflash = dflash_model

    def forward(self, **kwargs) -> torch.Tensor:
        if "hidden_state" in kwargs:
            return self._compute_logits(kwargs["hidden_state"],
                                        kwargs["embed_weight"])
        elif "raw_hidden" in kwargs:
            return self._combine_hidden(kwargs["raw_hidden"])
        else:
            return self._draft_forward(
                kwargs["noise_embedding"],
                kwargs["target_hidden"],
                kwargs["position_ids"],
                kwargs.get("attention_mask"),
            )

    def _draft_forward(
        self,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the DFlash model (no KV cache, no causal mask).

        Args:
            noise_embedding: (1, block_size, D) embedded noise block.
            target_hidden: (1, ctx_len, num_layers*D) raw concatenated
                           target hidden states (NOT yet projected).
            position_ids: (1, ctx_len + block_size) positions for RoPE
                          covering both context and noise.
            attention_mask: (1, 1, 1, ctx_len + block_size) additive float
                           bias (bf16): 0.0 for valid ctx/noise keys and
                           finfo(bf16).min for padding ctx keys. Added
                           directly onto attn_weights in eager attention.
        Returns:
            hidden_states: (1, block_size, D) – the draft model output
                           after the final norm (before lm_head).
        """
        return self.dflash(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )

    def _combine_hidden(self, raw_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target hidden states through fc + norm."""
        return self.dflash.hidden_norm(self.dflash.fc(raw_hidden))

    @staticmethod
    def _compute_logits(
        hidden_state: torch.Tensor,
        logits_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Draft logits: hidden @ W^T, where W is the target's OUTPUT
        projection (lm_head). For an untied target (gpt-oss) this is a
        distinct matrix from the input embedding; the proposer passes
        ``lm_head_weight_jax`` here, falling back to the embedding only when
        the target ties them."""
        return torch.nn.functional.linear(hidden_state, logits_weight)


class DFlashTorchaxWrapper:
    """Load the HF DFlash model on CPU, shard to TPU, expose JIT-compiled
    pure functions that the DFlash proposer can call."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.model: _DFlashRunner | None = None
        self.params: dict | None = None
        self.embed_weight_jax: jax.Array | None = None
        # Output projection (lm_head) weight used to turn draft hidden states
        # into draft logits. For a target with tie_word_embeddings=False (e.g.
        # gpt-oss) this is a DISTINCT matrix from the input embedding, so it
        # must be captured separately. Falls back to the input embedding when
        # the target ties them.
        self.lm_head_weight_jax: jax.Array | None = None

    def load(
        self,
        draft_model_path: str,
        target_model_state: Any,
    ) -> None:
        """Load HF DFlash model, shard weights to TPU, share embeddings."""

        logger.info("Loading DFlash PyTorch model via AutoModel from %s",
                    draft_model_path)

        with jax.default_device(jax.devices("cpu")[0]):
            hf_model = AutoModel.from_pretrained(
                draft_model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",  # pure-PyTorch ops for torchax
            )
            hf_model.eval()

        self.model = _DFlashRunner(hf_model)
        self.params = shard_model_to_tpu(self.model, self.mesh)
        # shard_model_to_tpu returns torchax tensors; convert to JAX view
        # so JAX JIT can trace through them.
        self.params = jax_view(self.params)

        # Share weights from the target model. The DFlash draft model owns NO
        # embedding / lm_head of its own; the custom modeling code consumes a
        # `noise_embedding` (built from the target INPUT embedding) and projects
        # its output through the target OUTPUT projection (lm_head). For an
        # untied target (gpt-oss: tie_word_embeddings=False) these are two
        # DISTINCT matrices, so we capture both: the input embedding for the
        # noise embedding, and lm_head for the draft logits. The HF DFlash
        # reference does exactly this: noise_embedding = target.embed_tokens(.),
        # draft_logits = target.lm_head(draft_hidden).
        #
        # Two backends pass different target_model_state shapes:
        #  - flax_nnx target: an nnx.State whose `.model.embed_tokens`/`.embed`
        #    exposes the embedding (`.embedding` or `.weight`).
        #  - vllm/torchax target (our gpt-oss path): a flat params dict of
        #    jax.Arrays keyed by torch-style fully-qualified names. The gpt-oss
        #    input embedding lives at `vllm_model.model.embedding.weight`
        #    (VocabParallelEmbedding, replicated global jax.Array [vocab, hidden]).
        if isinstance(target_model_state, dict):
            for key in (
                    "vllm_model.model.embedding.weight",
                    "vllm_model.model.embed_tokens.weight",
                    "model.embedding.weight",
                    "model.embed_tokens.weight",
            ):
                if key in target_model_state:
                    self.embed_weight_jax = target_model_state[key]
                    break
            # The draft logits use the target's OUTPUT projection (lm_head),
            # NOT the input embedding. These are the same weight only when the
            # target ties them; gpt-oss has tie_word_embeddings=False, so the
            # lm_head is a separate [vocab, hidden] matrix. Capture it here and
            # fall back to the input embedding for tied targets.
            for key in (
                    "vllm_model.lm_head.weight",
                    "lm_head.weight",
            ):
                if key in target_model_state:
                    self.lm_head_weight_jax = target_model_state[key]
                    break
        else:
            target_model = getattr(target_model_state, "model", None)
            embed = None
            if target_model is not None:
                embed = getattr(target_model, "embed_tokens", None)
                if embed is None:
                    embed = getattr(target_model, "embed", None)
            if embed is not None:
                if hasattr(embed, "embedding"):
                    w = embed.embedding
                    # Unwrap nnx.Param → raw jax.Array
                    self.embed_weight_jax = w.value if hasattr(w,
                                                               "value") else w
                elif hasattr(embed, "weight"):
                    w = embed.weight
                    if hasattr(w, "value"):
                        self.embed_weight_jax = w.value
                    elif isinstance(w, torch.Tensor):
                        self.embed_weight_jax = jax_view(w)
                    else:
                        self.embed_weight_jax = w
        if self.embed_weight_jax is None:
            raise RuntimeError(
                "Could not find target model embedding to share with DFlash")

        # Tied-embedding targets expose no separate lm_head weight; reuse the
        # input embedding for the logits projection in that case.
        if self.lm_head_weight_jax is None:
            logger.info(
                "No separate target lm_head weight found; assuming tied "
                "embeddings and using the input embedding for DFlash draft "
                "logits.")
            self.lm_head_weight_jax = self.embed_weight_jax

        logger.info("DFlash torchax wrapper loaded successfully.")

    def get_draft_forward_fn(self):
        """Return a JIT-compiled draft forward function.

        Signature::

            draft_forward(params, noise_input_ids, target_hidden,
                          position_ids, embed_weight,
                          attention_mask) -> hidden_states
        """
        model = self.model

        # Output is 3-D (N, block_size, D). MLP_DATA (= mesh 'data') is size 1
        # at DP=1 so the leading request axis is replicated (no divisibility
        # constraint on N); block + hidden stay replicated.
        hidden_sharding = NamedSharding(
            self.mesh, PartitionSpec(ShardingAxisName.MLP_DATA, None, None))

        @functools.partial(jax.jit, out_shardings=hidden_sharding)
        def draft_forward(
            params: dict,
            noise_input_ids: jax.Array,
            target_hidden: jax.Array,
            position_ids: jax.Array,
            embed_weight: jax.Array,
            attention_mask: jax.Array,
        ) -> jax.Array:
            with torchax.default_env():
                # All inputs carry a leading request axis N (>= 1):
                #   noise_input_ids (N, B), target_hidden (N, C, D'),
                #   position_ids (N, C+B), attention_mask (N, C+B).
                p = torch_view(params)
                noise_ids_t = torch_view(noise_input_ids)  # (N, B)
                embed_w_t = torch_view(embed_weight)
                noise_emb = torch.nn.functional.embedding(
                    noise_ids_t, embed_w_t)  # (N, B, D)

                target_h = torch_view(target_hidden)  # (N, C, D')
                pos = torch_view(position_ids)  # (N, C+B)
                # attention_mask is an additive float bias (bf16) of shape
                # (N, C+B). Reshape to (N, 1, 1, C+B) so it broadcasts over the
                # head + query axes of attn_weights (N, H, B, C+B) in the HF
                # eager_attention_forward add -- masks keys per request, never
                # queries.
                mask_t = torch_view(attention_mask)  # (N, C+B)
                mask = mask_t.reshape(mask_t.shape[0], 1, 1,
                                      mask_t.shape[1])  # (N,1,1,C+B)

                output = torch.func.functional_call(
                    model,
                    p,
                    kwargs={
                        "noise_embedding": noise_emb,
                        "target_hidden": target_h,
                        "position_ids": pos,
                        "attention_mask": mask,
                    },
                    tie_weights=False,
                )
                # output: (N, block_size, D)
                return jax_view(output)

        return draft_forward

    def get_combine_hidden_fn(self):
        """Return a JIT-compiled combine_hidden_states function.

        Signature::

            combine_fn(params, raw_hidden) -> projected_hidden
        """
        model = self.model

        hidden_sharding = NamedSharding(
            self.mesh, PartitionSpec(ShardingAxisName.MLP_DATA, None))

        @functools.partial(jax.jit, out_shardings=hidden_sharding)
        def combine_fn(
            params: dict,
            raw_hidden: jax.Array,
        ) -> jax.Array:
            with torchax.default_env():
                p = torch_view(params)
                h = torch_view(raw_hidden)
                out = torch.func.functional_call(
                    model,
                    p,
                    kwargs={"raw_hidden": h},
                    tie_weights=False,
                )
                return jax_view(out)

        return combine_fn

    def get_compute_logits_fn(self):
        """Return a JIT-compiled compute_logits function.

        Signature::

            logits_fn(params, hidden_states, logits_weight) -> logits

        ``logits_weight`` is the target's OUTPUT projection (lm_head), not the
        input embedding (they differ for an untied target like gpt-oss).
        """
        model = self.model

        # Batched: hidden in (N, num_spec, D) -> logits (N, num_spec, vocab).
        # Leading request axis on MLP_DATA (size 1 -> replicated at DP=1);
        # vocab stays tensor-parallel on MLP_TENSOR exactly as before.
        logits_sharding = NamedSharding(
            self.mesh,
            PartitionSpec(ShardingAxisName.MLP_DATA, None,
                          ShardingAxisName.MLP_TENSOR))

        @functools.partial(jax.jit, out_shardings=logits_sharding)
        def logits_fn(
            params: dict,
            hidden_states: jax.Array,
            logits_weight: jax.Array,
        ) -> jax.Array:
            with torchax.default_env():
                p = torch_view(params)
                h = torch_view(hidden_states)
                w = torch_view(logits_weight)
                out = torch.func.functional_call(
                    model,
                    p,
                    kwargs={
                        "hidden_state": h,
                        "embed_weight": w
                    },
                    tie_weights=False,
                )
                return jax_view(out)

        return logits_fn
