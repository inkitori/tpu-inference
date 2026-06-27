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
"""DFlash proposer for speculative decoding via torchax.

Stateless variant: each propose() call runs the HF DFlash PyTorch model
on the full accumulated context, trading K/V recomputation for the cost
of avoiding on-device KV cache management. Complements the JAX-native
proposer in ``spec_decode.jax.dflash``.
"""

import functools
from dataclasses import replace
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from vllm.config import VllmConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.logger import init_logger
from tpu_inference.utils import device_array

logger = init_logger(__name__)


class DFlashTorchaxProposer:
    """Stateless DFlash proposer running through torchax."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        runner: Any,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method

        self.runner = runner
        self.mesh = runner.mesh
        self.num_speculative_tokens = (
            self.speculative_config.num_speculative_tokens)

        hf_config = self.draft_model_config.hf_config
        self.block_size = getattr(hf_config, "block_size",
                                  self.num_speculative_tokens + 1)
        dflash_config = getattr(hf_config, "dflash_config", {})
        self.mask_token_id = dflash_config.get("mask_token_id", 0)
        self.hidden_size = hf_config.hidden_size
        self.num_layers = hf_config.num_hidden_layers

        target_layer_ids = dflash_config.get("target_layer_ids", None)
        num_target_layers = getattr(hf_config, "num_target_layers", None)
        if target_layer_ids is not None:
            self._num_target_layers = len(target_layer_ids)
        elif num_target_layers is not None:
            self._num_target_layers = num_target_layers
        else:
            self._num_target_layers = hf_config.num_hidden_layers
        target_hidden_size = getattr(hf_config, "target_hidden_size",
                                     self.hidden_size)
        self._raw_hidden_dim = self._num_target_layers * target_hidden_size

        self.rng_key = jax.random.key(self.vllm_config.model_config.seed)
        self.max_num_tokens = runner.max_num_tokens
        self.max_model_len = runner.max_model_len
        self.max_num_reqs = getattr(runner, "max_num_reqs",
                                    getattr(runner, "max_num_seqs", 1))

        # Per-slot proposer state (one entry per concurrent request slot).
        # Batched decode runs every slot in lock-step through a single
        # rectangular forward; each slot keeps its own context buffer + length.
        self._ctx_len: np.ndarray = np.zeros(self.max_num_reqs, dtype=np.int64)
        self._prev_seq_len: np.ndarray = np.zeros(self.max_num_reqs,
                                                  dtype=np.int64)
        self._last_req_id: list[Optional[str]] = [None] * self.max_num_reqs
        self._ctx_buf: Optional[jax.Array] = None
        self._wrapper = None
        self._draft_forward_fn = None
        self._compute_logits_fn = None
        self._params: Optional[dict] = None
        self._embed_weight: Optional[jax.Array] = None
        # Target output projection (lm_head); distinct from _embed_weight when
        # the target does not tie its embeddings (e.g. gpt-oss). Used for draft
        # logits, NOT for the noise embedding.
        self._lm_head_weight: Optional[jax.Array] = None

    def load_model(self, target_model: Any) -> None:
        """Load the DFlash draft model via torchax and share embeddings."""
        from tpu_inference.models.vllm.dflash import DFlashTorchaxWrapper
        self._wrapper = DFlashTorchaxWrapper(self.mesh)
        self._wrapper.load(self.draft_model_config.model, target_model)

        self._draft_forward_fn = self._wrapper.get_draft_forward_fn()
        self._compute_logits_fn = self._wrapper.get_compute_logits_fn()
        self._params = self._wrapper.params
        self._embed_weight = self._wrapper.embed_weight_jax
        self._lm_head_weight = self._wrapper.lm_head_weight_jax

        buf_len = self._next_padded_size(self.max_model_len)
        self._ctx_buf = jnp.zeros(
            (self.max_num_reqs, buf_len, self._raw_hidden_dim),
            dtype=jnp.bfloat16)

        logger.info(
            "DFlash torchax proposer loaded: context buffer shape %s",
            self._ctx_buf.shape,
        )

    def precompile(self) -> None:
        """Pre-warm JIT cache for every padded_ctx shape used at runtime.

        Draft proposal runs inside ``maybe_forbid_compile``, which raises
        when ``VLLM_XLA_CHECK_RECOMPILATION`` is on; this method must be
        invoked outside that guard before the first propose() call.
        """
        if self._draft_forward_fn is None:
            raise RuntimeError("precompile() called before load_model()")

        max_num_reqs = getattr(self.runner, "max_num_reqs",
                               getattr(self.runner, "max_num_seqs", 1))

        # Match the rounded-up size that prepare_inputs() actually uses,
        # otherwise non-power-of-two max_model_len leaves the largest shape
        # unwarmed.
        target_max = self._next_padded_size(self.max_model_len)
        padded_sizes: list[int] = []
        p = 16
        while p <= target_max:
            padded_sizes.append(p)
            p *= 2

        logger.info("Precompiling DFlash torchax for %d padded_ctx shapes...",
                    len(padded_sizes))

        # Warm every batch size the scheduler may produce so the batched
        # draft forward / sampler do not cold-compile at decode time. Both the
        # leading request axis N and the padded context width C are static in
        # the JIT signature, so we sweep N x C.
        batch_sizes = sorted({1, max_num_reqs})
        seq_lens = device_array(self.mesh,
                                np.zeros((max_num_reqs, ), dtype=np.int32))
        next_token_ids = device_array(
            self.mesh, np.zeros((max_num_reqs, ), dtype=np.int32))

        for n in batch_sizes:
            noise_input_ids = device_array(
                self.mesh,
                np.zeros((n, self.block_size), dtype=np.int32))
            for padded_ctx in padded_sizes:
                ctx_padded = device_array(
                    self.mesh,
                    jnp.zeros((n, padded_ctx, self._raw_hidden_dim),
                              dtype=jnp.bfloat16))
                position_ids = device_array(
                    self.mesh,
                    jnp.zeros((n, padded_ctx + self.block_size),
                              dtype=jnp.int32))
                # Additive float bias; all-zeros = "no padding masked" is a
                # valid warm-up case. Dtype/shape MUST match prepare_inputs()
                # so the JIT signature is shared (bf16, shape (N, C+B)).
                attention_mask = device_array(
                    self.mesh,
                    jnp.zeros((n, padded_ctx + self.block_size),
                              dtype=jnp.bfloat16))

                hidden = self._draft_forward_fn(
                    self._params,
                    noise_input_ids,
                    ctx_padded,
                    position_ids,
                    self._embed_weight,
                    attention_mask,
                )
                _ = self._sample_block_draft_tokens(self._params, hidden,
                                                    self._lm_head_weight)

            _ = self._build_noise_block(seq_lens[:n], next_token_ids[:n],
                                        self.mask_token_id, self.block_size)

        logger.info("DFlash torchax precompile complete.")

    @staticmethod
    def _next_padded_size(n: int) -> int:
        """Round n up to the next power-of-two, min 16."""
        if n <= 16:
            return 16
        p = 16
        while p < n:
            p *= 2
        return p

    @functools.partial(jax.jit, static_argnums=(0, 3, 4))
    def _build_noise_block(
        self,
        seq_lens: jax.Array,
        next_token_ids: jax.Array,
        mask_token_id: int,
        block_size: int,
    ) -> tuple[jax.Array, jax.Array]:
        # Batched: one noise block per request slot.
        #   seq_lens, next_token_ids: shape (N,)
        #   noise_input_ids:  (N, block_size) -- first col = the slot's next
        #                     token, rest = mask_token_id.
        #   noise_positions:  (N, block_size) -- [seq_len_i .. seq_len_i+B-1].
        num_reqs = next_token_ids.shape[0]
        noise_input_ids = jnp.full((num_reqs, block_size),
                                   mask_token_id,
                                   dtype=jnp.int32)
        noise_input_ids = noise_input_ids.at[:, 0].set(
            next_token_ids.astype(jnp.int32))
        noise_positions = (jnp.arange(block_size, dtype=jnp.int32)[None, :] +
                           seq_lens.astype(jnp.int32)[:, None])
        return noise_input_ids, noise_positions

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _sample_block_draft_tokens(
        self,
        params: dict,
        hidden_states: jax.Array,
        logits_weight: jax.Array,
    ) -> jax.Array:
        # logits_weight is the target's OUTPUT projection (lm_head), which is a
        # different matrix from the input embedding for untied targets.
        # hidden_states: (N, block_size, D). Draft positions 1..num_spec per
        # request -> (N, num_spec, D); argmax over vocab -> (N, num_spec).
        draft_hidden = hidden_states[:, 1:1 + self.num_speculative_tokens]
        logits = self._compute_logits_fn(params, draft_hidden, logits_weight)
        return jnp.argmax(logits, axis=-1)

    def prepare_inputs(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        next_token_ids: jax.Array,
        num_rejected_tokens: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, AttentionMetadata]:
        assert aux_hidden_states is not None and len(aux_hidden_states) > 0

        # Number of REAL requests this step (the padded tail of seq_lens is
        # ignored). The scheduler may batch up to max_num_reqs requests.
        num_reqs = self.runner.input_batch.num_reqs

        # Per-slot proposer state must reset when a slot's request changes;
        # otherwise the previous request's hidden states would be treated as
        # the new request's prefix.
        req_ids = self.runner.input_batch.req_ids
        for i in range(num_reqs):
            cur = req_ids[i] if i < len(req_ids) else None
            if cur != self._last_req_id[i]:
                self._ctx_len[i] = 0
                self._prev_seq_len[i] = 0
                self._last_req_id[i] = cur

        # accepted context length per request this step (host int array).
        seq_lens_host = np.asarray(jax.device_get(attn_metadata.seq_lens))
        # query_start_loc maps the flat ragged token axis of aux_hidden_states
        # back to per-request segments: req i occupies [qsl[i] : qsl[i+1]).
        qsl_host = np.asarray(jax.device_get(attn_metadata.query_start_loc))

        # raw_hidden: flat-ragged [total_tokens, raw_hidden_dim], ordered by
        # request via query_start_loc. NOT a batch axis.
        raw_hidden = jnp.concatenate(aux_hidden_states, axis=-1)

        # Append each request's newly-accepted hidden states (the LEADING rows
        # of its query segment) into that slot's context buffer. Determine the
        # per-slot new-token counts on host so the on-device update shape stays
        # static per slot.
        for i in range(num_reqs):
            seq_len = int(seq_lens_host[i])
            # Recompute/repair on a (rare) shrink, mirroring the single-seq
            # logic: if the accepted length regressed below our cached ctx_len
            # (e.g. partial-prefill rollback), trust the new seq_len.
            if self._prev_seq_len[i] > 0 and seq_len < self._ctx_len[i]:
                self._ctx_len[i] = seq_len
            self._prev_seq_len[i] = seq_len

            num_new = seq_len - int(self._ctx_len[i])
            if num_new <= 0:
                self._ctx_len[i] = seq_len
                continue
            end = min(int(self._ctx_len[i]) + num_new, self.max_model_len)
            n_copy = end - int(self._ctx_len[i])
            seg_start = int(qsl_host[i])
            new_raw = raw_hidden[seg_start:seg_start + n_copy].astype(
                jnp.bfloat16)
            # Update slot i, rows [_ctx_len[i] : _ctx_len[i]+n_copy).
            self._ctx_buf = lax.dynamic_update_slice(
                self._ctx_buf, new_raw[None, ...],
                (i, int(self._ctx_len[i]), 0))
            self._ctx_len[i] = end

        # All slots share one rectangular padded context width = power-of-two
        # over the max accepted length, so the downstream JIT cache only sees
        # ~log2(max_model_len) shapes regardless of batch size.
        ctx_lens = self._ctx_len[:num_reqs].astype(np.int32)
        max_ctx = int(ctx_lens.max()) if num_reqs > 0 else 1
        padded_ctx = self._next_padded_size(max(max_ctx, 1))
        # (N, padded_ctx, D)
        ctx_padded = self._ctx_buf[:num_reqs, :padded_ctx]

        # Per-slot positions/mask. Layout per row: context positions
        # [0..ctx_len_i-1, 0, 0, ...] then noise positions
        # [ctx_len_i .. ctx_len_i+block_size-1]. Padding ctx rows get position 0
        # and a large-negative additive bias so they contribute nothing.
        ctx_lens_j = jnp.asarray(ctx_lens)  # (N,)
        ar_ctx = jnp.arange(padded_ctx, dtype=jnp.int32)[None, :]  # (1, C)
        ctx_valid = ar_ctx < ctx_lens_j[:, None]  # (N, C)
        ctx_positions = jnp.where(ctx_valid, ar_ctx,
                                  jnp.zeros_like(ar_ctx))  # (N, C)
        noise_positions = (jnp.arange(self.block_size, dtype=jnp.int32)[None, :]
                           + ctx_lens_j[:, None])  # (N, B)
        position_ids = jnp.concatenate([ctx_positions, noise_positions],
                                       axis=1)  # (N, C+B)

        # Additive float bias (NOT a {0,1} multiplicative mask): the cached HF
        # DFlash draft adds attention_mask straight onto attn_weights
        # (eager_attention_forward: attn_weights = attn_weights + mask) with no
        # _prepare_4d_attention_mask conversion. So padding ctx keys must get a
        # large-negative bias (-> ~0 softmax weight) and every real ctx/noise
        # key must get 0.0. Use dtype-min (not literal -inf) to match HF's
        # AttentionMaskConverter and avoid NaN. Bias is bf16, shape (N, C+B);
        # reshaped to broadcast over the KEY axis of attn_weights
        # (N, H, B, C+B) downstream.
        neg = jnp.finfo(jnp.bfloat16).min
        ctx_mask = jnp.where(
            ctx_valid,
            jnp.zeros((num_reqs, padded_ctx), dtype=jnp.bfloat16),
            jnp.full((num_reqs, padded_ctx), neg, dtype=jnp.bfloat16),
        )  # (N, C)
        noise_mask = jnp.zeros((num_reqs, self.block_size), dtype=jnp.bfloat16)
        attention_mask = jnp.concatenate([ctx_mask, noise_mask],
                                         axis=1)  # (N, C+B)

        target_hidden_states = (ctx_padded, position_ids, attention_mask)

        seq_lens_arr = device_array(self.mesh, ctx_lens)  # (N,)
        # next_token_ids is padded to max_num_reqs; take the real rows.
        noise_input_ids, _ = self._build_noise_block(
            seq_lens_arr,
            next_token_ids[:num_reqs],
            self.mask_token_id,
            self.block_size,
        )  # (N, block_size)

        num_kv_cache_groups = len(self.runner.kv_cache_config.kv_cache_groups)
        draft_kv_cache_group_id = num_kv_cache_groups - 1
        block_tables = (
            self.runner.input_batch.block_table[draft_kv_cache_group_id].
            get_cpu_tensor().reshape(-1))
        # query_start_loc for the stateless draft forward: each request emits
        # block_size noise tokens, concatenated -> [0, B, 2B, ..., N*B].
        draft_query_start_loc = jnp.arange(
            num_reqs + 1, dtype=jnp.int32) * self.block_size
        draft_attn_metadata = replace(
            attn_metadata,
            input_positions=noise_positions,
            query_start_loc=draft_query_start_loc,
            block_tables=device_array(self.mesh, block_tables),
        )

        dummy_last_indices = jnp.zeros(num_reqs, dtype=jnp.int32)
        return (
            target_hidden_states,
            noise_input_ids,
            dummy_last_indices,
            draft_attn_metadata,
        )

    def propose(
        self,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attn_metadata: AttentionMetadata,
        last_token_indices: jax.Array,
        target_hidden_states,
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        """Generate all draft tokens in one stateless torchax forward pass."""
        ctx_padded, position_ids, attention_mask = target_hidden_states

        hidden_states = self._draft_forward_fn(
            self._params,
            input_ids,
            ctx_padded,
            position_ids,
            self._embed_weight,
            attention_mask,
        )

        # draft_token_ids: (N, num_speculative_tokens), one row per request.
        draft_token_ids = self._sample_block_draft_tokens(
            self._params, hidden_states, self._lm_head_weight)

        return kv_caches, draft_token_ids
