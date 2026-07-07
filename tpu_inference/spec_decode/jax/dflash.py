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
"""Batched DFlash proposer for speculative decoding on JAX/TPU.

DFlash drafts a whole block of ``block_size`` tokens in ONE non-causal draft
forward per step (no autoregressive loop like eagle3). This proposer keeps
the eagle3 interface (``prepare_inputs`` / ``propose`` with the same
signatures) so the SpeculativeDecodingManager and the async-scheduling path
are reused unchanged.

Per step, for each request with ``q_i`` scheduled tokens and ``n_i`` rejected
tokens (``a_i = q_i - n_i`` accepted rows, committed length ``N'``):

  flat draft stream: [ a_i ctx rows | B noise rows ]      (per request)
    ctx rows    <- freshly accepted target rows; their K/V (projected from
                   the target's aux hidden states) are written into the
                   draft's paged KV cache at positions [N-1, N'-1)
    noise rows  <- [bonus token, mask, ..., mask] at positions [N'-1, N'-1+B)

  kv_len = (N'-1) + B, and the paged-attention kernel's END-ALIGNED write of
  the ``a_i + B`` new K/V rows lands exactly on [N-1, N'-1+B). Attention is
  non-causal, so every noise token sees the full context plus the whole
  block. Draft tokens are argmax(target_lm_head(hidden[noise rows 1..S])).

The draft KV cache lives in the LAST ``num_draft_layers`` framework KV-cache
groups (standard block tables), so batching, slot management, and preemption
are handled by vLLM — the proposer holds no per-request state.
"""

import os
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from vllm.config import VllmConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import get_model
from tpu_inference.utils import device_array

logger = init_logger(__name__)


class DFlashProposer:
    """Block-diffusion (DFlash) drafter with framework-paged draft KV cache."""

    def __init__(
            self,
            vllm_config: VllmConfig,
            runner: Any,  # TPUModelRunner
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
        if self.num_speculative_tokens > self.block_size - 1:
            raise ValueError(
                f"num_speculative_tokens={self.num_speculative_tokens} "
                f"exceeds draft block_size-1={self.block_size - 1}")
        dflash_config = getattr(hf_config, "dflash_config", None) or {}
        self.mask_token_id = dflash_config.get("mask_token_id")
        assert self.mask_token_id is not None, (
            "DFlash draft config must provide dflash_config.mask_token_id")

        if runner.dp_size != 1:
            raise NotImplementedError(
                "DFlash speculative decoding does not support DP > 1 yet.")

        self.max_model_len = runner.max_model_len
        self.rng_key = jax.random.key(vllm_config.model_config.seed)

    def load_model(self, target_state: Any) -> None:
        """Load the draft model; share target embed_tokens and lm_head."""
        model = get_model(self.vllm_config,
                          self.rng_key,
                          self.mesh,
                          is_draft_model=True)
        self.model_fn = model.model_fn
        self.compute_logits_fn = model.compute_logits_fn
        self.combine_hidden_states_fn = model.combine_hidden_states_fn
        self.state = model.state
        self.model = model.model

        embed_w, lm_head_w = self._find_target_weights(target_state)
        # gpt-oss has untied embeddings: embed_tokens embeds the noise block,
        # lm_head projects draft hidden states to logits. Sharing references
        # (no copy) keeps the target's sharding.
        self.state.embed_tokens.value = embed_w
        self.state.lm_head.value = lm_head_w
        logger.info(
            "DFlash draft sharing target weights: embed_tokens%s lm_head%s",
            embed_w.shape, lm_head_w.shape)

        if isinstance(self.state, nnx.State):
            self.state_leaves = tuple(jax.tree_util.tree_leaves(self.state))
        else:
            self.state_leaves = self.state

    @staticmethod
    def _find_target_weights(target_state: Any):
        """Locate the target's input embedding and lm_head weights."""
        if hasattr(target_state, "items"):  # torchax params dict
            candidates = dict(target_state.items())
            embed_keys = [
                "vllm_model.model.embed_tokens.weight",
                "vllm_model.model.embedding.weight",  # gpt-oss
                "vllm_model.language_model.model.embed_tokens.weight",
            ]
            head_keys = [
                "vllm_model.lm_head.weight",
                "vllm_model.language_model.lm_head.weight",
            ]
            embed_w = next((candidates[k]
                            for k in embed_keys if k in candidates), None)
            lm_head_w = next((candidates[k]
                              for k in head_keys if k in candidates), None)
            if lm_head_w is None:
                lm_head_w = embed_w  # tied-embedding targets
            if embed_w is not None:
                return embed_w, lm_head_w
            raise RuntimeError(
                "DFlash: could not find target embed_tokens/lm_head in the "
                f"torchax state. Available keys (sample): "
                f"{[k for k in list(candidates) if 'embed' in k or 'head' in k][:10]}"
            )
        # flax_nnx target state
        from tpu_inference.models.jax.utils.weight_utils import get_param
        embed_w = lm_head_w = None
        for path in ("model.embed_tokens.weight", "model.embed.embedding",
                     "model.embed_tokens.embedding",
                     "embedder.input_embedding_table_VD"):
            try:
                embed_w = get_param(target_state, path).value
                break
            except ValueError:
                continue
        for path in ("lm_head.weight", "lm_head.input_embedding_table_DV"):
            try:
                lm_head_w = get_param(target_state, path).value
                break
            except ValueError:
                continue
        if embed_w is None:
            raise RuntimeError(
                "DFlash: could not locate target embedding in nnx state")
        if lm_head_w is None:
            lm_head_w = embed_w
        # gpt-oss JAX lm_head is stored (D, V); normalize to (V, D).
        if lm_head_w.shape[0] == embed_w.shape[1]:
            lm_head_w = lm_head_w.T
        return embed_w, lm_head_w

    def _draft_kv_cache_group_id(self) -> int:
        if getattr(self, "_draft_group_id", None) is None:
            groups = self.runner.kv_cache_config.kv_cache_groups
            for gid, group in enumerate(groups):
                if "draft_layer.0" in group.layer_names:
                    self._draft_group_id = gid
                    break
            else:
                raise RuntimeError(
                    "No KV cache group contains draft_layer.0; groups: " +
                    str([g.layer_names for g in groups]))
        return self._draft_group_id

    def prepare_inputs(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        last_sampled_token_id: jax.Array,
        next_prompt_token_id: jax.Array,
        is_in_prefill: jax.Array,
        num_rejected_tokens: jax.Array,
        num_reqs_dp: jax.Array,
    ) -> tuple[Any, jax.Array, jax.Array, AttentionMetadata]:
        assert aux_hidden_states, (
            "DFlash requires auxiliary hidden states from the target model.")

        # Use the block tables of whichever KV-cache group holds the draft
        # layers (they may be merged with same-spec target layers, e.g.
        # gpt-oss full-attention, rather than forming their own group).
        # The SpeculativeDecodingManager hands us the draft layer's
        # AttentionMetadata, whose block tables are already on device (part of
        # the runner's packed metadata blob); re-uploading the CPU block table
        # here would cost an extra host transfer per step.
        block_tables = attn_metadata.block_tables
        if block_tables is None:
            draft_kv_cache_group_id = self._draft_kv_cache_group_id()
            block_tables = self.runner.input_batch.block_table[
                draft_kv_cache_group_id].get_cpu_tensor().reshape(-1)
            block_tables = device_array(self.mesh, block_tables)

        return self._prepare_inputs(
            self.state_leaves,
            block_tables,
            attn_metadata,
            input_ids,
            tuple(aux_hidden_states),
            last_sampled_token_id,
            next_prompt_token_id,
            is_in_prefill,
            num_rejected_tokens,
            num_reqs_dp,
        )

    @jax.jit(static_argnums=(0, ))
    def _prepare_inputs(
        self,
        state_leaves: Any,
        block_tables: jax.Array,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        last_sampled_token_id: jax.Array,
        next_prompt_token_id: jax.Array,
        is_in_prefill: jax.Array,
        num_rejected_tokens: jax.Array,
        num_reqs_dp: jax.Array,
    ):
        return self._prepare_inputs_impl(
            state_leaves, block_tables, attn_metadata, input_ids,
            aux_hidden_states, last_sampled_token_id, next_prompt_token_id,
            is_in_prefill, num_rejected_tokens, num_reqs_dp)

    def _prepare_inputs_impl(
        self,
        state_leaves: Any,
        block_tables: jax.Array,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        last_sampled_token_id: jax.Array,
        next_prompt_token_id: jax.Array,
        is_in_prefill: jax.Array,
        num_rejected_tokens: jax.Array,
        num_reqs_dp: jax.Array,
    ):
        B = self.block_size
        S = self.num_speculative_tokens
        qsl = attn_metadata.query_start_loc  # (max_reqs + 1,)
        seq_lens = attn_metadata.seq_lens  # (max_reqs,)
        max_reqs = seq_lens.shape[0]
        T_in = input_ids.shape[0]
        T_out = T_in + B * max_reqs

        num_reqs = num_reqs_dp.reshape(-1)[0]
        req_ids = jnp.arange(max_reqs, dtype=jnp.int32)
        req_mask = req_ids < num_reqs

        q_lens = jnp.where(req_mask, qsl[1:] - qsl[:-1], 0)
        n_rej = jnp.where(req_mask,
                          num_rejected_tokens.reshape(-1)[:max_reqs], 0)
        # Accepted target rows -> fresh ctx rows for the draft cache.
        a = jnp.maximum(q_lens - n_rej, 0)
        m = jnp.where(req_mask, a + B, 0)
        out_qsl = jnp.concatenate(
            [jnp.zeros((1, ), jnp.int32),
             jnp.cumsum(m).astype(jnp.int32)])
        # Committed-minus-newest length: ctx covers [0, N'-1).
        new_seq_lens = jnp.where(req_mask, seq_lens - n_rej, 0)

        # Map each output row to (request, local offset).
        t = jnp.arange(T_out, dtype=jnp.int32)
        req_of = jnp.sum(t[:, None] >= out_qsl[None, 1:],
                         axis=1).astype(jnp.int32)
        req_of = jnp.clip(req_of, 0, max_reqs - 1)
        local = t - out_qsl[req_of]
        a_t = a[req_of]
        valid = t < out_qsl[max_reqs]
        is_ctx = valid & (local < a_t)
        is_noise = valid & (local >= a_t)
        noise_off = jnp.clip(local - a_t, 0, B - 1)

        # Gather source rows (rejection-trimmed) for ctx rows.
        src = jnp.clip(qsl[req_of] + local, 0, T_in - 1)

        raw = jnp.concatenate(aux_hidden_states, axis=-1)
        combined_all = self.combine_hidden_states_fn(state_leaves, raw)
        combined_out = jnp.where(is_ctx[:, None], combined_all[src],
                                 0).astype(combined_all.dtype)

        first_token = jnp.where(
            is_in_prefill.reshape(-1)[:max_reqs] != 0,
            next_prompt_token_id.reshape(-1)[:max_reqs],
            last_sampled_token_id.reshape(-1)[:max_reqs],
        ).astype(jnp.int32)
        ids_noise = jnp.where(noise_off == 0, first_token[req_of],
                              self.mask_token_id)
        ids_out = jnp.where(is_ctx, input_ids[src],
                            jnp.where(is_noise, ids_noise,
                                      0)).astype(jnp.int32)

        pos_noise = new_seq_lens[req_of] + noise_off
        positions_out = jnp.where(
            is_ctx, attn_metadata.input_positions[src],
            jnp.where(is_noise, pos_noise, 0)).astype(jnp.int32)

        # KV span = ctx [0, N'-1) plus the noise block. Clamped at
        # max_model_len: a request that close to the cap gets (rejectable)
        # garbage drafts but stays within its own pages.
        seq_lens_out = jnp.minimum(new_seq_lens + jnp.where(req_mask, B, 0),
                                   self.max_model_len).astype(jnp.int32)

        distribution = jnp.stack([
            jnp.zeros((), jnp.int32),
            jnp.zeros((), jnp.int32),
            num_reqs.astype(jnp.int32),
        ])

        draft_md = replace(
            attn_metadata,
            input_positions=positions_out,
            seq_lens=seq_lens_out,
            query_start_loc=out_qsl,
            request_distribution=distribution,
            block_tables=block_tables,
        )

        # Draft tokens come from noise rows 1..S (row 0 is the bonus token).
        last_token_indices = (out_qsl[:max_reqs] + a)[:, None] + 1 + jnp.arange(
            S, dtype=jnp.int32)[None, :]
        last_token_indices = jnp.clip(last_token_indices.reshape(-1), 0,
                                      T_out - 1)

        target_hidden_states = (combined_out, is_ctx)
        return target_hidden_states, ids_out, last_token_indices, draft_md

    def _prepare_split_impl(
        self,
        state_leaves: Any,
        block_tables: jax.Array,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        last_sampled_token_id: jax.Array,
        next_prompt_token_id: jax.Array,
        is_in_prefill: jax.Array,
        num_rejected_tokens: jax.Array,
        num_reqs_dp: jax.Array,
    ):
        """Split-stream layout: a compact ctx stream (size T_in) whose K/V
        are written up front, and a fixed-size noise stream (B * max_reqs)
        that alone runs through the draft transformer."""
        B = self.block_size
        S = self.num_speculative_tokens
        qsl = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_reqs = seq_lens.shape[0]
        T_in = input_ids.shape[0]
        T_noise = B * max_reqs

        num_reqs = num_reqs_dp.reshape(-1)[0]
        req_ids = jnp.arange(max_reqs, dtype=jnp.int32)
        req_mask = req_ids < num_reqs

        q_lens = jnp.where(req_mask, qsl[1:] - qsl[:-1], 0)
        n_rej = jnp.where(req_mask,
                          num_rejected_tokens.reshape(-1)[:max_reqs], 0)
        a = jnp.maximum(q_lens - n_rej, 0)
        new_seq_lens = jnp.where(req_mask, seq_lens - n_rej, 0)

        raw = jnp.concatenate(aux_hidden_states, axis=-1)
        combined_all = self.combine_hidden_states_fn(state_leaves, raw)

        distribution = jnp.stack([
            jnp.zeros((), jnp.int32),
            jnp.zeros((), jnp.int32),
            num_reqs.astype(jnp.int32),
        ])

        # ---- ctx stream: the rejection-trimmed accepted rows, compacted ----
        ctx_qsl = jnp.concatenate(
            [jnp.zeros((1, ), jnp.int32),
             jnp.cumsum(a).astype(jnp.int32)])
        t = jnp.arange(T_in, dtype=jnp.int32)
        req_of_c = jnp.sum(t[:, None] >= ctx_qsl[None, 1:],
                           axis=1).astype(jnp.int32)
        req_of_c = jnp.clip(req_of_c, 0, max_reqs - 1)
        local_c = t - ctx_qsl[req_of_c]
        valid_c = t < ctx_qsl[max_reqs]
        src_c = jnp.clip(qsl[req_of_c] + local_c, 0, T_in - 1)
        ctx_combined = jnp.where(valid_c[:, None], combined_all[src_c],
                                 0).astype(combined_all.dtype)
        ctx_positions = jnp.where(valid_c,
                                  attn_metadata.input_positions[src_c],
                                  0).astype(jnp.int32)
        # kv span for the ctx write ends at new_seq_lens (which is already
        # the committed-minus-newest length N'-1; see _prepare_inputs_impl):
        # the end-aligned a rows land on [N-1, N'-1), exactly matching the
        # combined-stream layout where kv_len = new_seq_lens + B covers ctx
        # then noise.
        ctx_seq_lens = jnp.where(req_mask, new_seq_lens, 0).astype(jnp.int32)
        ctx_md = replace(
            attn_metadata,
            input_positions=ctx_positions,
            seq_lens=ctx_seq_lens,
            query_start_loc=ctx_qsl,
            request_distribution=distribution,
            block_tables=block_tables,
        )

        # ---- noise stream: B rows per active request -----------------------
        noise_qsl = jnp.concatenate([
            jnp.zeros((1, ), jnp.int32),
            jnp.cumsum(jnp.where(req_mask, B, 0)).astype(jnp.int32)
        ])
        t2 = jnp.arange(T_noise, dtype=jnp.int32)
        req_of_n = jnp.sum(t2[:, None] >= noise_qsl[None, 1:],
                           axis=1).astype(jnp.int32)
        req_of_n = jnp.clip(req_of_n, 0, max_reqs - 1)
        noise_off = t2 - noise_qsl[req_of_n]
        valid_n = t2 < noise_qsl[max_reqs]

        first_token = jnp.where(
            is_in_prefill.reshape(-1)[:max_reqs] != 0,
            next_prompt_token_id.reshape(-1)[:max_reqs],
            last_sampled_token_id.reshape(-1)[:max_reqs],
        ).astype(jnp.int32)
        ids_noise = jnp.where(
            valid_n,
            jnp.where(noise_off == 0, first_token[req_of_n],
                      self.mask_token_id), 0).astype(jnp.int32)
        pos_noise = jnp.where(valid_n, new_seq_lens[req_of_n] + noise_off,
                              0).astype(jnp.int32)
        noise_seq_lens = jnp.minimum(
            new_seq_lens + jnp.where(req_mask, B, 0),
            self.max_model_len).astype(jnp.int32)
        noise_md = replace(
            attn_metadata,
            input_positions=pos_noise,
            seq_lens=noise_seq_lens,
            query_start_loc=noise_qsl,
            request_distribution=distribution,
            block_tables=block_tables,
        )

        # Draft tokens come from noise rows 1..S (row 0 is the bonus token).
        last_token_indices = noise_qsl[:max_reqs, None] + 1 + jnp.arange(
            S, dtype=jnp.int32)[None, :]
        last_token_indices = jnp.clip(last_token_indices.reshape(-1), 0,
                                      T_noise - 1)

        target_hidden_states = (ctx_combined, ctx_md)
        return target_hidden_states, ids_noise, last_token_indices, noise_md

    def propose(
        self,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attn_metadata: AttentionMetadata,
        last_token_indices: jax.Array,
        target_hidden_states,
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        return self._propose(
            self.state_leaves,
            kv_caches,
            input_ids,
            attn_metadata,
            last_token_indices,
            target_hidden_states,
            tuple(self.runner.layer_name_to_kvcache_index.items()),
        )

    @jax.jit(
        static_argnums=(0, 7),
        donate_argnames=("kv_caches", ),
    )
    def _propose(
        self,
        state_leaves: Any,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attn_metadata: AttentionMetadata,
        last_token_indices: jax.Array,
        target_hidden_states,
        layer_name_to_kvcache_index: tuple,
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        return self._propose_impl(state_leaves, kv_caches, input_ids,
                                  attn_metadata, last_token_indices,
                                  target_hidden_states,
                                  layer_name_to_kvcache_index)

    def _propose_impl(
        self,
        state_leaves: Any,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attn_metadata: AttentionMetadata,
        last_token_indices: jax.Array,
        target_hidden_states,
        layer_name_to_kvcache_index: tuple,
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        kv_caches, hidden_states, _, _ = self.model_fn(
            state_leaves,
            kv_caches,
            input_ids,
            target_hidden_states,
            attn_metadata,
            layer_name_to_kvcache_index,
            spec_step_idx=0,
        )

        sampled_hidden = hidden_states[last_token_indices]
        logits = self.compute_logits_fn(state_leaves, sampled_hidden, None)
        draft_token_ids = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        max_reqs = attn_metadata.seq_lens.shape[0]
        draft_token_ids = draft_token_ids.reshape(
            max_reqs, self.num_speculative_tokens)
        return kv_caches, draft_token_ids

    def prepare_and_propose(
        self,
        kv_caches: list[jax.Array],
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        last_sampled_token_id: jax.Array,
        packed_prefill_aux: jax.Array,
        num_rejected_tokens: jax.Array,
    ) -> tuple[list[jax.Array], jnp.ndarray, jnp.ndarray]:
        """Single-dispatch drafting step (prepare + draft forward + argmax +
        async next-token assembly). ``packed_prefill_aux`` packs
        ``[next_prompt_token_id | is_in_prefill | num_reqs_dp]`` into one
        int32 array so the manager pays one host transfer instead of three.
        """
        block_tables = attn_metadata.block_tables
        if block_tables is None:
            draft_kv_cache_group_id = self._draft_kv_cache_group_id()
            block_tables = self.runner.input_batch.block_table[
                draft_kv_cache_group_id].get_cpu_tensor().reshape(-1)
            block_tables = device_array(self.mesh, block_tables)
        return self._prepare_and_propose(
            self.state_leaves,
            kv_caches,
            block_tables,
            attn_metadata,
            input_ids,
            tuple(aux_hidden_states),
            last_sampled_token_id,
            packed_prefill_aux,
            num_rejected_tokens,
            tuple(self.runner.layer_name_to_kvcache_index.items()),
        )

    @jax.jit(
        static_argnums=(0, 10),
        donate_argnames=("kv_caches", ),
    )
    def _prepare_and_propose(
        self,
        state_leaves: Any,
        kv_caches: list[jax.Array],
        block_tables: jax.Array,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        last_sampled_token_id: jax.Array,
        packed_prefill_aux: jax.Array,
        num_rejected_tokens: jax.Array,
        layer_name_to_kvcache_index: tuple,
    ) -> tuple[list[jax.Array], jnp.ndarray, jnp.ndarray]:
        max_reqs = attn_metadata.seq_lens.shape[0]
        next_prompt_token_id = packed_prefill_aux[:max_reqs]
        is_in_prefill = packed_prefill_aux[max_reqs:2 * max_reqs]
        num_reqs_dp = packed_prefill_aux[2 * max_reqs:]

        # Split-stream drafting (ctx K/V written via a slim dummy-q kernel
        # pass + noise-only transformer) is DISABLED by default: it passes
        # single-request parity (bit-exact on 1 chip, cos 0.9999 on 8) but
        # under real multi-request serving the ragged tiny-q ctx-write pass
        # corrupts the draft cache (bench accept len 2.03 -> 1.26) and a
        # multi-request kernel repro hangs. Suspected RPA v3 edge case with
        # many tiny (0-4 row) non-causal segments. Keep for future debugging.
        prepare = (self._prepare_split_impl
                   if os.environ.get("SPEC_DFLASH_SPLIT", "0") == "1" else
                   self._prepare_inputs_impl)
        (target_hidden_states, ids_out, last_token_indices,
         draft_md) = prepare(
             state_leaves, block_tables, attn_metadata, input_ids,
             aux_hidden_states, last_sampled_token_id, next_prompt_token_id,
             is_in_prefill, num_rejected_tokens, num_reqs_dp)

        kv_caches, draft_token_ids = self._propose_impl(
            state_leaves, kv_caches, ids_out, draft_md, last_token_indices,
            target_hidden_states, layer_name_to_kvcache_index)

        # Async scheduling consumes [last_sampled | drafts] flattened per req
        # (see concat_last_sampled_tokens_and_draft_tokens).
        spec_next_tokens = jnp.concatenate(
            [last_sampled_token_id[:, None], draft_token_ids],
            axis=1).reshape(-1)
        return kv_caches, draft_token_ids, spec_next_tokens
