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
"""Fused greedy verification for speculative decoding.

The unfused path materializes bonus/target logit rows with shard_map'd
row-selects whose ``PartitionSpec(data)`` in_specs force an all-gather of the
vocab-sharded ``(padded_tokens, vocab)`` logits on every chip (~1ms each at
32 reqs x 8 tokens x 201k vocab), then runs argmax over the replicated rows.

For greedy verification none of that is needed: the only thing consumed from
the logits is their per-row argmax. This module computes ONE cross-shard
argmax on the still-sharded logits (XLA lowers it to a local argmax plus a
tiny cross-chip combine) and performs draft-token extraction, greedy
rejection, and last-sampled/num-rejected extraction in a single jitted
program — replacing five separately dispatched modules.

Only valid for greedy verification without logprobs; callers must fall back
to the unfused path otherwise.
"""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID, _get_segment_info)


@jax.jit(static_argnames=[
    "num_speculative_tokens", "max_num_reqs_per_dp_rank", "vocab_size", "mesh"
])
def fused_greedy_verify(
    mesh: jax.sharding.Mesh,
    logits: jax.Array,  # (padded_tokens, vocab), vocab-sharded
    input_ids: jax.Array,  # (padded_total_scheduled_tokens,)
    draft_lengths: jax.Array,  # (dp * padded_num_reqs_per_rank,)
    target_logits_indices: jax.Array,  # (dp * padded_logits_len_per_rank,)
    bonus_logits_indices: jax.Array,  # (dp * padded_num_reqs_per_rank,)
    final_logits_indices: jax.Array,  # (dp * padded_logits_len_per_rank,)
    num_speculative_tokens: int,
    max_num_reqs_per_dp_rank: int,
    vocab_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Greedy-verify draft tokens against target logits in one fused program.

    Returns:
        output_token_ids: same layout as ``RejectionSampler`` output —
            per rank ``[main_tokens (padded_logits_len), bonus (num_reqs)]``.
        last_sampled_tokens: (max_num_reqs,) last accepted/bonus token per seq.
        num_rejected_tokens: (max_num_reqs,) rejected count per seq.
    """
    # One argmax over the vocab-sharded logits; XLA emits a per-shard argmax
    # plus a small cross-chip combine (no logits all-gather).
    greedy_ids = jnp.argmax(logits, axis=-1).astype(jnp.int32)

    def _body(greedy_ids, input_ids, draft_lens, target_idx, bonus_idx,
              final_idx):
        # Draft token ids as fed to the verifier: the input id at the NEXT
        # scheduled position of each target row (see _extract_draft_token_ids).
        gathered_ids = input_ids[final_idx]
        draft_token_ids = gathered_ids[target_idx + 1]

        target_argmax = greedy_ids[target_idx]
        bonus_candidates = greedy_ids[bonus_idx]

        # --- greedy rejection (see _greedy_rejection_sample_with_segment) ---
        total_tokens = draft_token_ids.shape[0]
        batch_size = draft_lens.shape[0]
        segment_ids, group_indices = _get_segment_info(draft_lens,
                                                       total_tokens)
        mismatches = draft_token_ids != target_argmax
        large_value = total_tokens
        mismatch_indices = jnp.where(mismatches, group_indices, large_value)
        first_mismatch = jax.ops.segment_min(
            mismatch_indices.astype(jnp.int32),
            segment_ids,
            num_segments=batch_size,
            indices_are_sorted=True,
        )
        max_int = jnp.iinfo(jnp.int32).max
        first_mismatch = jnp.where(first_mismatch == max_int, large_value,
                                   first_mismatch)
        first_mismatch_b = jnp.repeat(first_mismatch,
                                      draft_lens,
                                      total_repeat_length=total_tokens)
        main_tokens = jnp.where(group_indices <= first_mismatch_b,
                                target_argmax, PLACEHOLDER_TOKEN_ID)
        all_accepted = first_mismatch == large_value
        should_get_bonus = all_accepted | (draft_lens == 0)
        bonus_tokens = jnp.where(should_get_bonus, bonus_candidates,
                                 PLACEHOLDER_TOKEN_ID)
        output = jnp.concatenate([main_tokens, bonus_tokens])

        # --- last sampled + num rejected (see _extract_last_sampled_tokens) --
        index_range = jax.lax.broadcasted_iota(
            jnp.int32, (batch_size, num_speculative_tokens), 1)
        valid_mask = index_range < draft_lens[:, None]
        segment_starts = jnp.pad(jnp.cumsum(draft_lens)[:-1], (1, 0),
                                 constant_values=0)
        main_indices = jnp.where(valid_mask, segment_starts[:, None] +
                                 index_range, 0)
        mt = jnp.where(valid_mask, main_tokens[main_indices],
                       PLACEHOLDER_TOKEN_ID)
        mt = jnp.where(mt < vocab_size, mt, PLACEHOLDER_TOKEN_ID)
        bt = jnp.where(bonus_tokens < vocab_size, bonus_tokens,
                       PLACEHOLDER_TOKEN_ID)
        num_valid_main = jnp.sum(mt != PLACEHOLDER_TOKEN_ID, axis=1)
        last_main_idx = jnp.maximum(num_valid_main - 1, 0)
        last_main = mt[jnp.arange(batch_size), last_main_idx]
        last_main = jnp.where(num_valid_main > 0, last_main,
                              PLACEHOLDER_TOKEN_ID)
        has_bonus = bt != PLACEHOLDER_TOKEN_ID
        last_sampled = jnp.where(has_bonus, bt, last_main)
        last_sampled = jnp.pad(last_sampled,
                               (0, max_num_reqs_per_dp_rank - batch_size),
                               constant_values=PLACEHOLDER_TOKEN_ID)
        num_rejected = jnp.where(
            draft_lens > 0,
            draft_lens + 1 - num_valid_main - has_bonus.astype(jnp.int32),
            jnp.zeros_like(draft_lens))
        num_rejected = jnp.pad(num_rejected,
                               (0, max_num_reqs_per_dp_rank - batch_size),
                               constant_values=0)
        return output, last_sampled, num_rejected

    data_spec = PartitionSpec(ShardingAxisName.ATTN_DATA)
    return jax.shard_map(
        _body,
        mesh=mesh,
        in_specs=(data_spec, ) * 6,
        out_specs=(data_spec, data_spec, data_spec),
    )(greedy_ids, input_ids, draft_lengths, target_logits_indices,
      bonus_logits_indices, final_logits_indices)
