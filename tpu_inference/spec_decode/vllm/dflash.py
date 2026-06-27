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

        # Context padding granularity. The persistent per-slot context buffer
        # _ctx_buf is sized to a multiple of this block, and every per-step
        # forward slices a width that is also a multiple of this block, so the
        # downstream JIT cache only ever sees ~max_model_len/_CTX_PAD_BLOCK
        # context shapes regardless of batch size. Using a fixed block (rather
        # than rounding up to the next power of two) keeps the persistent buffer
        # close to the real max_model_len need: pow2 rounding of e.g. 4224 -> 8192
        # nearly doubles the buffer (32 * D=14400 * bf16 = ~7 GiB replicated),
        # which OOMs at concurrency 32. This is a pure memory-layout / shape
        # change: padding rows are attention-masked to ~0 and consumed only by
        # per-position ops, so produced draft tokens are bit-identical.
        self._ctx_pad_block = 512
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
        # Guarantee at least one dead padding row ABOVE max_model_len: the
        # batched ctx write routes every invalid scatter cell to the last buffer
        # row, which must never be a valid write target (valid dst rows are
        # < max_model_len). If max_model_len lands exactly on the pad block,
        # bump by one block so buf_len - 1 stays a dead row.
        if buf_len <= self.max_model_len:
            buf_len += self._ctx_pad_block
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

        # Warm exactly the set of padded_ctx widths prepare_inputs() can emit:
        # every multiple of _ctx_pad_block from one block up to the rounded-up
        # max_model_len. Any runtime width is _next_padded_size(max_ctx) for
        # 1 <= max_ctx <= max_model_len, which is always a member of this set,
        # so propose() (running inside maybe_forbid_compile) never hits an
        # unwarmed shape under VLLM_XLA_CHECK_RECOMPILATION.
        target_max = self._next_padded_size(self.max_model_len)
        block = self._ctx_pad_block
        padded_sizes: list[int] = list(range(block, target_max + 1, block))

        logger.info("Precompiling DFlash torchax for %d padded_ctx shapes...",
                    len(padded_sizes))

        # Warm every batch size the scheduler may produce so the batched
        # draft forward / sampler do not cold-compile at decode time. Both the
        # leading request axis N and the padded context width C are static in
        # the JIT signature, so we sweep N x C. The active batch size
        # fluctuates over the full [1, max_num_reqs] range during serving, and
        # propose() runs inside maybe_forbid_compile -- any unwarmed N would
        # hard-fail under VLLM_XLA_CHECK_RECOMPILATION (else stall mid-decode).
        batch_sizes = list(range(1, max_num_reqs + 1))
        seq_lens = device_array(self.mesh,
                                np.zeros((max_num_reqs, ), dtype=np.int32))
        next_token_ids = device_array(
            self.mesh, np.zeros((max_num_reqs, ), dtype=np.int32))

        # The draft forward always receives the FULL persistent context buffer
        # and slices the active (n, padded_ctx) sub-block inside the jit (static
        # args), so warm with the real buffer (shape never varies); n and
        # padded_ctx drive the static trace key, matching prepare_inputs().
        for n in batch_sizes:
            noise_input_ids = device_array(
                self.mesh,
                np.zeros((n, self.block_size), dtype=np.int32))
            for padded_ctx in padded_sizes:
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
                    self._ctx_buf,
                    position_ids,
                    self._embed_weight,
                    attention_mask,
                    n,
                    padded_ctx,
                )
                _ = self._sample_block_draft_tokens(self._params, hidden,
                                                    self._lm_head_weight)

            _ = self._build_noise_block(seq_lens[:n], next_token_ids[:n],
                                        self.mask_token_id, self.block_size)

        # Warm the batched context write for every raw_hidden token bucket it
        # can see at runtime. _batched_ctx_write's only varying-shape input is
        # raw_hidden (leading dim == one of the runner's token-padding buckets);
        # the index/mask arrays share that leading dim. prepare_inputs() runs
        # inside maybe_forbid_compile, so an unwarmed bucket would hard-fail
        # under VLLM_XLA_CHECK_RECOMPILATION. Donation rebinds _ctx_buf each
        # call; the warm-up plan is an all-invalid (no-op) scatter so the buffer
        # contents are preserved.
        token_buckets = getattr(self.runner, "num_tokens_paddings", None)
        if token_buckets is None:
            token_buckets = [self.max_num_tokens]
        dead_row = self._ctx_buf.shape[1] - 1
        logger.info("Precompiling DFlash batched ctx write for %d buckets...",
                    len(token_buckets))
        for t in token_buckets:
            slot_idx = device_array(self.mesh, np.zeros(t, dtype=np.int32))
            dst_row = device_array(self.mesh,
                                   np.full(t, dead_row, dtype=np.int32))
            valid = device_array(self.mesh, np.zeros(t, dtype=bool))
            raw_dummy = jnp.zeros((t, self._raw_hidden_dim),
                                  dtype=jnp.bfloat16)
            self._ctx_buf = self._batched_ctx_write(self._ctx_buf, raw_dummy,
                                                    slot_idx, dst_row, valid)

        # Warm the slot-permutation gather (mirrors condense/swap moves). It has
        # exactly one static shape -- gather_src is always (max_num_reqs,) -- so
        # one identity warm-up covers every runtime move. prepare_inputs() runs
        # inside maybe_forbid_compile, so an unwarmed move step would hard-fail
        # under VLLM_XLA_CHECK_RECOMPILATION the first time a condense fires.
        identity_src = device_array(
            self.mesh, np.arange(self.max_num_reqs, dtype=np.int32))
        self._ctx_buf = self._permute_ctx_rows(self._ctx_buf, identity_src)

        logger.info("DFlash torchax precompile complete.")

    def _next_padded_size(self, n: int) -> int:
        """Round n up to the next multiple of the context pad block (min one block).

        Used for both the persistent _ctx_buf width and the per-step forward
        slice width; the two MUST share this granularity so the slice never
        exceeds the allocation. A fixed block (vs next-power-of-two) keeps the
        persistent buffer near the real max_model_len need without nearly
        doubling it. Numerics are unaffected (padding is attention-masked).
        """
        block = self._ctx_pad_block
        if n <= block:
            return block
        return ((n + block - 1) // block) * block

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

    @functools.partial(jax.jit, static_argnums=(0, ), donate_argnums=(1, ))
    def _batched_ctx_write(
        self,
        ctx_buf: jax.Array,
        raw_hidden: jax.Array,
        slot_idx: jax.Array,
        dst_row: jax.Array,
        valid_mask: jax.Array,
    ) -> jax.Array:
        """Single jitted, input-donated, batched in-place context write.

        Replaces the per-request eager ``lax.dynamic_update_slice`` loop (each
        eager call copied the whole multi-GiB ``_ctx_buf``). Donating
        ``ctx_buf`` lets XLA mutate it in place; the function returns the
        updated buffer and the caller immediately rebinds it, so donation is
        safe (mirrors the proven write-only microbench).

        Implemented as ONE masked scatter indexed over the SOURCE token axis of
        raw_hidden. Source row r (a flat-ragged target hidden state) is copied
        to ``ctx_buf[slot_idx[r], dst_row[r]]`` iff ``valid_mask[r]``. This
        naturally handles any per-request copy count -- including the large
        prefill step where a slot appends many rows -- without a fixed
        block-width cap, because the scatter axis IS the (bucketed) token axis.

        The only varying-shape input is ``raw_hidden`` (leading dim == one of
        the runner's token-padding buckets); the per-row index/mask arrays share
        that same leading dim. precompile() warms every bucket, so the recompile
        guard never fires at decode time.

        Bit-identical to the eager loop: invalid source rows (rows past a
        request's n_copy, rows of inactive/no-write slots, and padding rows of
        the bucketed raw_hidden) write the destination's CURRENT value back
        (read-modify-write), so untouched rows are bit-preserved. All invalid
        rows are routed by the caller to a single dead padding row (the last
        buffer row, guaranteed by load_model() to never be a valid target), so
        duplicate-index scatter ordering can never clobber a real write.

        Args:
          ctx_buf:    (max_num_reqs, buf_len, D) bf16 buffer (donated).
          raw_hidden: (total_tokens, D) flat-ragged target hidden states.
          slot_idx:   (total_tokens,) int32 destination slot per source row;
                      0 for invalid rows.
          dst_row:    (total_tokens,) int32 destination row per source row; for
                      valid rows == ctx_len[i] + offset, for invalid rows == the
                      dead last buffer row (buf_len - 1).
          valid_mask: (total_tokens,) bool; True iff this source row is copied.
        """
        src = raw_hidden.astype(ctx_buf.dtype)  # (T, D)
        cur = ctx_buf[slot_idx, dst_row]  # (T, D)
        merged = jnp.where(valid_mask[:, None], src, cur)
        return ctx_buf.at[slot_idx, dst_row].set(merged)

    @functools.partial(jax.jit, static_argnums=(0, ), donate_argnums=(1, ))
    def _permute_ctx_rows(
        self,
        ctx_buf: jax.Array,
        gather_src: jax.Array,
    ) -> jax.Array:
        """Reorder ctx_buf's leading (slot) axis by a full permutation.

        Mirrors a vLLM ``InputBatch.condense`` / ``swap_states`` slot move onto
        the proposer's per-slot context buffer: after compaction/reorder, the
        request that now occupies physical slot ``i`` previously lived in slot
        ``gather_src[i]`` (or ``i`` itself if it did not move / is brand new).
        A single leading-axis gather rebuilds the buffer in the new layout.

        Donating ``ctx_buf`` lets XLA reuse the storage; the gather materializes
        the reordered rows into the result before the donated input is reclaimed,
        so swaps and cyclic moves (src/dst overlap) are safe -- no aliasing.

        Shape never varies: ``gather_src`` is always (max_num_reqs,), so this has
        exactly one static trace (warmed once in precompile()).

        Args:
          ctx_buf:    (max_num_reqs, buf_len, D) bf16 buffer (donated).
          gather_src: (max_num_reqs,) int32 source slot for each destination
                      slot; the identity for unmoved/new slots.
        """
        return ctx_buf[gather_src]

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

        # Per-slot proposer state must FOLLOW its request when vLLM compacts the
        # batch. InputBatch.condense (and swap_states / _reorder_batch) run
        # BEFORE the drafter each step, moving a still-running request from one
        # physical slot to another (e.g. a finished low slot is backfilled by a
        # long-running high slot). The per-slot state (_ctx_buf row, _ctx_len,
        # _prev_seq_len) is keyed by physical slot, so a move desyncs it: the
        # destination slot's req_id changes while its state still describes the
        # old occupant. The OLD guard merely RESET _ctx_len[i]=0 on any req_id
        # mismatch -- discarding the moved-in request's accumulated context and,
        # worse, recomputing num_new against the full (large) accepted length on
        # the next step (out-of-range write / crash). Instead, MIRROR the move:
        # for each new slot i, find the OLD slot j that held the same req_id and
        # gather that slot's full state (buffer row + host scalars) into i. Only
        # genuinely-new req_ids (not present last step) are reset. This is purely
        # layout-based, so it transparently handles condense moves, swaps, and
        # chained/cyclic moves in one shot.
        req_ids = self.runner.input_batch.req_ids
        old_slot_of: dict = {}
        for j in range(self.max_num_reqs):
            rid = self._last_req_id[j]
            if rid is not None:
                old_slot_of[rid] = j

        # gather_src[i] = old slot whose request now occupies slot i (identity
        # for unmoved/new slots). Drives a single device gather over _ctx_buf.
        gather_src = np.arange(self.max_num_reqs, dtype=np.int32)
        moved = False
        new_ctx_len = self._ctx_len.copy()
        new_prev_seq_len = self._prev_seq_len.copy()
        new_last_req_id: list[Optional[str]] = [None] * self.max_num_reqs
        for i in range(self.max_num_reqs):
            cur = req_ids[i] if i < len(req_ids) else None
            new_last_req_id[i] = cur
            if cur is None:
                continue
            src = old_slot_of.get(cur)
            if src is None:
                # Brand-new request in this slot: reset its host state. Its
                # buffer row is left as-is (gather identity); ctx_len 0 means
                # every stale row is attention-masked, so contents are inert.
                new_ctx_len[i] = 0
                new_prev_seq_len[i] = 0
            elif src != i:
                # Request moved src -> i: carry its full state along.
                gather_src[i] = src
                new_ctx_len[i] = self._ctx_len[src]
                new_prev_seq_len[i] = self._prev_seq_len[src]
                moved = True
            # src == i: unmoved, keep existing state (identity gather).
        self._ctx_len = new_ctx_len
        self._prev_seq_len = new_prev_seq_len
        self._last_req_id = new_last_req_id
        if moved:
            self._ctx_buf = self._permute_ctx_rows(
                self._ctx_buf, device_array(self.mesh, gather_src))

        # accepted context length per request this step (host int array).
        seq_lens_host = np.asarray(jax.device_get(attn_metadata.seq_lens))
        # query_start_loc maps the flat ragged token axis of aux_hidden_states
        # back to per-request segments: req i occupies [qsl[i] : qsl[i+1]).
        qsl_host = np.asarray(jax.device_get(attn_metadata.query_start_loc))

        # raw_hidden: flat-ragged [total_tokens, raw_hidden_dim], ordered by
        # request via query_start_loc. NOT a batch axis.
        raw_hidden = jnp.concatenate(aux_hidden_states, axis=-1)

        # Append each request's newly-accepted hidden states (the LEADING rows
        # of its query segment) into that slot's context buffer. The per-slot
        # bookkeeping (shrink-repair, ctx_len bump) stays on host exactly as
        # before; only the DEVICE write is batched into a SINGLE jitted+donated
        # masked scatter (_batched_ctx_write) so XLA mutates the multi-GiB
        # buffer in place instead of copying it once per request.
        #
        # Build a per-source-row index plan over raw_hidden's flat token axis.
        # Every row defaults to a NO-OP write (routed to the dead last buffer
        # row, masked off); real copies overwrite their plan entries below.
        total_tokens = raw_hidden.shape[0]
        buf_len = self._ctx_buf.shape[1]
        dead_row = buf_len - 1
        slot_idx_host = np.zeros(total_tokens, dtype=np.int32)
        dst_row_host = np.full(total_tokens, dead_row, dtype=np.int32)
        valid_host = np.zeros(total_tokens, dtype=bool)
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
            # Hard backstop against overrunning raw_hidden's query segment for
            # this request: raw_hidden only holds THIS step's query rows
            # [qsl[i] : qsl[i+1]) for slot i. With the condense-move fix above,
            # _ctx_len[i] follows the request so num_new == seg_width in steady
            # state; clamping num_new to seg_width here makes that an invariant
            # rather than an assumption (any residual desync degrades acceptance
            # for that request but can never crash or write cross-request rows).
            # _ctx_len advances by exactly n_copy (== num_new after the clamp),
            # so bookkeeping stays consistent with what was written.
            seg_start = int(qsl_host[i])
            # Segment end = next request's start; for the last request (or a
            # degenerate query_start_loc that omits the trailing offset) it is
            # the end of raw_hidden. query_start_loc is the CSR offset form
            # (sized max_num_reqs + dp_size at runtime) so qsl_host[i+1] is
            # normally in-bounds; the fallback keeps this robust either way.
            seg_end = (int(qsl_host[i + 1])
                       if i + 1 < len(qsl_host) else total_tokens)
            seg_width = seg_end - seg_start
            num_new = min(num_new, seg_width)
            ctx_len_i = int(self._ctx_len[i])
            end = min(ctx_len_i + num_new, self.max_model_len)
            n_copy = end - ctx_len_i
            # Source rows [seg_start : seg_start+n_copy) -> slot i, dest rows
            # [ctx_len_i : ctx_len_i+n_copy). (Bit-identical to the eager
            # dynamic_update_slice that copied these same rows.)
            rows = slice(seg_start, seg_start + n_copy)
            slot_idx_host[rows] = i
            dst_row_host[rows] = ctx_len_i + np.arange(n_copy, dtype=np.int32)
            valid_host[rows] = True
            self._ctx_len[i] = end

        # One device write for all requests; donate _ctx_buf so it is updated in
        # place. Returned buffer immediately replaces the donated input.
        self._ctx_buf = self._batched_ctx_write(
            self._ctx_buf,
            raw_hidden,
            device_array(self.mesh, slot_idx_host),
            device_array(self.mesh, dst_row_host),
            device_array(self.mesh, valid_host),
        )

        # All slots share one rectangular padded context width = the max
        # accepted length rounded up to _ctx_pad_block, so the downstream JIT
        # cache only sees ~max_model_len/_ctx_pad_block shapes regardless of
        # batch size. This width never exceeds buf_len (both round up with the
        # same block, and max_ctx <= max_model_len), so the slice is in-bounds.
        ctx_lens = self._ctx_len[:num_reqs].astype(np.int32)
        max_ctx = int(ctx_lens.max()) if num_reqs > 0 else 1
        padded_ctx = self._next_padded_size(max(max_ctx, 1))
        # The active (N, padded_ctx, D) sub-block is NOT sliced here. We pass the
        # FULL persistent buffer to the jitted draft forward and slice inside
        # the jit (static num_reqs/padded_ctx) so XLA fuses the slice into the
        # fc matmul. Slicing eagerly here would materialize a multi-GiB copy of
        # the buffer every step (the c=32 OOM).
        ctx_full = self._ctx_buf

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

        # Carry the FULL context buffer plus the static (num_reqs, padded_ctx)
        # so the draft forward slices the active sub-block inside its jit.
        target_hidden_states = (ctx_full, position_ids, attention_mask,
                                num_reqs, padded_ctx)

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
        ctx_full, position_ids, attention_mask, num_reqs, padded_ctx = (
            target_hidden_states)

        hidden_states = self._draft_forward_fn(
            self._params,
            input_ids,
            ctx_full,
            position_ids,
            self._embed_weight,
            attention_mask,
            num_reqs,
            padded_ctx,
        )

        # draft_token_ids: (N, num_speculative_tokens), one row per request.
        draft_token_ids = self._sample_block_draft_tokens(
            self._params, hidden_states, self._lm_head_weight)

        return kv_caches, draft_token_ids
