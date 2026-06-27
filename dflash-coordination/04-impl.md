# 04 — IMPL: batched/multi-seq DFlash torchax decode

Phase: implementation. Date: 2026-06-27. Manager: impl.
GOAL blocker = DFlash decode hard-capped to 1 request. Add batch>1 so c=32 serves.

## The batch=1 assumptions (where they live)
The cap is GLUE-CODE, not a model limit. HF DFlash forward is fully batch-generic
(`bsz,q_len = hidden_states.shape[:-1]`). Single-seq assumptions live in:
- `runner/speculative_decoding_manager.py:111` `assert num_reqs<=1` (THE cap).
  `dp_size==1` (L109) + `not async` (L107) are INDEPENDENT — leave them.
- `spec_decode/vllm/dflash.py` — the crux: single-slot state machine
  (`_ctx_len`,`_prev_seq_len`,`_last_req_id`,`_ctx_buf (buf_len,D)`) +
  `prepare_inputs` reads `seq_lens[0]`, appends `raw_hidden[:n_copy]` (assumes ONE
  req contiguous at flat offset 0), builds 1-D positions/mask, `query_start_loc=[0,B]`.
- `models/vllm/dflash.py:get_draft_forward_fn` — hardcoded `unsqueeze(0)/squeeze(0)`,
  mask `reshape(1,1,1,-1)` (batch=1).
- `_sample_block_draft_tokens` slices `hidden[1:1+num_spec]` (axis0); `_build_noise_block`
  uses `next_token_ids[0]` only.
- Manager `propose_dflash_draft_token_ids` is ALREADY batch-aware (next-token loop,
  num_rejected_tokens, accepted_seq_lens per-req). host_extract_sampled_tokens ALREADY
  batch-correct. Only breakage = prepare_inputs collapses accepted_seq_lens to [0].

## Confirmed ragged layout (THE slicing recipe)
- `aux_hidden_states[L]` is FLAT-RAGGED on token axis `[total_tokens, D]`, ordered by
  request via `attn_metadata.query_start_loc` ([num_reqs+1] cu-offsets). NOT a batch axis.
  Sharded PartitionSpec(ATTN_DATA, None).
- `accepted_attn_metadata` passed to prepare_inputs has `seq_lens` = per-req accepted
  ctx len (accepted_seq_lens[i]); `query_start_loc` LEFT INTACT (full scheduled stream).
- Per req i: `num_new_i = accepted_seq_lens[i] - _ctx_len[i]`; rows to append =
  FIRST num_new_i rows of req i's segment = `raw_hidden[qsl[i] : qsl[i]+num_new_i]`
  (accepted tokens are the LEADING rows; rejected drafts/bonus are trailing — eagle3 conv).
- Per-slot buffer update: `lax.dynamic_update_slice(buf, new_rows[None], (slot_i, ctx_len_i, 0))`
  (static update shape required → pad num_new per slot).

## Design (per-slot state)
- `_ctx_buf` → `(max_num_reqs, buf_len, D)`. `_ctx_len/_prev_seq_len` → np int arrays
  [max_num_reqs]. `_last_req_id` → list[Optional[str]] len max_num_reqs.
- prepare_inputs: loop slots 0..num_reqs-1; per-slot reset on req-id change; append each
  req's accepted hidden via query_start_loc; pad all N ctx to common max_padded_ctx;
  build (N, C) positions, (N, C+B) additive mask, (N, B) noise ids.
- forward: drop unsqueeze/squeeze, carry N; mask reshape (N,1,1,C+B).
- _sample_block_draft_tokens: `hidden[:, 1:1+num_spec]` → argmax → (N, num_spec).
- precompile: warm with N=max_num_reqs batch shapes.

## Status
- [ ] implement  [ ] smoke serve @ max-num-seqs 32  [ ] coherent output  [ ] commit+push

## Commits
(pending)
