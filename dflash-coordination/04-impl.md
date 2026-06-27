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

## Adversarial review (L3) — 1 BUG found + fixed, rest clean
- BUG: precompile only warmed N in {1, max} but active batch fluctuates over
  [1,max] and N is a STATIC jit axis; propose() runs inside maybe_forbid_compile
  -> unwarmed N hard-fails under recompilation guard (else mid-decode stall).
  FIX: warm batch_sizes = range(1, max_num_reqs+1). (commit f9947f43)
- Clean: ragged slice (qsl[i]:qsl[i]+n_copy, leading rows = accepted), per-row
  positions/mask (ctx_valid per (slot,pos)), stale-buffer masking on req change,
  num_reqs source consistency, draft_token_ids (num_reqs, num_spec) alignment,
  noise-block/position consistency, out_shardings ranks, no leftover scalar reads.

## SMOKE RESULT — PASS (c=32, coherent)
- 32/32 concurrent greedy requests succeeded. Coherent output: "capital of
  France is Paris", "first president...George Washington", "gold is Au",
  "speed of light ~299,792,458 m/s", "author of Harry Potter is J.K. Rowling".
  Duplicate prompts produced identical greedy continuations (deterministic).
- The num_reqs<=1 assert is GONE; batched verification produces correct text
  across a full batch of 32 with variable content. No logic bug.
- Working config: max-model-len 1024, max-num-seqs 32, --gpu-memory-utilization
  0.75, EP, RAGGED_GATHER v1, --no-async, DRAFT_MODEL_IMPL_TYPE=torchax,
  num_speculative_tokens 7, HF_HOME=/home/enyouki/local_hf.
- HBM: KV cache ~1.6M tokens; per-chip ~23-24 / 31.25 GiB. _ctx_buf at len 1024
  ~0.88G fits. At default util / len 2048 it did NOT fit (1.76G _ctx_buf alloc
  failed with 970M free) — so for the c=32 SPEED bench at len 5120 the next
  manager MUST drop util and/or len, OR land the _ctx_buf scatter optimization.
- Wall time 230s reflects COLD XLA compiles (first run, SKIP_JAX_PRECOMPILE=1) —
  NOT a DFlash speed measurement. Speed bench is the next manager's job (warm).

## Status
- [x] implement  [x] adversarial review + fix  [x] smoke serve @ max-num-seqs 32
- [x] coherent output  [x] commit+push (code + coord)

## Smoke result so far
- Server LAUNCHES + serves at --max-num-seqs 32 (assert gone). Startup clean.
- First smoke attempt (max-model-len 2048, default util): batched path RUNS (no
  assert, no logic error) but RESOURCE_EXHAUSTED at the _ctx_buf
  dynamic_update_slice — tried to alloc 1.76G, only 970M free. This is the
  HBM-pressure SECOND risk the bench manager flagged, NOT a correctness bug.
  Root cause: vLLM memory profiler grabs ~all HBM for KV cache (2.04M tokens)
  before the proposer's full (32, buf_len, raw_hidden_dim) _ctx_buf is exercised.
  _ctx_buf ≈ 32 × buf_len × raw_hidden_dim(~13440) × 2B; at buf_len 2048 = 1.76G.
- PERF NOTE (follow-up, not blocking smoke): each prepare_inputs does up to
  num_reqs separate lax.dynamic_update_slice on the FULL _ctx_buf, eager =>
  full-buffer transient per call per step. A jitted batched scatter would avoid
  this. Next perf/opt manager should address.
- Retry config: max-model-len 1024, --gpu-memory-utilization 0.75 (shrinks KV +
  halves _ctx_buf to ~0.88G). Launch: scratchpad/serve.sh -> serve2.log.

## Commits
- 9410b793 batched DFlash decode (per-slot ctx buffers, ragged slicing, batched fwd)
- f9947f43 precompile warm all batch sizes 1..max_num_reqs
