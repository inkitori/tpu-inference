# 10 — IMPL: fix the DFlash slot/condense desync (c=32 out=4096 crash)

Phase: fix the batched condense/slot desync + verify across a real condense event.
Date: 2026-06-27. Manager: impl-condense. Builds on 09-bench-c32-v2 (the crash),
06-impl-hbm + 08-impl-perf (the batched/perf impl), 05-test-batched (perfect-draft method).

## TL;DR / VERDICT — FIXED + VERIFIED ⇒ NEEDS_BENCH
The c=32/out=4096 crash is gone AND the verify path stays lossless THROUGH a real
condense/backfill event. Two commits (the PROPER fix + an HBM follow-up the proper fix
itself required). On-TPU condense workload (util 0.75, the GOAL config): no crash,
perfect-draft 100% per-slot across condense, greedy answers token-identical 64/64, real
DFlash acceptance ~6 steady-state. ⇒ finally safe to re-bench A vs B at out=4096/c=32.

## WHICH FIX + WHY
Chose the **PROPER fix (scope option 2)**: make the proposer's per-slot state FOLLOW the
request through condense, not the minimal seg-width clamp alone. It was tractable (vLLM
exposes the slot layout via input_batch.req_ids) and it preserves acceptance through
backfill — directly protecting the speed goal. The minimal seg-width clamp is ALSO kept as
defense-in-depth (can never crash even on residual desync). NOTE: the PROPER fix was found
ALREADY COMMITTED at the start of this round (b94fc366, authored 03:54 in a prior session
before preemption); my work was (a) finding+fixing the HBM regression it introduced and
(b) the decisive on-TPU condense verification it had never had.

## WHAT CHANGED + WHERE (tpu_inference/spec_decode/vllm/dflash.py)

### b94fc366 — mirror condense slot-moves onto per-slot state (the desync fix)
- prepare_inputs(): before the copy loop, diff input_batch.req_ids vs cached _last_req_id
  to detect slot moves. For each new slot i, find the OLD slot j that held the same req_id
  and MIRROR the move onto all four state pieces: gather _ctx_buf rows + carry _ctx_len /
  _prev_seq_len / _last_req_id. Only genuinely-NEW req_ids reset (_ctx_len=0). Handles
  condense, swap, and chained/cyclic moves in one layout-based pass.
- Plus a defense-in-depth clamp: `num_new = min(num_new, seg_width)` where
  `seg_width = qsl[i+1]-qsl[i]` (fallback total_tokens) ⇒ "never overrun the query segment"
  is now an INVARIANT, not an assumption. (This alone == the 09 minimal candidate (b).)
- Root cause it kills: old guard reset _ctx_len[i]=0 on any req_id change, then num_new =
  seq_len-0 = full accepted length (3184) overran the 176-row query segment → the broadcast
  crash at dflash.py:421.

### 86f150cf — shrink the condense-move HBM transient (the regression THIS fix introduced)
- The b94fc366 move used `_permute_ctx_rows(ctx_buf, gather_src)` = `ctx_buf[gather_src]`, a
  FULL leading-axis gather. Even donated, an out-of-place gather MATERIALIZES a fresh full
  ~3.96 GiB buffer ⇒ ~4.58 GB transient. At util 0.75 (chip-0 headroom only ~0.4 GiB, per
  06-impl-hbm) this OOM-crashed the engine (RESOURCE_EXHAUSTED jit__permute_ctx_rows,
  dflash.py:452) once enough long reqs ran concurrently under condense. (At util 0.6 it ran
  clean — confirming it was purely this transient, NOT the desync logic.) This is why the
  prior session's condense probe (cond_pd.log) saw 33/64 ok then a cascade of 500s.
- Replaced `_permute_ctx_rows` with `_move_ctx_rows(ctx_buf, dst_slots, src_slots)` — a
  SPARSE in-place donated scatter (same pattern as the proven _batched_ctx_write):
  `ctx_buf.at[dst_slots].set(ctx_buf[src_slots])` gathers ONLY the K moved rows, not 32.
  K is bucketed to powers of two up to max_num_reqs ({1,2,4,8,16,32}). prepare_inputs host
  loop now collects only the moved (dst,src) pairs and pads to the bucket with self-copies
  of an UNMOVED slot p (NOT slot 0 — a (0,0) pad corrupts cycles touching slot 0 via
  duplicate-index scatter; caught by the numeric check). precompile() warms all K buckets.
- Common 1-2-move condense → K=2 bucket → ~248 MiB transient (vs ~3.96 GiB). Worst case
  (full permutation) is RESPECTED: top bucket = max_num_reqs, so it still works (gathers all
  32 only in that rare case) and NEVER drops a real move. Correctness > the memory win.

## VERIFICATION (the part the prior "proven" tests skipped — these EXERCISE condense)
Workload: scratchpad/fire_condense.py 64 — 64 reqs / 32 slots, output lengths spread
[32..1536] (min==max via min_tokens) ⇒ staggered finishes ⇒ freed slots backfilled by
still-running long reqs ⇒ real condense. Config: util 0.75, max-model-len 4224, c=32, EP,
v1 gather, no-async, torchax draft, 7 spec tokens (the GOAL bench config).

- **(A) NO CRASH at util 0.75.** 64/64 succeeded, engine up the whole workload. Log scan
  clean: no "could not broadcast", no RESOURCE_EXHAUSTED, no _move/_permute/EngineDead.
  finish-time spread 80.8s (perfect-draft) / 172.7s (real) ⇒ condense actually fired.
- **(B) PERFECT-DRAFT 100% PER-SLOT ACROSS CONDENSE.** DFLASH_PERFECT_DRAFT=1: EVERY metric
  window across the condense event (10 windows) = Mean accept len 8.00, per-pos 1.000×7,
  Accepted==Drafted, 100.0%. No window dipped < 8.00 ⇒ no slot mis-indexed by a move ⇒
  verify path stays aligned through slot moves (losslessness preserved through condense).
- **(C) GREEDY LOSSLESS through condense.** spec-on (real DFlash) vs target-only, same
  condense workload: factual ANSWER token-identical 64/64; 25/64 byte-identical full
  completions. Remaining divergences are post-answer high-entropy filler — and target-only
  is ITSELF nondeterministic for the same (prompt,len) (7/8 dup groups differ within
  target-only alone), so that filler drift is batch-position FP near-tie behavior present in
  target-only too, NOT a verifier regression. No step-1/step-2 answer divergence anywhere.
- **(D) REAL ACCEPTANCE UNDER CONDENSE.** Real DFlash mean accept len ~3.4→7.2 across
  windows, steady-state **~6.0–6.7**; per-pos degrades gracefully (busy 0.716→0.148, light
  ~0.96→0.64); avg draft accept ~35% (peak) to ~88% (light). Acceptance held up through
  backfill — the PROPER fix did NOT degrade acceptance (unlike the minimal-only fix would).
- HBM at init (util 0.75): total_hbm_used 39.73 GiB / cap 187.48; KV 1,972,451 tokens;
  max-conc 466.96x. OOM gone — stayed up under load.

## CORRECTNESS EVIDENCE (besides on-TPU)
- scratchpad/check_permute_inplace.py (CPU jax): _move_ctx_rows == old ctx_buf[gather_src]
  bit-identical for identity / single-move / swap / 3-cycle / full 6-cycle / condense-shift /
  padding-collision stress. PASS.
- Unit tests: tests/spec_decode/test_dflash_torchax.py 5/5; test_dflash.py 5/5.

## COMMITS
- b94fc366 fix DFlash slot/condense desync (mirror slot moves) — PUSHED (prior session)
- 86f150cf shrink DFlash condense-move HBM transient (sparse in-place scatter) — pushed this
  round
- (dflash-coordination/ docs committed alongside)

## FOLLOW-UP / NOTES FOR NEXT MANAGER
- NEEDS_BENCH: re-run A (DFlash) vs B (target-only) at in=1/out=4096/c=32, util 0.75, WARM
  cache, free-tpu between. This is FINALLY a trustworthy A number (the 07 "358 tok/s / 10.6x"
  was on a silently-corrupt run; this round proves the path is correct + crash-free under the
  bench workload). Expect A still slower than B until LEVER #1 (KV-cache the O(ctx) draft
  forward, 08-impl-perf) lands — but now measurable honestly.
- The minimal seg-width clamp + the proper move are BOTH live; the clamp is harmless belt-
  and-braces. No need to remove it.
- Worst-case full-permutation condense still gathers 32 rows (rare); if a future round wants
  to remove even that, a true row-by-row in-place loop would, but it's not worth it now.

## STATUS
- [x] root-cause confirmed (already-committed proper fix b94fc366)
- [x] found + fixed the HBM regression the proper fix introduced (86f150cf)
- [x] CPU numeric bit-identical (swap/cycle/padding) + unit tests 5/5 + 5/5
- [x] on-TPU condense verify: no crash @0.75, perfect-draft 100% across condense, greedy
      lossless, real accept ~6 ⇒ NEEDS_BENCH
