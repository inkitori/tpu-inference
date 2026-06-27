# 09 — BENCH c=32 v2 (re-verify after write fix) + a CRASH found at the bench point

Phase: gate + decisive c=32 speed bench, post-6b6acd49. Date: 2026-06-27. Manager: bench-v2.
Builds on 07-bench-c32 (10.6x slower), 08-impl-perf (donated write fix), 05-test-batched.

## TL;DR / VERDICT
- **GATE: PASS.** Perfect-draft at c=32 / SHORT outputs is 100% per-slot (mean 8.00, per-pos
  1.000×7, Accepted==Drafted) across 5 windows incl high-volume (672/644/1323). The donated
  `_ctx_buf` write (6b6acd49) did NOT break the verify path.
- **BENCH (Config A, DFlash ON, out=4096, c=32): DID NOT COMPLETE — DETERMINISTIC CRASH.**
  At ~69s into the c=32/out=4096 load (seqs grown to ~1500-3300 tokens) the engine dies with
  `ValueError: could not broadcast input array from shape (3184,) into shape (176,)` at
  **dflash.py:421** in `prepare_inputs`. NOT an OOM (kv_cache_usage 2.3% at crash). So NO valid
  A throughput number this round. Config B (target-only) not run (no point without A).
- **ROOT CAUSE: a PRE-EXISTING batched-correctness bug in DFlash, EXPOSED (not caused) by
  6b6acd49.** ⇒ **NEEDS_IMPL** (fix described below, then re-bench A vs B).

## STEP 1 — CORRECTNESS GATE (PASS)
Serve: `DFLASH_PERFECT_DRAFT=1 bash scratchpad/serve_dflash.sh 32 scratchpad/gate_serve.log`
(max-num-seqs 32, max-model-len 1024, util 0.75, EP, v1 gather, no-async, torchax draft).
Load: `python3 scratchpad/fire_concurrent.py ragged 32` ×2 → 64/64 OK, coherent prefixes
(primes 2,3,5,7,11; red/blue; Rayleigh scattering; Charles Babbage; 2+2=4).
Metrics (5 windows, ALL identical): `Mean accept len 8.00 | per-pos 1.000×7 | 100.0% |
Accepted==Drafted` (windows: 7/672/644/7/1323 accepted==drafted). ⇒ verify path correct at
full c=32 under short outputs.

## STEP 2 — SPEED BENCH (CRASHED before any measurement)
Serve A: `bash scratchpad/serve_A_dflash.sh` (DFlash ON, max-model-len 4224, c=32, util 0.75).
Came up clean (health 200 ~50s, HBM at init 39.73 GiB/8-chip, KV 1,972,451 tok, max-conc 466.96x).
WARMUP run (`bench_client.py --concurrency 32 --out 4096`): ttft 18.7s (cold compile, as expected),
but **crashed mid-stream at 03:37:38** — completion_tokens distinct = [538, 539, 4096] (truncated by
the engine death; min_tokens=4096 NOT honored because the engine died). Measured run NOT started
(engine dead). Saved: scratchpad/A_warmup_v2.out, scratchpad/A_serve.log.

### The crash (exact)
```
spec_decode/vllm/dflash.py:421, in prepare_inputs
    dst_row_host[rows] = ctx_len_i + np.arange(n_copy, dtype=np.int32)
ValueError: could not broadcast input array from shape (3184,) into shape (176,)
```
Call path: tpu_runner.sample_tokens → _sample_from_logits → speculative_decoding_manager
.propose_dflash_draft_token_ids:318 → drafter.prepare_inputs (dflash.py:421). kv_cache_usage 0.023.

## ROOT CAUSE (verified, two angles)
DFlash per-slot proposer state (`_ctx_len[i]`, `_last_req_id[i]`) is keyed by **physical slot
index i, NOT req_id**. `num_new = seq_len - ctx_len_i` drives the per-step copy width, where
`seq_len = accepted_seq_lens[i]` = `input_batch.num_tokens_no_spec[i]` = the request's FULL
running accepted length (speculative_decoding_manager.py:296-316). But `raw_hidden`'s segment for
slot i is only THIS step's query rows `[qsl[i] : qsl[i+1])` (a few rows).

The desync: when a request FINISHES mid-run, vLLM `InputBatch.condense` moves a still-growing
request DOWN into the freed slot, carrying its big `num_tokens_no_spec` (~3184). The slot-change
guard (dflash.py:366-371) sees `req_ids[i] != _last_req_id[i]` and resets `_ctx_len[i]=0`. Next
step: `num_new = 3184 - 0 = 3184`, but the query segment has ~176 rows ⇒ the host index plan
writes 3184 dst rows into a numpy slice that clamps to 176 ⇒ broadcast ValueError.

**Pre-existing, not a 6b6acd49 regression.** The per-request arithmetic is byte-identical old vs
new. OLD eager path: `raw_hidden[seg_start:seg_start+n_copy]` was a JAX slice that SILENTLY CLAMPED
to 176 rows (+ dynamic_update_slice clamps start) ⇒ no crash but it wrote WRONG rows (other reqs'
segments) into slot i = **silent draft-context corruption** for condensed requests. 6b6acd49's new
HOST index-plan (`np.arange(n_copy)` RHS vs numpy-clamped LHS) turned that silent corruption into a
hard crash. So the prior 07 bench (358 tok/s, "32/32 ok, all [4096]") COMPLETED only because the old
path silently corrupted instead of crashing — that 10.6x number was measured on a partially-corrupt
run. **The crash is good news: it surfaced a real latent batched bug.**

Why gate passed but bench crashed: condense fires only when a request finishes while another is
still large. Short outputs (gate, out≤48) finish together at similar lengths ⇒ no heterogeneous
slot reuse ⇒ benign. out=4096/c=32 finishes are staggered (lengths 1500-3300 at ~69s) ⇒ a finished
slot gets backfilled with an already-long request ⇒ reset under big seq_len ⇒ crash. Consistent.

## THE FIX (for the IMPL manager) — minimal, candidate (b)
Drive the copy width from the actual query-segment width, not the accepted-length delta.
In dflash.py prepare_inputs, replace `num_new = seq_len - ctx_len_i` (~line 408) with:
```python
seg_start = int(qsl_host[i])
seg_width = int(qsl_host[i + 1]) - seg_start   # rows actually in raw_hidden for req i
num_new = seg_width
```
Keep the rest (`if num_new<=0: continue`; `ctx_len_i=int(self._ctx_len[i]); end=min(ctx_len_i+num_new,
max_model_len); n_copy=end-ctx_len_i`; `self._ctx_len[i]=end`). `qsl_host[i+1]` is in-bounds:
query_start_loc is sized max_num_reqs+dp_size, tail padded to the cumsum last value
(tpu_runner.py ~2374). This appends exactly the leading accepted rows present this step and keeps
`end` == what was written, so bookkeeping stays consistent. (Candidate (a) `n_copy=min(n_copy,
seg_width)` also stops the crash but leaves `end` jumping to 3184 while only seg_width rows were
written ⇒ re-desync; (b) is the correct one.)

CAVEAT the IMPL manager MUST weigh: after a condense, the moved-in request's PRE-MOVE context
history lived in a DIFFERENT slot's _ctx_buf and is now lost. Fix (b) stops the crash and appends
correct CURRENT-step rows, but that request's draft context is now incomplete ⇒ its acceptance will
DEGRADE until it re-accumulates (NOT a losslessness break — the verifier still checks vs target).
A fuller fix carries per-slot _ctx_buf across condense (move/swap the buffer row with the request) or
keys proposer state by req_id. Recommend: land (b) to unblock the bench, then decide whether the
condense context-carry is needed for the speed verdict (degraded acceptance hurts the very metric we
test). RE-RUN THE GATE after the fix (perfect-draft at out≥512 to actually trigger condense), then A vs B.

## STEP 2 RE-BENCH PLAN (after fix)
Same as 07: A=serve_A_dflash.sh, B=serve_B_target.sh, free-tpu between, WARMUP+DISCARD then measured,
`bench_client.py --concurrency 32 --out 4096`. Prior baselines for reference (07, possibly-corrupt A):
A 358.65 sys tok/s / TPOT 0.317; B 3788.53 sys tok/s / TPOT 0.0084 (B is the honest target-only).

## HBM (Config A at init, before crash)
total_hbm_used 39.73 GiB / cap 187.48 (util 0.75); KV 1,972,451 tok; max-conc 466.96x. Fits.

## Status
- [x] gate PASS (short outputs)  [x] serve A up  [x] found+root-caused the out=4096 crash
- [ ] A measured (BLOCKED on fix)  [ ] B measured  [ ] A-vs-B verdict (BLOCKED on fix)

## Scripts / artifacts
- scratchpad/gate_serve.log (gate, 100% windows)
- scratchpad/A_serve.log (crash traceback @ dflash.py:421), scratchpad/A_warmup_v2.out (truncated)
- serve_A_dflash.sh / serve_B_target.sh / bench_client.py / fire_concurrent.py (unchanged from 07)
