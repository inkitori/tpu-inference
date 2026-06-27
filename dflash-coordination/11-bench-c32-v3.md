# 11 — BENCH c=32 v3 (FIRST trustworthy A-vs-B after condense fix)

Phase: decisive A-vs-B speed verdict on the now-correct path. Date: 2026-06-27.
Manager: bench-v3. Builds on 10-impl-condense (fix verified), 07-bench-c32 (corrupt
10.6x), 09-bench-c32-v2 (the crash). GOAL bench point: in=1, out=4096, c=32, WARM cache.

## VERDICT (decisive, TRUSTWORTHY): DFlash is ~2.90x SLOWER than target-only → NEEDS_IMPL
First A-vs-B run that did NOT crash and is NOT silently corrupt. DFlash now RUNS the full
out=4096/c=32 workload with backfill/condense (total=96 reqs over 32 slots) cleanly, 96/96
ok, all exactly 4096 tokens, acceptance healthy ~5-6.2/step. But it still LOSES the speed
goal: target-only is 2.90x faster on system output throughput and DFlash also LOSES on
latency here (87s vs 35s mean). The Phase-1 SPEED requirement still FAILS. The per-step
DFlash cost (TPOT 13.2x a target step) is not yet covered by ~6x acceptance because each
draft step recomputes the FULL context (O(ctx), no KV cache — LEVER #1, not yet done).

NOTE: this is a HUGE improvement over the corrupt 07 number (358 → 1310 tok/s, 3.7x faster
A) — the write fix (6b6acd49) + a non-corrupt path account for that. But still short of B.

## Headline numbers (warm cache, in=1, out=4096, c=32, total=96, 96/96 ok, 4096 tok each)
| Metric                       | A: DFlash ON | B: target-only | B/A (A penalty) |
|------------------------------|-------------:|---------------:|----------------:|
| SYSTEM output tok/s          | **1309.79**  | **3792.80**    | **2.90x**       |
| wall time (s)                | 300.21       | 103.67         | 2.90x           |
| mean per-req latency (s)     | **86.71**    | **34.55**      | 2.51x (A worse) |
| p50 / p99 latency (s)        | 82.27 / 126.08 | 34.59 / 34.63 |                |
| min / max latency (s)        | 44.52 / 126.08 | 34.44 / 34.63 |                |
| TPOT (s/token)               | **0.11122**  | **0.00843**    | 13.19x          |
| per-stream tok/s (1/TPOT)    | 8.99         | 118.59         |                 |
| ttft (s) mean / max          | 0.444 / 0.755 | 0.027 / 0.034 |                 |
| DFlash mean accept length    | ~5.0-6.2 @ full c=32 (tail to 6.7) | n/a |          |

- A acceptance (full-load windows): mean accept len 5.05, 5.82, 5.79, 5.86, 4.70-6.24;
  per-position 0.85→0.40 (e.g. 0.851,0.719,0.635,0.561,0.457,0.429,0.397); avg draft
  accept 58-75%. Tail windows (load < 32) climbed to 6.69-6.78. ⇒ real DFlash accept ~6,
  exactly as 10-impl-condense predicted. Acceptance is HEALTHY — this is a PERF gap.
- A latency varies (44-126s) because reqs finish staggered as concurrency falls; the
  system-throughput 2.90x is the apples-to-apples number. B latency is flat (~34.6s, all
  reqs identical) — no spec overhead, deterministic decode.

## NOTE on the "spec-decode wins latency" hypothesis — it does NOT hold here
The L2 brief flagged spec-decode often wins per-request latency even at throughput parity.
NOT the case at this bench point: A mean latency 86.7s vs B 34.6s — DFlash is 2.5x WORSE on
latency too. Reason: at c=32 the target step is already cheap and batched; DFlash's O(ctx)
draft recompute makes every step ~13x more expensive, and 6x acceptance only recovers ~4.5x,
so even single-stream effective speed (per-stream tok/s 8.99 vs 118.59) is far behind. Spec
decode would win latency only at LOW concurrency where the target step is the bottleneck —
not at c=32 where the draft recompute dominates.

## Configs (apples-to-apples; one at a time, free-tpu between) — VERIFIED
- **A (DFlash ON):** scratchpad/serve_A_dflash.sh — `--max-num-seqs 32 --max-model-len 4224
  --gpu-memory-utilization 0.75`, `--speculative-config {model: z-lab/gpt-oss-20b-DFlash,
  num_speculative_tokens: 7, method: dflash}`, DRAFT_MODEL_IMPL_TYPE=torchax. EP, RAGGED_GATHER
  v1, --no-async-scheduling, HF_HOME=/home/enyouki/local_hf.
- **B (target-only):** scratchpad/serve_B_target.sh — SAME target, SAME max-num-seqs 32, SAME
  max-model-len 4224, SAME util 0.75, EP, v1 gather, no-async. NO speculative-config.
- Both servers: prompt "Hi" → prompt_tokens=1 confirmed; both emit BYTE-IDENTICAL greedy text
  (", I am a student and I am") on the 8-tok sanity probe ⇒ same target, apples-to-apples.

## Method (followed the warm-cache bench discipline)
- tpu-env.sh SKIP_JAX_PRECOMPILE=1 ⇒ cold XLA compile on first use. Ran a WARMUP
  c=32/total=96/out=4096 for EACH config and DISCARDED it; measured on the now-warm cache.
- Load gen: scratchpad/bench_client_pool.py — WORKER POOL (NEW this round): concurrency=32
  workers pulling from a queue of TOTAL=96 reqs ⇒ holds 32 in flight while 96 total flow
  through ⇒ staggered finishes ⇒ REAL backfill/condense exercised (the bench-v3 requirement;
  the old bench_client.py fired exactly N and stopped). max_tokens=min_tokens=4096,
  ignore_eos=true, temperature=0, streaming, NO logprobs (spec-on logprobs crash known/oos).
- WARMUP evidence it was a real compile run (correctly discarded):
  - A_warmup: ttft max 8.52s (cold compile), 963.60 tok/s. A_measured: ttft max 0.755s,
    1309.79 tok/s ⇒ no compile leaked into measurement.
  - B_warmup: ttft max 5.15s (compile), 3595.61 tok/s. B_measured: ttft max 0.034s,
    3792.80 tok/s ⇒ warm.
- A measured: 96/96 ok, completion_tokens all [4096]. B measured: 96/96 ok, all [4096].
  NO crash in either (the 09 dflash.py:421 broadcast crash is GONE under condense — confirms
  10-impl-condense's fix holds at out=4096/c=32 with backfill).

## HBM (info)
- A (DFlash): total_hbm_used 39.73 GiB (8-chip) / cap 187.48; KV 1,972,451 tokens, max-conc
  466.96x. Fits at util 0.75, stayed up under full load + condense.
- B (target-only): total_hbm_used 24.07 GiB; KV 2,181,304 tokens, max-conc 516.41x (more
  slack, no _ctx_buf). HBM is NOT the bottleneck for the speed gap.

## WHY DFlash is slow — the remaining lever (unchanged root cause, now the ONLY one left)
The write fix (6b6acd49, LEVER #2) landed and helped (A 358→1310 tok/s vs the corrupt 07).
The DOMINANT remaining cost is **LEVER #1: the STATELESS draft recomputes fc (14400→2880) +
8 attn layers over the ENTIRE padded context EVERY step (O(context), grows toward 4608 at
out=4096; models/vllm/dflash.py, past_key_values=None/use_cache=False).** A target step
touches 1 new position; the DFlash draft reprocesses the whole sequence each step ⇒ TPOT
13.2x. To break even, a DFlash step must get under ~B_TPOT × accept ≈ 0.0084 × 6 ≈ 0.050s;
it is at 0.111s ⇒ ~2.2x too slow ⇒ KV-caching the draft context should close it (the draft
fc+attn would then touch only the new accepted rows, not the full O(ctx) buffer).

## Handoff → NEEDS_IMPL (LEVER #1)
KV-cache the DFlash draft forward so it stops recomputing fc + 8 attn over the full context
each step. Target: DFlash step < ~0.05s at long ctx (currently 0.111s). Then RE-RUN THIS
bench (same scripts: serve_A/serve_B, bench_client_pool.py --concurrency 32 --total 96
--out 4096, warmup+discard, free-tpu between). Expected: that single lever should flip A > B
(2.2x step speedup needed; KV cache removes the O(ctx) term entirely → step ~O(1) in ctx).

## Reproduce
- Serve A: `bash scratchpad/serve_A_dflash.sh` (wait "Application startup complete", ~75s)
- Serve B: `bash scratchpad/serve_B_target.sh`
- Warm + measure: `python3 scratchpad/bench_client_pool.py --port 8000 --concurrency 32
  --total 96 --out 4096 --tag <T>` (run twice; discard the first). free-tpu.sh between configs.
- Accept length: `grep "Mean acceptance length" <serve log>` (vLLM SpecDecoding metrics).

## Status
- [x] Config A serve up (no crash through full out=4096/c=32 + condense/backfill)
- [x] A warmup discarded + A measured (1309.79 tok/s, TPOT 0.111, accept ~6)
- [x] Config B serve up, B warmup discarded + B measured (3792.80 tok/s, TPOT 0.0084)
- [x] A-vs-B verdict: DFlash 2.90x SLOWER on throughput, 2.5x worse latency ⇒ NEEDS_IMPL
- [ ] LEVER #1 (KV-cache the O(ctx) draft recompute) — the next lever to flip A > B

## Scripts / artifacts
- scratchpad/bench_client_pool.py (NEW worker-pool load gen, total=96)
- scratchpad/A_serve_v3.log, scratchpad/B_serve_v3.log (serve logs + accept metrics)
- scratchpad/A_warmup_v3.out / A_measured_v3.out / B_warmup_v3.out / B_measured_v3.out
- serve_A_dflash.sh / serve_B_target.sh (unchanged from 07)
