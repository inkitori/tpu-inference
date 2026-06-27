# 14 — BENCH SWEEP: DFlash (KV-cache ON) num_speculative_tokens sweep vs target-only

Phase: decisive A-vs-B with `DFLASH_KV_CACHE=1` + a num_spec sweep. Date: 2026-06-27.
Manager: bench-sweep. Builds on 13-test-kvcache (flag-ON lossless, util 0.6 / drop
_ctx_buf gotcha), 12-impl-kvcache (the marginal-flip projection), 11-bench-c32-v3
(2.90x slower baseline; method + B=3793). GOAL bench point: in=1, out=4096, c=32, WARM.

## VERDICT (decisive): even the BEST num_spec is ~2.88x SLOWER than target-only → NEEDS_IMPL
The num_spec lever is EXHAUSTED. The KV-cache DFlash throughput PLATEAUS at ~1300 sys
out tok/s for num_spec ≥ 7 and DROPS for num_spec < 7. Best config (num_spec 7, the
cleanest of the plateau) = 1297 tok/s vs target-only 3752 tok/s ⇒ **2.89x slower on
throughput**, and 82.96s vs 34.93s mean latency ⇒ **2.37x WORSE on latency too**. The
"lower num_spec raises throughput" theory did NOT hold: lowering to 5 made it SLOWER
(acceptance fell faster than per-step cost). num_spec tuning cannot close a 2.9x gap.

## THE SWEEP TABLE (all DFlash KV-cache ON, util 0.6, c=32, out=4096, total=96, 96/96 ok, 4096 tok)
| Config            | sys out tok/s | TPOT (s) | mean lat (s) | p50/p99 lat | accept len (steady) | per-pos tail |
|-------------------|--------------:|---------:|-------------:|------------:|--------------------:|-------------|
| **B target-only** | **3752.20**   | 0.00852  | **34.93**    | 34.76/35.50 | n/a                 | n/a         |
| A num_spec=5      | 1226.57       | 0.09737  | 94.74        | 96.08/124.7 | ~4.4–4.6            | pos5 ~0.59  |
| **A num_spec=7**  | **1297.14**   | 0.10500  | 82.96        | 80.07/144.3 | ~5.5–6.0            | pos5-7 ~0.50 |
| A num_spec=10     | 1303.34       | 0.09741  | 88.03        | 91.13/116.8 | ~4.4–5.3            | pos8-10 = **0.000** |

- Throughput curve vs num_spec: 5 → 1227, 7 → 1297, 10 → 1303. PLATEAU at ~1300 for
  ns ≥ 7 (10 vs 7 = +6 tok/s = +0.5%, within noise); DROPS below 7. Optimum = **7**.
- **WHY it plateaus (the decisive mechanism):** at num_spec=10 the draft's per-position
  acceptance for positions 8/9/10 is EXACTLY 0.000 across the whole measured pass. The
  draft's EFFECTIVE acceptance depth is capped at ~7 (the draft block_size B=8). So
  num_spec ≥ 7 all degenerate to the same effective draft (extra positions are dead
  weight that never get accepted) ⇒ same ~1300 tok/s. ns=7 is the clean optimum (no
  wasted proposal slots). Going BELOW 7 throws away genuinely-accepted positions
  (pos 5–7 accept ~0.50–0.59) and loses throughput.
- A best (ns=7) LATENCY 82.96s also LOSES to B 34.93s ⇒ the "spec wins latency at
  parity" hypothesis still does NOT hold at c=32 (same as 11-bench-v3 found).

## B BASELINE — re-measured SAME-SESSION (clean apples-to-apples)
B (target-only, no spec, util 0.75, same target/len4224/c=32): **3752.20 sys out tok/s,
TPOT 0.00852s, mean lat 34.93s** (p50 34.76 / p99 35.50; ttft 0.027). Matches the prior
3792.80 (11-bench-v3) within ~1% noise ⇒ B is stable; the gap is real, not a B drift.

## WARMUP DISCIPLINE (each config: warmup discarded, then measured on warm cache)
tpu-env SKIP_JAX_PRECOMPILE=1 ⇒ cold XLA on first use. For EVERY config ran a full
c=32/total=96/out=4096 warmup and DISCARDED it; measured on the now-warm cache:
- B:    warmup ttft max 10.06s / 3480 tok/s → measured ttft max 0.034s / 3752 tok/s.
- ns7:  warmup ttft max 18.96s / 662 tok/s  → measured ttft max 7.96s / 1297 tok/s.
- ns5:  warmup ttft max 18.93s / 785 tok/s  → measured ttft max 6.79s / 1227 tok/s.
- ns10: warmup ttft max 51.44s / 884 tok/s  → measured ttft max 6.46s / 1303 tok/s.
The huge cold-warmup ttft (19–51s) vs the measured ttft confirms compiles landed in the
discarded run. (Measured ttft max ~6–8s is the per-config first-request residual compile
of an unwarmed shape, not the steady state — steady ttft mean is 0.25–0.49s.)

## NO-PREEMPTION EVIDENCE (the runs are FAIR — required by the brief)
Every DFlash serve log searched (entire file, case-insensitive) for preempt/preemption/
Preempted/recompute/recomputed/RESOURCE_EXHAUSTED/Cannot allocate/out of memory/OOM/
evict ⇒ **ZERO matches on all three (ns5, ns7, ns10).** "Waiting: 0 reqs" on every
window. HBM: total_hbm_used 41.98 GiB/chip, GPU KV cache 1,441,792 tokens, max-concurrency
341.33x for len4224 (≈10x the c=32 load) ⇒ KV blocks fit comfortably at util 0.6, no
KV-block pressure. The throughput gap is PURELY the draft step cost, not preemption.

## WHY DFlash is still slow — the lever after num_spec (root cause unchanged)
12-impl-kvcache's microbench predicted exactly this: the KV cache removed the O(C)
PROJECTION recompute (the 1.6–1.9x win), but the **O(C·B) ATTENTION-SCORE matmul** (draft
q over the full C+B keys) STAYS and now dominates (52 of 59ms @ C=4608). num_spec scales
B in that matmul, BUT lowering B also lowers acceptance ~proportionally, so throughput is
~flat-to-worse in num_spec — confirmed empirically (ns5 < ns7). The remaining lever is
HARDER: reduce the O(context) attention-score matmul itself (e.g. sparsify/window the
draft attention, or a cheaper score kernel) — not a config knob.

## CONFIGS / SCRIPTS (apples-to-apples, one at a time, free-tpu between) — VERIFIED
- A (DFlash KV-cache ON): scratchpad/serve_kv_sweep.sh — `NUMSPEC=<n> UTIL=0.6` →
  DFLASH_KV_CACHE=1, --max-num-seqs 32 --max-model-len 4224 --gpu-memory-utilization 0.6,
  speculative-config {z-lab/gpt-oss-20b-DFlash, num_speculative_tokens=<n>, method dflash},
  DRAFT_MODEL_IMPL_TYPE=torchax, EP, RAGGED_GATHER v1, --no-async, HF_HOME=/home/enyouki/local_hf.
  Cache live in every log: "DFlash KV cache enabled: k/v cache shape (8, 32, 4608, 8, 64)".
- B (target-only): scratchpad/serve_B_target.sh — same target/len/c, util 0.75, NO spec.
- Load gen: scratchpad/bench_client_pool.py --concurrency 32 --total 96 --out 4096
  (worker pool, min=max=4096 tok, ignore_eos, temp 0, stream, NO logprobs). free-tpu between.

## Reproduce
- Serve A: `NUMSPEC=7 UTIL=0.6 bash scratchpad/serve_kv_sweep.sh scratchpad/kv_ns7_serve.log`
  (wait "Application startup complete", ~65s).
- Serve B: `bash scratchpad/serve_B_target.sh`.
- Bench: `python3 scratchpad/bench_client_pool.py --port 8000 --concurrency 32 --total 96
  --out 4096 --tag <T>` (run twice, discard first). free-tpu.sh between configs.
- Accept length / per-pos: `grep "Mean acceptance length" <serve log>` (SpecDecoding metrics).

## STATUS
- [x] B re-measured same-session (3752.20 tok/s, matches prior 3793 ⇒ stable baseline)
- [x] num_spec sweep 7, 5, 10 (curve: 1297 / 1227 / 1303 ⇒ PLATEAU ns≥7, drop ns<7, opt=7)
- [x] best config (ns7) vs B: 2.89x slower throughput, 2.37x worse latency
- [x] no-preemption verified (zero signatures, all 3 runs), warmup discarded each config
- [x] mechanism: draft acceptance depth caps at ~7 (pos 8-10 accept 0.000) ⇒ num_spec lever DEAD
⇒ NEEDS_IMPL: next lever = reduce the O(ctx) draft attention-SCORE matmul (harder; not a knob).

## Artifacts
- scratchpad/serve_kv_sweep.sh (parameterized NUMSPEC/UTIL serve)
- scratchpad/kv_ns7_serve.log / kv_ns5_serve.log / kv_ns10_serve.log (serve + accept metrics)
- scratchpad/B_sweep_serve.log (B serve log)
- bench stdout: B_warmup/B_measured, A_ns{5,7,10}_warmup/_measured (in this session's run logs)
