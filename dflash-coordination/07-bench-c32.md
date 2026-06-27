# 07 — SPEED BENCH @ c=32 (DFlash vs target-only, branch `dflash`)

Phase: decisive A-vs-B speed verdict. Date: 2026-06-27. Manager: bench-c32.
GOAL bench point: input=1, output=4096, concurrency=32, WARM XLA cache.

## VERDICT (decisive): DFlash is ~10.6x SLOWER than target-only. → NEEDS_IMPL.
At the GOAL bench point DFlash does NOT beat target-only — it loses badly. Acceptance
is HEALTHY (mean ~5.5/step at full c=32) but the per-decode-step wall-clock cost of the
torchax DFlash draft+verify path is ~37x a target-only step, which ~5x acceptance cannot
compensate. This is a PERF problem, not a correctness/accept problem.

## Headline numbers (warm cache, in=1, out=4096, c=32, 32/32 ok, 4096 tok each)
| Metric                         | A: DFlash ON | B: target-only | B/A   |
|--------------------------------|-------------:|---------------:|------:|
| SYSTEM output tok/s            | **358.65**   | **3788.53**    | 10.6x |
| wall time (s)                  | 365.46       | 34.60          | 10.6x |
| mean per-req latency (s)       | 292.53       | 34.59          | 8.5x  |
| p50 / p99 latency (s)          | 310 / 365    | 34.6 / 34.6    |       |
| TPOT (s/token)                 | **0.31724**  | **0.00844**    | 37.6x |
| per-stream tok/s (1/TPOT)      | 3.15         | 118.49         |       |
| ttft (s)                       | 0.41         | 0.03           |       |
| DFlash mean accepted length    | ~5.5 @ full c=32; 4.33 avg over run | n/a | |

- A steady-state system gen throughput at full c=32: ~315–363 tok/s (Running:32 windows).
- A mean-accept per-position (full load): 0.85, 0.75, 0.67, 0.62, 0.58, 0.54, 0.52
  (very healthy — drops to ~2.4 only in the tail as reqs finish and load falls below 32).
- Per-req latency for A is ~8.5x not 10.6x because reqs finish staggered (accept rate falls
  as concurrency drops in the tail); the system-throughput 10.6x is the apples-to-apples #.

## Configs (apples-to-apples; one at a time, free-tpu between) — VERIFIED
- **A (DFlash ON):** scratchpad/serve_A_dflash.sh — `--max-num-seqs 32 --max-model-len 4224
  --gpu-memory-utilization 0.75`, `--speculative-config {model: z-lab/gpt-oss-20b-DFlash,
  num_speculative_tokens: 7, method: dflash}`, DRAFT_MODEL_IMPL_TYPE=torchax. EP,
  RAGGED_GATHER v1, --no-async-scheduling, HF_HOME=/home/enyouki/local_hf.
- **B (target-only):** scratchpad/serve_B_target.sh — SAME target, SAME --max-num-seqs 32,
  SAME --max-model-len 4224, SAME util 0.75. NO speculative-config.
- Both servers: prompt "Hi" → prompt_tokens=1 confirmed; both emit IDENTICAL greedy text
  (", I am a student and I am") on the 8-tok sanity probe ⇒ same target, apples-to-apples.

## Method (followed the bench discipline)
- tpu-env.sh SKIP_JAX_PRECOMPILE=1 ⇒ cold XLA compile on first use. Ran a WARMUP c=32/out=4096
  for EACH config and DISCARDED it; took the measured run on the now-warm cache.
- WARMUP evidence it was a real compile run (correctly discarded):
  - A_warmup: wall 667s, 196 tok/s (cold-compile dominated; measured A then 365s/359 tok/s).
  - B_warmup: ttft **19.7s** (compile), 2402 tok/s; measured B ttft **0.03s**, 3789 tok/s.
  - Measured runs show low/normal ttft ⇒ no compile leaked into the measurement.
- Load gen: scratchpad/bench_client.py — async 32-concurrent, max_tokens=min_tokens=4096,
  ignore_eos=true, temperature=0, streaming, NO logprobs (spec-on logprobs crash is known/oos).
  All 32 reqs returned exactly 4096 completion tokens in BOTH configs.

## HBM (info)
- A (DFlash): total_hbm_used 39.73 GiB (8-chip total) / cap 187.48; KV 1,972,451 tokens,
  max-conc 466.96x. Fits at util 0.75 (consistent with 06-impl-hbm).
- B (target-only): total_hbm_used 24.07 GiB; KV 2,181,304 tokens, max-conc 516.41x — more
  slack (no _ctx_buf), as expected. HBM is NOT the bottleneck for the speed gap.

## WHY DFlash is slow — bottleneck map (L3 code audit; for the IMPL manager)
Acceptance is fine; the per-step path is the problem. Top suspects (file:line):
1. **dflash.py:~324 `self._ctx_buf = lax.dynamic_update_slice(...)` — EAGER, non-donated,
   inside a per-request python loop (~305–327).** Functional update on the persistent
   ~4.0 GiB `_ctx_buf` (32,4608,14400) bf16 ⇒ XLA copies the FULL buffer; the per-request
   loop makes it up to 32 full-buffer copies/step (~100+ GiB device traffic/step) just to
   append a few hidden rows. KNOWN as a transient (06-impl-hbm chip task_d0ad4933) but its
   true cost is the per-request multiplier + missing donate_argnums. LIKELY co-dominant.
2. **HF draft is STATELESS / no KV cache (models/vllm/dflash.py fc @ ~177; past_key_values=
   None, use_cache=False).** Every step recomputes fc (14400→2880) + 8 attn layers over the
   ENTIRE padded context (grows toward 4608) — O(context) compute that GROWS with output len
   (~6 TFLOP for fc alone/step at long ctx). A target step touches 1 new position; DFlash
   reprocesses the whole sequence each step. Dominant & growing COMPUTE term at out=4096.
3. **Per-step host↔device syncs + ragged work (dflash.py:292/295 device_get seq_lens/qsl;
   299 concat full raw-hidden; 333 .tolist(); manager loop 259–271).** Multiple forced
   device syncs/step prevent pipelining; additive latency on top of (1)+(2).
- Single draft dispatch (NOT 7 sequential passes) — that part is fine.
- No per-step recompiles, no per-step all-gather — fine.
- NEW vs known: (a) the :324 write is in a 32x per-request loop; (b) missing donate_argnums
  ⇒ full copy not in-place; (c) the stateless full-context recompute is an unflagged O(ctx)
  compute cost that grows with output length; (d) several per-step device_get syncs.

## Handoff → NEEDS_IMPL (do NOT optimize from the bench seat)
Make a DFlash decode step cheap enough that ~5.5x acceptance nets a win (target step is
0.0084s; DFlash must get well under ~0.046s/step to break even). Highest-leverage levers,
in order: (1) jit + donate_argnums the _ctx_buf write (in-place, kill the per-step full-copy;
batch-scatter instead of the 32x python loop) — chip task_d0ad4933; (2) give the draft a
KV cache / stop reprocessing the full context each step (the O(ctx) compute term, worst at
out=4096); (3) remove the per-step device_get host syncs. Re-run THIS bench after each.

## Reproduce
- Serve A: `bash scratchpad/serve_A_dflash.sh`  (wait health 200, ~75s warm weights+compile)
- Serve B: `bash scratchpad/serve_B_target.sh`
- Warm + measure: `python scratchpad/bench_client.py --port 8000 --concurrency 32 --out 4096
  --tag <T>` (run twice; discard the first). free-tpu.sh between configs.
- Accept length: `grep "Mean acceptance length" <serve log>` (vLLM SpecDecoding metrics).
