# 15 — FEASIBILITY: can DFlash beat target-only at in=1/out=4096/c=32 AT ALL?

Phase: decisive go/no-go on the GOAL speed requirement. Date: 2026-06-27. Manager: feasibility.
Builds on 14-bench-sweep (num_spec DEAD, 2.89x wall), 12-impl-kvcache (cached draft ~59ms@C=4608,
52ms is the O(C·B) attn-score matmul), 11-bench-v3 (target-only B=3752 tok/s, TPOT 8.5ms).
Method: ISOLATED real-8-chip-mesh microbenchmarks (no out=4096 serve in the loop) + a calibrated
step-time model cross-checked against the measured serve.

## THE ONE QUESTION
Spec-decode trades 1 target DECODE step (1 tok/seq) for 1 target VERIFY step (k+1 tok/seq) that
yields ~accept_len tokens. It can only win if verify is NOT proportionally more expensive than the
savings. The decisive ratio: **R = T_verify(k=7) / T_decode32 vs accept_len.**

## VERDICT: ACHIEVABLE in principle (verify is NOT the ceiling), but needs TWO levers
**R = 1.58, accept_len ≈ 5.5–6, so R << accept_len.** The target verification forward is decisively
NOT the bottleneck — with a free draft and zero overhead, spec-decode would run ~5x faster than B.
Spec-decode is ECONOMICALLY VIABLE at c=32. The whole gap is the draft step + per-step overhead.

BUT: windowing the draft attention ALONE does **not** flip A>B. It cuts the step ~105ms→~64ms
(~2700 tok/s, 0.73x B) because **~40 ms/step of FIXED host/serialization overhead** (from
`--no-async`, redundant `_ctx_buf` write, un-overlapped jit dispatches, blocking device_gets,
32-iter host loops) dominates the remaining budget. Break-even is ~46 ms/step; fixed overhead
alone (~40 ms) nearly eats it. **To beat B you must window the draft attn AND cut the fixed
overhead.** With both, the projection clears 3752 with margin (parity at FIXED≈25ms, 1.27x B at
FIXED≈15ms). ⇒ NEEDS_IMPL, two coupled levers — NOT a single config knob, NOT a user goal change.

## DECISIVE MEASUREMENTS (isolated, REAL 8-chip v6e-8 mesh, TP8+EP, batch 32, median of 25 warm)
TARGET openai/gpt-oss-20b forward, isolated (booted runner, called model_fn with hand-built
AttentionMetadata; 2-group hybrid KV (sw-128 + full), RPA v3 Pallas kernel):

| forward shape (per-seq → total q) | C≈2048 | C≈4096 |
|-----------------------------------|-------:|-------:|
| T_decode32 — 1 tok/seq (32 tok)   | 5.07ms | 5.76ms |
| T_verify k=3 — 4 tok/seq (128 tok)| 6.24ms | 7.18ms |
| T_verify k=7 — 8 tok/seq (256 tok)| 8.08ms | 9.08ms |

**R = T_verify(k=7)/T_decode32 = 1.59x @C2048, 1.58x @C4096.** (k=3: 1.23x.)
Cross-check: isolated decode 5.1–5.8ms sits below serve TPOT 8.5ms (serve adds sampling/dispatch/
sched) — consistent. The verify forward scales sub-linearly with q-length (8x the tokens → 1.58x
the time): the target forward is bandwidth/weight-bound at c=32, so extra query tokens are nearly
free. THIS is why spec-decode is viable here.

## R vs accept_len — the core economic test
R = 1.58 << accept_len 5.5–6. ⇒ one verify forward costs 1.58 decode-forwards but yields ~5.8
tokens ⇒ a sufficiently cheap draft WINS BIG. The target verify is NOT the ceiling. The draft +
the serialized host path are the ceiling.

## T_draft breakdown (from 12-impl-kvcache real-mesh microbench) + linear fit
Cached draft forward: **fwd ≈ 3.1 + 10.4·(C/1000) ms** (offset 3.1ms, slope 10.4ms/1k ctx;
fits 8.83/13.56/26.01/47.04/52.04 @C=512/1024/2048/4096/4608 to <6%). The O(C·B) attention-SCORE
matmul (draft q over C+B keys) is ~88% of it (52 of 59ms @C=4608) and is the linear term. Full
draft total @C=4608 = kv_project 0.56 + write 6.52 + fwd 52.04 = 58.80ms.
**Windowing to W (score matmul over last W ctx instead of full C): fwd ≈ 3.1 + 10.4·(W/1000):**
W=256 → ~5.8ms, W=512 → ~8.5ms (≈8x / 5.5x cheaper than C=4096's 47ms). Code-EASY: the draft mask
is already an additive (N,1,1,C+B) bias added pre-softmax (models/vllm/dflash.py is_causal=False,
eager_attention_forward); windowing = mask positions <C−W with finfo(bf16).min. NO sparse kernel,
NO remote-code edit needed. The O(C·B) matmul still runs full-width though — true cost win needs the
matmul to actually skip masked keys (slice K/V to last W before the matmul), not just mask them.
NOTE: masking-only does NOT save flops; must SLICE the cached K/V to the window to get the speedup.

## THE UNMODELED ~40ms FIXED OVERHEAD (the real second lever — found this round)
Measured steady-state serve step at full c=32 ≈ **100–110 ms** (from kv_ns7_serve.log windows:
step = N·accept/sys_tok_s = 32·4.94/1468 = 108ms etc). Isolated device-component sum @C=4096 ≈
verify 9 + draft_fwd 47 + kv_proj 0.56 + write 6.5 ≈ **63 ms**. ⇒ **~40 ms/step is host-side and
UNMODELED by the microbenches**, all FIXED (does NOT shrink when the draft forward gets cheaper):
- `--no-async-scheduling` serializes verify→sampler→draft-prepare→propose on the host critical
  path with ZERO overlap (the microbench median-of-warm-calls amortized dispatch away).
- 4 blocking device→host syncs/step (device_get next_tokens, seq_lens, query_start_loc; .tolist()
  on draft ids) — each a round-trip + dispatch stall.
- Two 32-iteration Python loops in prepare_inputs (condense-mirror + per-req ctx-write-plan) + the
  32-elem list comps in propose.
- ~6+ un-overlapped jit dispatches/step (build_noise, ctx_write, kv_project, kv_write,
  draft_forward_cached, sample_block + move_* on condense).
- `_ctx_buf` STILL written every step on the KV-cache path (kept "for cross-check", flagged
  drop-next in 12-impl) = a redundant masked RMW over the multi-GiB buffer — DROP IT.
(The TPOT 0.105s in 14-bench is a tail artifact of the worker-pool draining 96 reqs through 32
slots at falling concurrency — it is NOT the decode-step time. Step time ≈ 105ms is the real one.)

## PROJECTIONS vs B = 3752 tok/s (model: step = FIXED + verify + kv + draft_fwd; tput = accept·32/step)
Model calibrated to the measured dense step (~105ms → ~1300–1800 tok/s; conservative, slightly
over-predicts current, so verdict is if anything pessimistic).
- **(a) FREE draft (T_draft=0), keep ~40ms fixed overhead:** step ≈ 51ms → ~3400 tok/s ≈ 0.9x B
  (still loses! fixed overhead alone keeps a free-draft spec under B). With the ~40ms overhead a
  FREE draft is NOT enough — proves the fixed overhead is a real, separate ceiling.
  Intrinsic ceiling (free draft AND 0 overhead, verify-only): ~19,000 tok/s = 5.2x B (the R result).
- **(b) WINDOWED draft (W=256, fwd 47→~6ms), keep ~40ms fixed overhead:** step ≈ 64ms →
  **~2750 tok/s = 0.73x B — STILL LOSES.** Windowing alone is necessary but NOT sufficient.
- **(b') WINDOWED draft (W=256) + REDUCE fixed overhead** (async + drop _ctx_buf + fuse/overlap
  dispatch + collapse device_gets): FIXED 42→25 → parity (~3760 tok/s); FIXED→15 → ~4780 tok/s =
  **1.27x B**; FIXED→8 → ~5900 tok/s = 1.57x B. ⇒ the combination clears B with workable margin.
- accept-drop sensitivity: even if windowing drops accept 6.0→4.0, the W=256 + low-overhead case
  still beats B (the matmul-cost win and overhead cut dominate the accept loss). W=256–512 is the
  sweet spot (W<128 risks accept collapse; W>512 gives back the matmul win).

## LATENCY angle (c=32)
At c=32 spec-decode also LOSES latency today (82s vs 35s) because the per-step cost (draft + ~40ms
host) is ~13x a target step and 6x accept only recovers ~4.5x. Spec wins latency only at LOW
concurrency (where the target step is the bottleneck and the host overhead is hidden). At c=32 the
verdict is throughput-bound; the same two levers (window + overhead) are what would also fix latency.

## VERDICT + RECOMMENDATION
ACHIEVABLE at c=32 — the target verify is NOT the ceiling (R 1.58 << accept 5.8). But it needs TWO
coupled levers, not one:
1. **Window/slice the draft attention-score matmul to W≈256** (slice cached K/V to last W before the
   matmul — masking alone won't save flops). ~8x cheaper draft forward. Code-easy (additive mask
   already exists; add the K/V slice).
2. **Cut the ~40ms fixed per-step overhead**: drop the redundant `_ctx_buf` write on the KV-cache
   path; reduce/overlap the 6+ jit dispatches; collapse the 4 blocking device_gets; and (Phase 2)
   enable async scheduling so verify/draft overlap. This is the bigger lever now — a FREE draft with
   today's 40ms overhead STILL loses, so overhead reduction is mandatory.
Together the projection clears B=3752 by ~1.0–1.6x (W=256, FIXED 8–25ms). num_spec stays at 7.

## STATUS checklist
- [x] T_decode32 isolated (5.07/5.76ms @C2048/4096) — cross-checks serve TPOT 8.5ms
- [x] T_verify k=7 / k=3 isolated (8.08/9.08 ; 6.24/7.18 ms) — R = 1.58
- [x] R vs accept_len: 1.58 << 5.8 ⇒ verify NOT the ceiling, spec-decode economically viable
- [x] T_draft breakdown + linear fit (3.1 + 10.4·C/1000; 88% is the O(C·B) score matmul)
- [x] reconciled serve step (~105ms) vs isolated sum (~63ms) ⇒ ~40ms FIXED host overhead found
- [x] projections (a) free-draft (b) windowed (b') windowed+low-overhead vs 3752
- [x] verdict: ACHIEVABLE via window W≈256 + ~40ms overhead cut (two levers)

## Artifacts
- L3 isolated target script: scratchpad/bench_target_verify.py (+ run3.log) — the T_decode/T_verify table
- This file's projection math reproduced inline (calibrated to measured 14-bench step ~105ms)
