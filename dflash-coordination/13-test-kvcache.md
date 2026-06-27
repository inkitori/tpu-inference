# 13 — TEST: KV-cache flag-ON serve path is LOSSLESS (DFLASH_KV_CACHE=1)

Phase: verify the NEW per-slot DFlash draft K/V cache (env `DFLASH_KV_CACHE=1`) is
lossless on the REAL serve path, ESPECIALLY through condense. Date: 2026-06-27.
Manager: test-kvcache. Builds on 12-impl-kvcache (the cache + flag),
10-impl-condense (the condense workload + per-slot-state migration it must follow),
05-test-batched (the perfect-draft per-slot method).

## TL;DR / VERDICT — FLAG-ON IS LOSSLESS ⇒ NEEDS_BENCH
With `DFLASH_KV_CACHE=1` ON, the GOAL serve config (c=32, max-model-len 4224,
EP, v1 gather, no-async, torchax draft, num_spec 7, method dflash):
- **Perfect-draft THROUGH CONDENSE: 100% per-slot** (21 windows all mean 8.00 /
  per-pos 1.000×7 / Accepted==Drafted) ⇒ the cache + its condense-migration keep
  verify alignment across slot moves. LOSSLESS through condense.
- **Greedy real-draft vs target-only: LOSSLESS** (24/64 byte-identical; ZERO
  step-1/2 / factual-answer divergences; all divergences deep, post-answer,
  swap-symmetric bf16 near-ties present in target-only too — same confound the
  no-cache round 10-impl-condense documented and accepted).
- **Real accepted length cache-ON ~6.3-6.9 steady-state** = at/above the ~5-6
  seen WITHOUT the cache ⇒ the cache did NOT degrade acceptance.
- The ~1-2 bf16 ULP hidden delta the cache claimed did NOT cause early divergence.

## HBM GOTCHA (NOT a correctness bug — but blocks util 0.75 on flag-ON)
At util 0.75 the flag-ON serve OOMs at the FIRST decode step:
`RESOURCE_EXHAUSTED jit__batched_ctx_write` needs 3.97 GiB, only 2.37 GiB free.
ROOT CAUSE: the flag-ON path allocates BOTH the old `_ctx_buf` (3.96 GiB) AND the
new K/V cache (2.25 GiB) — the impl (12-impl-kvcache, load_model dflash.py:165-167)
keeps `_ctx_buf` "alongside it this increment so writes can be cross-checked" and
explicitly flags "DROP _ctx_buf on flag-on path" as NOT-yet-done. So flag-ON pays
~6.2 GiB of draft buffers + the 3.97 GiB `_batched_ctx_write` JIT transient, which
no longer fits at util 0.75.
WORKAROUND for these correctness runs: util 0.6 (losslessness is independent of
util). FIX for the bench: impl manager should DROP `_ctx_buf` on the flag-on path
(net -1.7 GiB per 12-impl-kvcache's own HBM math) so flag-ON serves at util 0.75
like the no-cache path. Until then, flag-ON bench must use util ≤ 0.6 OR drop _ctx_buf.

## CONFIG (all runs)
serve_kvcache.sh (DFLASH_KV_CACHE=1, UTIL override default 0.6): max-model-len 4224,
max-num-seqs 32, EP, RAGGED_GATHER v1, --no-async, DRAFT_MODEL_IMPL_TYPE=torchax,
num_speculative_tokens 7, method dflash, HF_HOME=/home/enyouki/local_hf.
Target-only oracle: serve_B_target.sh (no spec, util 0.75, same target/len/c).
Workload: fire_condense.py / fire_condense_json.py 64 (64 reqs / 32 slots, output
lengths [32..1536] → staggered finishes → backfill → condense). Cache-live proof in
every flag-ON log: "DFlash KV cache enabled: k/v cache shape (8, 32, 4608, 8, 64)".

## TEST 1 — perfect-draft THROUGH CONDENSE, flag ON (util 0.6): PASS
DFLASH_PERFECT_DRAFT=1 + DFLASH_KV_CACHE=1. 64/64 ok, finish spread 68.7→322.1s
(253.5s ⇒ condense fired hard). 21 metrics windows, EVERY window:
Mean acceptance length 8.00 (min==max==8.00, zero dips), per-pos 1.000×7,
Accepted==Drafted, Avg 100.0%. (e.g. w1 Accepted 7399/Drafted 7399; w21 10388/10388.)
⇒ no slot mis-indexed by a condense move; the per-slot K/V cache write + condense-move
(`_batched_kv_write`/`_move_kv_rows`) keep verify aligned through backfill. LOSSLESS
through condense with the cache live. Log: scratchpad/kv_pd_serve_u06.log.
(The first attempt at util 0.75 OOM'd — see HBM GOTCHA above; not a correctness fail.)

## TEST 2 — greedy real-draft (cache ON) vs target-only: PASS (lossless)
spec-on (real DFlash, KV cache ON) vs target-only, same condense workload, greedy.
24/64 byte-identical. Divergence char-index distribution: 0 at char 0-5 (NO step-1/2
structural break), 1 below char 10, 13 in 11-40, 26 deep (>40); agreed-prefix median
149 / p75 590 / max 4086 chars. DECISIVE evidence it's bf16 near-tie not a bug:
- NO factual answer ever wrong (Paris / Jupiter / Au / Washington / primes 2,3,5,7,11
  identical both runs).
- SWAP SYMMETRY: every divergent continuation on the spec-on side ALSO appears on the
  target-only side for the same prompt (e.g. the Water-H2O branch pair shows on BOTH
  spec [21,53] and target [13,45]) ⇒ spec-on introduces no novel token, just lands on
  the other arm of the TARGET's OWN batch-position FP near-tie. The 4 "H2O mismatch"
  cases are exactly these symmetric Water branches.
This matches the no-cache round (then ~25-39% byte-identical, all answers correct,
divergences deep). ⇒ The KV cache did NOT regress losslessness. Files: specon_kv.json,
target_kv.json, diff_out.txt, kv_greedy_specon.log, kv_greedy_target.log.

## TEST 3 — real DFlash accepted length, cache ON, c=32: HEALTHY (no degradation)
16 windows, mean accept len range 3.41→6.96 (3.41/4.43 are cold XLA-warmup windows);
steady-state ≈ 6.3-6.96 (last windows 6.93/6.71/6.87/6.55/6.80/6.28). Per-pos clean
monotone-decreasing at steady state (e.g. 0.948,0.930,0.892,0.864,0.831,0.765,0.728,
avg ~85%). Steady-state ~6.3-6.9 is AT or ABOVE the ~5-6 without the cache ⇒ cache did
NOT degrade acceptance (if anything slightly higher).

## NEXT (for the bench manager)
NEEDS_BENCH: bench A(DFlash, DFLASH_KV_CACHE=1) vs B(target-only) at in=1/out=4096/
c=32, WARM cache, free-tpu between. TWO must-dos: (1) flag-ON serve OOMs at util 0.75
until `_ctx_buf` is dropped on the flag-on path — either DROP it first (HBM win,
12-impl-kvcache flagged it) or bench at util ≤ 0.6; (2) the flip was MARGINAL (~7%
short per 12-impl-kvcache's microbench projection), so SWEEP num_speculative_tokens
DOWN from 7 to clear it. Correctness is settled: flag-ON is lossless through condense.

## STATUS
- [x] Test 1 perfect-draft THROUGH condense, flag ON: 100% per-slot (21/21 windows 8.00)
- [x] Test 2 greedy real-draft cache-ON vs target-only: lossless (no answer/step-1-2 break)
- [x] Test 3 accepted length cache-ON: ~6.3-6.9 steady-state, no degradation
- [x] HBM gotcha documented (flag-ON double-buffer OOMs at util 0.75; util 0.6 workaround)
⇒ NEEDS_BENCH (drop _ctx_buf or util≤0.6 + num_spec sweep down).
