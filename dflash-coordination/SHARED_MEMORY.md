# SHARED MEMORY (read/write — HARD CAP 250 LINES)

> Rules for every agent manager:
> - Keep this file UNDER 250 lines. If you approach the cap, PRUNE stale/obsolete lines.
> - Only critical cross-manager facts go here. Per-phase detail goes in YOUR own file.
> - Entry format: `[YYYY-MM-DD][manager-id] fact`. Add new facts under the right section.
> - This is the one place a fresh manager learns what prior managers discovered. Keep it
>   true and lean.

## Current best-known status
- [2026-06-27][test-kvcache] **FLAG-ON (DFLASH_KV_CACHE=1) SERVE PATH VERIFIED LOSSLESS ⇒ NEEDS_BENCH.**
  GOAL config (c=32, len4224, EP, v1, no-async, torchax draft, num_spec 7). (1) PERFECT-DRAFT THROUGH
  CONDENSE 100% per-slot: 21 metrics windows ALL mean 8.00 / per-pos 1.000×7 / Accepted==Drafted across a
  hard condense event (fire_condense 64, finish spread 254s) ⇒ the per-slot K/V cache + its condense-move
  keep verify aligned through backfill. (2) GREEDY real-draft cache-ON vs target-only: LOSSLESS — 24/64
  byte-identical, ZERO step-1/2 / factual-answer divergences, all divergences deep post-answer + SWAP-
  SYMMETRIC (same branch appears on target-only too) = batch-position bf16 near-tie present in target-only
  too (same confound 10-impl-condense accepted). (3) ACCEPTED LENGTH cache-ON ~6.3-6.9 steady-state = AT/
  ABOVE the ~5-6 without the cache ⇒ cache did NOT degrade acceptance. **HBM GOTCHA (not a correctness
  bug, but BLOCKS the bench at util 0.75):** flag-ON allocates BOTH the old _ctx_buf (3.96 GiB) AND the new
  K/V cache (2.25 GiB) — 12-impl-kvcache keeps _ctx_buf "for cross-check" and flagged dropping it as not-
  done — so the jit__batched_ctx_write transient (3.97 GiB) OOMs at util 0.75 at FIRST decode. Workaround:
  serve flag-ON at util ≤ 0.6 (these correctness runs did; losslessness is util-independent). FIX for bench:
  DROP _ctx_buf on the flag-on path (net -1.7 GiB per impl's own math) → then 0.75 fits. ⇒ NEEDS_BENCH: A
  (KV_CACHE=1) vs B(target-only) at out=4096/c=32 (drop _ctx_buf OR util≤0.6) + sweep num_spec DOWN from 7
  to clear the ~7% marginal flip. Details: 13-test-kvcache.md.
- [2026-06-27][impl-kvcache] **LEVER #1 (KV-cache the O(ctx) draft recompute) LANDED + microbench-proven
  + token-equivalent ⇒ closes ~the whole 2.90x gap but MARGINAL (lands ~7% short of flipping A>B alone;
  num_spec-down is the next lever).** Flag-gated `DFLASH_KV_CACHE=1` (default OFF, live path UNCHANGED).
  Design B2: project K/V for ONLY newly-accepted rows (`_kv_project`, O(B)) + cache per-slot
  (_k_cache/_v_cache (L=8,32,buf~4608,KVH=8,hd=64) bf16) + a cache-consuming forward
  (`_draft_forward_cached`) that attends over [cached ctx K/V | fresh noise K/V] so fc+k/v-proj over
  the full context never recompute. Cache write `_batched_kv_write` + condense-move `_move_kv_rows`
  MIRROR the proven _batched_ctx_write/_move_ctx_rows idioms + plan arrays (so condense backfill is
  handled identically). **REAL 8-chip mesh N=32 microbench**: FULL draft step 88→98ms @C=4096-4608
  drops to **54→59ms CACHED = 1.6-1.9x** (the cache kills the O(C) projection; the O(C) attention-SCORE
  matmul stays + now dominates, 52 of 59ms @C=4608, so cached fwd is NOT flat in C). FLIP: cached_step/6
  = 8.99ms/accepted-tok @C=4096 vs target 8.40 ⇒ ~7% over break-even (lower-C steps already win). Token-
  equivalent: synthetic multi-step tie-free argmax IDENTICAL all steps (hidden ~1-2 bf16 ULP/8 layers =
  bf16 floor). 17 unit tests green; flag-off path verified unchanged. HBM: full cache replicated = 2.25
  GiB < the 3.96 GiB _ctx_buf it can REPLACE (net -1.7 GiB once _ctx_buf dropped; still written
  alongside as cross-check now). ⇒ NEEDS_TEST (real serve-path perfect-draft c=32 + greedy-vs-target
  with flag ON) then NEEDS_BENCH with num_spec tuned DOWN to clear the marginal flip. Commits ebb2b44c,
  39e2f91f, e1fa4115, d7a4300b, b5d5c7b4. Details: 12-impl-kvcache.md.
- [2026-06-27][bench-v3] **FIRST TRUSTWORTHY A-vs-B SPEED VERDICT: DFlash is ~2.90x SLOWER than
  target-only at the GOAL bench point (in=1, out=4096, c=32, WARM cache, total=96 reqs so
  backfill/condense fires). ⇒ NEEDS_IMPL.** A(DFlash)=1309.79 sys out tok/s, TPOT 0.111s, mean
  latency 86.7s; B(target-only)=3792.80 sys out tok/s, TPOT 0.0084s, mean latency 34.6s. NO CRASH
  (the 09 dflash.py:421 broadcast crash is GONE under real condense/backfill at out=4096/c=32 —
  confirms 10-impl-condense's fix holds end-to-end). 96/96 ok both configs, all exactly 4096 tok;
  same target (byte-identical greedy probe). DFlash accept HEALTHY ~5.0-6.2/step at full c=32 (tail
  to 6.7), per-pos 0.85→0.40 ⇒ PERF gap, not accept. DFlash ALSO loses latency here (2.5x worse) —
  the "spec wins latency" hypothesis does NOT hold at c=32 (target step already cheap+batched).
  Write fix (6b6acd49) helped (corrupt-07 358 → 1310, 3.7x), but the ONLY remaining lever is
  **LEVER #1: KV-cache the O(ctx) draft recompute** (draft recomputes fc+8 attn over the FULL
  context every step, TPOT 13.2x). Need ~2.2x step speedup to flip A>B; KV cache removes the O(ctx)
  term entirely. ⇒ the 07 "10.6x / 358 tok/s" A-number is now SUPERSEDED (it was silently corrupt);
  THIS is the honest baseline. HBM: A 39.73 / B 24.07 GiB (not the gap). Details: 11-bench-c32-v3.md.
- [2026-06-27][impl-condense] **CONDENSE/SLOT DESYNC FIXED + VERIFIED ACROSS A REAL CONDENSE EVENT ⇒
  NEEDS_BENCH.** The 09 out=4096/c=32 broadcast crash (dflash.py:421) is GONE. PROPER fix (b94fc366,
  found already-committed from a prior session): mirror vLLM InputBatch.condense slot-moves onto the
  proposer's per-slot state (gather _ctx_buf row + carry _ctx_len/_prev_seq_len/_last_req_id; only NEW
  req_ids reset) + a defense-in-depth `num_new=min(num_new, seg_width=qsl[i+1]-qsl[i])` clamp. That
  proper fix INTRODUCED an HBM regression (the full-buffer gather `_permute_ctx_rows` = ~4.58 GB
  transient → RESOURCE_EXHAUSTED at util 0.75 under condense, dflash.py:452); FIXED by 86f150cf:
  `_move_ctx_rows` = SPARSE in-place donated scatter `ctx_buf.at[dst_slots].set(ctx_buf[src_slots])`
  gathering only K moved rows (bucketed {1,2,4,8,16,32}); common 1-2-move condense → ~248 MiB (was
  ~3.96 GiB). Worst-case full permutation still correct (K=32, never drops a move). ON-TPU condense
  workload (fire_condense.py 64, util 0.75, max-model-len 4224, c=32 = the GOAL config): **(A) NO
  crash** (no broadcast, no OOM; 64/64; finish spread 80s ⇒ condense fired); **(B) perfect-draft 100%
  per-slot ACROSS condense** (10 windows all mean 8.00 / per-pos 1.000×7 / Accepted==Drafted ⇒ verify
  stays aligned through slot moves, lossless through condense); **(C) greedy answers token-identical
  64/64** spec-on vs target-only (filler-only divergence = batch-position FP drift present in
  target-only too); **(D) real DFlash accept under condense steady-state ~6.0-6.7** (per-pos busy
  0.716→0.148 / light ~0.96→0.64). CPU bit-identical check (swap/cycle/padding) PASS, unit tests 5/5 +
  5/5. ⇒ the 07 "358 tok/s / 10.6x" A-number stays UNRELIABLE (silently-corrupt run); re-bench A vs B
  at out=4096/c=32 is now FINALLY trustworthy. Details: 10-impl-condense.md.
- [2026-06-27][impl-perf] **LEVER #2 (write fix) LANDED — commit 6b6acd49.** Replaced the
  eager 32x lax.dynamic_update_slice loop in dflash.py prepare_inputs() with ONE jitted+donated
  masked-scatter `_batched_ctx_write` (in-place). ISOLATED microbench: per-step _ctx_buf WRITE
  ~234ms → **~2.2ms (~106x)** at decode on the real 8-chip mesh. **BIT-IDENTICAL** to the old
  loop (np.array_equal full buffer, all edge cases) — but it TOUCHED the proven write path ⇒
  per GOAL, NEEDS_TEST (re-run perfect-draft b=32 lossless before trusting). **HBM UNCHANGED**
  at GOAL (max_model_len=4224 → buf_len stays 4608, 3.96 GiB; the dead-row bump only fires when
  max_model_len is an exact multiple of 512). Unit tests 5/5. Per-step now ~17-101ms (was
  ~250-334ms), DOMINATED by the O(ctx) fc/attn forward = **LEVER #1 (KV-cache the context),
  next round** — breaks even only for C≲2k, 99ms at C=4608. Details: 08-impl-perf.md.
  Chip task_d0ad4933 (the donated-write task) is DONE.
- [2026-06-27][bench-c32] (SUPERSEDED by bench-v3 above — the "10.6x / 358 tok/s" A was a
  silently-corrupt run; B=3788.53 tok/s / TPOT 0.0084 was the honest target-only and matches
  bench-v3's B. Bottleneck map preserved in 07-bench-c32.md if needed.)
- [2026-06-26][L1-seed] Branch `dflash` has substantial prior DFlash work. `STATE.md`
  at repo root documents a claimed "working DFlash gpt-oss-20b serve recipe". Recent
  commits fixed: 0% accept (project draft logits through target lm_head, not input
  embedding); MXFP4→fp8_e4m3fn requant workaround for v6e MoE experts; phantom DFlash
  draft KV groups on torchax; draft mask docstring.
- [2026-06-26][L1-seed] UNKNOWN at start: is it CURRENTLY lossless? is it CURRENTLY
  faster than target-only at the required bench point? Do NOT trust that it works —
  ASSESS the real current state first.

## Confirmed facts
- [2026-06-26][test] CORRECTNESS DONE. Ladder#1 perfect-draft (DFLASH_PERFECT_DRAFT=1) →
  100% accept, mean accept len 8.00, per-pos 1.000x7 over 350+ tokens ⇒ verify/accept path
  correct, no off-by-one. Ladder#2 matched-shape stepwise (max_tokens=1, both servers):
  spec-on == target-only TOKEN-IDENTICAL 24 steps incl a 0.125-nat near-tie ⇒ LOSSLESS.
  Real DFlash accept ~mean 2.35-3.14, per-pos 0.74→0.04 (healthy). Details: 02-test.md.
- [2026-06-26][test] Committed+pushed diag `ba34e82b`: env-gated DFLASH_PERFECT_DRAFT in
  tpu_runner.py _sample_from_logits (off by default). Safe to leave in tree.
- [2026-06-26][test] GOTCHA for any logit probing: HTTP /v1/completions logprobs path is
  BUGGY for >1 token (IndexError serving.py:627). Use max_tokens=1 + logprobs<=20 (cap 20).
  Does NOT affect normal serving/bench. Also: single-shot vs stepwise(max_tokens=1) give
  different numerics → only compare matched request shapes for losslessness.
- [2026-06-26][research] Tree CLEAN (no uncommitted source; only untracked dflash-coordination/).
  Code on `dflash` looks COMPLETE/coherent for torchax spec-decode. Prior smoke test gave CORRECT
  output, ~2.89 mean accept len (~27% avg draft accept). Lossless+speed at the bench point NOT yet
  verified. Full notes: 01-research.md.
- [2026-06-26][research] Draft impl is DRAFT_MODEL_IMPL_TYPE=**torchax** (NOT vllm). Target is vllm
  (tpu-env.sh default MODEL_IMPL_TYPE=vllm). torchax draft selects spec_decode/vllm/dflash.py.
- [2026-06-26][research] Exact launch cmd (verbatim in STATE.md + memory):
  `env HF_HOME=/home/enyouki/local_hf DRAFT_MODEL_IMPL_TYPE=torchax RAGGED_GATHER_VERSION=v1 RAGGED_GATHER_REDUCE_VERSION=v1 ~/tpu-tooling/tpu-env.sh vllm serve openai/gpt-oss-20b --tensor-parallel-size 8 --enable-expert-parallel --no-async-scheduling --max-model-len 2048 --max-num-seqs 1 --speculative-config '{"model": "z-lab/gpt-oss-20b-DFlash", "num_speculative_tokens": 7, "method": "dflash"}'`
- [2026-06-26][research] PERFECT-DRAFT INJECTION POINT (GOAL ladder #1):
  tpu_inference/runner/tpu_runner.py `_sample_from_logits` ~L1659-1665. After draft_token_ids =
  _extract_draft_token_ids(...) (L1662), set `draft_token_ids = jnp.argmax(target_logits, axis=-1)`.
  Matches rejection_sampler.py:542 argmax → 100% accept. Run greedy (temp 0), batch size 1.
- [2026-06-26][research] Spec-decode CUDA/untraceable-op sweep: CLEAN. No cuda/triton/multinomial/
  nonzero/bincount/custom ops. Rejection sampler fully @jax.jit fixed-shape. (.tolist/device_get are
  host-side orchestration, by design.) Only out-of-repo risk: draft HF remote code, kept eager.
- [2026-06-26][research] Verify/accept-reject is the SHARED spec-decode rejection sampler
  (rejection_sample_greedy, accept = draft==target_argmax) — NOT DFlash-specific. PR#1868 added the
  proposer; "#1869"=a03d42e4 wired runner; "#1870"=fix cluster (5f4047cd lm_head proj = 0%-accept fix).

## Confirmed facts (batched)
- [2026-06-27][test-batched] **BATCHED CORRECTNESS DONE.** Perfect-draft per-slot PASS at
  uniform b=16 (448/448 accept), ragged b=16 (672/672), and ragged b=32 @ GOAL concurrency
  (1302/1302) — every metrics window mean accept len 8.00, per-pos 1.000×7, Accepted==Drafted.
  Aggregate metric is a valid per-slot detector under perfect-draft (one mis-indexed slot drags the
  global mean down). HBM FITS at c=32/len1024/util0.75 (no RESOURCE_EXHAUSTED). Details: 05-test-batched.md.
- [2026-06-27][test-batched] **BATCHED GREEDY LOSSLESS.** Matched-shape oracle (both servers same
  request shape, NO logprobs): step-1 batch=8 8/8 next-token IDENTICAL spec-on vs target-only (incl a
  0.062-nat near-tie slot, still matched); 8-step batch=8 8/8 byte-identical. The 4/8 deep divergences
  in free-running 64-tok completions are later-step request-shape numeric drift landing on
  high-entropy near-ties (the 02-test confound), NOT verifier bugs.
- [2026-06-27][test-batched] L3 code audit of per-slot _ctx_buf: CLEAN, no slot cross-contamination
  (covers the one bug class perfect-draft can't — drafter slot-mixing). Minor: dead num_rejected_tokens
  param in prepare_inputs; inert draft_attn_metadata (propose() never reads attn_metadata on torchax).
- [2026-06-27][test-batched] **NEW GOTCHA (worse than 02-test's):** requesting LOGPROBS on the
  SPEC-ON server CRASHES the engine — OverflowError in async_llm output_handler (logprobs.py:97
  detokenizes an out-of-range id in the top-k logprobs stream). Out-of-range id is in the LOGPROBS
  tensor, NOT the emitted sequence (sampled tokens fine). For spec-on probing use max_tokens=1 with
  NO logprobs. Worth fixing eventually but not a generation/verifier bug.

## Known blockers / gotchas
- [2026-06-27][impl-hbm] **HBM BLOCKER RESOLVED — SERVES at c=32 / max-model-len 4224 /
  util 0.75 (32/32 concurrent ~4000-tok greedy, coherent, no OOM).** This is enough for the
  GOAL bench point (in=1, out=4096 needs seq cap 4097). Two BIT-IDENTICAL fixes (NO re-test):
  (1) `395e90ce` right-size _ctx_buf: padding granularity pow2→multiple-of-512, buf_len at
  len4224 8192→4608 ⇒ 7.03→3.96 GiB/chip; (2) `2f62f0f1` fuse the per-step read-slice
  `_ctx_buf[:n,:c]` INSIDE the jitted draft forward (was an eager 3.96 GiB copy/step = the
  1st OOM site). Both change only memory layout, not values. _raw_hidden_dim = **14400**
  (5 target layers × 2880), NOT 13440. Unit tests now 5/5 (torchax) + 5/5 (jax-native).
  Details: 06-impl-hbm.md. ⇒ NEXT = NEEDS_BENCH (c=32 speed bench, util 0.75 len 4224).
- [2026-06-27][impl-hbm→impl-perf] Headroom was TIGHT (util 0.75 fits, 0.80 OOMs) due to the
  eager per-step write transient (old dflash.py:324). **RESOLVED by 6b6acd49** (jitted+donated
  in-place write) — the 3.96 GiB write transient is gone, which should also unlock util ≥ 0.80
  / longer context (a re-bench can confirm; not re-measured this round). chip task_d0ad4933 DONE.
- [2026-06-27][impl] **c=32 ASSERT BLOCKER RESOLVED.** Batched/multi-seq DFlash torchax decode
  IMPLEMENTED + smoke-verified. The num_reqs<=1 assert is gone; server serves --max-num-seqs 32
  and 32/32 concurrent greedy requests return COHERENT output (Paris / George Washington / Au /
  speed of light / J.K. Rowling; dup prompts -> identical greedy). dp_size==1 + no-async asserts
  KEPT (independent). Commits 9410b793 + f9947f43. Details: 04-impl.md. ⇒ NEXT = NEEDS_TEST:
  re-run perfect-draft machinery test AT batch>1 to prove batched verify still lossless, THEN the
  c=32 speed bench (03-bench's original job).
- [2026-06-26][research] v6e-only workarounds in place (not general): MXFP4→fp8_e4m3fn requant
  (layers/vllm/quantization/mxfp4.py); v1 ragged gather mandatory. EP mandatory (plain TP8 dies
  IndivisibleError). gcsfuse root-only → stage to local HF cache HF_HOME=/home/enyouki/local_hf.
  Draft HF remote code needs `datasets` installed in venv.

## Decisions (with rationale)
- [2026-06-27][impl] Batched DFlash = PER-SLOT context buffers. proposer state in
  spec_decode/vllm/dflash.py is now per-slot: _ctx_buf (max_num_reqs, buf_len, raw_hidden_dim),
  _ctx_len/_prev_seq_len np arrays, _last_req_id list. prepare_inputs loops over real requests,
  slices each req's accepted hidden from the FLAT-RAGGED aux_hidden_states via query_start_loc
  (raw_hidden[qsl[i] : qsl[i]+num_new_i], leading rows = accepted), appends to that slot's row,
  pads all N ctx to one rectangular max_padded_ctx with per-row (N,C+B) additive mask + (N,C+B)
  positions + (N,B) noise ids. Model forward (models/vllm/dflash.py) drops the hardcoded
  unsqueeze(0)/squeeze(0), carries N, mask reshape (N,1,1,C+B); out_shardings extended to 3-D
  (MLP_DATA size 1 @ DP=1 -> leading axis replicated; vocab stays TP on MLP_TENSOR=8). The shared
  rejection sampler + manager bookkeeping were ALREADY batch-aware. HF DFlash forward is fully
  batch-generic (no change). Rationale: mirrors eagle3's flat-ragged query_start_loc convention;
  no structural blocker, HF model already handles arbitrary bsz.

## Dead ends — do NOT repeat
- (none yet)
