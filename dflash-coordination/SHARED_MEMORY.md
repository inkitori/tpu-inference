# SHARED MEMORY (read/write — HARD CAP 250 LINES)

> Rules for every agent manager:
> - Keep this file UNDER 250 lines. If you approach the cap, PRUNE stale/obsolete lines.
> - Only critical cross-manager facts go here. Per-phase detail goes in YOUR own file.
> - Entry format: `[YYYY-MM-DD][manager-id] fact`. Add new facts under the right section.
> - This is the one place a fresh manager learns what prior managers discovered. Keep it
>   true and lean.

## Current best-known status
- [2026-06-27][impl-headshard] **HEAD-SHARD LANDED via jax.shard_map — cached draft fwd 83.8→9.7ms (8.6x),
  BIT-EXACT (max|diff| 0.0, greedy-argmax 100%), all-gathers 72→0, full context KEPT (accept stays ~6) ⇒
  the draft attn is NO LONGER the bottleneck; binding constraint is now Lever B (~40ms FIXED overhead).**
  Fix (commit 36ecb75c): replaced eager_attention_forward in _draft_forward_cached with `_sharded_attention`
  = jax.shard_map over the 'model' axis (each chip does nh/8=8 q-heads + kvh/8=1 kv-head: GQA expand, q@k^T·
  scaling+mask, f32 softmax→bf16, scores@v, sharded-o_proj contraction, psum). o_proj bias added by caller
  once. K/V cache + kv_project out KVH-axis sharded (condense-safe: writes/moves index slot/row not KVH).
  ROOT CAUSE the weight-sharding (88862fdb) + with_sharding_constraint (_pin, reverted cc735253) FAILED:
  GSPMD all-gathers q at `q_proj(hs).view(N,B,nh,hd)` (splits the model-sharded nh*hd axis → gathers) and at
  repeat_kv's reshape (the SCORES); a constraint only re-slices the already-gathered tensor (even +3 gathers).
  shard_map gives each chip no freedom to gather. PROJECTION vs B=3752 (full-ctx, step=FIXED+verify9.08+
  draft16.78; accept6): FIXED40(today) 0.78x, FIXED25 **1.01x**, FIXED15 **1.25x**, FIXED8 **1.51x**;
  accept5: FIXED15 1.04x. ⇒ head-shard alone (FIXED~40) still LOSES; **need Lever B to cut FIXED≤15-25ms —
  but now WITHOUT windowing (full ctx kept, accept ~6), which is the full-context win 16-impl asked for.**
  NEEDS_TEST: real serve perfect-draft-through-condense + greedy-vs-target lossless re-verify (attn formul.
  changed), THEN Lever B + serve bench. Probes pushed. Details: 19-impl-headshard.md.

- [2026-06-27][redteam-attn] **17's "FLOP-bound dead end" is FALSE — the draft attn is REPLICATION/
  MEMORY-bound and ~60x FIXABLE ⇒ ACHIEVABLE, NEEDS_IMPL.** Physics: useful attn FLOPs 1.55e11 ⇒ floor
  0.17ms @918TFLOP/s; measured eager 48.7ms = **0.35% of MXU peak** (a FLOP-bound kernel runs 30-70%,
  never 0.35%). DECOMPOSE (real 8-chip): repeat_kv alone = 9.5ms @ 2285 GB/s = **139% of single-chip HBM
  peak** (pure-memory smoking gun: the 8x GQA-expanded K/V is written then RE-READ by both matmuls);
  f32-score materialization = +8ms (secondary). FIX (measured, real mesh, correct to bf16 max|diff|
  1.9e-3): the draft attn runs REPLICATED (`PartitionSpec()`) so all 8 chips redundantly do all 64 heads
  — **HEAD-SHARD the 64 q-heads 8/chip on the `model` axis (SAME eager math) = 48.78→1.49ms (33x)**; +
  GQA-native K/V reads (KVH=8) + bf16 scores → **0.82ms (~60x)** @C=4608. 17 was wrong because its bench
  compared two REPLICATED formulations (eager + flash) and never sharded the embarrassingly-parallel head
  axis. Flash is NOT needed (replicated flash w/ sane tiling = 153ms WORSE than 17's 84; head-sharded
  flash 17.7ms still loses to eager head-sharded — tiny q-block B=8). Head-sharding KEEPS FULL CONTEXT ⇒
  accept stays ~6 (unlike windowing). PROJECTION (attn 52→~1ms; step=FIXED+verify9.08+draft~10): FIXED40
  (today) 0.86x, **FIXED25 1.16x, FIXED15 1.50x, FIXED8 1.88x B=3752** (accept→5 sensitivity still clears
  B once overhead cut). ⇒ this REVALIDATES 15's two-lever plan with a 60x (not 8x-windowing) draft lever;
  the ONLY remaining gap is Lever B (cut the ~40ms FIXED host overhead, already scoped). NO new draft
  model, NO sparse-attn research, NO dropping c=32. **NEEDS_IMPL: (1) head-shard the draft attn in
  models/vllm/dflash.py (shard Hq on MLP_TENSOR; exact per-head), (2) GQA-native+bf16 scores [keep softmax
  f32 — pure-bf16 softmax SIGSEGVs the v6e conv-emitter; use lax.dot_general], (3) Lever B overhead cut.**
  Microbenches committed (0333f8fa, 32264ef4); source untouched. Details: 18-redteam-attn.md.
- [2026-06-27][impl-window] **LEVER A (windowing) DEAD (kills accept 6→2.5, window-insensitive). LEVER B
  landed + sound (KEEP). Its open question "need a cheaper FULL-CONTEXT draft attn, not windowing" is now
  ANSWERED by impl-headshard (shard_map).** Lever B (numerics-NEUTRAL, KEEP): on the KV path _ctx_buf is
  written but never read — DROP its alloc (frees ~3.96 GiB → util 0.75 fits), drop per-step
  _batched_ctx_write + condense _move_ctx_rows + warms; BATCH the 2 device_gets into 1. Does NOT touch
  proposals (accept ~6). Flag-off byte-unchanged. Unit tests 17/17. Commits c620d4cd, 28f9a432, de781d72.
  Details: 16-impl-window.md.
- [2026-06-27][feasibility] **FEASIBLE at c=32 — target VERIFY is NOT the ceiling — but needs TWO
  levers, not one ⇒ NEEDS_IMPL.** Decisive isolated 8-chip microbench (target gpt-oss-20b, batch 32):
  T_decode32=5.8ms, T_verify(k=7)=9.1ms @C4096 ⇒ **R=1.58 << accept_len ~5.8** ⇒ one verify forward
  costs 1.58 decode-forwards but yields ~5.8 tokens (target forward is weight-bound at c=32, +8x query
  tokens = +58% time only). So a cheap-enough draft WINS BIG (intrinsic ceiling = free-draft+0-overhead
  = ~5x B). LEVER A: WINDOW/SLICE the draft attn-score matmul to W≈256 (slice cached K/V to last W
  BEFORE the matmul — masking-only won't save flops; mask is already additive so code-easy) → draft fwd
  47→~6ms @C4096 (fit: fwd=3.1+10.4·C/1000, 88% is the O(C·B) score matmul). **BUT WINDOWING ALONE
  LOSES (~2750 tok/s, 0.73x B)** because the serve step has **~40ms/step FIXED host overhead** (recon:
  real step ~105ms vs isolated device sum ~63ms) from `--no-async`, redundant `_ctx_buf` write (drop
  it), 6+ un-overlapped jit dispatches, 4 blocking device_gets, 32-iter host loops — NONE shrink with a
  cheaper draft. A FREE draft with today's 40ms overhead STILL loses ⇒ LEVER B: cut the fixed overhead
  (drop _ctx_buf write, fuse/overlap dispatch+device_gets; async = Phase 2) is MANDATORY + the bigger
  lever now. **WINDOW W≈256 + overhead→8-25ms ⇒ projection clears B=3752 by ~1.0-1.6x** (parity at
  FIXED25, 1.27x at FIXED15). num_spec stays 7. NOT a config knob, NOT a user goal change. (Note: 14-
  bench TPOT 0.105s is a worker-pool tail artifact, NOT the step time; step≈105ms is real.)
  Details: 15-feasibility.md.
- [2026-06-27][bench-sweep] **num_spec LEVER EXHAUSTED — KV-cache DFlash PLATEAUS ~1300 sys out tok/s,
  still ~2.89x SLOWER than target-only at the GOAL bench point ⇒ NEEDS_IMPL.** Decisive A(KV_CACHE=1,
  util 0.6)-vs-B(target-only, util 0.75) sweep at in=1/out=4096/c=32, WARM, total=96, 96/96 ok both,
  ZERO preemption/recompute/OOM on any run (KV blocks fit easily at 0.6: max-conc 341x). Sweep
  num_spec → sys out tok/s: **5→1226.57, 7→1297.14, 10→1303.34** (PLATEAU for ns≥7, DROPS below 7;
  opt=**7**). B re-measured SAME-SESSION = **3752.20 tok/s** (TPOT 0.00852, lat 34.93s) — matches prior
  3793 within 1%, gap is real. Best A (ns7) = 1297 tok/s / TPOT 0.105 / lat 82.96s ⇒ 2.89x slower
  throughput AND 2.37x worse latency (spec-wins-latency hypothesis still FALSE at c=32). MECHANISM
  (decisive): at ns=10 the draft's per-position accept for positions 8/9/10 is EXACTLY 0.000 — draft
  EFFECTIVE depth caps at ~7 (block_size B=8), so ns≥7 all degenerate to the same draft = same ~1300;
  ns<7 throws away genuinely-accepted pos 5-7. So "lower num_spec → higher throughput" theory is FALSE:
  cutting B lowers the O(C·B) attn-score matmul but lowers acceptance ~proportionally. 12-impl's
  microbench confirmed: KV cache killed the O(C) PROJECTION but the **O(C·B) ATTENTION-SCORE matmul**
  STAYS + now dominates (52/59ms @C=4608). ⇒ num_spec is DEAD as a lever; remaining lever is HARDER:
  reduce the O(ctx) draft attention-SCORE matmul itself (sparse/windowed draft attn or cheaper kernel),
  NOT a config knob. Details: 14-bench-sweep.md.
- [2026-06-27][test-kvcache] **FLAG-ON (DFLASH_KV_CACHE=1) SERVE PATH VERIFIED LOSSLESS ⇒ NEEDS_BENCH.**
  GOAL config (c=32, len4224, EP, v1, no-async, torchax draft, num_spec 7). (1) PERFECT-DRAFT THROUGH
  CONDENSE 100% per-slot: 21 metrics windows ALL mean 8.00 / per-pos 1.000×7 / Accepted==Drafted across a
  hard condense event (fire_condense 64, finish spread 254s) ⇒ the per-slot K/V cache + its condense-move
  keep verify aligned through backfill. (2) GREEDY real-draft cache-ON vs target-only: LOSSLESS — 24/64
  byte-identical, ZERO step-1/2 / factual-answer divergences, all divergences deep post-answer + SWAP-
  SYMMETRIC (same branch appears on target-only too) = batch-position bf16 near-tie present in target-only
  too (same confound 10-impl-condense accepted). (3) ACCEPTED LENGTH cache-ON ~6.3-6.9 steady-state = AT/
  ABOVE the ~5-6 without the cache ⇒ cache did NOT degrade acceptance. (HBM gotcha — both _ctx_buf + K/V
  cache allocated — RESOLVED by Lever B dropping _ctx_buf on the flag-on path.) NOTE: this lossless verify
  predates the head-shard attn-formulation change ⇒ a re-verify on the sharded path is needed (NEEDS_TEST).
  Details: 13-test-kvcache.md.
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
- [2026-06-27][bench-v3] Superseded by bench-sweep (B=3752 authoritative). Details: 11-bench-c32-v3.md.
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
- [2026-06-27][impl-perf] LEVER #2 (write fix) LANDED — 6b6acd49: per-step _ctx_buf write 234→2.2ms
  (~106x), bit-identical (jitted+donated masked-scatter _batched_ctx_write). Details: 08-impl-perf.md.
- [2026-06-26][L1-seed] Branch `dflash` has substantial prior DFlash work (STATE.md recipe). Correctness
  is DONE (see below); the open question was always SPEED at the bench point — now answered: BLOCKED (the
  draft step is FLOP-bound dense attn that no in-scope lever can cheapen at c=32/full-context).

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
- [2026-06-27][impl-window] WINDOWING the draft attn (W≤512): craters accept 6→2.5 (window-insensitive,
  intrinsic). Loses at every overhead. The draft genuinely uses its long context.
- [2026-06-27][impl-attnkernel→redteam-attn CORRECTED] SWAPPING the draft attn to a flash/ragged/paged
  KERNEL is dead (flash slower at this q=8 shape; target RPA is paged+hard-causal). BUT 17's broader
  "FLOP-bound, BLOCKED_USER" conclusion was WRONG — see redteam-attn above: the attn is REPLICATION-bound,
  head-sharding it (NOT a kernel swap) is ~60x. Do NOT pursue flash; DO head-shard + GQA-native the eager path.
- [2026-06-27][impl-headshard] HEAD-SHARDING the draft attn via WEIGHT-SHARDING ALONE or
  with_sharding_constraint (_pin) is DEAD on the torchax path — GSPMD all-gathers q back to 64 heads at the
  `q_proj(hs).view(N,B,nh,hd)` reshape (splits the model-sharded nh*hd axis) and at repeat_kv's reshape; a
  constraint just re-slices the already-gathered tensor (can even add gathers). USE jax.shard_map over the
  'model' axis (the landed fix, _sharded_attention) — it gives each chip no freedom to gather. 8.6x, bit-exact.
- [2026-06-27][bench-sweep] num_spec as a throughput lever: plateaus ~1300 (ns≥7 degenerate to same draft;
  ns<7 throws away accepted pos 5-7). Cutting B lowers accept ~proportionally to the matmul savings.
