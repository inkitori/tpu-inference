# SHARED MEMORY (read/write — HARD CAP 250 LINES)

> Rules for every agent manager:
> - Keep this file UNDER 250 lines. If you approach the cap, PRUNE stale/obsolete lines.
> - Only critical cross-manager facts go here. Per-phase detail goes in YOUR own file.
> - Entry format: `[YYYY-MM-DD][manager-id] fact`. Add new facts under the right section.
> - This is the one place a fresh manager learns what prior managers discovered. Keep it
>   true and lean.

## Current best-known status
- [2026-06-27][bench-v2] **GATE PASS, but BENCH BLOCKED: a DETERMINISTIC CRASH at the out=4096/c=32
  bench point.** Perfect-draft gate at c=32/SHORT outputs = 100% per-slot (mean 8.00, per-pos 1.000×7,
  5 windows) ⇒ 6b6acd49 did NOT break verify. BUT serving DFlash at out=4096/c=32 CRASHES ~69s in
  (seqs ~1500-3300 tok, kv 2.3% — NOT OOM): `ValueError: could not broadcast (3184,) into (176,)` at
  **dflash.py:421** in prepare_inputs. ROOT CAUSE = a PRE-EXISTING batched bug EXPOSED (not caused) by
  6b6acd49: `_ctx_len[i]` is keyed by SLOT, `num_new=seq_len-ctx_len_i` uses the FULL accepted length
  (num_tokens_no_spec); when a finished slot is backfilled by InputBatch.condense with a still-long
  request, the slot-change guard resets _ctx_len[i]=0 ⇒ num_new=whole-seq (3184) but raw_hidden's query
  segment has only ~176 rows ⇒ overrun. OLD eager path SILENTLY CLAMPED (wrote wrong rows = silent
  draft-context corruption, no crash); the new host index-plan turns it into a hard crash. ⇒ The 07
  "358 tok/s / 10.6x" A-number was measured on a SILENTLY-CORRUPT run; treat it as unreliable. B
  (target-only 3789 tok/s) is still honest. FIX (candidate b): drive copy width from the segment width
  `num_new = qsl[i+1]-qsl[i]` (not seq_len-ctx_len), keep end=ctx_len+num_new. CAVEAT: that stops the
  crash but a condensed-in request LOSES its pre-move _ctx_buf history ⇒ acceptance degrades for it
  (NOT a losslessness break) — a fuller fix carries _ctx_buf across condense or keys state by req_id.
  ⇒ **NEEDS_IMPL** (fix the slot/condense desync, then re-GATE at out≥512 to trigger condense, then
  re-bench A vs B). Details: 09-bench-c32-v2.md.
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
- [2026-06-27][bench-c32] **SPEED VERDICT: DFlash is ~10.6x SLOWER than target-only at the
  GOAL bench point (in=1, out=4096, c=32, WARM cache).** A(DFlash)=358.65 sys out tok/s,
  TPOT 0.317s; B(target-only)=3788.53 sys out tok/s, TPOT 0.0084s. Per-step is ~37x slower.
  Acceptance is HEALTHY (mean ~5.5/step at full c=32, per-pos 0.85→0.52) ⇒ this is a PERF
  bug, not accept/correctness. ⇒ Phase-1 SPEED requirement FAILS as-is → NEEDS_IMPL (perf).
  Bottlenecks (07-bench-c32.md): (1) dflash.py:~324 eager non-donated _ctx_buf
  dynamic_update_slice inside a 32x per-req loop (full ~4GiB copy/step); (2) STATELESS draft
  recomputes fc+8 attn over the FULL context every step (O(ctx), grows with out len, no KV
  cache); (3) per-step device_get host syncs. Both servers gave byte-identical greedy probe
  output (same target). HBM is NOT the gap (A 39.7 / B 24.1 GiB total).
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
