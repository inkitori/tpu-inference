# SHARED MEMORY (read/write — HARD CAP 250 LINES)

> Rules for every agent manager:
> - Keep this file UNDER 250 lines. If you approach the cap, PRUNE stale/obsolete lines.
> - Only critical cross-manager facts go here. Per-phase detail goes in YOUR own file.
> - Entry format: `[YYYY-MM-DD][manager-id] fact`. Add new facts under the right section.
> - This is the one place a fresh manager learns what prior managers discovered. Keep it
>   true and lean.

## Current best-known status
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

## Known blockers / gotchas
- [2026-06-26][research] v6e-only workarounds in place (not general): MXFP4→fp8_e4m3fn requant
  (layers/vllm/quantization/mxfp4.py); v1 ragged gather mandatory. EP mandatory (plain TP8 dies
  IndivisibleError). gcsfuse root-only → stage to local HF cache HF_HOME=/home/enyouki/local_hf.
  Draft HF remote code needs `datasets` installed in venv.

## Decisions (with rationale)
- (none yet)

## Dead ends — do NOT repeat
- (none yet)
