# DFlash on dflash-oss — working notes / session handoff

_Last updated: 2026-07-07 (session 3). Remove this file before any upstream merge._

**Goal**: maximize per-user TPS (ShareGPT, c=32, greedy, ignore-eos) for
gpt-oss-20b + z-lab/gpt-oss-20b-DFlash on v6e-8; stretch goal 2x the no-spec
baseline. Session 3 also brought up the 120b variant (user request).

## SESSION 3 HEADLINE RESULTS (all warm, 128 prompts, c=32)

| config                        | mean TPOT | median | accept len | TPS/user | vs baseline |
|-------------------------------|-----------|--------|------------|----------|-------------|
| 20b baseline async (no spec)  | 5.28      | 5.27   | –          | 189      | 1.00x       |
| 20b DFlash session-2 state    | 6.17      | 5.84   | 2.22       | 162      | 0.86x       |
| **20b DFlash now (NS=3)**     | **4.35**  | 4.20   | 2.09       | **230**  | **1.21x**   |
| 120b baseline async (no spec) | 12.45     | 12.63  | –          | 80       | 1.00x       |
| **120b DFlash (NS=3)**        | **8.30**  | 8.01   | 2.47       | **120**  | **1.50x**   |
| 120b DFlash NS=5              | 8.52      | 8.03   | 2.79       | 117      | wash        |

Final 256-prompt validation (fully warm): 20b DFlash mean 4.45 / median 4.28,
accept 2.09, 5969 tok/s output; 120b DFlash mean 8.35-8.66 / median 7.97,
accept 2.42. Noise-row-only draft o_proj/MLP (SPEC_DFLASH_NOISE_MLP) measured
a WASH — the draft matmuls are weight-streaming-bound at these row counts;
gated off. This also bounds what split-stream drafting could ever win.

Serve: `bash scripts/serve_dflash.sh --async-scheduling` (now defaults
NUM_SPEC=3, VLLM_TPU_BUCKET_PADDING_GAP=32, TARGET_F32_LOGITS=1).
120b: `DFLASH_TARGET=openai/gpt-oss-120b DFLASH_DRAFT=z-lab/gpt-oss-120b-DFlash`.
120b draft was downloaded and uploaded to the GCS bucket
(models--z-lab--gpt-oss-120b-DFlash; block_size=10, aux layers [1,9,17,25,33]).
Bench: `DATASET=<sharegpt.json> bash scripts/bench_dflash.sh <tag> 128 32`;
run 3-4x after every restart — fine-grained buckets (gap=32) keep hitting cold
XLA compiles for ~2-3 runs (SKIP_JAX_PRECOMPILE=1). Judge medians until P99
settles < ~15ms.

## Why 2x is out of reach on THIS workload (measured, not vibes)

1. **Acceptance ceiling**: `scripts/dflash_dev/ceiling_probe.py` runs the pure
   HF reference spec loop (HF draft vs HF target, CPU) on the same ShareGPT
   prompts with forced continuation: **accept len 2.46** over 307 steps (20b,
   full block 8). Serving measures 2.20-2.22 at NS=7 → we're at ~90% of the
   intrinsic ceiling. The z-lab 4.2-5.1 numbers are chat/harmony evals; raw
   ShareGPT + ignore-eos is simply hard to draft. Verifier-numerics hunting is
   capped at +0.24 tok/step (f32 logits probe: no change; bf16-MoE probe
   session 2: no change).
2. **Verify floor**: the 128-token verify forward costs ~6.0ms (20b) vs the
   32-token baseline forward inside its 5.28ms step. gmm weight streaming
   (~1.75ms) + collectives (~0.95) + RPA (~0.86) + MoE plumbing (~1.3).
   With accept ~2.1-2.5, ratio_max ≈ accept × baseline_step / (verify +
   propose + extras) lands ≈ 1.5-1.7x. 120b confirms: bigger fwd amortizes
   spec overheads better → 1.50x with the same code.

## What session 3 changed (all committed on dflash-oss)

1. **Fused greedy verify** (`spec_decode/jax/fused_verify.py`): the old path
   all-gathered the (256, 201k) vocab-sharded logits TWICE via shard_map'd
   row-selects (~1ms each) before argmax. Now ONE dispatch does row-select +
   f32 lm_head matmul + cross-shard argmax + greedy rejection + last-sampled
   extraction (`fused_logits_greedy_verify`; logits never materialize outside
   it — `_can_defer_spec_logits` in tpu_runner skips the standalone
   compute_logits). CPU-verified bit-identical vs the old chain.
   Fallbacks: logprobs / sampling / lora / grammar / SPEC_PERFECT_DRAFT.
2. **Host-path cuts** (~15 device_puts + ~13 dispatches/step → ~4 + ~8):
   spec metadata arrays, positions, async substitution + rejection-subtract
   indices, and the drafter aux [next_prompt|is_in_prefill|num_reqs_dp] all
   ride the packed DeviceBuffer blob (one H2D/step). Subtract+substitute
   fused into `_apply_prev_step_corrections_fn`. Drafter reuses the draft
   group's on-device block tables (manager picks `draft_layer.0` metadata).
   DFlash drafting = ONE dispatch (`prepare_and_propose`, includes async
   [last_sampled|drafts] assembly). `general_device_put` batches pytrees into
   one `jax.device_put`. Sampling-metadata dummy cached across steps.
3. **NUM_SPEC=3 + bucket gap 32**: verify width 256→128. Positions 3-6 accept
   at <8%/3%/2%/1% (20b) and don't pay for their verify width. Sweep:
   NS=3 4.35 mean / NS=4 4.62 / NS=7 5.02. 120b: NS=3 8.30 vs NS=5 8.52
   (accept 2.47→2.79 doesn't cover +64 verify tokens).
4. **TARGET_F32_LOGITS=1**: jax-native f32 lm_head matmul for the verifier
   (bypasses torchax LogitsProcessor call), draft compute_logits f32 to
   match. No acceptance change; slightly faster + fewer dispatches.
5. **120b bring-up**: worked first try via config-driven paths (aux layers,
   block size, requant gate all generic). Accept len 2.47 at NS=3 —
   noticeably better drafter than 20b's (67/47/33% per-position vs 58/30/15).

## Step anatomy after all this (20b NS=3, from profile)

verify fwd 6.05ms (op-sum 5.17: gmm 1.75, psum+AR 0.95, RPA 0.86, gathers
0.32, copies 0.31, select_reduce 0.21, sort 0.16, ...) + fused verify 0.09 +
prepare_and_propose 1.81 (AR 0.55, RPA 0.43, copies 0.26) + corrections/split
~0.1 + host gaps ~0.8-1.5ms. Step ≈ 9.1ms, accept 2.09 → TPOT ~4.35.

## Remaining ideas (diminishing returns, ~0.3-0.5ms each)

1. Draft "option d": keep the legacy combined stream through attention
   (write semantics proven) but gather noise rows before o_proj+MLP each
   layer (ctx rows' hidden states are never consumed — their K/V come from
   combined_ctx). Saves ~1/3 of draft o/mlp matmul + AR payload.
2. Verify epilogue: the ~1.3ms of MoE plumbing (gather_fusion/copies/sort)
   inside step_fun — RAGGED_GATHER v1 path; needs a kernel-level dig.
3. Host: ~8 dispatches/step remain (~0.3ms host each): fold the blob
   jnp.split into the corrections jit; move _modify_prev_results loops off
   the critical path.
4. **SPLIT-STREAM DRAFTING (gated off, `SPEC_DFLASH_SPLIT=1` to enable): DO
   NOT re-enable without fixing** — writes ctx K/V via a slim dummy-q kernel
   pass then runs the transformer on noise rows only. Passes 1-chip parity
   bit-exact and 8-chip cos>0.9999 (`scripts/dflash_dev/split_parity.py`) but
   under real multi-request serving acceptance collapses 2.03→1.26 (pos-0
   58%→21%) and a multi-request repro hangs the RPA v3 kernel. Suspect:
   ragged non-causal calls with many tiny (0-4 row) q segments. The
   CPU prepare-impl A/B (legacy vs split layouts) passes bit-exact.

## Session-2 knowledge that still applies

- Acceptance debugging chain (parity_test.py, replay/dump tooling,
  SPEC_DFLASH_DUMP env) — all still works; dump forces the legacy
  (non-fused) drafting path automatically.
- Env gotchas: `pkill -9 -f "[v]llm serve"` (bracket trick); engine dies
  permanently on first EngineCore exception → full restart; HF_HOME on GCS
  is read-only (HF_MODULES_CACHE elsewhere); Mxfp4Config(dequantize=True)
  for HF CPU loads; disk ~48G — no dequantized checkpoints on disk.
- fp8-requant target numerics cost ~11% argmax flips vs HF but bf16-MoE
  doesn't fix acceptance (session 2) — consistent with the ceiling finding.

## Bench hygiene for final numbers

For publishable numbers serve with SKIP_JAX_PRECOMPILE=0 (precompile all
buckets; slower boot) or run 4+ warm benches. The bench JSONs live in the
session scratchpad `bench/` dir (mean/median/p99 TPOT + per-position accept).
