# DFlash on dflash-oss — working notes / session handoff

_Last updated: 2026-07-07 (session 2). Remove this file before any upstream merge._

**Goal**: clean DFlash spec decode (target `openai/gpt-oss-20b`, draft
`z-lab/gpt-oss-20b-DFlash`) + async scheduling on v6e-8, maximize per-user TPS at 32
concurrent, ShareGPT via `vllm bench serve`, must beat no-spec baseline clearly.

## SESSION 2 HEADLINE: the "acceptance bug" is RESOLVED — there is no bug.

Chain of evidence (all scripts in `scripts/dflash_dev/`, all committed):

1. **Parity test** (`parity_test.py`, now PARITY_P env for ctx length): our
   `DFlashDraftModel` matches the HF reference at cos 0.9997+ for BOTH a
   short (P=13) and long (P=300, multi-KV-block/multi-page) context, fresh and
   cached steps. Draft model + weights + YaRN RoPE + non-causal hd64 kernel +
   paged end-aligned cache: all correct.
2. **Serving dump + HF replay** (`SPEC_DFLASH_DUMP=<dir>` env on the server →
   `replay_dump.py`): feeding the DUMPED serving aux features through the HF
   reference draft gives 1.15 accepted/step vs our served 1.05 — identical.
   Shared embed/lm_head are bit-exact vs the checkpoint; fc+hidden_norm
   (combine) matches at cos 1.0. Stream assembly (positions/is_ctx/noise
   block/lti) verified correct in the dumps. Draft side fully exonerated.
3. **Aux capture check** (`aux_check.py`; NOTE: HF CPU load needs
   `Mxfp4Config(dequantize=True)` — `quantization_config=None` silently gives
   RANDOM MoE weights): captured aux ≈ HF ground truth at cos 0.998-0.9995
   (right layers, right rows). Serving target greedy == true-HF argmax 88.8%
   of tokens (fp8-MoE numeric noise flips ~11%).
4. **Truth-feature replay** (`truth_replay.py`): swapping ground-truth
   features for TPU features improves replay only 1.15 → 1.23 accepted/step.
   Feature noise is NOT the bottleneck; the serving VERIFIER's argmax flips
   are what truncate acceptance chains (compounded per position).
5. **Reference e2e** (`reference_e2e.py`, z-lab spec_generate loop, pure HF
   CPU): on our exact 18-token raw prompt the checkpoint's TRUE capability is
   **1.84 accepted/step (2.84 incl bonus)** — NOT ~6.
6. **The "expected 6+" was a benchmark artifact**: origin/dflash's 6.3-6.9 was
   measured at in=1/out=4096 (degenerate free-running repetitive text). Its
   own real-prompt smoke was 2.35-3.14 accept len ≈ what we see. z-lab README
   claims accept len 4.2-5.1 on CHAT-formatted evals (Math500/GSM8K/
   HumanEval/MT-Bench, SGLang H200 bf16, medium reasoning).

Measured acceptance on dflash-oss serving (fp8-requant target, greedy):
raw 1-req prompt 1.05 acc/step (pos-0 59%); chat math request 1.24; 4-req
concurrent 1.05; ShareGPT c=32 `vllm bench` workload **accept len 1.58,
rate 8.2%, pos-0 34.7%** (ignore-eos forces post-EOS OOD text + raw
non-harmony ShareGPT turns → intrinsically hard to draft).

## FINAL BENCH RESULTS (ShareGPT c=32, greedy, warm server, 128 prompts;
## JSONs in bench_results/)

| config                  | mean TPOT | median | output tok/s | accept len |
|-------------------------|-----------|--------|--------------|------------|
| baseline sync           | 8.47 ms   | 8.44   | 2538         | –          |
| **baseline async**      | **5.35**  | 5.31   | **3783**     | –          |
| dflash sync fp8         | 17.0*     | 9.52   | 1877         | 2.21       |
| **dflash async fp8**    | **6.24**  | 5.91   | **3194**     | 2.22       |
| dflash async bf16-MoE   | 6.89      | 6.75   | 2894         | 2.20       |

*sync dflash mean polluted by residual compiles (P99 277ms); median is
representative. Per-user TPS ≈ 1000/TPOT: async baseline ≈ 187, async dflash
≈ 160.

**VERDICT: DFlash async beats the sync baseline by ~26% but TRAILS the async
baseline by ~17%.** At accept len 2.22 the spec step (≈13.1ms = 5.91×2.22)
costs ~2.5x a plain decode step (5.31ms); the acceptance doesn't cover it.
Even at the reference ceiling (~2.9 tokens/step) it would win by only ~10%.
Async scheduling itself works perfectly for dflash (token-identical output,
same acceptance, 17.0→6.24ms mean TPOT — the sync spec path was host-bound).

## Acceptance-noise attribution (closed)

- bf16-MoE target (`MXFP4_REQUANT_DTYPE=bfloat16 MOE_NO_LHS_QUANT=1`, both
  env gates committed): acceptance UNCHANGED (2.20 vs 2.22; smoke prompt
  1.00 vs 1.05 acc/step) and ~10% slower. Target MoE quantization noise is
  NOT what limits acceptance. The residual serving-vs-reference gap
  (1.05 vs 1.84 acc/step on the probe prompt) lives elsewhere (attention
  kernel / router bf16 numerics — unattributed, low ROI).
- Scale-less bf16 gmm path produces GARBAGE output on v6e (unquantized rhs
  through gmm_v2 broken there) — do not use; f16 rhs fails Mosaic lowering
  ("Invalid vector type for load"). The working combo is bf16-with-block-
  scales + normalize-to-1.0 patch in quantize_tensor (wide-float targets
  only) + MOE_NO_LHS_QUANT=1.

## What it would take for DFlash to win at c=32 (future work)

1. Cut the spec-step overhead (~7.8ms over baseline step): profile; fuse
   ctx-KV projection across draft layers, skip MLP for ctx rows, 2-stage
   sharded argmax for draft logits, prepare_inputs donation/sharding audit.
2. A stronger drafter for this traffic (accept 2.2 is intrinsic on
   ShareGPT/ignore-eos; z-lab's 4.2-5.1 numbers are chat/harmony evals).
3. NUM_SPEC<7 is nearly free to try but projected marginal (draft fwd cost is
   block_size-bound, only verify width shrinks; pos rates 60/32/16/7/3/2/1%).

## Architecture implemented (unchanged from session 1, all pushed)

- `tpu_inference/models/jax/dflash.py` — DFlashDraftModel (flax nnx): qwen3
  arch, biases, q/k-norm, YaRN via GptOssRotaryEmbedding (params in
  hf_config.rope_parameters), ctx-row K/V override each layer, non-causal
  paged attention with end-aligned write, embed/lm_head shared from target
  (gpt-oss unties them), compute_logits slices to true vocab.
- `tpu_inference/spec_decode/jax/dflash.py` — DFlashProposer: batched, jitted
  prepare/propose, eagle3-compatible signatures, draft KV in framework paged
  cache (group found via draft_layer.0), `a = q_lens - num_rejected`,
  `new_seq_lens = seq_lens - num_rejected`, noise at new_seq_lens + b.
- kernels hd64 `use_causal_mask` flag; runner dispatch; kv spec registration;
  precompile helpers; aux hook via set_eagle3_aux_hidden_state_layers (+1
  convention verified BY VALUE vs HF, aux = x+residual after layers
  1,6,11,16,21); v6e requant gate; SPEC_PERFECT_DRAFT=1 diagnostic (verify
  path proven 175/175).
- NEW session 2: `SPEC_DFLASH_DUMP=<dir>` + `SPEC_DFLASH_DUMP_STEPS` env on
  the server dumps per-step drafter inputs/outputs npz
  (runner/utils.py:dump_dflash_step, hooked in speculative_decoding_manager).

## Serve/bench recipes

- Serve: `bash scripts/serve_dflash.sh [--no-async-scheduling|--async-scheduling]`
  (EP mandatory; RAGGED_GATHER_VERSION=v1 on v6e; MAX_MODEL_LEN=2048
  MAX_NUM_SEQS=32 NUM_SPEC=7 env-overridable). Boot ~4-5 min.
- Baseline: same line minus --speculative-config.
- Bench: `DATASET=<sharegpt.json> bash scripts/bench_dflash.sh <tag> <n> <c>`;
  ShareGPT at
  `/tmp/claude-2001/-home-enyouki-tpu-inference/df1a8d85-8c04-4be3-b316-1fb928b7c908/scratchpad/sharegpt.json`
  (re-download URL in scripts; node has internet). WARM UP first (cold XLA).
- Acceptance: `curl -s localhost:8000/metrics | grep spec_decode` or the bench
  "Speculative Decoding" block.
- Offline forensic loop: serve with SPEC_DFLASH_DUMP → replay_dump.py /
  truth_replay.py / aux_check.py / reference_e2e.py (all CPU, HF_HOME=gcs
  mount, HF_MODULES_CACHE=/tmp/hfmods, OMP_NUM_THREADS=32+).

## Session-2 status: benches DONE (see FINAL BENCH RESULTS above), async
## scheduling verified, acceptance question closed. Remaining work is the
## optimization list under "What it would take for DFlash to win".

## Environment gotchas (rediscovery tax)

- `pkill -f "vllm serve"` kills YOUR OWN shell — use `pkill -9 -f "[v]llm serve"`.
- HF_HOME=/tmp/gcs/bucket is READ-ONLY → HF_MODULES_CACHE elsewhere for
  trust_remote_code; HF CPU gpt-oss load MUST use Mxfp4Config(dequantize=True).
- vllm serve engine dies permanently on first EngineCore exception → full
  restart (~4-5 min).
- TPU "already in use": `~/tpu-tooling/free-tpu.sh`.
- Disk on / is ~48G free — do NOT save a dequantized 40GB checkpoint; use the
  in-process MXFP4_DEQUANT_BF16 path instead.
- KV cache capacity at MAX_MODEL_LEN=2048/MAX_NUM_SEQS=32: watch
  num_requests_waiting{reason=capacity} — effective concurrency can be <32.
