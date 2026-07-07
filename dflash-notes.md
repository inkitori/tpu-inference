# DFlash on dflash-oss — working notes / session handoff

_Last updated: 2026-07-07 (session 1). Remove this file before any upstream merge._

**Goal**: clean DFlash spec decode (target `openai/gpt-oss-20b`, draft
`z-lab/gpt-oss-20b-DFlash`) + async scheduling on v6e-8, maximize per-user TPS at 32
concurrent, ShareGPT via `vllm bench serve`, must beat no-spec baseline clearly.
`origin/dflash` = old bloated torchax attempt (reference only — its keeper fixes are
already re-implemented here cleanly).

## Architecture implemented (all committed + pushed on `dflash-oss`)

- **`tpu_inference/models/jax/dflash.py`** — `DFlashDraftModel` (flax nnx): qwen3 arch,
  8 layers, attention **biases**, per-head q/k-norm, **YaRN RoPE via
  `GptOssRotaryEmbedding`** (params live in `hf_config.rope_parameters` — modern
  transformers dropped top-level `rope_theta`!). Per step the flat stream per request is
  `[a_i accepted ctx rows | 8 noise rows]`; ctx rows' K/V are **overridden** each layer
  with `k/v_proj(hidden_norm(fc(concat aux)))` (+k_norm+RoPE at true positions) so the
  paged-attention kernel's **end-aligned write** lands ctx K/V at `[N-1, N'-1)` and noise
  K/V at `[N'-1, N'+7)`; attention is **non-causal** (`use_causal_mask=False`). ctx-row
  outputs are garbage and discarded (cross-token flow only via overridden K/V — safe).
  `embed_tokens`/`lm_head` are zeros placeholders **shared from the target** at load
  (gpt-oss unties them: embed for noise, lm_head for logits — using embed for logits =
  0% acceptance; that was origin/dflash's big bug). `compute_logits` slices to true
  vocab (padded lm_head tail rows are zeros → argmax could pick padded id).
- **`tpu_inference/spec_decode/jax/dflash.py`** — `DFlashProposer`: fully batched,
  jitted `_prepare_inputs`/`_propose`, **eagle3-compatible signatures** so
  `SpeculativeDecodingManager` + async path are reused unchanged (vLLM's `use_eagle()`
  returns True for dflash). No per-request host state. Draft KV in the framework paged
  cache; block tables via `_draft_kv_cache_group_id()` (finds group containing
  `draft_layer.0` — draft layers **merge into the gpt-oss full-attention group**, do NOT
  assume last group). Key math in `_prepare_inputs`:
  `a = q_lens - num_rejected`, `new_seq_lens = seq_lens - num_rejected` (= N'-1),
  noise positions `new_seq_lens + b`, `kv_len = new_seq_lens + 8`, out stream size
  `T_in + 8*max_reqs` (static per bucket), draft ids from noise rows 1..7.
- **`kernels/.../kernel_hd64.py`** — added `use_causal_mask` static flag (non-causal =
  mask only `k_span < kv_len`); plumbed through
  `attention_interface.{attention,sharded_ragged_paged_attention}`. **Tested vs ref on
  TPU** (`tests/kernels/ragged_paged_attention_kernel_v3_hd64_test.py -k non_causal`).
- **Wiring**: runner dispatch (`method=="dflash"` before `use_eagle()`);
  `kv_cache_manager.get_kv_cache_spec` registers `draft_layer.{0..7}` **outside** the
  impl if/else (torchax targets take the else-branch!); `compilation_manager`
  `_precompile_dflash_helpers` chains real prepare outputs into propose;
  `vllm_model_wrapper` fires eagle3 aux hook for dflash (captures x+residual **after**
  target layers 1,6,11,16,21 — log line "auxiliary layers from config: (2,7,12,17,22)"
  confirms, +1 conversion is done by upstream `get_eagle3_aux_layers_from_config`);
  mxfp4→fp8_e4m3fn requant gated `get_tpu_version() < 7`; draft load does NOT clobber
  global quant config; offline HF cache resolution in `get_model_weights_files`.
- **`SPEC_PERFECT_DRAFT=1`** env diagnostic in `tpu_runner._sample_from_logits`:
  replaces draft ids with target argmax (verify-path sanity check).

## Current status

- **Serving works end-to-end** (greedy chat + raw completions are coherent/correct),
  no crashes, 32-way config boots. `bash scripts/serve_dflash.sh --no-async-scheduling`.
- **Verify side proven correct: `SPEC_PERFECT_DRAFT=1` gives 175/175 = 100% acceptance.**
- **OPEN BUG: real draft acceptance is only ~20%** (accept/draft-token; ~1.3-1.5
  accepted/step +1 bonus ≈ 2.3-2.5 tokens/step; positions decay 59%/37%/20%/9%...).
  Expected ~6+ tokens/step (origin/dflash measured accept ~6.3-6.9 with the SAME
  checkpoints, so the draft is capable — my draft-side computation has a numeric bug).
  Same ~20% on chat-formatted and raw prompts → not a prompt-distribution artifact.

## Next step (in progress): draft parity test

`scripts/dflash_dev/parity_test.py` — runs HF reference draft (torch CPU,
trust_remote_code; needs `datasets` pip pkg — installed; needs
`HF_MODULES_CACHE=<writable>` because HF_HOME is a read-only gcsfuse mount) vs our
`DFlashDraftModel` (TPU, 1-chip mesh with all `MESH_AXIS_NAMES` sized 1), REAL weights
both sides, identical inputs (random ctx features + injected noise embeddings via a
fake embed table), two chained steps (fresh-write then cached-ctx). Compare block
hidden outputs. Status: script debugged up to model construction; last fix appended
(`vllm_config.model_config.dtype = jnp.bfloat16` before direct construction). Run:

    HF_MODULES_CACHE=/tmp/hfmods ~/tpu-tooling/tpu-env.sh python scripts/dflash_dev/parity_test.py

(kill any server first: `pkill -9 -f "[v]llm serve"` — NOTE the [v] trick, plain
pkill matches your own shell and kills it.)

**If step1 mismatches** → static math bug (suspects: weight mapping/transposes; bias
reshape; q/k-norm; YaRN details; fc/hidden_norm; jnp.take on sharded embed).
**If only step2 mismatches** → paged cache write/read (end-alignment, positions, crop
semantics). If BOTH match → bug is in prepare_inputs stream assembly or aux capture on
the serving path (then instrument serving: dump prepare_inputs intermediates for a
1-request run and check `a`, `new_seq_lens`, positions, is_ctx layout; also verify aux
hidden values against HF gpt-oss `output_hidden_states` on a short prompt).

Verified-correct already (don't re-suspect): HF vs vLLM aux layer convention (+1, both
= output of layers 1,6,11,16,21, ascending concat order); noise[0]=bonus token at
position N'-1; ctx rows exclude bonus; rejection-trim = keep FIRST a_i rows;
prefix-cache hits are safe (end-aligned write + reused draft blocks); scheduler
allocates lookahead num_spec+1 for dflash; EAGLEConfig copies all draft attrs, arch
stays `DFlashDraftModel`.

## Serve/bench recipes

- Serve: `scripts/serve_dflash.sh [--no-async-scheduling|--async-scheduling]`
  (EP mandatory for gpt-oss mxfp4; RAGGED_GATHER_VERSION=v1 v6e; MAX_MODEL_LEN=2048
  MAX_NUM_SEQS=32 NUM_SPEC=7 env-overridable). First boot ~4-5 min.
- Baseline (no spec): same vllm serve line minus --speculative-config.
- Bench: `scripts/bench_dflash.sh <tag> <num_prompts> <concurrency>` — ShareGPT json
  expected at the session scratchpad (re-download:
  `https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json`,
  set DATASET=). Node HAS internet. Warm up once before measuring (SKIP_JAX_PRECOMPILE=1
  ⇒ cold XLA compiles on first hits).
- Acceptance: `curl -s localhost:8000/metrics | grep spec_decode` →
  num_drafts/num_draft_tokens/num_accepted_tokens (+ per-pos).
- Smoke: `scripts/dflash_dev/smoke.sh`.

## Remaining roadmap

1. Fix the acceptance bug (parity test above).
2. Baseline vs DFlash ShareGPT bench @ c=32 (`vllm bench serve`), report per-user TPS
   (≈1000/mean-TPOT) + total throughput.
3. Async scheduling: should already work (manager gates use_eagle() which passes for
   dflash; propose returns device arrays) — flip `--async-scheduling`, verify, bench.
4. Optimize: profile (examples/tpu_profiling.py); candidates: fuse ctx-KV projection
   across layers (one GEMM), skip MLP for ctx rows, distribution/bucket tuning,
   num-tokens padding buckets for the draft stream, donate/sharding audit of
   prepare_inputs, logits argmax via 2-stage sharded argmax instead of full gather.
5. Unit tests: port parity script into tests/ with tiny synthetic weights (no
   checkpoint dependency); add proposer stream-assembly unit test (pure jnp, CPU).

## Environment gotchas (rediscovery tax)

- `pkill -f "vllm serve"` kills YOUR OWN shell (pattern matches its cmdline) — use
  `pkill -9 -f "[v]llm serve"`.
- HF_HOME=/tmp/gcs/bucket is READ-ONLY; anything needing writes (HF dynamic modules)
  needs HF_MODULES_CACHE elsewhere.
- vllm serve engine dies permanently on first EngineCore exception → full restart
  (~4-5 min) per debug cycle.
- TPU "already in use": leftover EngineCore pid in the error message; kill it or
  `~/tpu-tooling/free-tpu.sh`.
- Torchax gpt-oss param names: `vllm_model.model.embedding.weight` (not embed_tokens),
  `vllm_model.lm_head.weight`; both (201088, 2880) bf16, no vocab padding.
- Draft checkpoint ships ONLY layers/fc/hidden_norm/norm — no embed, no lm_head; q/k/v
  AND o_proj have biases (o_proj bias exists too).
- KV groups for gpt-oss+dflash: [sliding 12 layers], [full 12 + draft 8 merged? — in
  practice the draft layers ended up findable via group.layer_names; always search].
- `jax.jit(static_argnums=...)` on methods: index 0 is `self` — count from there.
