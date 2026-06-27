# 01 — RESEARCH / CURRENT-STATE ASSESSMENT (DFlash gpt-oss-20b, branch `dflash`)

Phase: read-only investigation. Date: 2026-06-26.

## Verdict
Code on `dflash` looks **complete and coherent** for the torchax spec-decode path.
A prior smoke test already produced **correct output** with ~2.89 mean accept length
(~27% avg draft accept). No WIP/uncommitted source. Next step = run the perfect-draft
machinery test (GOAL ladder #1) before any real bench. STATUS → NEEDS_TEST.

## Exact known-good launch command (from repo-root STATE.md, mirrored in user memory)
```bash
env HF_HOME=/home/enyouki/local_hf DRAFT_MODEL_IMPL_TYPE=torchax \
  RAGGED_GATHER_VERSION=v1 RAGGED_GATHER_REDUCE_VERSION=v1 \
  ~/tpu-tooling/tpu-env.sh vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 8 --enable-expert-parallel --no-async-scheduling \
  --max-model-len 2048 --max-num-seqs 1 \
  --speculative-config '{"model": "z-lab/gpt-oss-20b-DFlash", "num_speculative_tokens": 7, "method": "dflash"}'
```
- Target `openai/gpt-oss-20b` (impl = vllm, set implicitly by `tpu-env.sh` default `MODEL_IMPL_TYPE=vllm`).
- **Draft impl = `DRAFT_MODEL_IMPL_TYPE=torchax`** (NOT vllm — corrects the phase brief; this is the
  draft impl type, and it selects the torchax proposer `spec_decode/vllm/dflash.py`). Both STATE.md
  and memory agree on `torchax`.
- Spec: `num_speculative_tokens=7`, `method=dflash`. Draft `block_size=8` (= num_spec+1).
- EP mandatory; `--no-async-scheduling` + `--max-num-seqs 1` (DFlash asserts sync + dp_size==1 + single seq).

## What the 3 "PRs" actually are
- **#1868** (real upstream, commit `ee7aa8c9`): adds the DFlash drafter + **proposer** foundation
  (torchax `DFlashTorchaxProposer`, JAX-native `DFlashProposer`, models, dflash attention iface, tests).
  No runner integration.
- **"#1869"** = local commit `a03d42e4` "wire DFlash into the torchax pipeline (concurrency 1)":
  integration — runner dispatch, gpt-oss aux-hidden capture + embedding share.
- **"#1870"** = local follow-up fix cluster: `5f4047cd` (lm_head projection → 0%-accept fix),
  `8486757a` (MXFP4→fp8_e4m3fn requant, v6e-only workaround), `3b051a24` (skip phantom draft KV groups),
  `924e6c87`/`abb70c19` (draft mask additive-bias fix + docstring).
- NOTE: torchax DFlash files were later moved under `vllm/` (commit `867e5a06`): paths are now
  `spec_decode/vllm/dflash.py` and `models/vllm/dflash.py`.
- **Verification/accept-reject is NOT DFlash-specific** — DFlash reuses the shared spec-decode
  rejection sampler. Only the draft *dispatch* is new.

## Wiring map (torchax path, method="dflash")
- LOAD: `spec_decode/vllm/dflash.py:96` `DFlashTorchaxProposer.load_model` →
  `models/vllm/dflash.py:127` `DFlashTorchaxWrapper.load` (AutoModel, trust_remote_code,
  attn_implementation="eager"). **gpt-oss lm_head fix at `models/vllm/dflash.py:184-190`** —
  distinct lm_head weight for tie_word_embeddings=False (commit 5f4047cd).
- PROPOSE: `runner/speculative_decoding_manager.py:240` `propose_dflash_draft_token_ids` →
  `spec_decode/vllm/dflash.py:328` `.propose` → `_sample_block_draft_tokens` (:208, argmax).
  **Greedy proposer: returns draft token IDs only, NO draft probs.**
- VERIFY: `runner/tpu_runner.py:1606` `_sample_from_logits` (spec branch :1645-1673) →
  `layers/jax/sample/rejection_sampler.py` `rejection_sample_greedy` (:~516; argmax :542; mismatch :553,
  accept = draft == target_argmax).

## INJECTION POINT for perfect-draft test (GOAL ladder #1)
**File:** `tpu_inference/runner/tpu_runner.py`, fn `TPUModelRunner._sample_from_logits`, ~lines 1659-1665.
```python
target_logits   = self._select_from_array_fn(logits, spec_decode_metadata.target_logits_indices)  # :1659 [N_flat, vocab]
draft_token_ids = self._extract_draft_token_ids(...)                                               # :1662 [N_flat] int32
# PERFECT-DRAFT INJECT (test only):
# draft_token_ids = jnp.argmax(target_logits, axis=-1)
next_tokens = self.rejection_sampler(draft_token_ids=draft_token_ids, target_logits=target_logits, ...) # :1665
```
- Setting `draft_token_ids = jnp.argmax(target_logits, axis=-1)` exactly matches
  `target_logits_argmax` (rejection_sampler.py:542) → `mismatches` all-False (line 553) → **100% accept**.
- Shapes/sharding: both 1-D-flat over `num_draft_tokens_flat`, `PartitionSpec(ATTN_DATA)`.
- This site tests the real `_extract_draft_token_ids` plumbing. Deeper fallback: set
  `draft_token_ids = target_logits_argmax` right after rejection_sampler.py:542 (bypasses extraction).
- MUST run greedy (temperature 0 / do_sampling False) and batch size 1.

## Risks / gotchas
- **MXFP4→fp8_e4m3fn requant** (`layers/vllm/quantization/mxfp4.py`) is a v6e-only workaround,
  not general (no native FP4 Mosaic lowering on v6e). Not for merge as-is.
- **v1 ragged gather** mandatory on v6e (v2 SparseCore gather fails to lower at decode).
- gcsfuse mount root-only → must stage to local HF cache `HF_HOME=/home/enyouki/local_hf`.
- Draft HF remote code imports `datasets` — must be installed in the venv.
- **CUDA/untraceable-op sweep of the spec path: CLEAN.** No `.cuda`, triton, multinomial, nonzero,
  bincount, custom ops. `.tolist()`/`device_get` calls are host-side orchestration (outside traces),
  by design. Rejection sampler is fully `@jax.jit` fixed-shape. Caveat: the draft's external HF
  modeling code (trust_remote_code) is out of repo, but forced `attn_implementation="eager"` keeps it
  traceable.
- Greedy-only: stochastic rejection path exists but DFlash passes draft_probs=None, so only greedy
  `draft==target_argmax` is exercised — fine for the lossless requirement.

## Open questions for next phase (NOT yet verified)
- Is it CURRENTLY lossless vs target-only at greedy? (perfect-draft test answers the machinery half.)
- Is it CURRENTLY faster than target-only at input=1/output=4096/concurrency=32 warm cache? (bench phase.)
