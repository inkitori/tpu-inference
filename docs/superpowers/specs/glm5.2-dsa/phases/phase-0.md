# Phase 0 — env, HF oracle, harness

**Status:** **DONE** (committed `5d31ba05`: `tests/models/jax/glm_moe_dsa_harness.py`,
`tests/models/jax/test_glm_moe_dsa.py`, 26 gates green on the v6e-8 — tiny config, `build_hf_oracle` eager
forward, `t2j_weights`, `maxabs`, `weight_checksum`/`assert_identical_weights`, `make_glm_mesh` 6-axis,
1/multi-device round-trips). The two items marked **(ADD)** below remain.
**Precondition:** none (foundational).
**Core anchors:** §B1, §B2, §B3, §B10 (tiny), §B11 (medium), §F7 (converter), §H1 (oracle), §J1, §J3 (env).

## Deliverables
- **py3.12 venv** with the pinned stack (core §J3); verify `jax.devices()` returns **8** chips in one process.
- **HF-eager oracle fixture** (torch CPU, random weights, no download) per core §H1 — `_from_config` +
  `attn_implementation="eager"` + `experts_implementation="eager"` mandatory; deterministic init **including
  buffers** (`e_score_correction_bias`).
- **(ADD) Incremental HF decode oracle.** The committed `build_hf_oracle` runs a single eager `forward`
  only. Add an HF **decode** reference — `use_cache=True` / `past_key_values` / `cache_position`, stepped one
  token at a time — so the harness can express **prefill-vs-decode equivalence** (Phase 1c) and **greedy
  `generate()`** parity (Phases 1a/2). Without it, no decode gate is expressible (the precise hole that let
  V4 ship a prefill-only validation).
- **Tiny config** (core §B10) and **(ADD) medium config** (core §B11) as harness fixtures.
- **Harness helpers** (core §F7): `convert_hf_weights`/`t2j_weights` (build on the existing `utils.py:78`
  `t2j`, do not shadow it), `maxabs` (upcast fp32 inside), identical-weights checksum asserted before every
  parity run.
- **Mesh fixtures:** both a 1-device fixture and a multi-device (up to 8-chip) fixture — do **not** inherit
  the single-device assert in `tests/models/jax/conftest.py`. Use construction tests for the DeepSeek
  regression gate; do **not** depend on the broken-import `test_deepseek_v3.py`.

## Acceptance gates (numeric)
- `jax.devices()` returns **8** chips on the v6e-8.
- HF-eager oracle runs a finite forward on the tiny config; oracle prints `transformers 5.12.1`.
- A trivial JAX op round-trips TPU → host numpy.
- **(ADD)** the incremental HF decode oracle runs: per-step logits == a single full forward at the same
  length, fp32.
- **(ADD)** the medium config instantiates and runs an HF-eager forward on CPU.

## Phase-specific risks & fixtures
- **Env friction** — py3.10 can't install pinned `jax==0.10.1`; mitigated by the py3.12 venv (core §J3).
- **RoPE-freqs fixtures** (land here / Phase 1): freqs-table-vs-bucket reshape; `freqs_cis` float64-vs-torch.
