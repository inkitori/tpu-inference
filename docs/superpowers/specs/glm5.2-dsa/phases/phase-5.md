# Phase 5 — fp8 + perf structure

The fp8 path and the performance structure, validated with **no real weights** (real shapes, random
weights), so Phase R only reconciles, never first-contacts. Reuses the **existing** repo fp8 machinery —
stop routing GLM around the loader (core §F8). Gated behind Phases 3–4 and its own correctness ladder.

**Precondition:** 4 (continuous batching + the decode loop) + 2 (the DSA gather is the rewrite target).
**Core anchors:** §A3, §B7, §B8, §E3, §F8, §G8, §H4, §H6, §H11, §H13, §I7, §J2.

## Deliverables
- **fp8 structure.** Reuse `layers/jax/quantization/fp8.py`, `get_tpu_quantization_config`
  (`layers/jax/quantization/__init__.py:24`), `MLAEinsum.load_weights`, block dequant + the fork's
  `deepseek_v4_loader.py` recipe. Quantize **random** weights to e4m3 + 128-block UE8M0; validate the fp8
  indexer (fp32-pinned scoring/head-sum, downcast after top-k — core §E3), fp8 MLA, and **fp8 KV cache** vs
  the bf16 jnp-ref within a measured fp8 floor; 8-chip N-dev==1-dev. The DSA indexer fp8 path has **zero
  repo implementation** — port it (Triton ref `triton_fp8_mqa_logits.py:128-129,149-150`).
- **(FIX) Wire UE8M0 decode into the fp8 dequant.** ⚠️ The fp8 path **first-contacts UE8M0 at Phase R unless
  fixed here.** Today's dequant does `tensor_q.astype(float32) * scale` (a plain-float multiply,
  `layers/common/quantization/__init__.py:161`; `MLAEinsum.load_weights` hand-rolls the same). The on-disk
  `weight_scale_inv` is UE8M0 **exponents** (core §B7), so a float multiply is wrong by orders of magnitude.
  The decode helper **already exists** (`e8m0_to_fp32`, `__init__.py:63`) but is wired **only** into the
  MXFP4 path. **Wire it (or port vLLM's `_upcast_e8m0_to_fp32`) into the fp8 blockwise/MLA scale path** + add
  a `float8_e8m0fnu`/uint8-scale branch in `dequantize_tensor` and `MLAEinsum.load_weights`.
- **(FIX) On-disk UE8M0 fp8 fixture.** Random *in-process* quantization produces float32 scales and
  round-trips self-consistently **by construction** — it can never catch the UE8M0 bug. Write a tiny
  medium-config safetensors with `weight_scale_inv` stored as `uint8`/`float8_e8m0fnu` (real key names, real
  `[ceil(out/128),ceil(in/128)]` block layout) and load it through the **actual disk path**, asserting the
  UE8M0 branch fires and no scale is read as a float (the bitcast-not-astype rule, core §B7).
- **Perf structure (real shapes ≠ real weights).** The DSA gather **DMA-scalar-prefetch rewrite**
  (O(`index_topk`), not O(N) — core §G8): the **gen-6 idiom** — `mla/v2/kernel.py:2454`
  `PrefetchScalarGridSpec`/`num_scalar_prefetch`, `:661` `make_async_copy`; `ragged_paged_attention/v3` —
  SMEM-prefetch the per-request-local index array, kv as an HBM operand, per-row `make_async_copy` in a
  statically-unrolled loop, size-0 copy for the `-1` sentinel; softmax math unchanged, only kv-gather
  construction changes. **GLM-shaped `tuned_params` keys** (`actual_num_q_heads=64`,
  `actual_lkv_dim=512`, `actual_r_dim=64`, real `page_size`/dtype — the inherited 128-head/fp8 table matches
  **none** of GLM's shapes and silently falls to defaults) + a bounded sweep at real GLM dims; a dedicated
  `sparse_attn` tuner (subclass `KernelTunerBase`). Guardrails: full correctness suite green, tuned-vs-baseline
  ≤5% regression (modeled on `mla_tuned_vs_baseline_test.py:131-135`), frozen kernel commit, human-reviewed
  apply-back.
- **(FIX) KV-capacity gate.** Explicit per-token-byte budget (MLA 576/tok·L; indexer 128/tok·**full**-L, 21
  of 78, core §B8) **+ the RoPE sin/cos cache bytes** (`[max_position_embeddings, head_dim]` ≈ 537 MB at 1M —
  core §B2) + a **hard startup gate** bounding `max_model_len × max_num_seqs` against profiled HBM (the repo
  has none — OOM surfaces at allocation). State the bf16-KV `max_model_len` ceiling; fp8 KV is the
  long-context lever.

## Acceptance gates (numeric)
- fp8 indexer/MLA/KV within the measured fp8 floor vs bf16 jnp-ref; 8-chip N-dev==1-dev.
- Gather decode latency scales with `index_topk` not N (≥2× at N=8×topk vs the one-hot bridge).
- GLM-shaped tuned-vs-baseline ≤5% regression green.
- KV-capacity startup gate enforced (incl. RoPE-cache bytes); full correctness suite still green.
- **(FIX)** the on-disk UE8M0 fixture loads with the UE8M0 branch firing (no scale read as float).
- All at **real shapes, random weights**.

## Phase-specific risks & fixtures
- **fp8-on-disk blocker** — the load path built here is the Phase-R prerequisite (core §B7); the UE8M0 fix
  is what makes "Phase R reconciles, not first-contacts" actually true for fp8.
- **No KV-capacity budget / startup gate** — OOM at allocation; the gate above closes it.
- **CPU losslessness ≠ v6e losslessness** — on-device cast-equivalence for fp8 (core §H11c).
- **Tuned-param reuse is a no-op for GLM v1** (64 heads + bf16 never match the 128-head/fp8 keys).
