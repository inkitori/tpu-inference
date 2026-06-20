# Phase 2 — DSA (lightning indexer + sparse selection)

The DSA indexer + top-k selection (GLM/V3.2 form, core §A3/§E) **and** the adapted sparse-attention kernel,
built and validated on TPU, single- and multi-device. The §A5 dense==sparse equivalence is the bridge from
the 1a/1c dense backbone into the sparse path.

**Precondition:** 1a (dense backbone + jnp-ref MLA), 1b (S1 fix on the mesh), 1c (decode spine — DSA decode
builds on the dense decode path).
**Core anchors:** §A1, §A3, §A5, §B5, §B6, §B8, §C1 (k_norm row), §D1–§D3, §D5, §E1–§E6, §F3, §G2, §G4,
§G5, §G7, §G8, §H4, §H5, §H7, §H10, §H11, §I6, §I7, §I8.

## Deliverables
- **`GlmMoeDsaIndexer`** (core §A3, §E1, §E3, §E5; §D2 rotate-half): `wq_b` from the shared q-LoRA residual,
  `wk` + `k_norm` (LayerNorm), `weights_proj`, rope-first split, rotate-half RoPE, `relu` on scores, weighted
  head-sum, **a causal additive mask added to the fp32 `index_scores` BEFORE `top_k`** (strict `>` cutoff,
  core §E5), `top_k=min(index_topk, T)`. Support `indexer_types=="shared"` (reuse previous top-k). Keep a
  pure-`jnp` indexer + `-inf`-mask sparse path as the fp32 oracle.
- **Port + adapt the fork's already-wired Mosaic `sparse_attn` kernel** (core §G7, §G8): drop `attn_sink`,
  drop the compressor (direct `k_norm(wk(x))`), keep the one-hot gather + `-1` sentinel; wire it to the new
  indexer-KV cache + `topk_indices` (core §I7, §I8) and give it the paged surface.
- **(FIX) DSA decode state — built and validated here, not deferred.** Today's code *discards* the sparse
  path (`kv_cache_manager.py:613-620`; `mla_attention.py:318-320`; `deepseek_v4_attention.py:99-108`). Build
  the indexer-KV cache (one row appended per decode step) + `topk_indices` in `AttentionMetadata`, and
  validate **per-step append + per-step top-k re-selection** against the incremental HF decode oracle, plus
  **prefill↔decode equivalence with DSA on** (seq ≤ topk where sparse==dense, then seq > topk). Cover the
  `-1`-sentinel topk-buffer wrap boundary vs recompute-from-scratch.
- **(FIX) Top-k / sentinel boundary fixtures:** seq exactly == `index_topk`, topk±1, seq==1, seq==0 (K-clamp
  / empty-tensor / dense==sparse all hold); an **all-`-1`-selection row** (first decode token / empty
  selection) must yield **finite (non-NaN) softmax** (reset `m_max` to finite; modeled on the fork's
  `test_sparse_attn_all_invalid`); a **post-topk causal re-mask**
  so a `top_k` tie on `-inf` cannot return a future slot (the pre-topk mask alone was insufficient in the
  fork); `index_topk` not a multiple of 128 with M>1 in the kernel BlockSpec (tiny's `index_topk=64`).
- **(FIX) Synthetic well-separated indexer-weights selection gate.** Random `normal*0.02` weights produce
  near-tied indexer scores → top-k set-equality passes vacuously (core §H11a), so DSA's core selection
  function is otherwise unvalidated until real weights. Hand-construct `wq_b`/`wk`/`weights_proj` so a
  **designated token set wins by a wide margin**, then assert the selected set is **exactly** that set
  (no checkpoint needed). This makes the selection gate load-bearing pre-R.
- **(FIX) Structural injected-error fixture.** The injected-1%-error tests prove sensitivity to *magnitude*
  perturbations; add a **structural** one — flip the causal cutoff `>`→`>=` (core §E5), or move the `relu`
  past the head-sum — and assert the index-set / `index_scores` gate trips. (Magnitude perturbations don't
  characterize the discrete bug class that dominates selection.)
- **(FIX) Warmup precompile primers.** Add the new `topk_indices` `AttentionMetadata` leaf + the indexer-KV
  cache (core §I7, §I8) to **every** precompile primer (`_precompile_backbone_helper` builds
  `AttentionMetadata` with no indexer field), or the first real DSA request recompiles at serving time.
- **Indexer parity** (core §H5, §H7): top-k boundary-aware tie-tolerant set equality + `index_scores` at
  1e-3 fp32 (computed from HF submodules — HF's indexer returns indices only).
- **Validation ladder vs HF eager (on TPU):** (1) seq ≤ `index_topk`: **sparse == dense at fp32 (≈1e-6)**
  (core §A5) + top-k contains all valid tokens; (2) seq > `index_topk`: full backbone forward, fp32 math
  gate 1e-3 + bf16 shipped + argmax ≥0.95, selected index set matches HF (tie-tolerant).
- **Multi-device sparse gate:** rerun the ladder on the 8-chip mesh; N-dev == 1-dev at fp32; rerun the decode
  + boundary fixtures on the mesh.
- **(FIX) O(N)-gather scaling smoke = EXIT gate** (core §G8). The one-hot gather reads every resident KV row
  (O(N), not O(`index_topk`)). Sweep context length N at fixed `index_topk` (N≈topk and N≫topk) and **measure
  the O(N) cliff**: either decode-step latency is already flat in N, or the DMA-scalar-prefetch rewrite
  (Phase 5) is **scheduled before any long-context claim**.

## Acceptance gates (numeric)
- seq ≤ topk: **sparse==dense fp32 ≈1e-6 on TPU** (no HF dep) + top-k contains all valid tokens.
- seq > topk: fp32 math gate 1e-3 + bf16 floor + argmax ≥0.95 + **boundary-aware tie-tolerant index-set
  equality** vs HF + `index_scores` 1e-3 fp32; multi-device sparse N-dev==1-dev.
- **(FIX)** DSA decode: per-step append + top-k re-selection vs the incremental oracle; prefill↔decode
  equivalence with DSA on.
- **(FIX)** exact-`index_topk` + all-`-1`-row finite softmax + post-topk re-mask + not-128-mult fixtures green.
- **(FIX)** synthetic well-separated-weights selection gate: selected set **exactly** the designed set.
- **(FIX)** structural injected-error (causal `>`→`>=` / relu misplacement) **trips** the index-set/scores gate.
- **(FIX)** O(N)-gather cliff measured (rewrite scheduled if not flat).

## Phase-specific risks & fixtures
- **Adapted sparse kernel** — non-paged, may hit tile-alignment; validate kernel-vs-jnp-ref; **severable v1
  fallback** (core §J4: DSA jnp-ref only, kernel → v1.1).
- **Indexer cache mis-modeled as a "hybrid second group"** → it is bespoke, core §I7.
- **Top-k set equality flaky under bf16/multi-device reduction order** → boundary-aware tie-tolerant compare
  + separate `index_scores` 1e-3 gate (core §H7).
