# Phase R — real-checkpoint reconciliation (the ONLY real-weight phase)

GLM 5.2 generation correct against the real `zai-org/GLM-5.2-FP8` checkpoint. Every structural / decode /
batching / fp8 / perf / multi-host-**orchestration** concern is already retired on small/medium configs and
dummy weights, so this phase **reconciles only the four genuinely weight-dependent items** — it does not
first-contact a runtime, a sharding mode, a loader path, or a host topology.

**Precondition:** ALL of Phases 0–5 **+ Mh** (multi-host orchestration proven on dummy weights). The fp8
load path (Phase 5, incl. the UE8M0 wiring) and the multi-host slice (Mh) are already green.
**Core anchors:** §B3, §B7, §B8, §B9, §D2, §E1, §E3, §E4, §E5, §F6, §F8, §G7, §H2, §H4, §H6, §H10, §H11,
§H12, §H13, §I4, §I7, §J2.

## Deliverables (the four genuinely-weight-dependent items)
1. **fp8 real-weight load.** The Phase-5 path (with UE8M0 wired, core §B7) now reads the real e4m3 +
   `weight_scale_inv` UE8M0 128-block checkpoint. UE8M0-as-uint8 (bitcast, not astype). *(The bf16 repo is
   ~1.5 TB / multi-node — never used here.)*
2. **Indexer-RoPE HF-vs-vLLM ground truth.** GLM-5.2 sets `indexer_rope_interleave=true`; HF runs rotate-half,
   vLLM interleaved (core §D2, §B9). Only trained `wq_b`/`wk` adjudicate — decide by indexer top-k behavior
   on real weights, in **fp32**. Reconcile vLLM's fp8 indexer ReLU location (Triton ref readable; final
   confirm on the real GPU path).
3. **Real-shape/-depth numerical reconciliation.** Single-seq + batched generation matches the
   **jnp-ref kernel oracle** (the pure-`jnp` fp32 answer key built across Phases 1a+2, core §H2/§H13, in
   `tests/models/jax/test_glm_moe_dsa.py` per §H12) within the **bf16 floor re-measured at real shapes** (run the jnp-ref in
   bf16 to derive it — 78-layer noise compounds differently; the Phase-1a depth-slope characterization
   *predicts* this, don't reuse the tiny floor) + argmax ≥0.95. HF-eager is impractical at real shapes; the
   jnp-ref is the math reference here.
4. **Real top-k selection fidelity.** Trained weights give well-separated scores where a wrong relu/scale
   changes the selected set (random near-ties hid this in dev; the Phase-2 synthetic-weights gate de-risked
   it, but real-weight confirmation lands here — core §H11a).
   - **(FIX) Hadamard + fp8 selection-set comparison.** The HF oracle is a documented **approximation** that
     skips the real indexer's Hadamard transform + fp8 scoring (core §H10), so the real selected set can
     differ from the HF-eager set even when our impl is correct. Compare the selected top-k set against the
     real **vLLM/GPU fp8 + Hadamard** path, not just HF — adjudicate boundary divergences by which path the
     trained weights were optimized for.

- **Multi-host slice** sized for weights + KV (the real model never fits one v6e-8, core §J2); real
  expert/tensor parallelism. The **orchestration is pre-validated by Phase Mh**, so R's multi-host work is
  "run real weights on the proven slice," not first bring-up.
- **(FIX) Confirm the FP8 repo carries the tokenizer / chat template.** GLM-5.2's chat template + special
  tokens ship in the checkpoint repo's `tokenizer_config.json`/`chat_template.jinja`; the **weights** repo
  is `zai-org/GLM-5.2-FP8`. Confirm it carries them (or source them), else the served model produces correct
  token IDs with **wrong chat framing**.

## Acceptance gates (numeric)
- Single-seq + batched generation matches the jnp-ref oracle within the **re-measured bf16 floor at real
  shapes** + argmax ≥0.95.
- Indexer-RoPE HF-vs-vLLM ground truth **decided** against real weights (fp32); ReLU location confirmed.
- **(FIX)** Hadamard+fp8 selected-set comparison vs the real vLLM path completed; boundary divergences
  adjudicated.
- Multi-host slice gate green (thinner — orchestration proven in Mh).
- FP8 repo tokenizer/chat-template confirmed present (or sourced).
- The full serving stack runs the real checkpoint end-to-end.

## Phase-specific risks & fixtures
- **Multi-host-only modes** are mostly retired by Phase Mh — what remains is real-weight residency + real
  expert/tensor parallelism on the proven slice (a thin gate, not "more of the same" first contact).
- **Real model never fits the dev box** (~744B → 1.5 TB bf16 / 744 GB fp8 vs 256 GB) → this is the only
  phase on the multi-host slice (core §J2).
