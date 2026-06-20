# GLM 5.2 (DSA) on JAX/TPU — Design spec

Bring up **GLM 5.2** — `GlmMoeDsaForCausalLM` (`model_type="glm_moe_dsa"`, DeepSeek-V3.2-style **MLA +
DSA** at GLM-5 scale) — as a **native-JAX** model in `tpu-inference`, developed and validated **on TPU with
the real kernels from the start**, and made **fully servable** (concurrent decode, continuous batching, fp8).

## How to use this spec (read this first)

This spec is split so that **working on one phase loads only what that phase needs**:

- **[`core.md`](core.md)** — every invariant that more than one phase depends on, stated **once**
  (architecture, exact config, norms, RoPE, DSA deltas, weight names, code-reuse map, testing methodology,
  sharding contract, hardware reality). Organized as anchors §A–§J.
- **[`phases/phase-N.md`](phases/)** — a self-contained **work order** per phase: deliverables, numeric
  acceptance gates, phase-specific risks/fixtures, the `core.md` anchors it depends on, and its precondition.
- **[`history.md`](history.md)** — provenance and rationale (the V4 bring-up lessons, the audit that drove
  the scope, *why* decisions were made). **Not needed to implement** — read only when you need the "why".

**A phase implementer loads `core.md` + their one `phases/phase-N.md`.** Nothing references another phase
file; cross-phase dependencies are stated as a one-line precondition.

**Navigating with grep** (the structure is uniform on purpose):
- Resolve a cited anchor → its definition: `grep -n "§I4" core.md` (citation and definition share the **`§`** token).
- A phase's required anchors: the `**Core anchors:**` line near the top of each `phase-N.md`.
- A phase's adversarial-review fixes: `grep -n "(FIX)" phases/phase-N.md`.
- A phase's gates / scope: the uniform headers `## Deliverables`, `## Acceptance gates (numeric)`, `## Phase-specific risks & fixtures`.
- A code symbol's home: cites are `path/file.py:line` (line numbers drift — re-locate by symbol).

## Phase plan (TPU-first; the giant checkpoint loads LAST)

Validate decode, concurrent decode, continuous batching, fp8 and perf **structure** on small/medium configs
**first**; load the real giant checkpoint **last** (the only phase that genuinely needs real weights).

| Phase | What | Surface |
|---|---|---|
| **[0](phases/phase-0.md)** | env, HF oracle, harness — **DONE** (`5d31ba05`) | tiny + medium config |
| **[1a](phases/phase-1a.md)** | single-device dense-MLA backbone (real kernel + fp32 jnp-ref oracle) — **DONE** (`b5bc0101`) | 1 chip |
| **[1b](phases/phase-1b.md)** | multi-device S1 gate (uninit-HBM-on-reshard) — **EP + TP modes** | 8 chips |
| **[1c](phases/phase-1c.md)** | single-sequence decode spine (prefill↔decode equivalence) | 1 + 8 chips |
| **[2](phases/phase-2.md)** | DSA indexer + sparse kernel + DSA decode state | 1 + 8 chips |
| **[3](phases/phase-3.md)** | concurrent decode (`max_num_seqs>1`) | 8 chips |
| **[4](phases/phase-4.md)** | continuous batching + the serving lifecycle | 8 chips |
| **[5](phases/phase-5.md)** | fp8 + perf structure (real shapes, random weights) | 8 chips |
| **[Mh](phases/phase-Mh.md)** | multi-host **structure** on dummy weights (orchestration, no real weights) | 2-host slice |
| **[R](phases/phase-R.md)** | real-checkpoint reconciliation (the ONLY real-weight phase) | multi-host slice |

Hard-gates: **1a → 1b**, **1a → 1c** (never debug math on the mesh). Each production phase is single-chip-correct
first, then re-gated on the 8-chip mesh, then (only at Phase R) multi-host.

## Status

Phase 0 (`5d31ba05`) and **Phase 1a** (`b5bc0101`) are committed on `glm5.2-dsa`. Phase 1a delivered the
full dense GLM backbone in `tpu_inference/models/jax/glm_moe_dsa.py` (both MLA paths: the non-absorbed
fp32 jnp-ref answer key + the absorbed `mla/v2` Mosaic kernel, validated against each other at ~1.6e-7),
the converter, and ~75 numeric gates (full-forward fp32 ≤1e-3 vs HF, token-exact greedy generate, medium
real-shape + 1M-compile coverage, bf16-floor depth characterization). Everything else is designed and
ready for implementation planning. Architecture facts are verified against **transformers 5.12.1**;
re-locate every cited symbol by name (line numbers drift).

**Verified spec corrections (from Phase 1a):** (1) the current `mla/v2` kernel has **no** `pages_per_seq`-keyed
"wide-program / SMEM-OOM minibatch-split" branch — it is one static-tuned program path and its flash gather
loop is **`kv_len`-bounded**, not `pages_per_seq`-bounded; so the 1a `max_model_len=1M` variant exercises
RoPE-width + cache allocation/addressing only, and the true long-context flash **compute** is first-contacted
by **real long sequences in Phase 1c**, not by a wide `pages_per_seq` at short seq. (2) `max_position_embeddings`
is not read by the JAX model (RoPE width comes from the positions array; cache width from caller allocation).
(3) the bf16 numeric floor is **lm_head-projection-dominated and approximately flat with depth** (~1e-2 at
hidden=6144/vocab=2048); the real L=78 floor will be **higher** (larger production vocab → more lm_head
accumulation; trained weights; full MoE) — use ~1.1e-2 as a conservative lower bound.

> Supersedes the single-file `2026-06-19-glm5.2-dsa-jax-tpu-design.md` (preserved in git history). The
> predating research companions under `../research/` contain superseded claims — **this spec is
> authoritative.**
