# Phase Mh — multi-host structure (dummy weights)

**New phase, inserted between 5 and R.** Validate multi-host **orchestration** on dummy/random weights
**before** the real checkpoint, mirroring how Phases 3/4/5 pull decode/batching/fp8 *structure* ahead of the
giant load. This separates the multi-host-only failure modes from real-weight reconciliation, so Phase R
first-contacts **neither** (the rationale and the V4 evidence are in `history.md` — the fork reproduced its
worst multi-host bug on random weights, and every genuinely multi-host bug it hit was weight-independent).

**Precondition:** 5 (full correctness / decode / batching / fp8 / perf structure green on the single-host
8-chip mesh). Needs a **>1-host slice**, but **not** the real weights.
**Core anchors:** §B11 (medium config), §F8 (loader, `--load-format dummy`), §H6 (the multi-device gate +
localization, now across the host boundary), §H11, §H13, §I1–§I9 (the full sharding contract + new DSA
tensors must survive cross-host), §J2 (slice sized for medium, not the real model).

## In-repo precedent (this is not greenfield)
- `PathwaysDummyModelLoader.create_dummy_weights_on_tpu()` random-inits weights **directly onto the
  multi-host TPU mesh** (no checkpoint, no CPU staging); `--load-format dummy` auto-routes to
  `jax_dummy`/`pathways_dummy` (`model_loader.py:269-273`).
- The repo already runs a multi-host CI gate — `.buildkite/pipeline_dev.yml` `tpu7x_demo_multihost` /
  `pipeline_jax.yml` `test_26` — Ray-based, 2-host/16-chip, against a **small** MoE (Qwen3-30B-A3B). Point
  `run_multihost.sh` + the Ray executor at GLM **medium** with `--load-format dummy`.

## Deliverables
- Bring up the **medium config across a 2-host slice** with dummy weights.
- **Weight-independent orchestration gates:** Ray/Pathways placement-group + per-worker mesh construction
  succeeds; PP transfer-server connectivity; **per-host load symmetry** (the fork's asymmetric-load wedge —
  assert no host silently takes a dummy-zeros fallback); host-side-vs-device launch-group ordering (the
  fork's RoPE-precompute "scheckne" core-halt — assert RoPE-freq precompute is host-numpy, no SPMD launch-id
  mismatch); cross-host KV-transfer server init.
- **Cross-host value-invariance** (the multi-host analogue of the Phase-1b S1 gate): same dummy weights,
  **single-host-8-chip result == 2-host result at fp32**, reusing the core §H6 per-weight checksum probe
  across the host boundary — catches a cross-host sharding/transfer corruption **on random weights**, before
  the checkpoint.

## Acceptance gates (numeric)
- 2-host slice boots; medium config runs end-to-end on dummy weights.
- **Cross-host N-dev==1-dev value-invariance** on the medium config (dummy weights), fp32.
- NaN-poison clean across hosts; per-weight checksum identical host-vs-host.
- Orchestration gates green (placement groups, transfer servers, load symmetry, launch-group ordering) —
  all **without real weights**.

## Phase-specific risks & fixtures
- **Multi-host-only failure modes** (cross-host collective ordering, slice-builder/transfer races, Ray init,
  per-host load asymmetry) — these were never separated out in the fork; pulled forward here onto dummy
  weights so they don't co-occur with real-weight bring-up in Phase R.
- If no >1-host dev slice can be provisioned, this phase's deferral becomes a **resource** constraint — state
  it explicitly and accept the Phase-R pile-up risk; do **not** claim these modes "only surface at R."
