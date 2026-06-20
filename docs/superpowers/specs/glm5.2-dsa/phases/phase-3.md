# Phase 3 — concurrent decode (`max_num_seqs>1`)

Correct multi-request decode — the production batch dimension — validated before the giant model. Reuses the
repo's existing multi-request serving path (`persistent_batch_manager.py`, `input_batch.py`,
`tpu_runner.py::_prepare_inputs`, the per-seq-looping MLA kernel).

**Precondition:** 1b (the S1 weight-load half of the corruption retired) + 2 (DSA decode state, the bespoke
indexer-KV cache, and `topk_indices` built). Phase 3 extends value-invariance to the **batch** dimension.
**Core anchors:** §A4, §B10, §B11, §H6, §H8, §H11, §I7, §I8, §J1.

## Deliverables
- **Per-request-local `topk_indices`** (0..seq_len−1) + block_table demux per `seq_idx` (core §I8) — a
  selected index never crosses a request boundary; the kernel demuxes via `cu_kv_lens` + `block_table`.
- The **bespoke indexer-KV cache machinery** participating in block allocation (core §I7 — **not** the
  hybrid path).
- **Ragged multi-seq isolation:** each per-seq output **bit-identical** to its serial run, incl. a 3-seq
  case for the `n_active` off-by-one (fork `e4cb7564`).
- **Activation-sharding half — close the *other* half of the `max_num_seqs>1` corruption.** Phase-1b retired
  only the weight-load uninit-HBM half; gate the **runtime `num_reqs < attn_dp` activation-sharding** path that routes MoE through
  a dense einsum reading uninit HBM for un-owned tokens (fork `fc03c1f9`). **Until this phase is green, a
  loud startup guard must reject `max_num_seqs>1`** (silent-corruption footgun).
- **Pad-bucket decode hazards** (the fork's worst saga): `L_real < T_pad` prefill→decode parity (decode
  seeded from **padding** tokens = the "pad-token attractor"; slice to `L_real` via a **traced** scalar, core
  §H11b); the `is_decode` detector vs runner T-padding.

## Acceptance gates (numeric)
- `max_num_seqs>1` decode on the v6e-8: **N-dev==1-dev fp32**; NaN-poison clean.
- Per-request outputs **bit-stable vs single-sequence runs** of the same prompts (value-invariance extended
  to the batch dim).
- Ragged 3-seq bit-identical-vs-serial + pad-bucket `L_real<T_pad` fixtures green.
- **Activation-sharding** `num_reqs<attn_dp` path gated; loud guard rejects `max_num_seqs>1` until green.

## Phase-specific risks & fixtures
- **`max_num_seqs>1` two-ingredient corruption** — (a) retired by 1b; (b) the runtime activation-sharding
  (`num_reqs<attn_dp`) path, gated here.
- **Pad-bucket / ragged-batch decode bugs** (pad-token KV attractor, `is_decode` vs T-padding, cross-seq
  attention leak) — invisible to exact-length single-seq fixtures.
