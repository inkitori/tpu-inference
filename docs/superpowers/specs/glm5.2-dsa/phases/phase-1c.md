# Phase 1c — single-sequence decode spine

The autoregressive / KV-cache path for the dense MLA backbone — a **distinct failure surface from prefill**
(core §H13 guardrail 4) and exactly what broke in V4. Built and validated on tiny **and** medium, single-
and multi-device, **before** any DSA or giant-model work.

**Precondition:** **1a hard-gates 1c.** Needs 1a's dense backbone + the Phase-0 incremental decode oracle.
(1b's S1 fix is reused on the mesh sub-step, but 1c follows 1a directly.)
**Core anchors:** §A2, §B10, §B11, §G5, §H6, §H8, §H11, §H13, §I2, §J1.

## Deliverables
- **Prefill↔decode equivalence** (the invariant V4's exact-length CPU parity missed): token N's logits from
  a length-N prefill must equal its logits from a length-(N−1) prefill + 1 decode step. Exercises
  `MlaCase.DECODE`/`BATCHED_DECODE`, the in-kernel KV-write index `[kv_len−q_len:kv_len]` (never stressed in
  prefill, where `kv_len−q_len==0`), and decode positions. Use **peaked logits** (core §H11a) so the check
  has teeth.
- **Page-boundary fixture.** Prompt length P mid-page, then decode across ≥1 page boundary (`page_size=16`
  tiny / `128` medium). Add a **fused-loop variant** to catch the frozen-`block_tables` + **silent
  out-of-range page-index clamp** in the kernel (a boundary crossing reads/writes the wrong page, no error).
- **Traced-position discipline.** Decode position must be a **traced** scalar, not a Python int: assert the
  JIT cache grows by **exactly 1** across consecutive decode positions (no per-position recompile; modeled on
  the fork's `test_buffer_decode_jit_cache_hits_across_positions`). Decode
  `seq_len`/position off-by-one: KV write index == `seq_len−1`, position == cached_len.
- **Donated-buffer determinism** (pre-empt the V4 decode "Heisenbug"). Under `jax.jit(donate_argnums=...)`,
  R1 then R2 on identical input must be **byte-identical**, + an HLO assertion that donated kv_caches survive
  with `input_output_alias`. (XLA write-elision under donation cost the fork ~6 sessions; prefer a
  Pallas-kernel write over `at[].set`.) Same cached prefill artifact run twice → **no SIGSEGV**.
- **Decode on the mesh.** Re-run prefill↔decode equivalence + NaN-poison on the 8-chip mesh: **N-dev==1-dev
  over a multi-step decode** (V4's decode bug lived on the multi-device decode path). No size-1 token-axis
  `with_sharding_constraint` gather (halts the core).
- **Greedy generation on the mesh** matches single-device (token-exact).
- *(Shared with Phase 1a: the `max_model_len=1M` wide-kernel variant — see phase-1a; it forces the same wide
  `pages_per_seq` MLA program the decode path will hit at real context length.)*

## Acceptance gates (numeric)
- **Prefill↔decode equivalence** (fp32 1e-3 + **argmax-exact**) on tiny + medium, single- and multi-device
  (N-dev==1-dev over multi-step).
- Page-boundary + fused-loop variants clean.
- Traced-position JIT-cache-grows-by-exactly-1 assertion holds.
- **Donated-buffer R1==R2 byte-identical + HLO `input_output_alias`.**
- Mesh greedy generation == single-device (token-exact).

## Phase-specific risks & fixtures
- **Decode validated only after the giant model** (the V4 trap) → this whole phase, on the mesh, before R.
- **Donated-buffer "Heisenbug"** — XLA write-elision under `donate_argnums`; re-arms if indexer-KV (Phase 2)
  is later written via `at[].set` rather than a Pallas kernel.
- **Silent page-index clamp** — the fused-loop variant catches it.
