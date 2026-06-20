# Phase 4 — continuous batching + the serving lifecycle

The production serving runtime, correct end-to-end on the small config — the whole reason "servable" is the
goal. The repo's high-throughput path: `tpu_runner.py` gates `is_decode_only and enable_continue_decode →
_execute_continue_decode → runner/decode_loop.py::continue_decode` (a `jax.lax.while_loop` whose carry
includes the full `AttentionMetadata`, rebuilt each step). v1 leaves `enable_continue_decode=False`, so
Phases 0–2 are unaffected.

**Precondition:** 3 (concurrent decode + per-request-local `topk_indices` + the bespoke indexer-KV cache).
**Core anchors:** §A4, §A5, §B8, §G6, §H8, §H9, §H11, §I7, §I8, §I9, §J1.

## Deliverables
- **Continuous-batching matrix** (each a cheap fp32 self-consistency gate, same spirit as §A5):
  - **G2 chunked prefill** — a chunk's queries top-k over `[0, num_computed+chunk)`; gate: 2-chunk == single-shot.
  - **G3 prefix caching** — shared-prefix → identical top-k; indexer keys live in reused blocks; the
    query-dependent top-k is **never** cached.
  - **G4 preemption** — recompute, not swap (`swap_blocks` is a TPU stub that raises); preempt-then-resume ==
    uninterrupted; also fix `ContinuousFreeQueue` missing `prepend_n` (block leak on preemption).
  - **G5 persistent batch** — per-slot indexer state moves in lockstep with `add/remove/condense/swap`;
    bit-stable under reorder/condense.
  - **G6 speculative decoding** — eagle3/ngram rejected-draft indexer-KV writes roll back like the seqlen
    fixup (`_subtract_num_rejected_tokens`); gate: DSA + eagle3/ngram with rejection == DSA-only greedy for
    accepted tokens. (G8 structured/guided decoding is orthogonal — applied at logits downstream; no DSA
    handling needed.)
- **In-graph decode loop.** Compute the indexer top-k **device-side inside** `decode_loop.py`'s `lax.while_loop`
  (a naively-added `topk_indices` becomes a frozen step-0 **closure constant — silently stale**); the
  indexer-KV cache is **carried + donated** like `kvc`; the `-1`-sentinel topk buffer is loop-carried,
  fixed-shape; the loop block budget is host-clamped (`_get_min_remaining_slots`).
- **(FIX) End-to-end servability smoke.** Spin GLM up through the real serving entrypoint (random weights,
  small config) and complete a request — sampling (temp/top_k/top_p), streaming, stop strings, EOS/length →
  finish_reason, `logprobs=5`, `min_tokens`. **Write the model registration** (`_MODEL_REGISTRY` +
  `_PP_DISABLED_MODELS`, core §G6) — the e2e smoke is the first path that resolves the arch through the
  registry — and **assert arch resolution lands on the flax_nnx GLM path** (the registry silently falls back
  to torchax if the GLM entry is missing/broken → you can "serve" the *wrong model* at HTTP 200). Unsupported sampler params (penalties/min_p/logit_bias/bad_words/seed) are **rejected,
  not silently no-op'd** (core §H9).
- **(FIX) Abort mid-decode.** Verified-broken on two paths: the disagg `EngineCoreRequestType.ABORT` is a
  no-op (`core_tpu.py:775-779`, TODO), and the in-graph `continue_decode` `cond_fn` (`decode_loop.py:216-220`)
  checks only `i < max_decode_steps` + the EOS flag — **no abort signal**, so a cancelled/disconnected
  request decodes the full step budget holding batch/KV slots. Thread an abort signal into the loop carry
  (or document/cap `max_decode_steps` as the abort-latency bound); close the disagg ABORT TODO before any
  disagg serving. Gate: an abort-mid-decode case in the servability smoke frees the slot promptly.
- **(FIX) Positive observability.** The runner has only init-time logging (no per-step/per-request metrics).
  Assert the inherited vLLM `SchedulerStats` surface (decode throughput, finish-reason counts) is wired for
  GLM, and add **DSA-specific** counters (indexer top-k cost / O(N)-gather latency / indexer-KV occupancy /
  compile-cache-miss). Do **not** gate on liveness/health metrics (core §H11d) — gate on values upstream of
  the defensive `nan_to_num`.
- **(FIX) Runtime joint MLA+indexer-KV backpressure.** The Phase-5 capacity gate is a *startup* bound;
  runtime overflow currently falls to bare `assert end_idx <= max_model_len` (`tpu_runner.py:1102` etc.,
  crashes the worker, not the request). Confirm those asserts are unreachable given frontend validation
  (backstops), and add the growing indexer-cache contribution to the preemption trigger so the **joint**
  MLA+indexer-KV budget drives backpressure, not just MLA KV.
- **Sampler + generation gates** (core §H8): sampler unit tests vs a numpy ref; sampled-token determinism +
  **batch-invariance** (a greedy request's tokens independent of co-batched random requests); EOS/stop
  halting; logprobs/prompt_logprobs.

## Acceptance gates (numeric)
- Continuous-batching matrix passes (fp32 self-consistency) on tiny + medium with `max_num_seqs>1`:
  chunked-prefill==single-shot; shared-prefix identical top-k; preempt-resume==uninterrupted;
  batch-reorder bit-stable; spec-decode==greedy-on-accepted.
- DSA correct on `_execute_continue_decode` (jnp-ref fp32 over multi-step).
- Served-request e2e smoke completes with correct finish_reason/logprobs + **arch-resolution lands on
  flax_nnx GLM**; unsupported sampler params **rejected** (not no-op).
- **(FIX)** abort-mid-decode frees the slot promptly (both the default and `continue_decode` paths).
- **(FIX)** GLM `SchedulerStats` + DSA occupancy/compile-miss counters surfaced.
- Sampler unit + sampled-token determinism + batch-invariance + EOS/stop gates green.

## Phase-specific risks & fixtures
- **Silent model-resolution fallback** to torchax (wrong model, HTTP 200) → the arch-resolution assertion.
- **Sampler features silently no-op** → reject + schedule (core §H9).
- **Generation never validated pre-real-weights** (forward `argmax≥0.95` ≠ token-sequence correctness) →
  the greedy/sampler/EOS gates here + the Phase-1a greedy `generate()` gate.
