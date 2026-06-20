# GLM 5.2 (DSA) — Production-Readiness Adversarial Audit

**Date:** 2026-06-20
**Purpose:** Pressure-test the design spec (`../specs/2026-06-19-glm5.2-dsa-jax-tpu-design.md`)
against the goal of a **fully correct, servable, highly performant, OpenRouter-ready**
GLM 5.2 — doing **as much work as possible on small/medium configs before loading the
giant real weights**, while still validating full correctness against references.
**Method:** 9 parallel adversarial/research agents reading the spec, the repo, the
`claude-deepseek-v4` fork (2923 commits + `HANDOFF_*` postmortems), and vLLM. Each agent
was told to assume the spec under-specifies its dimension and to prove it with code evidence.

> Sibling trees on this machine: repo `/Users/enyouki/open-inference/tpu-inference`,
> fork `/Users/enyouki/open-inference/claude-deepseek-v4` (checkout under
> `work/tpu-inference/tpu_inference/`), vLLM `/Users/enyouki/open-inference/vllm`.
> Line numbers drift — re-locate by symbol.

---

## 0. Headline diagnosis

The spec is **excellent on prefill math, the S1/uninit-HBM/reshard class, and DSA *forward*
correctness**. It is **structurally wrong for the user's goal in two ways**:

1. **It validates only `forward` (prefill).** The word "generation" first appears in **Phase 3**,
   gated on **real weights** (spec line 206). Even *single-sequence decode* — KV-cache append,
   the prefill→decode handoff, the absorbed-MLA decode branch — is never gated before the giant
   model loads. **This is the exact `claude-deepseek-v4` failure:** the fork validated prefill,
   declared correctness off it, and decode was what broke (fork `e6788e7f`: *"prefill-everything
   is COHERENT; decode collapses at step 1"*; `5e37b9da`: a decode bug the *"CPU parity test
   missed because it uses exact-length"*; `fc03c1f9`: concurrent decode *"CONFIRMED BROKEN,"*
   walled off with `--max-num-seqs=1` rather than fixed).

2. **It loads the giant checkpoint (Phase 3) *before* every production item** (Phase 4 concurrent
   decode, 5 continuous batching, 6 fp8). Of everything deferred past the giant load, **only four
   items genuinely need real weights** (§9 below). All the rest — decode, concurrent decode,
   continuous batching, the in-graph loop, fp8 *structure*, the O(N)→O(topk) gather rewrite, even
   GLM-shaped kernel tuning (real **shapes** ≠ real **weights**) — can and should be validated on
   small/medium configs first.

Two further facts the spec never states, each load-bearing:

- **The real servable checkpoint is fp8 on disk** (`zai-org/GLM-5.2-FP8`: e4m3 + `weight_scale_inv`
  UE8M0 128-block). The spec's "v1 loads bf16" assumption only holds for a ~1.5 TB bf16 repo that
  needs multi-node. **You cannot load the real weights without an fp8/dequant path** → fp8 is a
  *load-time prerequisite*, not a Phase-6 nicety.
- **The real model does not fit the validation surface.** GLM 5.2 is ~744B params (≈1.5 TB bf16 /
  744 GB fp8); a v6e-8 is 256 GB. **Weights, not KV, are the binding constraint.** The dev box
  validates *correctness + shape-driven capacity math*, never real-model residency — the real model
  needs a multi-host slice. The spec mentions multi-host for *correctness* (S1) but never for
  *capacity*.

---

## 1. Decode path & KV cache (BLOCKER — the central gap)

**Verdict: the spec does not validate decode before the giant-model load.** Evidence:

- Every v1 exit (Phases 0/1a/1b/2) checks a single `forward`; "generation" only at Phase 3 (real
  weights, line 206) and even that compares to the jnp-ref, not HF `generate()`.
- **No per-token `slot_mapping` in the JAX path.** The KV-write index is computed *inside* the
  kernel as `[kv_len - q_len : kv_len]` from `seq_lens` + `block_tables`
  (`kernels/mla/v2/kernel.py`). In prefill `q_len==kv_len` ⇒ `kv_len-q_len==0`, so this arithmetic
  and the `MlaCase.DECODE`/`BATCHED_DECODE` masking branch are **never exercised until decode**.
- **Fused decode loop freezes `block_tables`** and the kernel **silently clamps** an out-of-range
  page index to the last page (instead of erroring) — a page-boundary crossing mid-decode reads/writes
  the **wrong page with no failure signal** (`decode_loop.py` + kernel clamp).
- **DSA decode state is actively discarded today**, not merely unbuilt: vLLM's indexer cache is
  collapsed to a dense MLA slot (`kv_cache_manager.py:613-620`), `_topk_indices` is dropped
  (`mla_attention.py:318-320`), the TPU DSA forward is a no-op passthrough
  (`deepseek_v4_attention.py:99-108`).
- Latent teardown bugs to fix before any long-decode test: `ContinuousFreeQueue` missing
  `prepend_n` (block leak on preemption); queue injection may not cover the non-disagg path.

**Recommended changes**
- **Phase 0 harness:** add an **incremental HF decode oracle** (`use_cache`/`past_key_values`/
  `cache_position`, N-step loop). This unblocks every decode gate.
- **New decode gate before the mesh work (proposed Phase 1c, hard-gates 1b):** single-device
  **prefill-vs-decode equivalence** on the tiny config — step-N decode logits == length-N prefill
  logits (fp32 ≈1e-3 + argmax-exact). Exercises `MlaCase.DECODE`, the `kv_len-q_len` write/mask,
  decode positions.
- **Page-boundary fixture** (P mid-page, P+K crosses ≥1 boundary at `page_size=16`), plus a
  **fused-loop variant** to catch the frozen-`block_tables` / silent-clamp hazard.
- **Pull DSA decode state into Phase 2:** build the indexer-KV cache + `topk_indices`-in-metadata
  and validate per-step append + per-step top-k re-selection vs the incremental oracle.
- **Run the decode-equivalence + NaN-poison gate on the 8-chip mesh in 1b** (N-dev==1-dev over
  multi-step decode) — this is where the fork's decode bug actually lived.
- Reframe Phase 3 as "**re-confirm** decode on real weights," not "first decode validation."

---

## 2. Phase sequencing (resequence so the giant model is last)

**Verdict: ordering is incompatible with the thesis.** Phase 3 (real checkpoint) sits *between*
its prerequisite (Phase 1b's S1 fix) and the dependents (4/5/6) for no stated dependency reason —
Phase 1b is green on the tiny config with random weights, so everything gated only on 1b can be
built on the tiny config.

**Per-item: does it genuinely need real weights?**

| Item (current phase) | Real weights? | Why / rebuttal |
|---|---|---|
| Single-seq decode (never scheduled pre-3) | **NO** | forward at seq=1 + KV read; pure plumbing |
| Concurrent decode `max_num_seqs>1` (P4) | **NO** | spec's own exit (line 315) never mentions weights; it's coords/isolation |
| Continuous batching + in-graph loop (P5) | **NO** | control-flow/shape discipline; exit (line 320) is jnp-ref fp32 |
| fp8 *structure* (P6) | **NO** | quantize random weights; fp8-vs-bf16 jnp-ref within a floor |
| DSA gather O(N)→O(topk) rewrite | **NO** | length-driven; "no regression vs jnp-ref" |
| GLM-shaped kernel tuning | **NO** (real *shapes*) | timing is weight-value-independent; needs real dims, not weights |
| Multi-host *structure* arming | partial (real *sizes*) | collective ordering armable at real shapes |
| **Indexer-RoPE HF-vs-vLLM ground truth** (§9 #5) | **YES** | only trained weights adjudicate the `indexer_rope_interleave` flag |
| **Real-shape/-depth numerical reconciliation + bf16 floor** | **YES** | 78-layer noise compounding; needs real weights |
| **Full 78-layer end-to-end generation** | **YES** | real depth + weights by definition |
| **vLLM-fp8 ReLU-location confirmation** (§9 #4) | **YES** (confirm only) | compiled deep_gemm; confirm on real GPU path |

**Net:** only the four bold items are legitimately last. Everything else moves ahead of the
giant-model load. Re-scope §1 non-goals (lines 37–42), the "v1 finish line" (line 29), and the §11
opener (line 308), which all frame production as post-v1. Decouple "real shapes" from "real weights"
explicitly in §10/§13.

---

## 3. Quantization reality (fp8-on-disk is a load-time prerequisite)

**Verdict: inadequate and mis-sequenced; one hard blocker.**

- **fp8-on-disk blocker (very high confidence):** `zai-org/GLM-5.2-FP8` ships e4m3 + 128×128
  block-quant, scales `weight_scale_inv` (UE8M0). The repo's **generic loader upcasts to bf16**;
  native fp8 load (no upcast) only happens on the **`Fp8Config` path**, which GLM currently does
  *not* take. The spec routes GLM *around* the fp8 loader (§5/§8). DSA indexer fp8 has **zero
  implementation** in the repo — must be ported from scratch.
- **The repo already has the machinery the spec treats as Phase-6 greenfield:**
  `layers/jax/quantization/fp8.py`, `get_tpu_quantization_config` (`model_loader.py:318`),
  `MLAEinsum.load_weights` (`deepseek_v3.py:512-583`), block dequant
  (`layers/common/quantization/fp8.py:149-162`); the V4 fork has a working dequant recipe
  (`deepseek_v4_loader.py`). Post-load is **dequant-then-requant to the TPU matmul layout**, not
  dequant-to-bf16.
- **fp8 is validatable on the small config early:** quantize tiny random weights → e4m3 + 128-block
  UE8M0 → existing `Fp8Config` path → compare to bf16 jnp-ref within an fp8 floor. No real weights,
  no batching needed.
- vLLM fp8 indexer ReLU **is** readable (Triton ref): `triton_fp8_mqa_logits.py:128-129,149-150`
  (`scores = tl.maximum(scores, 0.0)`) — cite this in §9 #4 instead of "confirm empirically."
- Omitted: `quantization_config` absent from §3; no dequant-on-load milestone; the UE8M0-as-uint8
  rule (bitcast not astype; never on-device `float8_e8m0fnu`); 128-block scale axis/layout
  (replicated, not head-sharded); **fp8 KV cache** (`--kv-cache-dtype fp8_e4m3`); the
  under-specified absorbed-MLA fp8 mode; activation quant (`activation_scheme: "dynamic"`).

**Recommended changes**
- Add a **fp8-weight-load + small-config fp8-validation rung ahead of the giant-model phase**, reusing
  the existing `Fp8Config`/`MLAEinsum`/fork-loader machinery; stop routing GLM around the fp8 loader.
- Record the real `quantization_config` + UE8M0-uint8 rule in §3; cite the Triton ReLU ref.

---

## 4. Generation & sampling correctness (separate from forward parity)

**Verdict: not validated; first "generation" check is Phase 3 on real weights.**

- `argmax≥0.95` forward-parity ≠ greedy generation: ~1-in-20 wrong top-1 cascades under
  autoregression; two models at 0.95 forward-parity can diverge wildly when generating.
- The on-TPU sampler (`layers/jax/sample/sampling.py::sample`) implements **only temp/top_k/top_p
  (+logprobs)**. **No penalties, no min_p, no logit_bias, no bad_words, no allowed_token_ids, no
  per-request seed** — a single **global RNG stream** (`tpu_runner.py:539`), split per step. TPU
  **replaces the entire worker** (`tpu_platform.py:314-315`); there is no per-platform sampler hook,
  so every sampling feature is the backend's responsibility or it **silently no-ops** for an
  OpenRouter-facing server.
- `do_sampling` is **batch-level static** (`not all_greedy`): one random request flips the whole
  batch onto the sampling HLO + shared RNG; the spec's logits-value-invariance gate does **not**
  cover **sampled-token** batch-invariance.
- Untested: prefill→first-decode-token handoff (off-by-one in last-token slice); EOS/stop/min_tokens
  halting (`decode_loop.py` EOS path); logprobs / prompt_logprobs (device-side; `tpu_input_batch.py:153`).
- Precedent the spec abandons: the V4 fork *did* gate multi-step generation (a temp=0 `vllm serve`
  Fibonacci probe `2d8ca139`/`b2333a4b`; an N=12 decode-trajectory self-consistency test
  `test_deepseek_v4.py::TestRealConfigDecodeStability`).

**Recommended changes (all on the small config, against the Phase-0 HF oracle)**
- **Greedy `generate()`-vs-HF token-exact gate** (Phase 1a, dense-equiv regime; near-free) + a
  seq>topk variant in Phase 2. Highest-value missing gate.
- **Sampler unit tests** vs a numpy reference (temp/top_k/top_p transforms, greedy=argmax,
  `is_greedy` short-circuit, fixed-seed `categorical` reproducibility across both RNG paths) +
  injected-error test.
- **Document & decide sampler scope:** pin ordering (temp→top_k→top_p) vs vLLM; declare
  penalties/min_p/logit_bias either out-of-scope-for-v1 (and assert the server rejects/ignores) or
  schedule them.
- **Sampled-token determinism + batch-invariance** gate (fixed greedy request identical regardless
  of co-batched random requests); decide per-request `SamplingParams.seed` support (`fold_in`).
- **EOS/stop + logprobs** gates; prefill→first-decode-token handoff gate.

---

## 5. OpenRouter / servability

**Verdict: not validated as servable, and not early.**

- **Silent-fallback trap:** two arch-recognition layers behave oppositely. vLLM's frontend registry
  (`registry.py:122`) already maps `GlmMoeDsaForCausalLM` → so startup passes the *loud* guard. The
  tpu-inference registry (`model_loader.py`) **silently falls back to the torchax path**
  (`get_model:612-624`) if the GLM entry is missing/broken → you can "successfully serve" the **wrong
  model**. Needs an explicit assertion that resolution lands on the flax_nnx GLM path.
- The rest of the serving lifecycle (SSE framing, incremental detok, stop strings, stop-token/EOS/
  length→finish_reason, max_model_len validation, health, metrics, cancellation, error→HTTP) is
  **host-side and inherited** — which makes the absence of even **one served-request smoke test on
  the small config** the glaring, cheap-to-fix gap.
- `prompt_logprobs` (echo) and `min_tokens` EOS-suppression are device-side and unverified for GLM.

**Recommended changes**
- Add an **end-to-end served-request smoke** on the small config (random weights, through the serving
  entrypoint, completing a request with `logprobs=5` and a `min_tokens` probe) — the e2e harness
  already exists (`tests/e2e/`).
- Add an **arch-resolution assertion** (resolution lands on flax_nnx GLM, not torchax fallback) as a
  Phase-0/1 gate.

---

## 6. Continuous batching / serving-stack integration

**Verdict: not adequately covered; §12's headline claim is materially wrong.**

- **§12 "register as a second cache group [that] flows through the existing hybrid path" is wrong on
  the TPU native-JAX path.** `get_kv_cache_spec` has two branches (`kv_cache_manager.py:471`): the
  **native-JAX/`flax_nnx` branch** (`:484-565`, the GLM target) emits one MLA spec per layer from
  `hf_config` and has **no indexer awareness**; the `is_cache_for_ds_v4` detection +
  `_hybrid_uniform_page_size_bytes` live **only in the vLLM/torchax `else:` branch** (`:566-668`).
  Even in vLLM, the indexer spec is **co-bundled** into one group (gated on a `SlidingWindowMLASpec`
  layer GLM doesn't have), **not** a separate group. The hybrid path's only real tenant is
  **non-paged mamba** (one recurrent slot/request via `mamba_state_indices`, bypassing
  `block_tables`); `_hybrid_uniform_page_size_bytes` *forces* a single page size. A per-token, paged,
  growing raw-key indexer cache is unlike anything this code has run. Supporting blockers:
  `initialize_kv_cache` sizes blocks from one `page_size_bytes` (`:795-886`); `create_kv_caches`
  takes one model-wide `use_mla`; KV-transfer hard-raises for `len(block_ids)>1` (`:1202-1206`);
  there is **no `slot_mapping`** in the JAX path.
- **§11 Phase-5 mechanism imprecise:** in `decode_loop.py` the `lax.while_loop` carry is a flat tuple
  and `AttentionMetadata` is rebuilt each step from carried (`pos`,`sl`) **+ closure-constant**
  (`block_tables`,`query_start_loc`,`request_distribution`) values — a naively-added `topk_indices`
  becomes a **frozen step-0 closure constant (silently stale)**, not loop-carried. Fix needs
  device-side recompute in `_run_one_step` or explicit `_pack`/`_unpack`; the indexer-KV cache must
  also be **carried + donated** like `kvc`; the loop block budget is **host-clamped**.
- **G7 — Phase-1b retires only HALF of `fc03c1f9`.** The concurrent-decode MoE corruption has two
  ingredients: (a) weight-load uninit-HBM (the S1 half Phase-1b fixes) **and** (b) a **runtime
  `num_reqs < attn_dp` activation-sharding** path that routes MoE through a dense einsum reading
  uninit HBM for un-owned tokens (silent garbage on co-batched requests, HTTP 200). The spec's
  "Phase-1b retires this" claim is incomplete.
- Omitted serving features, each invisible to single-seq: **chunked prefill** × indexer causal
  pre-topk mask (a chunk's queries must top-k over the whole prefix-so-far; needs earlier chunks'
  keys written + coords offset by `num_computed_tokens`); **prefix caching** × indexer-key validity
  (reused blocks must contain indexer keys; top-k itself must never be cached); **preemption**
  rebuild (recompute-resume gets fresh blocks → rebuild indexer state; `swap_blocks` is a TPU stub
  that raises); **persistent-batch per-slot indexer state** (add/remove/condense/swap + ragged
  packing, modeled on `mamba_state_indices`); **speculative decoding** rollback of rejected drafts'
  indexer writes (eagle3/ngram are wired; GLM's own MTP head is dropped). Structured decoding is
  orthogonal (applies at logits) — no action.

**Recommended changes**
- **Rewrite §12's indexer-cache claim** to describe the **bespoke** TPU machinery actually required
  (native-path per-layer spec entry, dedicated allocation branch, own block table in
  `MultiGroupBlockTable` — which *does* support per-group `block_sizes` — own write op, new
  per-token-index `AttentionMetadata` field modeled on `mamba_state_indices`; lift the
  `len(block_ids)>1` `NotImplementedError`).
- Add a **§12.x continuous-batching-for-DSA** subsection with one acceptance gate each (chunked-prefill
  bit-equal to single-shot; shared-prefix identical top-k; preempt/resume equivalence; per-request
  bit-stability under batch reorder/condense; spec-decode-vs-greedy on accepted tokens) — **all cheap
  fp32 self-consistency gates on the small config with `max_num_seqs>1`, no HF dependency** (same
  spirit as §2's dense≡sparse boundary check).
- Strengthen the Phase-4 S1 framing (G7) + a **loud startup guard** against `max_num_seqs>1` before
  that phase is green.
- **State explicitly that the small config + `max_num_seqs>1` is the validation surface for the
  entire continuous-batching matrix** — mirroring how §2 uses short-seq dense≡sparse to retire the
  sparse-plumbing class.

---

## 7. Medium-config tier & vLLM static-divergence oracle

**Verdict: the spec leaves correctness on the table (single oracle, single config tier).**

- **Tiny config can't reach production code paths.** `page_size=16` ⇒ `bkv_sz % 128 != 0` ⇒ the MLA
  kernel's **shipped fast masking path** (`kernel.py:581-602`) is **never tested** (tiny only runs
  the slow reference `:605-613`); the transposed-cache layout (`:92` `assert page_size % 128 == 0`)
  is unreachable; MoE expert distribution is degenerate (`8//8=1`, `sparse_moe.py:69-79`), so the
  `(EP,EP,local)` reshard tiling never tiles non-trivially; `index_topk=2048` and 64-head/model-axis
  divisibility never manifest.
- **vLLM can't be a cheap runtime oracle:** CPU hard-blocks **MLA** *and* **sparse attention**
  (`cpu.py:83-86` `NotImplementedError`); the DSA indexer is **fp8/CUDA+DeepGEMM-only** with no
  native fallback (`sparse_attn_indexer.py:337`). Neither the MacBook nor the v6e-8 has an NVIDIA GPU.
  (`GlmMoeDsaForCausalLM` *is* registered in vLLM → `deepseek_v2`; `--load-format dummy` exists.)

**Recommended changes**
- **Add a "medium" config tier** at real per-layer dims (hidden 6144, 64 heads, real head dims
  qk_nope=192/qk_rope=64/v=256, kv_lora=512, index_head=128, index_n_heads=32, `page_size=128`,
  `index_topk=2048`, `index_topk_freq=4`/`offset=3`) with **reduced depth + experts + moe_inter**
  (e.g. L6 = 3 dense + 3 sparse, 16 experts, moe_inter 512, vocab ~2048 → ~1.7B params / ~7 GB fp32,
  CPU-eager-feasible). Run two seq regimes around `index_topk` (~2040 ≤topk, ~3000 >topk). Slots into
  Phase 1a (fast masking path, 64-head divisibility) and Phase 2 (real index_topk boundary).
  **Caveat:** medium *random-weight* buys real-**shape** coverage, not real-**weight** top-k
  selection fidelity (random near-ties mask selection bugs — see §10).
- **Add a static vLLM-divergence gate** (no GPU, no weights): a unit test importing vLLM's
  `deepseek_v2` indexer/RoPE construction and asserting our config choices on the documented
  divergence points — indexer `is_neox_style`/rotate-half (`deepseek_v2.py:1028`), q_a/kv_a
  layernorm eps (1e-6 vs vLLM 1e-5), mscale-never-on-indexer — **match HF and consciously differ
  from vLLM**, pinning §9 #5 as a tracked assertion now.
- Re-arm the Phase-1b S1 stress fixture at **medium dims** (experts not divisible by the mesh axis,
  e.g. 10/14 on an 8-chip TP axis) so it tiles non-trivially.

---

## 8. KV memory, long-context capacity, gather scaling

**Verdict: nearly silent on the whole reason DSA exists.**

KV per-token-per-layer: MLA latent = 576 numbers; indexer key = 128 (single head). 78 layers
(indexer "full" layers = 21 of 78 on the real schedule).

| Context | MLA latent (bf16) | Indexer all-78L (bf16) | Dual total | vLLM fp8 (full-only) |
|---|---|---|---|---|
| 32k | 2.74 GB | 0.61 GB | **3.35 GB** | 1.6 GB |
| 128k | 10.97 GB | 2.44 GB | **13.4 GB** | 6.6 GB |
| 1M | 87.75 GB | 19.5 GB | **107 GB** | 52.7 GB |

The bf16/all-layers plan is ~2× vLLM's fp8/full-only layout. **Binding constraint is weights** (744B
params; v6e HBM = 32 GB/chip in-code, `utils.py:181-190`; 31.25 GiB usable measured by the fork) —
the real model needs a multi-host slice sized for **weights + KV**.

- **No KV byte budget, no startup capacity gate** anywhere (§5/§10/§12/§13). The repo has the inputs
  (`tpu_worker.py:420` available-HBM; `get_attention_page_size_bytes`) but no upfront
  `max_model_len`-vs-HBM check — OOM surfaces at allocation (`kv_cache_manager.py:701`).
- **Indexer cache layer-set unresolved (3.7× swing):** vLLM allocates it only on "full" layers
  (`deepseek_v2.py:1023`, 21/78); the spec implies all 78. HF-version disagreement: transformers
  **5.9.0** builds the indexer **every** layer (keys cached per layer; "shared" is only a forward-time
  `skip_topk`); the spec's 5.12.1 claim (`indexer=None` on shared, `:406-407`) is **unverified
  locally** (5.12.1 venv was on the TPU box). Pin against 5.12.1 + the real checkpoint before sizing.
- **O(N) gather mis-sequenced:** the ported one-hot `onehot[K,N] @ kv[N,D]` reads every resident KV
  row (fork `kernels/sparse_attn/kernel.py:87-91`). It's length-driven, **validatable on the tiny
  config by sweeping N** — should be a **Phase-2 exit gate**, not deferred to Phase 5. The fork never
  stress-tested it (MAX_LEN=4096, called the O(N) read "negligible," deferred the DMA path to a
  large-N regime it never reached). vLLM's real `flashmla_sparse` does a true top-K DMA gather
  (~index_topk rows), so the bridge is *architecturally* different from the reference it must match.
- **No fp8 KV cache** — but for DSA it is a **capacity enabler** (halves the dominant MLA term), not
  just perf. **1M `max_model_len`** feasibility never addressed (block table sizes
  `ceil(max_model_len/block_size)` slots with no HBM check).

**Recommended changes**
- Add a **KV-capacity section** with explicit per-token-byte arithmetic + a **hard startup gate**
  bounding `max_model_len × max_num_seqs` against profiled HBM (the repo has none).
- **Resequence the gather N-sweep into Phase-2 exit** (tiny config): latency flat in N after the
  DMA-prefetch rewrite, or measure the O(N) cliff and schedule the rewrite before any long-context
  claim.
- **Resolve the indexer-cache layer-set** (21 vs 78) against 5.12.1; extend the weight-map golden
  test to the cache, not just the weights.
- Promote **fp8 KV** to a capacity lever with an early feasibility note (state the bf16-KV
  `max_model_len` ceiling; mirror vLLM's fp8 layouts MLA 656 B/tok, indexer 132 B/tok ue8m0).
- State the **real-model capacity reality** in §1/§11 (744B → multi-host slice; dev box validates
  correctness + shape-math, not residency).

---

## 9. What genuinely CANNOT be done without real weights (legitimately last)

1. **Indexer-RoPE HF-vs-vLLM ground truth** (§9 #5) — only trained `wq_b`/`wk` reveal which top-k
   convention the model expects.
2. **Real top-k *selection* fidelity** — random near-tie scores make any boundary swap "tie-tolerant";
   trained weights produce well-separated scores where a wrong relu/scale changes the selected set.
3. **Real-shape/-depth numerical reconciliation + re-measured bf16 floor; full 78-layer end-to-end
   generation** — needs real weights + depth.
4. **vLLM-fp8 ReLU-location confirmation** — fused in compiled deep_gemm (Triton ref readable, but
   final confirm wants the real GPU path).

Everything else the spec defers to real weights is **needlessly** deferred.

---

## 10. Cross-cutting methodology guardrails (the meta-bugs that burned the most sessions)

1. **Random small-weight repros actively mislead.** Both the S1 and decode-collapse arcs chased the
   *wrong layer for ~6 sessions* because `normal*0.02` weights produce bf16 near-ties that mimic
   structural bugs. Discriminator that settled it: **fp32-vs-bf16 A/B on confident/peaked logits**
   (fork `s1_cpu_dtype_disambig.py`: fp32 relErr 2.6e-4 vs bf16 0.227). **Any decode-correctness
   fixture must use peaked logits + fp32/bf16 A/B + cross-process (2 fresh engines) determinism**, or
   it false-greens/false-reds.
2. **CPU losslessness ≠ v6e losslessness** (an fp4→fp8 cast bit-exact on CPU, `5.2e5 max|Δ|` on v6e —
   unresolved). Add **on-device** cast-equivalence checks for any new dtype path.
3. **Trace-time-baked guards silently never fire** (a "slice to real length" guard compiled away
   because warmup used a full bucket so `L_real==T`). Any slice/clamp-to-real-length must key on a
   **traced** scalar.
4. **Weak smoke probes false-green.** The "Paris" probe passed while three decode bugs were live;
   `compute_logits`'s `nan_to_num` returns HTTP 200 on garbage; `completion_tokens`/`ends_clean` read
   healthy on corrupted output. **Never gate on liveness/health metrics.** Use sustained-decode +
   ragged-batch + cross-process determinism probes.

---

## 11. V4 stress-fixture suite to ADD (cheap, pre-giant-model, tiny config unless noted)

Beyond the 5 bugs the spec already captures (S1, `max_num_seqs>1` MoE corruption, E2003 one-hot
gather, gmm_v2 zero_init disproof, decode-collapse on empty attn_dp shards):

**A. Padded-bucket / decode (absent — highest priority):**
1. `L_real < T_pad` prefill→decode parity (`L_real∈{1,4,6,8}` × `T_pad∈{16,32,64}`) vs exact-length
   run — **the pad-token KV attractor** (fork `2ac33061`/`6245ea84`).
2. Decode position is **traced**: JIT cache grows by exactly 1 across 4 consecutive positions
   (fork `test_buffer_decode_jit_cache_hits_across_positions`).
3. Indexer-KV append + `-1`-sentinel wrap boundary parity vs recompute-from-scratch.
4. Decode `seq_len`/position off-by-one: KV write index == seq_len-1, position == cached_len
   (fork `5e26c85e`).

**B. Ragged-batch / multi-seq (absent):**
5. 2–3 concurrent prompts of different lengths → each per-seq slice **bit-identical** to its serial
   run (the 3-seq case catches `n_active` off-by-one; fork `e4cb7564`). Also exercises Phase-4
   `topk_indices` per-request-local isolation.

**C. Top-k / sentinel boundary (partially covered — add exact boundaries):**
6. Seq length **exactly == `index_topk`**, plus topk±1, seq==1, seq==0 (K-clamp / empty-tensor /
   dense==sparse hold at the boundary).
7. **All-`-1`-selection row** (first decode token / empty selection) → assert finite (non-NaN)
   softmax output (fork `test_sparse_attn_all_invalid`; high-value — the `-1` sentinel one-hot path
   keeps this).
8. **Post-topk tie-break leakage**: `-inf`-masked future positions present + ties possible → assert
   no selected index is causally invalid (the pre-topk mask alone was insufficient in the fork;
   `top_k` can still return a `-inf`-tied future index → post-topk re-mask).
9. `index_topk` not a multiple of 128 with M>1 in the kernel BlockSpec (tiny's `index_topk=64`).

**D. RoPE / freqs sizing (absent):**
10. Bucket > `max_model_len` RoPE reshape: prefill a prompt whose padded bucket exceeds the
    freqs-table length; assert reshape succeeds (fork `d495e5cf`).
11. `test_freqs_cis_matches_torch` host-NumPy **float64** vs torch (`original_seq_len∈{0,64}`) + the
    `lo==hi` ramp guard. (GLM has no YaRN/mscale² — consistent with the fork.)

**E. Donation / compiler (CPU-simulatable under `donate_argnums`, absent):**
12. **Donated-buffer decode determinism:** R1 then R2 on identical input must be byte-identical +
    HLO assertion that donated kv_caches survive with `input_output_alias`. Pre-empts the V4 decode
    **"Heisenbug"** — XLA write-elision under `donate_argnums` (~6 wasted sessions; re-arms if GLM
    writes indexer-KV via `at[].set` instead of a Pallas kernel).
13. Cached-artifact 2nd execution: run the same compiled prefill artifact twice → no SIGSEGV
    (fork: JIT-boundary reshard + donated kv → SIGSEGV without `with_sharding_constraint(b, P())`).
14. **No size-1 token-axis wsc-gather:** compile-only assertion (a `with_sharding_constraint` that
    gathers a size-1 decode token axis halts the core — proven ~8× in the fork). Directly constrains
    GLM's token-axis-sharded `topk_indices` (`P(ATTN_DATA, None)`, §12).

**F. Multi-device (Phase 1b/2, 8-chip):**
15. No-fallback ragged-gather compile check: a dynamic-range `ragged_gather`/`ragged_scatter` inside
    `shard_map`, compile-only — the in-code "falls back to plain gather" comment is **stale/wrong**;
    SparseCore is live on v6e and crashes. Relevant if the per-request-local `topk_indices` demux
    uses ragged gather/scatter.

**Other numerics to carry into the kernel port:** gather from bf16 then upcast the *gathered* result
(not full KV — bit-identical, halves traffic); `0.0 * (-inf) = NaN` in any multiply-by-zero barrier
(`where(cond, b+kv, b)` instead); defensive final `nan_to_num` at the sampler boundary **but never
relied on** (it's what masked the `max_num_seqs` corruption as HTTP 200).
