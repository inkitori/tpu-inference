# DeepSeek-V4-Flash on TPU v6e-8 (torchax) — Design Spec

**Date:** 2026-06-21
**Status:** Draft for review
**Branch:** `ds-v4-flash-torchax`
**Scope of this doc:** Overview of the whole port + **Phase 1 in full detail**. Later phases (2–5) are
sketched with exit gates and get their own specs when reached.

---

## 1. Goal & Constraints

Serve `deepseek-ai/DeepSeek-V4-Flash` (**text-only**) on a single **TPU v6e-8** (Trillium, 8 chips,
~31.25 GiB HBM/chip) through the **torchax / vLLM** path (`MODEL_IMPL_TYPE=vllm`), producing **accurate,
coherent** output, with the long-term goal of production serving on OpenRouter.

**Hard constraints:**
- **Quantization stays on.** Experts in **FP4** (`float4_e2m1fn`), linears in **block FP8** (e4m3, ue8m0
  scales). This is the only configuration that fits HBM (confirmed: 187/230 GiB at load). Requantizing
  experts to FP8 overflows HBM.
- **Never download full weights.** They are mounted read-only at `/home/enyouki/dsv4-weights` (gcsfuse of
  `gs://personal-mark-eu/vllm/hub/models--deepseek-ai--DeepSeek-V4-Flash/`).
- **torchax path only** (`MODEL_IMPL_TYPE=vllm`), not the pure-JAX model path.
- **Text-only.** Multimodal disabled. **MTP (multi-token prediction) stays disabled** throughout.
- **Port GPU-only / untraceable ops** to JAX/torchax lowerings that reproduce the pure-Torch reference's
  exact numerics, whenever one surfaces in a trace.
- **Correctness target:** long context (32K+). **Bar:** coherence first, then numeric parity vs the vLLM
  GPU/ROCm reference (per-component, then full-forward logits).
- When stuck, **consult tpu-inference PRs first**.

**Reference environment (verified):** jax/jaxlib 0.10.1, torch 2.10.0+cpu, torchax 0.0.13, ml_dtypes 0.5.4,
vLLM @ `ecf9d8352` (AMD/ROCm DSV4 variant forced). 8× TPU v6 lite reachable in-process.

---

## 2. Current State (as of 2026-06-21)

**Confirmed working (committed on `ds-v4-flash-torchax`):**
- Model constructs and **loads weights**; full param manifest registered for all 43 layers (the fix in
  `deepseek_v4_attention.py` calls the real `super().__init__()` with `torch.cuda.Event` monkey-patched).
- **HBM fits** (Risk-1 cleared): 187.14/229.97 GiB; FP4 experts preserved through the requant pipeline.
- **Mesh** active: DP-attention + expert-parallel across 8 chips.
- **KV cache** spec working: `fp8_ds_mla`, 576 B slots, `MLAAttentionSpec(model_version="deepseek_v4",
  alignment=576)`.
- **mHC ops** implemented as pure torch traced by torchax: `VllmHCHeadOp.forward_tpu` (gated RMS-norm
  collapse) and `VllmMHCFusedPostPreOp.forward_tpu` (composes `mhc_post_torch` + `mhc_pre_torch`).
- **Model wiring:** reuses vLLM's own `DeepseekV4ForCausalLM` **AMD variant**
  (`vllm.models.deepseek_v4.amd.model`), patched in via `_maybe_patch_for_deepseek_v4`
  (`vllm_model_wrapper.py:94`) and `patch_deepseek_v4_mla_cls()`. MTP buffer stubbed out.
- **5 Pallas kernels vendored** (from open PRs #2903/#2858) in
  `tpu_inference/kernels/experimental/deepseek_v4/`: `mla.py`, `mla_swa.py`, `compressor.py`,
  `compress_norm_rope.py`, `compress_store.py` — **complete impls with tests, but NOT wired** into the
  attention forward.

**Stubbed / blocked:**
- **`VllmDeepseekV4MLAAttention.forward` is a pass-through stub** (`deepseek_v4_attention.py:118`) →
  attention does nothing → output is currently incorrect. `forward_mqa`, `_o_proj` raise
  `NotImplementedError`; `process_weights_after_loading` is `pass`. **This is the central gap.**
- **First forward dies in `gmm_v2.py`** on the FP4 MoE matmul (see §3 — diagnosis corrected).
- **Indexer (#2905) not in tree**, and broken in the PR (missing `distribution`, wrong kwargs, cache-format
  mismatch).

**PR landscape:** #2903 / #2858 / #2905 / #2950 / #1906 are **all open, none merged**; no newer DSV4 work
supersedes the vendored copies. **Keep the vendored copies**; watch #2903 to reconcile `mla*.py` when it
lands. Closest **working precedent** on this path: **DeepSeek-R1 under `MODEL_IMPL_TYPE=vllm`**, which uses
`VllmMLAAttention` + `VllmMultiHeadLatentAttentionWrapper` (`layers/vllm/custom_ops/mla_attention.py`) with
a real `forward` registered via `register_oot` — the pattern Phase 1 should imitate for *how* to structure
the forward.

---

## 3. Corrected FP4-GMM Analysis (replaces the wrong handoff diagnosis)

> `DSV4_HANDOFF.md` claims "native FP4 matmul is v7-only" and proposes "port the unpack to integer/bitwise
> ops." **Both are wrong.** Verified against live v6e hardware and the jax matmul-support gate.

**What's actually true:**
- Native `float4_e2m1fn` systolic matmul exists on **no** TPU generation (the `is_matmul_supported` gate in
  jax `_src/pallas/mosaic/tpu_info.py` never lists `f4` in any branch, including `case 7`). On v6e even
  `f8e4m3fn × f8e4m3fn` is unsupported.
- **Real root cause:** `VllmMxfp4MoEMethod` requantizes experts with **`REQUANTIZED_BLOCK_SIZE = 512`**
  (`mxfp4.py:57,170-171`). v6e's `mxu_column_size = 256`. Because `512 ≥ 256`,
  `should_dequantize_before_matmul` returns **False** (`gmm_v2.py:~198-215`) → the kernel takes the
  **quantized-matmul** path → tries `block_rhs.astype(f8e4m3fn)` on a native-`f4` vector (`gmm_v2.py:~500`,
  guarded by the failing `is_matmul_supported` at `~498`) → Mosaic on v6e cannot compile the
  `vector<...xf4E2M1FN>` unpack. The failing MLIR op is the sub-byte `tpu.unpack_subelements` (the bitcast
  unpack at `gmm_v2.py:~390-392`).
- **Working precedent in this same repo:** the **NVFP4** path uses block size **16** (`nvfp4.py:444`,
  `< 256`) → `should_dequantize_before_matmul` True → the **dequant-in-VMEM** branch (`gmm_v2.py:~398-409`)
  casts `f4 → bf16` inside VMEM and runs a clean **bf16 × bf16** matmul, never calling `is_matmul_supported`.
  Same native `float4_e2m1fn` weights, same bitcast unpack — and it works on v6e today.

**Corrected fix (Phase 1):** make the mxfp4 path take the dequant-in-VMEM branch, exactly like NVFP4 — set
the mxfp4 requant block size **below 256** (e.g. 32, the native MXFP4 group) in
`VllmMxfp4MoEMethod.process_weights_after_loading` (`mxfp4.py:170-171`). One-knob change; experts stay FP4 in
HBM; numerics are correct because the e2m1 decode is handled by the native dtype in VMEM (the same mechanism
as `u8_unpack_e2m1`, `quantization/__init__.py:54-60`).

**Do NOT** implement the handoff's "unpack nibbles with integer/bitwise ops" — that conflates int4 and e2m1
semantics and corrupts FP4 numerics.

**Validate empirically as step 1 of Phase 1:**
1. Confirm DSV4 dispatches to `VllmMxfp4MoEMethod` (not `Nvfp4`) — `deepseek_v4_fp8.py:76` indicates yes.
2. Use the **AOT compile gate** (§6) to confirm the block-size change makes `gmm_v2` compile on v6e *before*
   any full run.
3. Confirm sub-256 block accuracy/perf is acceptable (smaller blocks = finer scales = better accuracy, more
   dequant overhead; NVFP4 at block 16 is direct evidence it's viable).

**Corrections to other handoff levers:** `MOE_REQUANTIZE_WEIGHT_DTYPE=fp4` is **inert** for V4 (only read in
the fp8 path); what keeps experts FP4 is the checkpoint's `expert_dtype`. PR #1906's pre-quant load is a
**no-op for V4** (DSV3/fp8 path only). Don't rely on either.

---

## 4. Architecture Overview

**Model facts** (from checkpoint `config.json`): 43 transformer layers (+1 MTP, disabled). `hidden=4096`,
`num_attention_heads=64`, `head_dim=512` (`nope=448` + `rope=64`), `kv_heads=1` (shared 512-d latent),
`q_lora_rank=1024`, `o_lora_rank=1024`, `o_groups=8`, `sliding_window=128`, `rms_norm_eps=1e-6`. MoE: **256
routed + 1 shared expert, top-6**, `moe_inter=2048`, `sqrtsoftplus` scoring + `noaux_tc` selection,
hash-routing (`tid2eid`) in the first 3 layers. RoPE: dual-theta GPT-J interleaved — `rope_theta=10000`
(dense) vs `compress_rope_theta=160000` (compressed), YaRN factor 16. Vocab 129,280. Context 4M native.

**The three attention regimes** (per-layer `compress_ratio`, base clamps to `max(1, ratio)`):
- **Dense (ratio→1): layers 0, 1, 42** (3 layers) — pure sliding-window-128 causal attention; no
  compressor, no indexer.
- **CSA / lightning (ratio 4): 20 layers** — lightning **indexer** selects top-512 tokens; attention attends
  `SWA-window ∪ topk`.
- **HCA / compress (ratio 128): 20 layers** — **compressor** writes a compressed KV cache; attention attends
  `SWA-window ∪ compressed` via `kv_lens_to_attend`.

(The handoff's "41 of 43" was wrong; it's **3 dense / 20 CSA / 20 HCA**.)

**Attention dataflow** (reference: vLLM `deepseek_v4/attention.py` + `amd/rocm.py`; shapes `N`=tokens,
`H=64`, `D=512`, `Dr=64`, `Dn=448`):

| Step | Op | Shapes / notes |
|---|---|---|
| 0 | prealloc `o_padded` | `[N, padded_heads=64, 512]` |
| 1 | `fused_wqa_wkv(hidden)` → `qr_kv` | `[N, 1024+512]`; FP8-block `MergedColumnParallelLinear`, `disable_tp` (replicated) |
| 2 | split `[1024,512]`→`qr,kv`; `fused_q_kv_rmsnorm(qr,kv,q_norm.w,kv_norm.w,eps)` | `q_norm`=RMSNorm(1024), `kv_norm`=RMSNorm(512), **with** weight; out bf16 |
| 3 | `wq_b(qr)` → `[N,64,512]` | `ColumnParallelLinear`, FP8-block |
| 4 | fused: per-head **weight-free** RMSNorm over 512 + **GPT-J interleaved RoPE** (last 64) + KV RoPE + fp8 quant + paged insert | use `self.rotary_emb` (per-layer theta); `is_neox_style=False`; `cos_sin_cache [max_pos,64]` (cos\|sin) |
| 5 | compressor / indexer | **sparse layers only — skipped in Phase 1** |
| 6 | `forward_mqa(q,kv,positions,o_padded)` → attention kernel | writes `o_padded [N,64,512]` |
| 7 | `_o_proj` | inverse GPT-J RoPE (last 64, fp32) → group view `[N,8,heads_per_group·512]` → `wo_a` BMM `einsum("tgd,grd->tgr")` → `[N,8,1024]` → `wo_b` `RowParallel(8192→4096)` → `[N,4096]` |

**Attention sink:** per-head `attn_sink [padded_heads]` fp32 (`-inf` default, first `n_local_heads` filled by
loader), folded into softmax as an extra logit column then dropped. `mla_swa` has **no** sink argument;
`mla.py` does (`attention_sinks` arg).

---

## 5. Phased Roadmap

Each phase has a single exit gate. Phase 0 (test harness) is built first and reused by all.

**Phase 0 — Test harness** (see §6). Synthetic small-config factory, multi-chip mesh fixtures, pure-torch
reference + golden capture/replay, first-divergence locator, AOT compile gate. **Exit:** harness can run a
synthetic DSV4 mini-model on the 8-chip mesh and diff any component against a golden.

**Phase 1 — First light: FP4 GMM fix + dense attention** (see §7). **Exit:** coherent output on a real-weight
prompt of **≤128 tokens** (one sliding window), and the AOT compile gate passes for the MoE GMM.

**Phase 2 — Short-context numeric parity (+ attention sinks).** Add per-head sinks (switch dense layers to
`mla.py` which takes `attention_sinks`, or add a sink column to `mla_swa`). Per-component parity vs the vLLM
references (qnorm-rope-kv-insert, inverse-rope o_proj, sparse softmax+sink, router `sqrtsoftplus`+`noaux_tc`,
mHC), then full-forward short-context logit parity. **Exit:** short-context logits match the GPU reference
within the §6 tolerance ladder.

**Phase 3 — Compressor (HCA / 20 ratio-128 layers).** Wire `compressor_forward` into the step (project
`kv_score`, write compressed KV + state caches). Build the per-layer `compress_ratio` **dispatcher** (no PR
provides it): HCA → compressor + `mla.py` with `kv_lens_to_attend`; dense → `mla_swa`. Parity vs
`test_compressor_kv_cache.py`, then medium-context full-forward. **Exit:** long-range context via compression
is correct.

**Phase 4 — Lightning indexer (CSA / 20 ratio-4 layers) — hardest, main schedule risk.** Bring in #2905,
rework the broken bits (missing `distribution`, wrong kwargs, cache-format mismatch), with the **power-of-two
scale rounding** flagged in PR review. Compute top-512 selection, fold `topk_indices` + the SWA window into
one ragged set for `mla.py` (`combine_topk_swa_indices`). Parity vs `test_rocm_triton_attn_dsv4.py`, then
full-forward at 32K+. **Exit:** all 43 layers correct → full long context.

**Phase 5 — Production serving & hardening.** Move from `offline_inference.py` to vLLM serve
(OpenAI-compatible). Throughput / batching / chunked-prefill, kernel perf (the deferred "batch decode q
tokens" TODOs in `mla*.py`), stability under load, full eval suite (MMLU/GSM8K) within tolerance, OpenRouter
wiring. **Exit:** production-ready serving.

> **Coherence caveat:** dense-only (Phase 1) is coherent **only for ≤128-token sequences**. Beyond the window,
> the 20+20 sparse layers go blind to long-range context → fluent-but-wrong output **with no error** (this is
> the silent-wrongness risk). Production coherence is not reached until **Phase 4**. Phases 1–2 are validation
> milestones; Phase 3 is partial long-range; Phase 4 delivers the target.

---

## 6. Testing Strategy (Phase 0)

**Two non-negotiable principles** (also in repo `CLAUDE.md`):
1. **Test with synthetic small-config weights — do not load the full model for routine testing.** Reserve the
   full (~187 GiB) load **only** for milestone *coherence* smokes.
2. **Test on the real multi-chip mesh from the start** (TP=8 + expert-parallel + DP-attention) so
   sharding/collective bugs surface immediately.

### 6.1 The synthetic / real split
- **Numerical accuracy & parity → synthetic small-config weights.** Parity is implementation-vs-reference on
  *identical* weights, so trained values are irrelevant. Run the reference and the TPU impl on the **same**
  synthetic weights+inputs; diff within tolerance.
- **Coherence → real weights** (random weights produce gibberish by definition). Milestone gates only.
- **Real-weight component confidence → single-layer extraction** (load one layer's weights, not 43×257),
  using `SkipLayersModelLoaderForTest` / `num_layers_to_load_for_test`.

### 6.2 Synthetic config constraints
A DSV4 *mini* config must (a) include all three regimes (≥1 dense, ≥1 CSA, ≥1 HCA layer), (b) use the **real
quant formats** (FP4 e2m1 experts, FP8 e4m3 block linears, ue8m0 scales), and (c) have dims **divisible by the
mesh**: `num_experts % 8 == 0`, `num_heads % 8 == 0`, so it shards identically to production. Suggested:
~4 layers, 16–32 experts, hidden 256–512, 8–16 heads.

### 6.3 Harness components (highest leverage first)

**A. AOT compile gate — `jit(f).lower(avals).compile()`.** Feed `jax.ShapeDtypeStruct` dummies (pattern at
`runner/compilation_manager.py:96`); `.compile()` runs the Mosaic passes and **surfaces the FP4 GMM
MosaicError with no weights, in seconds**. Wrap each kernel/layer. Pair with `jax.make_jaxpr` to inspect the
op graph. **`.lower()` alone is insufficient** — it only serializes; you must call `.compile()`. Use as a CI
pre-flight before any full run.

**B. First-divergence locator — torchax `debug_accuracy_for_each_op`.** Built into torchax 0.0.13: set
`torchax.default_env().config.debug_accuracy_for_each_op = True`; every torch op is re-run on CPU and
`allclose`-compared to the JAX result, dropping into pdb at the **first** diverging op. Zero-build,
op-granularity. **Eager-mode only** — run the model **un-jitted** (around `step_fun_impl`-style calls, not the
jitted `jit_step_func`). The hardcoded `atol=1e-3` is wrong for FP4/FP8 (too tight) and bf16 norms — expect to
patch the threshold or filter ops. Does **not** see inside Pallas kernels (op = whole kernel call).

**C. Pure-torch reference + golden capture/replay (must BUILD — does not exist).** The vLLM AMD model **cannot
run on CPU** (`aten::mm.dtype` unimplemented, custom CUDA ops absent, `float4_e2m1fn.float()` raises). Build a
small **pure-torch bf16/fp32 reference** reusing vLLM's dequant utils: `break_fp4_bytes`
(`nvfp4_emulation_utils.py:328`, CPU-runnable) for FP4 experts, `_upcast_e8m0_to_fp32` (`fp8_utils.py:1049`)
for ue8m0 scales, `.float()` for FP8 linears; standard RMSNorm + GPT-J/YaRN RoPE + dense MLA
`(Q@Kᵀ·scale).softmax@V` + pure-torch MoE gather/topk + `mhc_pre_torch`/`mhc_post_torch` (note: `HCHeadOp` has
**no** `forward_native` — reimplement the head collapse in torch). Capture goldens at clean module boundaries
(attn_norm/attn/ffn_norm/ffn/mhc_*/hc_head) and **persist** them (`.npz`); TPU tests replay against the saved
goldens — no reference re-run, no full load.

**D. Pallas `interpret=True` — kernel math off-TPU.** `pl.pallas_call(..., interpret=True)` runs the numpy ref
path for the **new** DSV4 kernels (compressor, indexer top-k, sparse gather, sinks off-by-one, grouped
o_proj). `pl.debug_print` works inside. **Caveats:** validates *math only*, not tiling/VMEM/layout/compile (a
kernel can pass interpret and still fail Mosaic — exactly the FP4 case); and **`gmm_v2` is TPU-only even under
interpret** (it calls `pltpu.get_tpu_info()` at trace time).

**E. CPU-vs-TPU bisection — three explicit lanes.** (1) pure-JAX-on-CPU for module math (mHC/RoPE/norms/
routing/projections) — bit-identical to TPU; (2) **8-device host CPU mesh**
(`--xla_force_host_platform_device_count=8`) for TP/EP partition-spec + collective correctness; (3)
`interpret=True` for MLA-kernel math. **`gmm_v2` and all Mosaic-compile bugs stay TPU-only** — CPU bisection
**cannot** reproduce the current FP4 blocker; it's for *future* math/sharding bugs.

**F. Reference-free invariants — `checkify` + asserts + `jax_debug_nans`.** In-jit invariant checks via
`checkify` (`index_checks | nan_checks` for the indexer's top-512 gather OOB; custom checks for router top-k
renorm sum, Sinkhorn doubly-stochastic row/col≈1, attention probs≤1). `JAX_DEBUG_NANS=1` as a first-NaN
tripwire (sqrtsoftplus overflow, all-`-inf` softmax rows, Sinkhorn div-by-zero). **All three stop at the
Pallas boundary** — guard kernel inputs/outputs, and use `pl.debug_print` for in-kernel NaNs.

**G. Metamorphic / property tests (no golden needed).** mHC Sinkhorn double-stochasticity (4×4, 20 iters,
`hc_post_alpha=2.0`); **expert permutation invariance** (permuting expert IDs+weights leaves MoE output
invariant — catches EP gather/scatter index bugs); causality/SWA masking (future tokens can't change earlier
outputs; ≤128 tok ⇒ window covers whole sequence); RoPE per-pair L2 norm preservation.

**H. Sharding oracle (corrected).** Plain `jax.eval_shape` returns `.sharding = None` — **not** a sharding
oracle. Use `jax.jit(f, out_shardings=sh).eval_shape(...)` or assert on the **committed** output:
`out.sharding == NamedSharding(mesh, P(...))`. **Shard-equivalence is exact here**: with
`jax_threefry_partitionable=True` (default), 8-way-sharded RNG is bit-identical to single-device — so
single-vs-sharded tests can assert exact equality. **Assert that flag** at the top of such tests.

**I. Mosaic / HLO dumps for diagnosis.** `LIBTPU_INIT_ARGS="--xla_mosaic_dump_to=/path
--xla_mosaic_enable_dump_debug_info"` (a **libtpu** flag, set before process start) shows which MLIR op Mosaic
chokes on; `XLA_FLAGS="--xla_dump_to=/path --xla_dump_hlo_as_text"` for HLO. Use to confirm the FP4 fix routes
through the dequant-in-VMEM path.

**J. Recompile / perf guard.** Turn on `VLLM_XLA_CHECK_RECOMPILATION=1` (already wired,
`compilation_manager.py:62`) to catch accidental recompiles / silent fallbacks from ragged/dynamic shapes.
Persistent compilation cache is already enabled (helps 2nd-run warmup, not the I/O-bound 12-min load). Reserve
the full load for coherence smokes.

**K. `poison_tpu_memory()`** (`tests/test_utils.py:240-264`) before a new kernel to surface uninitialized
VMEM/SMEM reads as NaN — high value for the compressor/indexer kernels with manual scratch.

### 6.4 Tolerance ladder
- bf16 exact-math ops (norms, projections sans quant): `rtol=atol≈1e-5`.
- RoPE: near bit-exact (reference asserts bit-exact on the RoPE portion).
- FP8 block GEMM / FP4 MoE GEMM: `rtol=atol≈0.1` (matches `mla_test.py:394`).
- Full-forward logits: per-token top-1 agreement + loose logit `atol`; tighten as components are proven.

### 6.5 Reuse inventory (don't reinvent)
| Need | Reuse | Path |
|---|---|---|
| Random small-config weights | `JaxDummyModelLoader` | `models/jax/utils/weight_utils.py:1079-1147` (verify/adapt for the vllm/torchax loader) |
| Load only first N layers | `SkipLayersModelLoaderForTest`, `num_layers_to_load_for_test` | `tests/models/jax/conftest.py:84-109` |
| Multi-chip mesh factory | `get_spmd_mesh(num_devices, enable_attn_dp)` | `tests/layers/common/utils.py:26-42` |
| Kernel test skeleton | synthetic inputs + ref + `assert_allclose` | `tests/kernels/deepseek_v4/mla_test.py:200-395` |
| Compile cache opt-out | `@pytest.mark.disable_jax_cache` | `tests/conftest.py:100-154` |
| Uninitialized-mem detector | `poison_tpu_memory()` | `tests/test_utils.py:240-264` |
| Named numeric references | router/sink/inv-rope/compressor/fp8-GEMM/mxfp4-dequant | see §8 reference list |
| **Golden persist/replay** | **does not exist — BUILD** | new `tests/.../golden_*.py` |

---

## 7. Phase 1 — Detailed Design

**Objective:** get the first real forward to run and produce **coherent ≤128-token output**, proving the full
pipeline (FP4 MoE, mHC, attention dataflow, o_proj) end-to-end. **Not** in scope: attention sinks, long
context, numeric parity (those are Phase 2+).

### 7.1 Deliverable 1 — FP4 GMM fix
Change the mxfp4 requant block size to `< 256` (e.g. 32) in `VllmMxfp4MoEMethod.process_weights_after_loading`
(`mxfp4.py:170-171`) so the GMM takes the dequant-in-VMEM branch (§3). **Validation order:** (1) AOT compile
gate confirms `gmm_v2` compiles on v6e with the new block size; (2) Mosaic dump confirms it routes through the
bitcast→bf16 path, not the f8 cast; (3) a `gmm_v2` micro-test with mxfp4 experts at block 32 on the real mesh
passes; (4) confirm experts still fit HBM and the dispatch is `VllmMxfp4MoEMethod`.

### 7.2 Deliverable 2 — attention forward (dense-SWA, all layers)
Implement in `tpu_inference/layers/vllm/custom_ops/deepseek_v4_attention.py` (follow the R1
`VllmMLAAttention` structure for *how* to write it; keep the existing class-patch wiring since loading already
works):

- **`forward`** — orchestrate steps 0–4, 6–7 from §4. **Route all 43 layers through `mla_swa`** (skip
  step 5). Use `self.rotary_emb` per layer (correct per-layer theta even when forced dense).
- **`forward_mqa(q, kv, positions, o_padded)`** — call
  `mla_sliding_window_ragged_paged_attention` (`mla_swa.py:932`):
  ```
  (q [max_tok, num_q_heads, head_dim] bf16,
   new_kv [max_tok, head_dim] bf16,            # raw bf16; kernel quantizes internally
   cache_kv [num_pages, ...] uint8 (donated),  # DSv4 FP8
   kv_lens i32, page_indices i32, cu_q_lens i32, distribution i32[3],
   *, sm_scale, sliding_window=128, logical_page_size, mask_value,
      chunk_prefill_size, num_kv_pages_per_block, num_queries_per_block,
      vmem_limit_bytes, unnormalized_output=False)
   -> (out [max_tok, num_q_heads, head_dim], updated_cache_kv, l, m)
  ```
  The kernel **quantizes its own KV** and **writes the cache** and applies **both causal and sliding-window
  masks** internally. Handle the decode/prefill split as `rocm.py` does. **No sink** in this kernel (Phase 2).
- **`_o_proj`** — inverse GPT-J RoPE on the last 64 dims per head in fp32 (`x'=x·cos+y·sin`,
  `y'=y·cos−x·sin`; even/odd interleaved) → group view `[N, 8, heads_per_group·512]` → `wo_a` BMM
  `einsum("tgd,grd->tgr")` (`bmm_batch_size = n_local_groups = 8`) → `[N, 8, 1024]` → `wo_b`
  `RowParallelLinear(8192→4096)` on the flattened result.
- **`process_weights_after_loading`** — implement any required weight reshaping/sharding for `wo_a`'s BMM and
  the LoRA projections (currently `pass`).

### 7.3 Why dense-only is coherent for ≤128 tokens
For `pos < 128` the SWA mask never fires, so `mla_swa` attends to the full causal prefix — mathematically the
same as full attention. The SWA cache holds the real (uncompressed) per-token KV for **every** layer; the
compressor/indexer write to **separate** caches the SWA path never reads, so routing sparse layers through
`mla_swa` reads correct, complete KV. Within one window the sparse contribution is a subset of what SWA already
covers → redundant. (The one fidelity gap vs reference is the missing sink → small softmax drift, coherent but
not parity — closed in Phase 2.)

### 7.4 Phase 1 testing (uses the Phase 0 harness)
1. **AOT compile gate** on the full forward with synthetic mini-config — catches untraceable/Mosaic errors
   before any run.
2. **Component parity** (synthetic, multi-chip) vs pure-torch goldens: `fused_wqa_wkv`, q/kv-norm, step-4
   qnorm+RoPE+insert, `mla_swa` math (vs `interpret=True` + the dense-MLA reference), `_o_proj` (vs
   `test_fused_inv_rope_fp8_quant.py:177`), MoE (vs router ref + `break_fp4_bytes` dequant), mHC.
3. **First-divergence locator** (`debug_accuracy_for_each_op`, eager) on the mini-model to catch any silent op
   mismatch end-to-end.
4. **Invariants:** causality/SWA masking, RoPE norm preservation, MoE top-6, output sharding spec.
5. **Coherence smoke (milestone gate):** real weights, a ≤128-token prompt, output must read sensibly.

### 7.5 Phase 1 risks
- **Block-size accuracy/perf** unknown until measured (mitigated: NVFP4 precedent at block 16).
- **`mla_swa` cache/quant assumptions** must match the KV-cache spec (`fp8_ds_mla`, 576 B); verify
  `logical_page_size`/alignment wiring.
- **Decode vs prefill** ragged metadata (`distribution`, `cu_q_lens`, `page_indices`) must be built correctly
  for the dense path — most likely source of first-forward bugs.
- **Silent wrongness > 128 tokens** — keep Phase 1 smokes strictly ≤128 tokens; add a dev-time guard/log when
  a sequence exceeds the window in dense-only mode.

---

## 8. Open Questions & Risks (whole project)

- **Indexer (#2905) rework (Phase 4)** is the largest unknown — broken in the PR, on-chip top-k, scale
  power-of-two rounding, ragged `combine_topk_swa_indices`. Main schedule risk.
- **Attention-sink numerics (Phase 2):** add a sink column to `mla_swa` vs switch dense layers to `mla.py`
  (which already takes `attention_sinks`) — decide in Phase 2.
- **mHC has no torch reference** — must be reproduced from vLLM `kernels/mhc/tilelang*` (Sinkhorn 20 iters,
  `hc_post_alpha=2.0`); covered partly by the double-stochasticity invariant until a golden exists.
- **`JaxDummyModelLoader` is the JAX-path loader** — confirm/adapt a dummy-weight path for the vllm/torchax
  loader, or extract weights via `SkipLayersModelLoaderForTest`.
- **Perf (Phase 5):** the `mla*.py` "batch decode q tokens" TODOs and v6e's bf16-upcast MXU throughput.
- **Hash routing** (first 3 layers, `tid2eid`) and **shared expert** correctness — verify in MoE parity.

### Named numeric references (for parity tests)
- Router `sqrtsoftplus`+`noaux_tc`: `vllm/.../fused_moe/router/fused_topk_bias_router.py:59`
- Sparse-MLA softmax + sink: `vllm/tests/kernels/attention/test_rocm_triton_attn_dsv4.py:77-87`
- Inverse-RoPE `o_proj` (+ fp8 quant): `vllm/tests/kernels/test_fused_inv_rope_fp8_quant.py:177`
- KV compressor: `vllm/tests/kernels/test_compressor_kv_cache.py:460`
- FP8 block GEMM: `vllm/tests/kernels/quant_utils.py:91`
- MXFP4 dequant (`[0,.5,1,1.5,2,3,4,6]`): `vllm/.../nvfp4_emulation_utils.py:328`
- Q-path RMSNorm-no-weight + RoPE + KV-insert: `vllm/tests/.../test_fused_deepseek_v4_qnorm_rope_kv_insert.py`
