# DeepSeek-V4-Flash on TPU v6e-8 (torchax / MODEL_IMPL_TYPE=vllm) — Working Notes

Status legend: ✅ confirmed · ⚠️ risk/open · 🔨 to build/integrate

## Environment (confirmed)
- TPU: v6e-8, 8 chips, 1 core/chip (`jax 0.10.1`, devices = "TPU v6 lite").
- venv: `/home/enyouki/.venv` (must `source` it; system python lacks jax).
- vLLM: editable at `/home/enyouki/vllm`, `0.23.1rc1.dev193+gecf9d8352`. **Already contains `DeepseekV4ForCausalLM`** (registry → `vllm.models.deepseek_v4`, nvidia branch canonical).
- tpu-inference: editable at `/home/enyouki/tpu-inference`, current branch `ds-v4-flash-torchax`.
- Weights: gcsfuse mount `personal-mark-eu` at `/tmp/gcs/bucket` with `--only-dir vllm` → model at
  `/tmp/gcs/bucket/hub/models--deepseek-ai--DeepSeek-V4-Flash/`. refs/main snapshot = `553034d7…`.
  46 safetensors shards (~155 GB, FP4 experts + FP8 linears). ⚠️ **Mount is root-only (uid=0, no allow_other)** —
  `enyouki` gets EPERM. Must remount with `allow_other` (or run server as root) before loading. Read-only.
- Set `HF_HOME=/tmp/gcs/bucket` (so `$HF_HOME/hub/models--…` resolves). Never download full weights.

## Config (from bucket config.json — matches spec)
hidden 4096 · 43 layers all-MoE · vocab 129280 · head_dim 512 (qk_rope 64 → nope 448) · 64 heads · 1 KV head ·
q_lora 1024 · o_lora 1024 · o_groups 8 · index_head_dim 128 · index_n_heads 64 · index_topk 512 · sliding_window 128 ·
256 routed experts top-6 + 1 shared · moe_inter 2048 · scoring_func sqrtsoftplus · topk_method noaux_tc ·
routed_scaling 1.5 · norm_topk_prob true · swiglu_limit 10 · num_hash_layers 3 · hc_mult 4 · hc_sinkhorn_iters 20 ·
fp8 e4m3 block [128,128] ue8m0 + expert_dtype fp4 · rope_theta 10000 / compress_rope_theta 160000 · YaRN factor 16,
orig_max_pos 65536 · MTP num_nextn_predict_layers 1 (skip) · compress_ratios[44] ∈ {0,4,128} (last entry = MTP layer).
Per layer: `compress_ratio = max(1, compress_ratios[i])` → checkpoint 0 ⇒ ratio 1 (dense SWA).

## CORRECTED SCOPE (vs spec) — this is integration, not greenfield
The spec undersold reuse. There is a real DSV4 PR cluster on **vllm-project/tpu-inference** (upstream of the
`inkitori` fork at origin). The hard kernels already have implementations.

### Git reality
- `origin` = `git@github.com:inkitori/tpu-inference.git` (fork). No `upstream` remote yet.
- Local `main`/`ds-v4-flash-torchax`/`glm5.2-dsa-torchax` all at **stale** `2d4e886f` — does NOT contain merged DSV4 skeleton.
- Merged DSV4 skeleton commits exist in local object store (reachable from `glm5.2-dsa`, which user said to IGNORE):
  `ef69b9b6` #2792 skeleton · `0dd4e0ec` #2829 vLLM-refactor import fix · `520f1b14` #2839 codeowners.
  Clean way to get them: add upstream `vllm-project/tpu-inference`, fetch main.

### PR inventory (themes → PRs)
- **Merged skeleton (foundation):** #2792 (`custom_ops/deepseek_v4_attention.py`, `custom_ops/mhc.py`,
  `quantization/deepseek_v4_fp8.py`, wires model into wrapper, `runner/kv_cache_manager.py`; confirmed boots a server),
  #2829 (adapt to vLLM `DeepseekV4Attention` collapse — old `DeepseekV4MLA*` symbols are stale), #2839.
- **Open kernel PRs (the hard parts):**
  - #2903 main MLA attention kernels: `kernels/experimental/deepseek_v4/mla_swa.py` (SWA cache write+attend) + `mla.py`
    (combine SWA out with compressed-KV cache; reads compressed cache, doesn't write it).
  - #2858 KV compressor (writes compressed cache): `kernels/experimental/deepseek_v4/{compressor,compress_norm_rope,compress_store}.py`.
  - #2905 Lightning Indexer: `kernels/experimental/deepseek_v4/streamindex_topk.py` (on-chip running top-k in VMEM) +
    `custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py` (torchax↔jax interop, metadata bucketing). (#2811 = older, superseded — ignore.)
  - #2950 mHC head-collapse op TPU impl (`VllmHCHeadOp.forward_tpu`) extends `custom_ops/mhc.py`.
  - These 3 kernel PRs each add `kernels/experimental/deepseek_v4/__init__.py` → conflict there; reconcile on integrate.
  - Runtime coupling: #2858 writes cache → #2903 mla.py reads; #2905 indexer feeds top-k.
- **FP4 MoE stack:** merged #1756 (FP8 method can requant→FP4, patches `megablox/gmm_v2.py` for FP4 GMM),
  #2894 (`MOE_REQUANTIZE_CLIP_PERCENTILE`), #2719 (fast NVFP4 path `quantization/nvfp4.py`);
  open #1906 (pre-quantized FP4 load: `MOE_SKIP_REQUANTIZE`, uint8-packed fp4, `float4_e2m1fn` unpack; DSV3 converter — V4 variant likely needed).
- **Sparse MLA reference (closed, not merged):** #2457 top-k MLA on `mla/v1` (current is `mla/v2`).
- **Reference only (user says ignore branch):** `glm5.2-dsa` has DSA design specs `docs/superpowers/specs/glm5.2-dsa/phase-2.md`
  + `custom_ops/mla_attention.py` sparse hooks — useful design ref for indexer→topk→sparse-MLA dataflow.

## torchax/vLLM path mechanics (confirmed)
- Select path: `MODEL_IMPL_TYPE=vllm` (env; under `auto`, DeepSeek routes to JAX path — must force vllm).
  `models/common/model_loader.py:get_model()` → `get_vllm_model()` → `VllmModelWrapper`.
- Tracing boundary: `models/vllm/vllm_model_wrapper.py` — builds real vLLM module on CPU, wraps `_VllmRunner`,
  `shard_model_to_tpu` (t2j + sharded torchax tensors), `jit_step_func()` runs `torch.func.functional_call`
  under `torchax.default_env()` → every torch op lowered to JAX. KV caches via `vllm_model_wrapper_context`.
- Two override mechanisms:
  - `@register_function` (torchax op-dispatch override) — used for SDPA only today.
  - `register_oot` (vLLM out-of-tree class replacement) — Linear/Embedding/FusedMoE/RoPE/MLA/GDN swapped here
    (`layers/vllm/custom_ops/*`, fired from `layers/vllm/__init__.py`).
- Registration of DeepseekV4 is **automatic** via vLLM ModelRegistry (no tpu-inference allowlist). Real porting
  surface = the custom_ops + kernels, not a registry.
- ⚠️ The vLLM model is almost entirely fused custom ops (CUDA `torch.ops._C.*`, `torch.ops.vllm.*`, DeepGEMM,
  FlashMLA, cute-dsl, Triton). torchax can't trace these → each must be replaced by a JAX impl (register_function /
  model patch) or the corresponding tpu_inference custom_op/kernel. This is what the PR cluster provides.

### Reusable kernels (entry points)
- MLA v2 Pallas: `kernels/mla/v2/kernel.py:1398 mla_ragged_paged_attention(...)`; backend `layers/vllm/backends/flash_attn_mla.py`.
- Block-FP8 linear: `kernels/quantized_matmul/blockwise_kernel.py:25 quantized_matmul_kernel(...)`.
- MoE GMM: `kernels/megablox/gmm_v2.py:1130 gmm_v2(...)`; layer `layers/common/fused_moe_gmm.py:426 fused_moe_func(...)`.
  ⚠️ gmm_v2 has no first-class `float4_e2m1fn`; fp4 experts flow via mxfp4 method / #1756 requant patch — validate.
- e8m0/mxfp4 helpers: `layers/common/quantization/__init__.py` (`e8m0_to_fp32`, `u8_unpack_e2m1`,
  `quantize_tensor_to_mxfp4_packed`, `dequantize_tensor_from_mxfp4_packed`; MXFP4_BLOCK_SIZE=32).

### ⚠️ HBM trap (confirmed)
`MOE_REQUANTIZE_WEIGHT_DTYPE` default `"float8_e4m3fn"` → `process_fp8_moe_weights`
(`layers/common/process_weights/moe_weights.py:513`) dequants experts to fp8 ⇒ ~33 GiB/chip ⇒ won't fit ~29.8 usable.
Must keep experts FP4: route via mxfp4 path or set `MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn`/`fp4` (or #1906 `MOE_SKIP_REQUANTIZE`).

## Numeric ground-truth references (vLLM torch impls — the parity gold)
- fp8 block GEMM: `tests/kernels/quant_utils.py:91 native_w8a8_block_matmul` (matmul-then-scale, fp32 accum).
- MXFP4 dequant: `nvfp4_emulation_utils.py:328 break_fp4_bytes` (table [0,.5,1,1.5,2,3,4,6]) + `fp8_utils.py:1049 _upcast_e8m0_to_fp32`.
- router sqrtsoftplus+noaux_tc: `fused_moe/router/fused_topk_bias_router.py:59` (`scores=sqrt(softplus(logits))`,
  bias selection-only, gather unbiased, renorm, ×routed_scaling last).
- sparse-MLA softmax+sink: `tests/kernels/attention/test_rocm_triton_attn_dsv4.py:77-87` (sink = extra pre-softmax
  unscaled logit whose mass is dropped from output; off-by-one softmax).
- inverse-RoPE o_proj: `tests/kernels/test_fused_inv_rope_fp8_quant.py:177`.
- KV compressor: `tests/kernels/test_compressor_kv_cache.py:460 _reference_kv_compress_norm_rope`.
- mHC: no torch ref — reproduce from vLLM `model_executor/kernels/mhc/tilelang*` (Sinkhorn `hc_sinkhorn_iters=20`,
  `hc_post_alpha=2.0`, streams `(T, hc_mult, H)`).
- RoPE: GPT-J interleaved (`is_neox_style=False`), YaRN, base = compress_rope_theta when ratio>1 else rope_theta;
  mscale collapses to 1.0. cos at [:half], sin at [half:].
- weight-name mapper: vLLM `_make_deepseek_v4_weights_mapper` (model.py:1221) + `stacked_params_mapping`.

## PLAN (proposed — incremental, spec decision B)
0. Fix bucket access (remount allow_other) + env (HF_HOME, MODEL_IMPL_TYPE=vllm, FP4 requant guard, text-only, TP+EP=8).
1. Get merged skeleton: add upstream remote, base work on up-to-date upstream main; verify vLLM-LKG ↔ installed vLLM (ecf9d8352) compat.
2. Boot skeleton, attempt load → **validate FP4 expert GMM + HBM fit on v6e (Risk 1, day-1 gate)**.
3. Integrate kernels incrementally with parity tests each: MLA SWA/dense (#2903) → KV compressor (#2858) →
   lightning indexer + decode re-selection (#2905) → mHC head op (#2950) → FP4 load (#1906 adapted).
4. Per-component parity vs vLLM torch refs above; then short-context full-forward logits parity; then coherence smoke.
5. Adversarial review agent on every kernel port/integration.

## Open decisions for user
- A. If FP4 GMM fails on v6e: INT4 requant (accuracy hit) vs scale to v6e-16/v7.
- B. Bring-up order — recommend dense-first as above.
- C. Golden reference — use vLLM torch refs (above) + PR parity tests; GPU golden logits only if needed.

## Progress log
- Git: added `upstream` = vllm-project/tpu-inference; fast-forwarded `ds-v4-flash-torchax` 2d4e886f → upstream/main `2b09274e` (673 commits). Skeleton present; installed vLLM ecf9d8352 compatible (MLAAttentionSpec has compress_ratio/cache_dtype_str/alignment/model_version). PR branches fetched as local refs: `pr-2903 pr-2858 pr-2905 pr-2950 pr-1906`.
- Bucket: 2nd gcsfuse mount as enyouki (read-only) at `/home/enyouki/dsv4-weights` (cache /dev/shm/dsv4cache). Model snapshot (refs/main): `/home/enyouki/dsv4-weights/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/553034d7dd9e06c2eeaee68cf85a17d6d4754cf0` (46 shards + index + tokenizer). The systemd root mount at /tmp/gcs/bucket is left untouched. VM has 1.4TB RAM.
- Launcher: `/home/enyouki/dsv4_run/boot.sh` → `examples/offline_inference.py`. Required env/flags discovered: `MODEL_IMPL_TYPE=vllm`, `NEW_MODEL_DESIGN=1`, `--tensor-parallel-size 8 --enable-expert-parallel --additional-config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true,"expert_parallelism":8,"tensor_parallelism":1}},"replicate_attn_weights":"True","sparse_matmul":"True"}'` (canonical DeepSeek recipe from scripts/multihost/benchmarks/jax/run_deepseek_v3_1k_1k.sh). FP4 guard: `MOE_REQUANTIZE_WEIGHT_DTYPE=fp4`.
- Boot attempt 1: reached weight loading. CONFIRMED: FP4 experts preserved, DP+EP mesh, GMM EP kernel, 31.25 GiB/chip. BLOCKED at `KeyError: layers.0.attn.attn_sink` (amd/model.py:687) — stub `VllmDeepseekV4MLAAttention.__init__` registers no params. Need #2903 attention module (declares params + forward). attn_sink loader is TP head-sliced into params_dict[name][:n].
- mHC state in merged skeleton: pre/post → vLLM torch impls (mhc_pre_torch/mhc_post_torch, traceable); head op = pass-through stub (#2950); fused_post_pre = NotImplementedError (NO torch fallback in vLLM mhc_kernels — must implement, e.g. compose from pre+post or port tilelang).
