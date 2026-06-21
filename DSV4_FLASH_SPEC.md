# DeepSeek-V4-Flash on TPU v6e-8 — Bring-up Spec (torchax / MODEL_IMPL_TYPE=vllm)

## Goal
Serve `deepseek-ai/DeepSeek-V4-Flash` **text-only** on v6e-8 with accurate/coherent output, weights kept in
native quant (FP4 experts + FP8 linears — see Risk 1/2). Run vLLM's PyTorch model
(`/home/enyouki/vllm/vllm/models/deepseek_v4`, arch `DeepseekV4ForCausalLM`) through **torchax**; replace every
non-traceable CUDA/Triton/CuTe-DSL/DeepGEMM/FlashMLA op with JAX/Pallas, reusing this branch's kernels.
Weights are pre-mounted from `gs://personal-mark-eu/vllm/hub/models--deepseek-ai--DeepSeek-V4-Flash/` (gcsfuse,
read-only) — never download full weights. **Scope: this branch only** (ignore `glm5.2-dsa`, `mlx`).

## Key config facts (from bucket `config.json`; HF arch `DeepseekV4ForCausalLM`)
hidden 4096 · 43 layers (all MoE) · vocab 129280 · head_dim 512 (448 nope + 64 rope) · heads 64 · KV heads 1
(shared 512-d K=V latent, **no kv_b reconstruction**) · q_lora 1024 · o_groups 8 / o_lora 1024 · sliding_window 128 ·
experts 256 top-6 + 1 shared, moe_inter 2048 · `scoring_func=sqrtsoftplus`, `topk_method=noaux_tc`, routed_scale 1.5,
swiglu_limit 10 · `num_hash_layers=3` (tid2eid token-id routing) · mHC `hc_mult=4`, `hc_sinkhorn_iters=20` ·
quant fp8 e4m3 block [128,128] `scale_fmt=ue8m0` + `expert_dtype=fp4` · rope_theta 10000 / compress_rope_theta 160000,
YaRN factor 16 · `compress_ratios[44]` ∈ {0,4,128} per layer · MTP `num_nextn_predict_layers=1` (skip — spec-decode only).

## Baseline on THIS branch (reuse as-is)
- MLA v2 Pallas kernel `kernels/mla/v2` (`mla_ragged_paged_attention`) + `layers/vllm/backends/flash_attn_mla.py`.
- Block-FP8 linear `kernels/quantized_matmul/blockwise_kernel.py` — fp8 stays in HBM, upcasts in-VMEM (runs v6e).
- MoE GMM `kernels/megablox/gmm_v2.py` + `layers/common/fused_moe_gmm.py`; `float4_e2m1fn` supported, unpacked+upcast
  in-VMEM (`gmm_v2.py:355,446`) → functionally runs on v6e (see Risk 1).
- vLLM quant methods `layers/vllm/quantization/{fp8,mxfp4}.py`; e8m0 helpers in `layers/common/quantization/__init__.py`
  (`e8m0_to_fp32`, `u8_unpack_e2m1`, `quantize_tensor_to_mxfp4_packed`).
- torchax infra: `models/vllm/vllm_model_wrapper.py`, `layers/vllm/ops/` (`@register_function`),
  `layers/vllm/custom_ops/{rope,mla_attention,fused_moe,linear,embedding}.py`.
- `models/jax/deepseek_v3.py` — numeric reference for RoPE/router conventions only (not reused as model).

## To add / modify (build)
Register `DeepseekV4ForCausalLM` for the vllm/torchax path; provide JAX replacements for:

1. **Sparse-MLA attention (the crux)** — per-layer regime keyed on `compress_ratios` (0/4/128):
   shared 512-d K=V latent + output **inverse-RoPE**; sliding-window(128) mask every layer; **attention sinks**
   (per-head, off-by-one softmax); weight-free per-head Q RMSNorm; **grouped low-rank o_proj** (block-diag [8,1024,4096]).
   - ratio 0: sliding-window dense MLA only (KV cache 128), base theta, no YaRN.
   - ratio 4: + **lightning indexer/NSA** — fp8 MQA logits, per-row **top-512** select, indexer KV cache + per-step
     decode re-selection; compress theta + YaRN.
   - ratio 128: + **KV compressor** (gated pooling + learned APE + RoPE + fp8/fp4 quant + paged write), no indexer.
   Build on MLA v2 kernel; new Pallas: compressor, indexer (logits+topk+k-quant-cache), sparse gather.

2. **MoE** — gmm_v2 FP4 experts; router `sqrtsoftplus` + `noaux_tc` e_score_correction_bias select + `norm_topk_prob` +
   routed_scale 1.5; swiglu clamp 10; **hash routing via `tid2eid`** for first 3 layers; separate shared expert.

3. **mHC (Manifold-Constrained Hyper-Connections)** — 4 parallel residual streams; pre(4→1)/post(1→4) mixers +
   `fused_post_pre` + `hc_head`; 4×4 comb matrix → doubly-stochastic via **20 Sinkhorn iters**; per-layer
   attn/ffn RMSNorm fused into the mixers.

4. **Quant glue** — decode UE8M0 scales by **bitcast** (not `.astype`); keep experts **FP4** (do NOT requant to fp8,
   Risk 2); FP8 block for generic linears; `wo_a` fp8→bf16 at load.

5. **Weights + RoPE + launch** — map vLLM deepseek_v4 weight names; DeepseekV4 YaRN **dual-theta** rope (GPT-J
   interleaved). Mount bucket via gcsfuse, point HF cache at it. Launch text-only, MTP/spec-decode disabled,
   TP+EP across all 8 chips (mandatory — model does not fit on fewer).

## Risks (ranked)
1. **FP4 experts on v6e (highest).** HBM fits ONLY if experts stay FP4 (~16–18 GiB/chip); FP8 experts ≈ 33 GiB/chip >
   ~29.8 usable → won't fit. FP4 gmm path exists but support matrix marks it **v7** and it's untested on v6e (it
   upcasts in VMEM, so may functionally run). **Validate FP4 GMM on v6e on day 1.** If it fails: INT4 requant
   (fits ~18 GiB/chip, v6e-supported, accuracy hit) or scale to v6e-16 / v7. (NB: "keep FP8 quant in effect" holds for
   the linears; experts must be FP4 — pure-FP8 experts overflow HBM.)
2. **Requant default widens experts.** `MOE_REQUANTIZE_WEIGHT_DTYPE` defaults to `float8_e4m3fn` → blows HBM. Must set
   `fp4` / bypass expert requant.
3. **Sparse path silently wrong.** Dense-MLA is numerically correct only at short context; compressor/indexer/
   sliding-window run in every forward and, if stubbed, give fluent-but-wrong output (no error). Indexer KV-write index +
   decode re-selection are the classic landmine — only exercised at decode, not prefill. Gate numerically vs reference.
4. **Kernel volume / parity.** ~8–10 new Pallas/JAX kernels (compressor, indexer, sparse gather, sinks, grouped o_proj,
   mHC sinkhorn, sqrtsoftplus+hash router, FP4 GMM); each needs a parity test vs vLLM `nvidia/` reference. Largest effort.
5. **v6e fp8/fp4 throughput** uses the bf16-upcast MXU path (correctness fine; slower than v7-native).

## Open decisions (for review)
- **A.** If FP4 GMM fails on v6e: INT4 expert requant vs scale to v6e-16/v7?
- **B.** Bring-up order — recommend: short-context dense-equivalent first (validate MLA+MoE+mHC end-to-end), then add
  compressor → indexer → decode re-selection.
- **C.** Reference for numeric parity — run vLLM `nvidia/` on a GPU for golden logits, or trust config-derived math?

## Verification
Per-component parity vs vLLM `nvidia/` reference (router, mHC, compressor, indexer, MLA, o_proj) → full-forward logits
parity at short context → coherence smoke (generate) on bucket weights. Use adversarial review agents on every kernel port.
