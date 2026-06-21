# tpu-inference — project notes for Claude

## DeepSeek-V4-Flash bring-up (active work)

Porting `deepseek-ai/DeepSeek-V4-Flash` (text-only) to **TPU v6e-8** via the **torchax** path
(`MODEL_IMPL_TYPE=vllm`), keeping experts in **FP4** and linears in **FP8** (the only way it fits HBM).
Branch: `ds-v4-flash-torchax`. Weights are mounted read-only at `/home/enyouki/dsv4-weights`
(gcsfuse of the `personal-mark-eu` bucket) — **never download the full weights**.

Handoff docs: `DSV4_HANDOFF.md`, `DSV4_BRINGUP_NOTES.md`, `DSV4_FLASH_SPEC.md`.
Specs/plans live under `docs/superpowers/specs/`.

> **Caution:** the FP4-GMM diagnosis in `DSV4_HANDOFF.md` is **wrong** (it claims "native FP4 matmul
> is v7-only"). The real root cause is the mxfp4 requant **block size (512 ≥ v6e mxu_column_size 256)**
> forcing the quantized-matmul path instead of dequant-in-VMEM. See the current spec for the corrected
> analysis before trusting the handoff on quantization.

## Testing requirements (apply to ALL DSV4 phases and every spec written for them)

These are **hard requirements** for every component we build and every spec authored for this port:

1. **Test with synthetic small-config weights — do NOT load the full model for routine testing.**
   - Numerical-accuracy / parity tests MUST use randomly-initialized weights on a **small DeepSeek-V4
     config** (few layers, few experts, reduced hidden/head dims) expressed in the **real quant formats**
     (FP4 e2m1 experts, FP8 e4m3 block-scaled linears, ue8m0 scales). This instantiates in seconds.
   - Parity means: run the vLLM torch/GPU reference and the TPU implementation on the **same synthetic
     weights and inputs**, compare within atol/rtol. Correctness needs no trained weights.
   - Reserve the full (~187 GiB) model load **only** for milestone *coherence* smoke tests (real text must
     read sensibly) — never for iterative development.

2. **Test on the real multi-chip TPU mesh from the start.**
   - Every test runs on the **production sharding config** (TP=8 + expert-parallel + DP-attention),
     the same mesh used for full-model serving, so sharding / collective bugs surface immediately.
   - Never validate single-chip-only and defer sharding correctness to later.

## General gotchas

- Ignore the `glm5.2-dsa*` branches — unrelated/broken.
- Don't reinstall vLLM (pinned editable install at `/home/enyouki/vllm`).
- When a GPU/CUDA-only or untraceable op surfaces in a TPU trace, port it to a JAX/torchax lowering that
  reproduces the pure-Torch reference's exact numerics.
