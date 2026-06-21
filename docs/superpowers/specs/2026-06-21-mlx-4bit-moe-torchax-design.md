# MLX affine 4-bit MoE on the torchax (vLLM) path — Design

- **Date:** 2026-06-21
- **Status:** Approved, ready for planning
- **Target model:** `mlx-community/Qwen3-30B-A3B-4bit`
- **Supersedes:** the JAX-path MLX work (`docs/superpowers/specs/2026-06-21-mlx-4bit-moe-loading-design.md` and plan, `HY3_SPEC.md`, `.sdd/`) — all obsolete.

## Goal

Serve `mlx-community/Qwen3-30B-A3B-4bit` through the **torchax path** (`MODEL_IMPL_TYPE=vllm`) with accurate, coherent outputs. Weights stay **4-bit (packed uint32) + scales + biases in HBM**.

**Staged delivery (user-confirmed):**
1. **Correct baseline** — MoE experts dequant-at-load to bf16, run the existing unquantized grouped-matmul. Proves the whole pipeline (registration, weight transform, routing) and brings up the real 30B with coherent output. Experts bf16 in HBM (~7.6GB/chip, fits).
2. **In-kernel optimized end-state** — experts kept as uint32-packed int4 + per-group scale + per-group bias in HBM (true 4-bit footprint, ~2GB/chip), dequantized **inside the `gmm_v2` grouped-matmul kernel** for only the routed experts (bandwidth- and compute-optimal). Attention linears stay dequant-in-`apply` (small, negligible).

### Key kernel finding (verified)
`gmm_v2` **already** (a) bitcast-unpacks uint32-packed int4 weights in-kernel along K, and (b) applies a per-group `rhs_scale [G, num_blocks, 1, N]` inside the k-loop. The **only** missing piece for MLX affine is a **per-group additive bias**: the existing `rhs_bias [G,1,N]` is per-output-channel (a normal MLP bias), not per-group. We add a parallel `rhs_groupbias [G, num_blocks, 1, N]` and accumulate `acc_n += groupbias · groupsum(block_lhs)` in the k-loop before the fused activation (Approach B). This works for the fused w13 (gate/up+SiLU) and w2 calls; an outside-kernel correction (Approach A) does not, because w13's activation is fused inside the kernel.

### Non-goals
- Quantizing `embed_tokens` / `lm_head` at runtime (dequant-on-load to bf16 instead — they are tiny and not routed by `get_quant_method`).
- Keeping attention linears in-kernel (they stay dequant-in-`apply`; ~2B of 30B params, not the bottleneck).
- The JAX path. All prior JAX MLX code is removed.

## Background

### The model (verified from the real checkpoint)
`Qwen3MoeForCausalLM`: 48 layers (all MoE), 128 experts, top-8, `hidden=2048`, `moe_intermediate=768`, untied embeddings. MLX affine 4-bit, `group_size=64`, uniform (no per-module overrides). Quantized: attention q/k/v/o, the 128 experts (stored **stacked** as `mlp.switch_mlp.{gate,up,down}_proj`), router `mlp.gate`, `embed_tokens`, `lm_head`. Plain bf16: all norms.

Each quantized tensor = `.weight` (uint32, 8 nibbles/word packed along the **input** dim) + `.scales` + `.biases` (bf16, `[out, in//64]`). Dequant is affine: `w = scale·q + bias` per group of 64 along the input dim. (`element 0 = low nibble`; flat input index = `word*8 + k`.)

### The torchax path
`MODEL_IMPL_TYPE=vllm` runs vLLM's real PyTorch `Qwen3MoeForCausalLM` under torchax: weights load on CPU, then shard to TPU as torchax JAX arrays; the forward is traced into one XLA graph. Quant methods on this path **cross into JAX**: `jax_view(x)`/`jax_view(layer.param)` → express dequant+matmul in `jnp`/`jax.lax` → `torch_view(result)`. **AWQ** (`tpu_inference/layers/vllm/quantization/awq.py`) is a near-exact template — also 4-bit packed, also dequant-in-`apply` via jnp.

### The checkpoint ↔ vLLM mismatch
vLLM expects **per-expert, separate** names (`mlp.experts.{e}.{gate,up,down}_proj.*`) and fuses qkv / gate_up / experts itself. MLX stores experts **stacked** under `switch_mlp`. vLLM's expert mapping (`make_expert_params_mapping`) is suffix-agnostic and routes `.weight`/`.scales`/`.biases` automatically (proven by AWQ MoE), but only for per-expert names. So we must rename + un-stack before vLLM's loader sees the stream.

## Strategy

Implement a standard vLLM quant method (config + linear method + MoE method) mirroring AWQ, plus a load-time weight-stream transform that bridges naming/layout. Reuse `mlx_unpack`/`mlx_dequantize` (jnp) from `tpu_inference/layers/common/quantization/__init__.py`. Deliver in two stages (above): a correct dequant-at-load baseline first, then the in-kernel optimized path for the experts (the bulk of the params). Attention linears use AWQ-style dequant-in-`apply` throughout (4-bit footprint, bf16 bandwidth — negligible since they are small).

## Components

### 1. Method registration (mirror AWQ)
- `tpu_inference/layers/common/quant_methods.py`: add `MLX = "mlx"`.
- `tpu_inference/layers/vllm/quantization/__init__.py` (`get_tpu_quantization_config`): **detect MLX format** — hf quant config (`quantization`/`quantization_config`) has `group_size`+`bits` and no recognized `quant_method` → select `VllmMlxConfig`; add to `method_to_config`.
- `tpu_inference/platforms/tpu_platform.py`: add `"mlx"` to `supported_quantization`.
- New `tpu_inference/layers/vllm/quantization/mlx.py`: `VllmMlxConfig`, `VllmMlxLinearMethod`, `VllmMlxMoEMethod`.

### 2. `VllmMlxConfig`
Subclass `(QuantizationConfig, VllmQuantConfig)`, `@register_quantization_config(MLX)`, `get_name() → "mlx"`. `from_config` parses `group_size`/`bits`. `get_quant_method(layer, prefix)` dispatches by `match layer`: `LinearBase` (attention q/k/v/o + router gate) → `VllmMlxLinearMethod`; `FusedMoE` → `VllmMlxMoEMethod` (after setting `layer.moe_config`); else `None`.

### 3. `VllmMlxLinearMethod` (mirror `awq.py:91-232`)
- `create_weights`: register packed `weight` (uint32 `[out, in//8]`) + `scales`/`biases` (bf16 `[out, in//64]`) with the right weight-loader/sharding attrs. Handle vLLM's qkv/gate_up **output-dim fusion** (output_sizes / n_shards as AWQ does); fusion is along the output dim while packing is along the input dim, so no packed-offset arithmetic is needed.
- `process_weights_after_loading`: `t2j` params, shard along the output dim, store packed (no dequant here — stay 4-bit in HBM).
- `apply`: `jax_view` packed `weight`+`scales`+`biases` → `mlx_dequantize` → bf16 `[out,in]` → `jnp.einsum` contract `in` → `torch_view`. Wrap in `jax.named_scope`. Mirror AWQ's `_apply_fused` / `_apply_split` for fused vs split params.

### 4. `VllmMlxMoEMethod` (mirror `awq.py:235-501`)
`create_weights` (both stages): register stacked packed `w13_qweight`/`w13_scales`/`w13_biases` (+ `w2_*`) using AWQ's param-name convention so the suffix-agnostic expert mapping routes per-expert `.weight`/`.scales`/`.biases` slices. Shapes: `w13_qweight [E, 2I, H//8]`, `w13_scales/biases [E, 2I, H//64]`, `w2_qweight [E, H, I//8]`, `w2_scales/biases [E, H, I//64]`. `is_monolithic = True`.

**Stage 1 (baseline):**
- `process_weights_after_loading`: `jax_view` packed experts → dequant uint32→bf16 (jnp) → `process_moe_weights`/`shard_moe_weights` → store bf16 `w13_weight`/`w2_weight`.
- `apply_monolithic`: mirror `VllmUnquantizedFusedMoEMethod.apply_monolithic` — pass bf16 weights + `w13_weight_scale=None` to `vllm_moe_apply`. Correct, serves; experts bf16 in HBM.

**Stage 2 (in-kernel optimized):**
- Extend `gmm_v2` with a per-group `rhs_groupbias [G, num_blocks, 1, N]` input (clone of the `rhs_scale` block-spec/index-map); accumulate `acc_n += groupbias · jnp.sum(block_lhs, axis=1, keepdims=True)` in the k-loop before `apply_act_fn` (hook points: scale loop `gmm_v2.py:457-462`, bias add `:471-474`, k-loop `:419`). Add a kernel-level unit test vs a numpy affine grouped-matmul reference.
- `process_weights_after_loading`: lay out packed int4 weight + per-group scale + per-group bias in the `[G, num_blocks, 1, N]` form `process_moe_weights` produces for scale; shard; keep packed (true 4-bit in HBM).
- `apply_monolithic`: pass packed weight + scale + groupbias straight to `vllm_moe_apply`/`fused_moe_func` (no XLA dequant). Kernel unpacks int4, applies per-group scale + groupbias, fuses activation, computes only routed experts.
- Confirm the active `MoEBackend` routes through `gmm_v2` (`GMM_EP`/`GMM_TP`), not the `DENSE_MAT`/`MEGABLX_GMM` cases that currently `raise NotImplementedError` in `process_moe_weights`.

### 5. Weight-stream transform (`tpu_inference/models/vllm/vllm_model_loader.py`)
Override `load_weights`/`get_all_weights` on `IncrementalModelLoader` to wrap the `(name, tensor)` generator before `model.load_weights`, gated on MLX detection:
- **Experts:** rename `mlp.switch_mlp.{proj}` → `mlp.experts.{e}.{proj}`, un-stack the leading 128-dim into per-expert tensors; keep `weight` packed (uint32) and carry `.scales`/`.biases`. vLLM re-stacks them into the quant params.
- **Attention q/k/v/o + router gate:** pass through packed + scales/biases (names already match vLLM).
- **embed_tokens + lm_head:** **dequant to bf16** plain `.weight`, drop scales/biases (not routed by `get_quant_method`; small).
- **Norms:** pass through.

### 6. Detection / config injection
MLX `config.json` has no `quant_method`, so vLLM's own detection won't fire. `get_tpu_quantization_config` sets `vllm_config.quant_config = VllmMlxConfig(...)` directly (it's already called at `vllm_model_wrapper.py:127`), so per-layer `get_quant_method` dispatch uses our instance. **Verify first** that vLLM's `VllmConfig` construction tolerates the bare `quantization_config` (no `quant_method`) without crashing; if it does crash, patch/strip the HF config in an early hook.

## Cleanup (full)

Remove (revert the 5 JAX commits' changes + untracked artifacts):
- `tpu_inference/layers/jax/quantization/int4.py` (delete).
- `tpu_inference/layers/jax/quantization/__init__.py` int4 registration + MLX detection (revert).
- MLX hooks in `tpu_inference/models/jax/utils/weight_utils.py` (`_is_mlx_packed`, `_get_active_int4_config`, `_normalize_mlx_switch_mlp_keys`, uint32 view-map entry) (revert).
- `tpu_inference/models/common/model_loader.py` `enable_weights_track` toggle (revert; verify it's only for the JAX MLX path).
- JAX int4 tests: `tests/models/jax/test_qwen3_moe_mlx_int4_e2e.py` + committed `test_int4_*.py` (delete).
- Obsolete docs: `HY3_SPEC.md`, `.sdd/`, the JAX-era spec/plan under `docs/superpowers/`.

Keep:
- `tpu_inference/layers/common/quantization/__init__.py` `mlx_unpack`/`mlx_dequantize` (reused by the new apply).
- `tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py` shim (unrelated vLLM-version fallback).
- `tests/utils/mlx_synthetic.py` (adapt for the torchax test).

## Testing

1. **Harness sanity:** `MODEL_IMPL_TYPE=vllm` offline_inference with `Qwen/Qwen3-0.6B` → coherent text (no MLX).
2. **Numerical correctness:** adapt `tests/utils/mlx_synthetic.py` to build a tiny MLX int4 Qwen3-MoE checkpoint + a bf16 reference from the same golden weights; new `tests/models/vllm/test_qwen3_moe_mlx_int4_e2e.py` loads the synthetic checkpoint through the vllm path and asserts logits match the bf16 reference within bf16 tolerance.
3. **Real model:** `MODEL_IMPL_TYPE=vllm SKIP_JAX_PRECOMPILE=1 python examples/offline_inference.py --model mlx-community/Qwen3-30B-A3B-4bit --tensor-parallel-size 8` → coherent completions.

Run env: `.venv` (vllm 0.22.0, jax 0.9.2, 8×v6e). Tests via `MODEL_IMPL_TYPE=vllm python -m pytest -s -v -x <path>`.

## Risks / open questions

1. vLLM `VllmConfig` tolerance of bare `quantization_config` (no `quant_method`) — verify before building anything; patch HF config early if it crashes.
2. uint32 through safetensors → torch → `jax_view` — confirm dtype survives; view as int32 in unpack if needed.
3. Exact vLLM parameter classes / weight-loader wiring for sharding MLX `.scales`/`.biases` along the output dim (follow AWQ's parameter choices closely).
4. qkv/gate_up fusion of packed weights carrying `.scales`/`.biases` through vLLM's `stacked_params_mapping` (AWQ proves it works; verify suffixes route).
5. Stage 1 dequants all 128 experts/step (correct but bandwidth-suboptimal) — resolved by Stage 2 (in-kernel dequant of only routed experts). Stage 2 risk is isolated to the `gmm_v2` `rhs_groupbias` addition, guarded by a kernel unit test and the synthetic e2e; keep it zero-overhead when absent so existing callers are unaffected.

> If problems arise, consult tpu-inference PRs first.
