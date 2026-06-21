# MLX 4-bit MoE loading on tpu-inference — Design Spec

**Goal:** Load `mlx-community/Qwen3-30B-A3B-4bit` and run it on tpu-inference, keeping
weights **4-bit in HBM** and dequantizing **on the fly** (in-kernel) so memory stays small.
This is the scoped bring-up rung toward the larger `HY3_SPEC.md` (295B Hy3 MoE) — same MLX
quant path, smaller/known architecture.

## 1. Target & scope

- **Model:** `mlx-community/Qwen3-30B-A3B-4bit` — arch `Qwen3MoeForCausalLM` (already
  implemented in `models/jax/qwen3_moe.py`, **no new model code**). 48 layers, hidden 2048,
  128 experts / 8 active, moe_intermediate 768, vocab 151936, untied embeddings. ~17 GB / 4 shards.
- **Two-phase delivery:**
  - **Phase 1 (correctness):** weights stay packed uint32 in HBM; dequant in XLA at apply time
    (`w = scales·q + biases` → bf16) → existing einsum / `gmm_v2`. Verified vs HF reference.
  - **Phase 2 (performant):** extend `gmm_v2` so the expert matmul stays int4-in-kernel; add the
    affine **bias term** (the one genuinely new piece of math).
- **Out of scope:** new architectures; DeepSeek router gap (that's Hy3); a dense int4 kernel for
  attention (those stay on XLA-dequant-on-the-fly — still packed in HBM).

## 2. MLX 4-bit affine format (verified against the real repo)

Each quantized `Linear` → **3 tensors**:
- `…weight` — **uint32**, packed 8 four-bit values/word, **element 0 = low nibble** (little-endian),
  shape `[out, in/8]`.
- `…scales` — **bf16**, shape `[out, in/group_size]` = `[out, in/64]`.
- `…biases` — **bf16**, shape `[out, in/64]`. **Real float offset, not an integer zero-point.**

Dequant (per group of 64 along the **input** dim): **`w = scales·q + biases`**. Scales may be
**negative**; bias is a real offset.

- **Experts are STACKED**, leading dim = 128, **separate** gate/up/down (MLX does NOT fuse gate+up):
  `model.layers.N.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}`.
- **Everything linear is quantized** in the base repo, incl. `mlp.gate` (router), `embed_tokens`,
  `lm_head`. Only norms (incl. QK-norm `q_norm`/`k_norm`) stay bf16 (lone `.weight`, no scales).
- **Mixed precision (Instruct variant):** `config["quantization"]` carries top-level
  `{group_size:64, bits:4}` **plus dotted per-module overrides** (e.g. `…mlp.gate → {bits:8}`).
  Loader must read **per-module bit-width**; pack factor = `32/bits` (8 for 4-bit, 4 for 8-bit),
  `in = weight.shape[-1]·pack_factor`, `num_groups = in/group_size == scales.shape[-1]`.

## 3. The core technical problem — the affine bias term

Symmetric quant lets you "matmul the ints, then rescale." MLX is **asymmetric**, so the `+bias`
doesn't factor out:

```
out[m,o] = Σ_i x[m,i]·(scale[o,g]·q[o,i] + bias[o,g])     g = i//64
         = Σ_i x[m,i]·scale[o,g]·q[o,i]      (the scaled matmul — already handled)
         + Σ_g bias[o,g]·(Σ_{i∈g} x[m,i])    (DATA-DEPENDENT correction — the new term)
```

The second term depends on activations `x`, so it can't be a constant output bias. It is a small
matmul of **group-summed activations** against a `[E, num_groups, N]` bias tensor.

**Good news from the kernel survey:** `gmm_v2`'s existing per-K-block scale path
(`rhs_scale` shape `[E, num_blocks, 1, N]`) already == MLX's group-64 per-output-channel scale
layout, and it already keeps int4 packed + bitcasts nibbles + upcasts to fp8 for the MXU. Its
existing `rhs_bias` is a per-output-channel **post-accumulation** constant — the **wrong** kind —
so the only real kernel gap is the per-input-group bias term above (+ negative scales + nibble order).

## 4. Where it plugs in (tiers)

Quantization is a **strategy object** (`quant_method`), not scattered `if quantized` checks. Only
the two tiers that touch raw bytes get format-specific code.

| Tier | File | Change |
|------|------|--------|
| Model arch | `models/jax/qwen3_moe.py` | **none** |
| MoE layer | `layers/jax/moe/moe.py` (`JaxMoE`) | **none** (already hosts `quant_method`) |
| Dispatch | `layers/jax/quantization/__init__.py` | detect MLX `"quantization"` block → register `int4`; add to method dict + TPU allow-list |
| Quant method | `layers/jax/quantization/int4.py` *(new)* | `Int4Config` + `Int4LinearMethod` + `Int4FusedMoEMethod`, mirroring `fp8.py` |
| Loader | `models/jax/utils/weight_utils.py` | uint32-aware: skip bf16-cast + transpose for packed weights; carry `scales`/`biases`; stack experts |
| Kernel | `kernels/megablox/gmm_v2.py` | Phase 2: MLX nibble order, group-64 scales, negative scale, **+ affine bias term** |

**Backend-routing guard:** the int4 path only "inherits `JaxMoE` for free" if the model routes
experts through the GMM backend (GMM_TP/GMM_EP → `gmm_v2`) that FP8 already uses. If config lands
on `FUSED_MOE`, the quant method must guard/redirect — a check in the method, not a `JaxMoE` edit.

## 5. Components

1. **Config detection** — recognize MLX's top-level `"quantization": {group_size, bits[, mode]}`
   (no `quant_method` key) + dotted per-module overrides; normalize to method `int4`. Add
   `int4: Int4Config` to `method_to_config` and to the TPU allow-list.
2. **Quant methods** (mirror FP8 lifecycle `create_weights_jax / load_weights /
   process_weights_after_loading / apply_jax`):
   - `Int4FusedMoEMethod` — experts. `create_weights` declares packed uint32 + bf16 scales/biases
     params; `process_weights_after_loading` stacks/reshapes to the kernel's leading-dim-E layout;
     `apply_jax` hands packed weight + scales + biases to `gmm_v2`.
   - `Int4LinearMethod` — dense linears (attention qkv/o_proj, lm_head, embed). Phase 1: XLA dequant.
3. **uint32-aware loader** — base loader force-casts to bf16 (corrupts uint32) and transposes
   unconditionally (destroys packing). Add: `keep_original_dtype` for packed-weight keys; skip
   transpose for them; `uint32` entry in `DTYPE_VIEW_MAP`; load `scales`/`biases`; dispatch on
   key suffix (`.scales`/`.biases` present ⇒ quantized; lone bf16 `.weight` ⇒ norm). Experts arrive
   pre-stacked (leading dim 128) — preserve, don't expect per-expert files.

## 6. Phases

- **Phase 1:** packed in HBM; `apply_jax` does `w = scales·q_unpacked + biases` → bf16, then existing
  matmul. Correct, validates loader + format. Not yet bandwidth-optimal (matmul runs bf16).
- **Phase 2:** extend `gmm_v2` for the expert matmul — reuse the scale path; add the bias term via
  **Approach A** first: compute `Σ_g bias[o,g]·groupsum(x)` as a separate cheap `dot_general` outside
  the kernel and add to the output (kernel stays nearly unchanged, hard math isolated in plain JAX,
  easy to test). Graduate to fully-in-kernel (**Approach B**) only if profiling demands it.

## 7. Sharding

Packed weights pack along the **input** dim → the contraction dim may be sharded only in **multiples
of 64** (word×group boundary). Clean axes: **output dim** + **expert-parallel** — what the existing
MoE EP/TP sharding already prefers. Don't promise fine-grained TP along the contracted dim.

## 8. Testing

1. **Golden dequant:** numpy `w = scales·q + biases` vs `mlx.core.dequantize` on real tensors from
   the repo (and confirm nibble order against `pltpu.bitcast`).
2. **Adversarial synthetic** MLX-format checkpoint (few layers, ~8 experts, group-64) with
   deliberately **negative scales + nonzero bias** — catches symmetric-only assumptions and the bias
   term. Used for kernel iteration without the 17 GB download.
3. **End-to-end:** logits / greedy-decode parity vs HF transformers on a few prompts.

## 9. Risks

- Negative scales through the int4→fp8 upcast path (verify the existing path tolerates them).
- Mixed-precision Instruct variant (8-bit router gate) — read per-module bits from config.
- **Iteration hardware:** ~17 GB at 4-bit needs a v6e (32 GB) or multi-chip v5e to load; kernel
  iteration uses the synthetic tiny config regardless.
