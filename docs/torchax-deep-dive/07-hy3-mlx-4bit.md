# 07 — MLX 4-bit quantization through the torchax route (Hy3-preview)

> **Scope.** This doc covers the **MLX-specific** plumbing for serving an MLX-format
> 4-bit MoE checkpoint (Qwen3-30B-A3B-4bit style, and the Hunyuan **`HYV3ForCausalLM`**
> "hy3-preview") via the torchax/vLLM route (`MODEL_IMPL_TYPE=vllm`). The general
> quant framework (`VllmQuantConfig` / `VllmQuantizationMethod` / the incremental
> loader) is documented elsewhere; here we focus on what MLX adds on top.
>
> Strategy in one line: **keep-4bit + dequant-in-forward.** Quantized linears
> dequant to bf16 inside the XLA `apply`; the MoE keeps the dominant expert weight
> (`w13`) as *packed int4 in HBM* and dequantizes **inside the GMM kernel**.
>
> All claims below are anchored to `path:line`. Verified against the `hy3` branch.

---

## 0. TL;DR map

```
HF MLX checkpoint (uint32-packed 4-bit + per-group scales + per-group biases)
        │
        │  IncrementalModelLoader.get_all_weights()  ── is_mlx_quantized()? ──► transform_mlx_weights()
        │     vllm_model_loader.py:88-109                                        mlx_weight_transform.py
        ▼
   rewritten (name, tensor) stream:
     • switch_mlp.{g,u,d}_proj  → per-expert experts.{e}.*   (STAY uint32 packed)
     • embed_tokens / lm_head   → single bf16 .weight        (dequant at load)
     • router.gate (Hy3, 8-bit) → single bf16 .weight        (dequant at load)
     • router.expert_bias (Hy3) → renamed to .expert_bias
     • shared_mlp.* (Hy3)       → keep infix, STAY 4-bit
        │
        │  vLLM model build → get_quant_method() per layer  (mlx.py:124-139)
        ▼
   ┌─────────────────────────────┐   ┌──────────────────────────────────────────┐
   │ LinearBase                  │   │ FusedMoE                                  │
   │  VllmMLXLinearMethod         │   │  VllmMLXMoEMethod                          │
   │  keep uint32, dequant @apply │   │  w13: keep int4 → dequant in gmm_v2       │
   │  y = einsum(x, scale*q+bias) │   │  w2 : int4-in-kernel OR bf16-at-load      │
   └─────────────────────────────┘   └──────────────────────────────────────────┘
                                              │
                                              ▼  moe_apply → fused_moe_func → gmm_wrapper → gmm_v2
                                       w = scale*q + groupbias  reconstructed in the k-loop
```

Key files:
- `tpu_inference/layers/vllm/quantization/mlx.py` — config + linear method + MoE method.
- `tpu_inference/models/vllm/mlx_weight_transform.py` — the load-time name/layout rewrite.
- `tpu_inference/models/vllm/vllm_model_loader.py:88` — where the transform hooks the stream.
- `tpu_inference/layers/common/quantization/__init__.py:271,281` — `mlx_unpack` / `mlx_dequantize`.
- `tpu_inference/layers/common/process_weights/moe_weights.py` — the `groupbias` rails.
- `tpu_inference/kernels/megablox/gmm_v2.py` — in-kernel dequant (`rhs_scale`, `rhs_groupbias`).

---

## 1. The MLX quant config & how `mlx` is selected

### 1.1 What an MLX checkpoint looks like

MLX checkpoints carry a quant block with `group_size` + `bits` and crucially **no
`quant_method` key** (that key is what vLLM uses to dispatch to AWQ/GPTQ/etc.). This
absence is the detection signal:

`mlx.py:54-61`
```python
def is_mlx_quantized(hf_config) -> bool:
    """MLX checkpoints carry a quant block with group_size+bits and NO quant_method."""
    for attr in ("quantization_config", "quantization"):
        q = getattr(hf_config, attr, None)
        if isinstance(q, dict) and "group_size" in q and "bits" in q \
                and "quant_method" not in q:
            return True
    return False
```

Because there's no `quant_method`, vLLM leaves `model_config.quantization` **unset**.
Two places fix that up:

1. **TpuPlatform advertises `"mlx"`** — `tpu_inference/platforms/tpu_platform.py:94-95`
   lists `"mlx"` in `supported_quantization`, so vLLM accepts `--quantization mlx` and
   accepts a model whose effective quant resolves to `mlx`.
2. **The TPU quant-config selector forces the route** — when `quantization is None`
   but the config *is* MLX, it sets it to `"mlx"`:

   `tpu_inference/layers/vllm/quantization/__init__.py:42-44, 52`
   ```python
   if model_config.quantization is None and is_mlx_quantized(model_config.hf_config):
       model_config.quantization = quant_methods.MLX
   ...
   quant_methods.MLX: VllmMLXConfig,   # the dispatch table entry
   ```

`quant_methods.MLX == "mlx"` (`tpu_inference/layers/common/quant_methods.py:20`).
`VllmMLXConfig` is also registered with vLLM via the decorator
`@register_quantization_config(MLX)` (`mlx.py:64`), so vLLM's own
`get_quantization_config("mlx")` resolves to it.

### 1.2 `VllmMLXConfig`

`mlx.py:65-139`. Subclasses both vLLM's `QuantizationConfig` and the repo's
`VllmQuantConfig` (the latter provides `get_linear_config`/`get_moe_config`/`mesh`,
documented in the general-quant doc). Scheme parameters:

`mlx.py:67-75`
```python
self.group_size = group_size          # e.g. 64
self.bits = bits                      # 4
self.pack_factor = 32 // bits         # 8 (eight 4-bit nibbles per uint32 word)
self.modules_to_not_convert = ...     # is_layer_skipped → unquantized method
```

**Per-module overrides (Hy3).** `from_config` (`mlx.py:82-111`) handles MLX's habit
of carrying per-module quant overrides as dict-valued config entries keyed by the
full module path. The **only** supported override is the 8-bit `*.mlp.router.gate`,
which is dequantized to bf16 at load. Anything else (a different module, or a
different bit-width) **raises fail-fast** (`mlx.py:92-108`) rather than silently
mis-loading.

**Method dispatch** (`get_quant_method`, `mlx.py:124-139`):
- `LinearBase` → `VllmMLXLinearMethod` (or `VllmUnquantizedLinearMethod` if the prefix
  is in `modules_to_not_convert` via `is_layer_skipped`).
- `FusedMoE` → `VllmMLXMoEMethod`.

### 1.3 The MLX affine scheme (group size, packing, layout)

MLX is **affine** quantization: `w = scale * q + bias`, where `q` is an **unsigned**
4-bit code `[0, 15]`, and `(scale, bias)` is a *per-group* affine pair (one pair per
`group_size` contiguous input elements). Contrast with AWQ (`(q - z) * s`, packs along
*output*) — MLX packs along the **input** dim and is affine.

Layout for a linear weight (`mlx.py:142-156` docstring):
```
weight  : [out, in // pack_factor]  uint32   (packed_dim = 1 = INPUT)
scales  : [out, in // group_size]   bf16     (affine scale)
biases  : [out, in // group_size]   bf16     (affine bias)
```

The packing/dequant primitives live in
`tpu_inference/layers/common/quantization/__init__.py`:

`__init__.py:271-288`
```python
def mlx_unpack(packed, bits):                       # uint32 → int32 codes, low-nibble-first
    per_word = 32 // bits
    shifts = jnp.arange(per_word, dtype=jnp.uint32) * bits
    vals = (packed[..., None] >> shifts) & mask     # element 0 = LOW bits
    return vals.reshape(*packed.shape[:-1], -1).astype(jnp.int32)

def mlx_dequantize(packed, scales, biases, group_size, bits):
    q = mlx_unpack(packed, bits)                    # [..., out, in]
    scales_e = jnp.repeat(scales, group_size, axis=-1)
    biases_e = jnp.repeat(biases, group_size, axis=-1)
    return (q*scales_e + biases_e).astype(jnp.bfloat16)   # AFFINE: w = scale*q + bias
```

> **Nibble order is low-first.** Element `k` of a word lives in bits `[k*4, k*4+4)`.
> Tests pin this exactly: `test_mlx_dequant.py::test_unpack_low_nibble_first`
> (`0x76543210 → [0..7]`), and a *negative control* in
> `test_mlx_dequant_independent.py:371` proves a reversed-nibble unpack is detected.

---

## 2. The MLX weight transform (`mlx_weight_transform.py`)

### 2.1 Where it hooks the loader

The repo uses a custom `IncrementalModelLoader` (registered as `tpu_streaming_loader`).
Its `get_all_weights` wraps the raw `(name, tensor)` stream:

`vllm_model_loader.py:88-109`
```python
weights = super().get_all_weights(model_config, model)
if is_mlx_quantized(hf_config):
    q = (getattr(hf_config, "quantization_config", None)
         or getattr(hf_config, "quantization"))
    return transform_mlx_weights(weights, group_size=q["group_size"],
                                 bits=q["bits"], num_experts=hf_config.num_experts)
return weights
```

Non-MLX checkpoints pass straight through. The transform is a pure generator over the
stream — it runs **before** the torchax env is active and before vLLM's
`weight_loader`s consume the tensors, so it deals in **plain CPU torch tensors**.

### 2.2 What it rewrites (`transform_mlx_weights`, `mlx_weight_transform.py:104-192`)

| Pattern | Action | Stays 4-bit? | Anchor |
|---|---|---|---|
| `...mlp.switch_mlp.{g,u,d}_proj.{weight,scales,biases}` (stacked `[E,…]`) | un-stack axis 0 into per-expert `...mlp.experts.{e}.{proj}.{suffix}` | **yes** (uint32) | `:122-130` |
| `...mlp.shared_mlp.{g,u,d}_proj.*` (Hy3 shared expert) | **keep** the `shared_mlp.` infix | **yes** | `:132-144` |
| `...mlp.router.gate.{weight,scales,biases}` (Hy3, 8-bit) | buffer triplet → emit one **bf16** `.weight` | no (dequant) | `:146-158` |
| `...mlp.router.expert_bias` (Hy3) | rename → `...mlp.expert_bias` | n/a | `:160-163` |
| `model.embed_tokens.*` / `lm_head.*` | buffer triplet → emit one **bf16** `.weight` | no (dequant) | `:165-177` |
| everything else (attn q/k/v/o, norms, dense mlp) | pass through unchanged | — | `:179` |

Key subtleties:

- **Un-stacking keeps packing.** `tensor[e].contiguous()` preserves uint32 dtype and
  the per-expert `[out, in*]` shape (`:127-129`). The downstream `VllmMLXMoEMethod`
  is what re-stacks them into `w13_*`/`w2_*` and decides int4-vs-bf16.

- **Dequant-at-load uses `t2j`/`j2t`, not torchax views.** `_dequant_to_bf16`
  (`:77-101`) crosses into JAX via `t2j` (the AWQ/FP8 load-time idiom), runs
  `mlx_dequantize` in XLA, then materializes back to a **plain** CPU torch tensor via
  `j2t` (`:101`). Using `torch_view` here would yield a torchax-wrapped tensor; the
  consuming `weight_loader` ends in `param.data.copy_(loaded_weight)` against a plain
  CPU param while the torchax env is **disabled**, which would assert
  ("torchax Tensors can only do math within the torchax environment"). The long
  comment at `:77-101` is the canonical explanation of this load-time tensor-domain
  rule and is worth reading before touching any load-time dequant.

- **Triplet integrity.** Embed/lm_head/router-gate are buffered until `weight+scales+
  biases` all arrive (`:151,:172`). A leftover-buffer pass at the end (`:186-192`)
  passes through any already-bf16 embed/head that shipped without scales, but a
  *router gate* with an incomplete triplet is a corrupt checkpoint → **raise**
  (`:187-190`).

- **Why router-gate→bf16:** vLLM's `GateLinear` (the MoE router) is forced unquantized
  fp32; the MLX checkpoint ships it 8-bit. vLLM later strips `router.` → `mlp.gate`,
  hence the rename of `expert_bias` too (vLLM only remaps `router.gate`, so without the
  rename the bias would `KeyError`). See the file's module docstring `:30-48`.

> **No `NEW_MODEL_DESIGN` gating on the MLX path.** Grep of `mlx.py`,
> `mlx_weight_transform.py`, `moe_weights.py`, `interface/moe.py` finds **zero**
> `NEW_MODEL_DESIGN` references. MLX selection is driven purely by the checkpoint's
> quant block (§1.1), not by that env var. (The env var gates the JAX/flax route's new
> model rewrites, which is a different code path.)

---

## 3. Dequant-in-forward — the linear method

`VllmMLXLinearMethod` (`mlx.py:142-298`). Keeps the weight uint32-packed all the way
to `apply`, where it dequantizes in XLA and matmuls in bf16. **No fused quantized
matmul** for linears — it's *dequant-to-bf16-then-einsum*.

### 3.1 `create_weights` (`mlx.py:163-201`)

Registers three vLLM parameters with the packing/grouping made explicit to vLLM's
sharding machinery:

```python
weight = PackedvLLMParameter(  [out, in//pf] uint32, output_dim=0,
                                input_dim=1, packed_dim=1, packed_factor=pf)
scales = GroupQuantScaleParameter([out, in//gs] bf16, output_dim=0, input_dim=1)
biases = GroupQuantScaleParameter([out, in//gs] bf16, output_dim=0, input_dim=1)
```

### 3.2 `process_weights_after_loading` (`mlx.py:203-273`)

**Stays packed** (no unpack/dequant here). Two steps, mirroring AWQ but on the three
MLX tensors directly:

1. **Fused-projection reorder** (`:255, :263-265`): if this is a fused QKV / merged
   gate_up, reorder output dim 0 from contiguous concat `[q|k|v]` into
   interleaved-by-shard layout, so the apply-time slice recovers each projection.
2. **Shard** each tensor along output dim 0 (or input dim for RowParallel) with the
   layer's `weight_sharding`, via `jax.device_put(NamedSharding(mesh, wsh))` (`:266`).

Two **fail-fast guards** worth knowing (they encode correctness invariants):

- **RowParallel input-dim sharding guard** (`:234-246`). For RowParallel the spec is
  `P(None, ATTN_HEAD)` — it shards **axis 1**, which is *packed* (one uint32 = 8
  nibbles) *and grouped* (one affine pair per `gs` inputs). Splitting is correct only
  if **each shard owns whole words AND whole groups** for the same contiguous input
  range, i.e. both `in//pf` and `in//gs` must be divisible by the input shard count.
  The assert fires otherwise. (Qwen3-30B `o_proj` at tp=8: `in=4096 → 512 (÷8✓),
  64 (÷8✓)`.) Pinned numerically by
  `test_mlx_linear_method.py::test_rowparallel_input_dim_sharding_dequant_consistency`
  (asserts `dequant(full) == concat(dequant(per-shard))`).
- **Fused-only guard** (`:252-254`): MLX keeps a *single* packed Parameter per tensor
  and always uses the fused-style apply; the per-projection split path (AWQ's
  `_apply_split`) isn't built, so a *non-fused* multi-projection layer asserts rather
  than mis-slice.

### 3.3 `apply` — dequant in XLA (`mlx.py:275-298`)

```python
weight = mlx_dequantize(jax_view(layer.weight), jax_view(layer.scales),
                        jax_view(layer.biases),
                        group_size=..., bits=...)        # → [out, in] bf16
outs = jnp.einsum("bd,fd->bf", x_jax, weight)            # contract the INPUT dim
... optional + bias ...
outs = slice_sharded_tensor_for_concatenation(outs, output_sizes, n_shards)
return torch_view(jnp.concatenate(outs, axis=-1))        # split fused back into projs
```

So a linear's forward is: unpack nibbles → affine reconstruct → bf16 einsum → split
fused output. Pinned by `test_mlx_linear.py` and the fusion/reorder branches of
`test_mlx_linear_method.py::test_apply_matches_golden`.

---

## 4. MoE specifics — `VllmMLXMoEMethod`

This is the heart of the "keep-4bit" strategy. `mlx.py:347-583`. The headline:
**w13 stays packed int4 in HBM and is dequantized *inside* the GMM kernel**; **w2 is
int4-in-kernel when it shards cleanly, otherwise bf16-at-load**. The two run as
separate `gmm_v2` calls so a mixed int4-w13 / bf16-w2 forward is well-defined
(`mlx.py:347-372` docstring).

### 4.1 Stacked param layout (`create_weights`, `mlx.py:394-429`)

```
w13_weight  uint32 [E, 2I, H//pf]   (gate→w1 first I rows, up→w3 next I rows; packed along H)
w13_scales  bf16   [E, 2I, H//gs]
w13_biases  bf16   [E, 2I, H//gs]
w2_weight   uint32 [E,  H, I//pf]   (down_proj; packed along I)
w2_scales   bf16   [E,  H, I//gs]
w2_biases   bf16   [E,  H, I//gs]
```

Params are tagged `quant_method = FusedMoeWeightScaleSupported.GROUP.value` (`:409`)
so vLLM's `FusedMoE.weight_loader` routes them through the group-scale code path.

### 4.2 The biases loader hack (`_make_mlx_moe_bias_loader`, `mlx.py:301-344`)

This is an MLX-specific gotcha. vLLM's `FusedMoE.weight_loader` routes **by substring**:
names containing `scale`/`zero`/`offset` hit the group-scale branch, names with
`weight` hit the weight branch, and **everything else is silently dropped**
(`return False`). The MLX affine biases get mapped (via
`make_expert_params_mapping`) into params named `w13_biases`/`w2_biases` — matching
**none** of those substrings. Stock vLLM would silently drop them, turning dequant
into `scale*q` (no `+bias`) and corrupting **every** expert.

The fix attaches a **custom per-param `weight_loader`** (set at `:427`) that replicates
the group-scale loader prologue (global→local expert id, `is_transposed` shard-dim
flip) and routes biases through the exact same internal helper the scale branch uses
(`layer._load_model_weight_or_group_weight_scale`, `:334`). `SHARD_ID_TO_SHARDED_DIM
= {w1:0, w3:0, w2:1}` (`:321`).

> **Lesson for adding affine MoE quant:** any affine/zero-point tensor whose param name
> doesn't contain vLLM's routing substrings needs a custom loader, or it silently
> vanishes.

### 4.3 `process_weights_after_loading` — the int4 fold (`mlx.py:431-551`)

The MLX checkpoint codes are **unsigned `[0,15]`**, but the GMM kernel's matmul is
**signed int4 `[-8,7]`**. The fix shifts codes by `-8` and folds the offset back into
the groupbias, exactly preserving the affine value:

`mlx.py:485-492`
```python
w13_codes_u = mlx_unpack(w13q, bits)                 # unsigned [0,15], int32
w13_codes   = (w13_codes_u - 8).astype(jnp.int4)     # signed [-8,7]
w13_scale   = w13s.astype(jnp.float32)
# (q-8)*scale + (bias + 8*scale) == q*scale + bias
w13_groupbias = w13b.astype(jnp.float32) + 8.0 * w13_scale
```

This `(scale, signed-code, groupbias)` triplet then flows through the **same**
`process_moe_weights` / `shard_moe_weights` pipeline that the symmetric-int4 W4A8 MoE
uses (`:520-530`), with the affine `+ groupbias` riding the new groupbias rails (§4.5).

**w2 decision (int4 vs bf16)** — `mlx.py:475-479`:
```python
w2_num_blocks = int(w2_scales.shape[-1])  # I // gs
w2_keep_int4 = (self.moe_backend == MoEBackend.GMM_EP        # EP shards on expert axis
                or w13_reorder_size <= 1                     # no TP sharding
                or w2_num_blocks % w13_reorder_size == 0)    # blocks divide MLP-TP degree
self._w2_int4 = w2_keep_int4
```
Why: in **GMM_TP** the w2 per-group scale/groupbias shard on the *block* dim, so w2 can
stay int4 only when `num_blocks(w2) = I/gs` divides the MLP tensor-parallel degree
(Hy3: `1536/64=24`, `24 % 8 == 0` → int4). For **Qwen3-30B at tp=8** `768/64=12`,
`12 % 8 != 0` → **w2 falls back to bf16-at-load** (`:505-511` runs `mlx_dequantize`).
In **GMM_EP** the scale/groupbias shard on the *expert* axis, so block divisibility is
irrelevant and w2 always stays int4.

The whole transform is wrapped in a single `@jax.jit` (`:481`) and ends with
`jax.effects_barrier()` (`:551`) to release intermediate buffers before the next layer
(mirrors the unquantized path's cross-layer barrier).

### 4.4 `apply_monolithic` — straight to the kernel (`mlx.py:553-583`)

No dequant happens here. It assembles a `FusedMoEWeights` with **packed int4 + scale +
affine groupbias** and hands it to `vllm_moe_apply` → `moe_apply` → `gmm_v2`:

```python
w13_weight=jax_view(layer.w13_weight).astype(jnp.int4),   # defensively re-cast to int4
w13_weight_scale=jax_view(layer.w13_weight_scale),
w13_groupbias=jax_view(layer.w13_groupbias),
w2_weight = ... (int4 + scale + groupbias)  OR  (bf16, scale=None, groupbias=None)
```

> **int4-through-torch-Parameter caveat (`:532-536`):** a torchax-wrapped `int4`
> survives the `torch.nn.Parameter` round-trip but is *reported* as int8 (the
> underlying jax buffer stays int4); `apply_monolithic` re-casts to `int4` defensively
> — same as the W4A8 method.

### 4.5 The `w13_groupbias` rails (process / shard / kernel)

The affine `+ groupbias` term is a **first-class** addition to `FusedMoEWeights`:

`process_weights/moe_weights.py:40-57`
```python
# w13_groupbias / w2_groupbias: per-quant-block affine bias for MLX-style affine
# quant (w = scale*q + groupbias). Rides the SAME rails as the per-group SCALE
# (NOT the per-channel w13_bias/w2_bias MLP-bias rails): it's part of weight
# reconstruction inside the k-loop, so it applies on EVERY shard, exactly like scale.
w13_groupbias: ... | None = None
w2_groupbias:  ... | None = None
```

The crucial design call: **groupbias rides the SCALE rails, not the bias rails.**
Concretely, in `process_moe_weights` it gets the *same* `swapaxes(1,2)` +
`expand_dims(2)` as the scale → lands in `[E, num_blocks, 1, N]`
(`moe_weights.py:291-298`), the *same* w13 reorder (`concat_dim=3`, not 2, because it's
4-D like the scale — `:405-409, :432-436`), and the *same* sharding spec
(`shard_moe_weights`, `:500-510` for GMM_TP; `:478-483` for GMM_EP/FUSED_MOE).

The w2 grouped tensors (scale **and** groupbias) use a block-dim-aware spec computed
from their *own* block dim — replicate when single-block, else shard on the block dim
(`moe_weights.py:488-499, :519-522`). This is the same mechanism that backs the §4.3
w2-int4 divisibility decision.

> **Guard:** `FUSED_MOE` backend does **not** support affine groupbias (it reshapes
> the scale to 5-D but has no groupbias reshape), so a non-None groupbias **asserts**
> there (`moe_weights.py:314-317`). MLX affine MoE therefore requires **GMM_TP or
> GMM_EP**. Backend selection: `interface/moe.py:29-58`
> (`USE_MOE_EP_KERNEL` → FUSED_MOE; `moe.use_ep` → GMM_EP; else GMM_TP).

### 4.6 In-kernel dequant (`gmm_v2`)

The packed-int4 weight, per-group scale, and groupbias flow:
`moe_apply` (`moe.py:127-158`, passes `w1_groupbias=weights.w13_groupbias`,
`w2_groupbias=weights.w2_groupbias`) → `fused_moe_func` → `gmm_wrapper`
(`fused_moe_gmm.py:129-172`) → `gmm_v2(rhs_scale=..., rhs_groupbias=...)`.

`gmm_v2` (`kernels/megablox/gmm_v2.py`) reconstructs the weight **inside the k-loop**:
matmul the (full-precision) LHS against the int4 RHS codes, scale by the per-group
`rhs_scale`, then add the affine term `groupbias · sum(lhs over the block)`:

`gmm_v2.py` (unquantized-LHS / affine path, ~`:449-471`)
```python
block_acc  = matmul(lhs[:, k0:k1], int4_rhs[k0:k1, ...], preferred=f32)
block_acc *= rhs_scale[b_id, :, n0:n1]                      # per-group scale
block_acc += rhs_groupbias[b_id, :, n0:n1] * sum(lhs[:, k0:k1])   # affine + bias
```
So it is **dequant-to-bf16/fp32 reconstruct-in-kernel then matmul**, *not* a fully
fused int×int matmul — the int4 weight lives in HBM (8× smaller), is read into the
kernel, and reconstructed block-by-block on chip. (`b_id` is the per-quant-block
index; `groupbias·Σlhs` is the algebraic expansion of `Σ (scale·q + groupbias)·x`.)

> **Why LHS stays full-precision on the affine path** (`fused_moe_gmm.py:138-156`):
> `gmm_v2`'s quantized-LHS fast path strides the k-loop by the *512-wide* LHS quant
> block and indexes the RHS scale/groupbias by `start_k // rhs_block`. When the RHS
> block is finer than 512 (MLX `gs=64`), the per-block scale/groupbias collapses to
> `b_id=0` and silently corrupts the result. So the wrapper sets
> `maybe_quantize_lhs = rhs_groupbias is None` — affine paths keep the LHS full
> precision; every non-affine path (bf16, symmetric W4A8) is left byte-identical.

### 4.7 Relationship to W4A8 (symmetric int4 MoE)

MLX's int4 MoE deliberately **mirrors** `VllmCompressedTensorsW4A8MoEMethod`
(`tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8.py`):
both push int4 codes + per-group scale through `process_moe_weights` → `gmm_v2`. The
**only** structural difference is that MLX is *affine* — it adds the `w13_groupbias` /
`w2_groupbias` term (W4A8 is symmetric, `groupbias=None`). That's why the groupbias
rails were added rather than overloading the existing per-channel `w13_bias` rails:
groupbias is part of *weight reconstruction* (must apply on every shard, like scale),
whereas `w13_bias` is an MLP output bias.

---

## 5. Numerical-equivalence guarantees (tests)

The in-kernel int4 path is asserted **token-exact** against a bf16 reference; the
load-time and in-kernel dequant must agree. Most illuminating tests:

| Test | Asserts | Anchor |
|---|---|---|
| `test_mlx_dequant.py::test_mlx_dequantize_matches_affine_golden` | `mlx_dequantize == w=scale*q+bias` golden (incl. negative scale) | `:10-21` |
| `test_mlx_dequant.py::test_unpack_low_nibble_first` | nibble order is low-first (`0x76543210→[0..7]`) | `:35-39` |
| `test_mlx_dequant_independent.py::test_negative_control_reversed_nibble_order_is_detected` | the bf16-grid check has teeth (reversed unpack is caught) | `:371-411` |
| `test_mlx_linear_method.py::test_rowparallel_input_dim_sharding_dequant_consistency` | `dequant(full) == concat(dequant(per-shard))` for RowParallel (whole words+groups) | `:193-242` |
| `test_mlx_moe_method.py::test_process_weights_dequant_matches_golden_experts` | w13 **and** w2: signed-int4 fold + non-zero groupbias reconstruct golden; wrong fold / dropped bias must fail | `:241-358` |
| `test_mlx_weight_transform.py::test_experts_unstacked_and_renamed_kept_packed` | switch_mlp → per-expert, stays uint32 | `:39-57` |
| `test_mlx_weight_transform.py::test_embed_and_lm_head_dequantized_to_bf16_weight` | embed/lm_head → plain bf16 `torch.Tensor` at load | `:60-81` |
| `test_qwen3_moe_mlx_int4_e2e.py::test_synthetic_mlx_moe_logits_match_bf16_reference` (tp=1,2) | greedy tokens **exactly** match bf16 ref — proves sharded w13+w2 groupbias is applied correctly | `:76-150` |
| `test_qwen3_moe_mlx_int4_e2e.py::test_real_mlx_30b_auto_loader_serves_coherently` | real `Qwen3-30B-A3B-4bit`, `load_format="auto"` selects streaming loader, coherent ASCII output | `:157-182` |

(Full test inventory: `tests/layers/common/quantization/test_mlx_dequant*.py`,
`tests/layers/vllm/quantization/test_mlx_*.py`,
`tests/models/vllm/test_mlx_weight_transform.py`,
`tests/models/vllm/test_qwen3_moe_mlx_int4_e2e.py`,
`tests/utils/test_mlx_synthetic.py`, `tests/kernels/megablox/test_gmm_v2_groupbias.py`.)

---

## 6. Status, open questions, and gotchas for the next engineer

- **Stage naming.** The MoE docstrings reference "Stage-1" (dequant-at-load → bf16)
  and "Stage-2/3" (w13 in-kernel int4 + optional w2 int4). Current code is the hybrid:
  w13 always int4-in-kernel, w2 conditional (§4.3). The history (`feat(quant): MLX MoE
  Stage-2 …`, commits `31315914`, `981b1639`) confirms in-kernel w13 + `rhs_groupbias`
  landed together.
- **Linears never go int4-in-kernel.** Only the MoE keeps packed int4 to HBM; dense
  linears (`VllmMLXLinearMethod`) always dequant to bf16 in `apply` (§3.3). If a
  fused int4 linear matmul is ever wanted, that's net-new.
- **FUSED_MOE backend is unsupported for MLX** (affine groupbias assert,
  `moe_weights.py:314-317`). MLX needs GMM_TP/GMM_EP.
- **w2 path is data-dependent.** Whether w2 is int4 or bf16 depends on
  `I/gs % MLP-TP-degree` and the backend — don't assume w2 is always one or the other
  when reasoning about HBM or numerics.
- **Couldn't fully read** the exact gmm_v2 line numbers in this doc (the kernel is
  large and the `:449-471` range is from a sub-agent trace, not a direct quote here);
  the *behavior* (scale then `groupbias·Σlhs` in the k-loop, signed int4 RHS, LHS kept
  full-precision on affine) is verified via `fused_moe_gmm.py:138-156` and the e2e
  exact-match test. If you touch the kernel, re-read `gmm_v2.py` around the
  `has_groupbias` blocks directly.
- **`HYV3ForCausalLM` model class** lives in vLLM (the transform's `shared_mlp`/router
  handling targets `HYV3MoEFused`/`HYV3FeedForward`); this doc covers the tpu-inference
  side only. The Hy3 chat template defaults to *no-think*; `reasoning_effort` toggles
  thinking (per project memory) — orthogonal to quant.
