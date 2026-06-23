# 05 — A Concrete End-to-End Trace: Qwen2 on the torchax Route

> **Read this first.** This is the "walk me through one real model, start to finish"
> doc. We pick the simplest interesting dense model — **Qwen2** — and trace it from
> the moment vLLM instantiates its PyTorch `nn.Module` tree all the way down to the
> Pallas attention kernel on TPU, under the **torchax route** (`MODEL_IMPL_TYPE=vllm`).
>
> Where deeper machinery (layer patching, the JIT wrapper, kernels, weight loading)
> is documented in sibling docs, we state the fact and cite the anchor rather than
> re-deriving it. The *trace itself* is concrete and verified against code.

All anchors are `path:line`. Two repos:
- **tpu-inference** (this repo): `/home/enyouki/tpu-inference`, package `tpu_inference/`.
- **vLLM** (editable): `/home/enyouki/vllm/vllm`.

---

## 0. Which model class actually runs? (and a routing gotcha)

On the torchax route, **no JAX rewrite is involved**. vLLM's own PyTorch model class
is instantiated unchanged and executed as JAX via torchax. For Qwen2 that class is
**`Qwen2ForCausalLM`**, defined in:

- `/home/enyouki/vllm/vllm/model_executor/models/qwen2.py:527`

with the module tree:

```
Qwen2ForCausalLM                      qwen2.py:527
└─ model: Qwen2Model                  qwen2.py:358
   ├─ embed_tokens: VocabParallelEmbedding   qwen2.py:391
   ├─ layers[N]: Qwen2DecoderLayer    qwen2.py:239
   │  ├─ input_layernorm: RMSNorm     qwen2.py:287
   │  ├─ self_attn: Qwen2Attention    qwen2.py:120
   │  │  ├─ qkv_proj: QKVParallelLinear      qwen2.py:159
   │  │  ├─ o_proj:   RowParallelLinear      qwen2.py:168
   │  │  ├─ rotary_emb: <get_rope(...)>      qwen2.py:181
   │  │  └─ attn: Attention (vLLM wrapper)   qwen2.py:192
   │  ├─ post_attention_layernorm: RMSNorm   qwen2.py:288
   │  └─ mlp: Qwen2MLP                qwen2.py:280
   │     ├─ gate_up_proj: MergedColumnParallelLinear  qwen2.py:93
   │     ├─ down_proj:    RowParallelLinear           qwen2.py:100
   │     └─ act_fn: SiluAndMul        qwen2.py:111
   └─ norm: RMSNorm                   qwen2.py:415
└─ lm_head: ParallelLMHead (or tied embed_tokens)     qwen2.py:554
└─ logits_processor: LogitsProcessor  qwen2.py:567
```

### Routing gotcha — you must *ask* for the torchax route for Qwen2

`tpu_inference/models/common/model_loader.py::get_model()` dispatches on
`envs.MODEL_IMPL_TYPE` (`model_loader.py:463`). The `case "vllm"` branch
(`model_loader.py:502`) is the torchax route. But **`auto` does not pick `vllm` for
Qwen2.** `resolve_model_architecture()` returns:

```python
# model_loader.py:563
impl = "vllm" if arch in _VLLM_PREFERRED_ARCHITECTURES else "flax_nnx"
```

and `_VLLM_PREFERRED_ARCHITECTURES` is only `{GptOssForCausalLM, Qwen3MoeForCausalLM}`
(`model_loader.py:51`). Qwen2 **has a JAX rewrite** at
`tpu_inference/models/jax/qwen2.py` (registered `model_loader.py:93`), so `auto`
resolves Qwen2 to **`flax_nnx`**, not torchax. A code comment even flags this:
`"currently Qwen2ForCausalLM is using legacy model implementation"`
(`model_loader.py:153`).

**Conclusion:** to run Qwen2 through the torchax route described here you set
`MODEL_IMPL_TYPE=vllm` explicitly. That is exactly the route this doc traces — it is
the same machinery a *new* model (with no JAX rewrite) would ride by default. So Qwen2
is the perfect concrete teaching example: simple, dense, and exercises the full
torchax pipeline.

---

## 1. The one idea that makes everything click

vLLM's Qwen2 is **plain PyTorch**. It never knows it's on a TPU. Two mechanisms make
it run as sharded JAX:

1. **Class patching (`register_oot`).** When vLLM constructs the module tree, certain
   layer classes have been replaced by TPU subclasses so that their *weights* are
   converted to sharded `jax.Array`s and their `apply()` runs a JAX matmul. This is
   `tpu_inference/layers/vllm/` (see **doc 03 — layer patching**). The patches are
   registered by importing the package, triggered by the `register_layers` vLLM plugin
   entry point (`tpu_inference/layers/vllm/__init__.py:21`, registered in
   `setup.py`).

2. **torchax tracing.** The actual `forward()` runs inside a torchax env
   (`torchax.default_env()`), so *every* torch op that isn't explicitly patched —
   `RMSNorm`, `SiluAndMul`, the residual adds, the standard `RotaryEmbedding` math —
   is lowered op-by-op to JAX and compiled by XLA. There is **no per-op kernel** for
   these; they're just traced (see **doc 02 — torchax**).

So the layer-by-layer mapping below has two flavors:
- **Patched** layers → a specific TPU op/method/kernel (embedding, linear, attention).
- **Traced** layers → "plain torch, lowered to JAX by torchax" (norm, activation,
  standard RoPE).

Both are correct and both run on TPU. The distinction matters when you add a new
model: you only need to *patch* the structural layers; the pointwise math comes along
for free via tracing.

---

## 2. Layer-by-layer mapping table

| vLLM module (Qwen2) | What happens on TPU | TPU mechanism / file |
|---|---|---|
| `VocabParallelEmbedding` / `ParallelLMHead` (`qwen2.py:391`,`558`) | **Patched.** Weights sharded as `jax.Array`; gather is a JAX op. | `VllmVocabParallelEmbedding` / `VllmParallelLMHead` `layers/vllm/custom_ops/embedding.py:23,41`; method `VllmUnquantizedEmbeddingMethod` `layers/vllm/quantization/unquantized.py` |
| `RMSNorm` (`qwen2.py:287,288,415`; q/k_norm only if `qk_norm`) | **Traced.** Standard torch RMSNorm `forward_native`, lowered to JAX by torchax. **No custom kernel.** | not patched in `tpu_inference/`; runs under `torchax.default_env()` |
| `rotary_emb` from `get_rope(...)` (`qwen2.py:181`) → `RotaryEmbedding` | **Traced** for Qwen2 (NEOX-style `RotaryEmbedding`, `vllm/.../rotary_embedding/base.py:118`). The TPU RoPE custom op only patches the **Deepseek** non-NEOX variant. | `rope.py` patches only `DeepseekScalingRotaryEmbedding` (`layers/vllm/custom_ops/rope.py:32`); plain Qwen2 RoPE is traced, **not** routed here |
| `QKVParallelLinear` (`qwen2.py:159`) | **Patched.** `super().forward` → `quant_method.apply` → `jnp.einsum` sharded matmul. Out-dim sharded over heads. | `VllmColumnParallelLinear` base `layers/vllm/custom_ops/linear.py:32`; `VllmUnquantizedLinearMethod.apply` `layers/vllm/quantization/unquantized.py:242`; einsum `layers/common/quantization/unquantized.py:60,89` |
| `o_proj`, `down_proj` = `RowParallelLinear` (`qwen2.py:168,100`) | **Patched.** Same path; in-dim sharded, reduce-scatter implicit in JAX sharding. | `VllmRowParallelLinear` `layers/vllm/custom_ops/linear.py:22` |
| `gate_up_proj` = `MergedColumnParallelLinear` (`qwen2.py:93`) | **Patched** (subclass of ColumnParallel). Fused gate+up, out-dim sharded. | same linear method as QKV; fused matmul `_apply_fused` `layers/common/quantization/unquantized.py:41` |
| `SiluAndMul` (`qwen2.py:111`) | **Traced.** `silu(x[:half]) * x[half:]`, lowered to JAX. **No custom kernel.** | not patched; runs under torchax env |
| `attn = Attention(...)` (`qwen2.py:192`) → `self.attn(q,k,v)` (`qwen2.py:234`) | **Backend-swapped.** vLLM `Attention.impl` is the TPU Pallas backend; calls `ragged_paged_attention`. | `PallasAttentionBackendImpl` `layers/vllm/backends/flash_attn.py`; kernel `kernels/ragged_paged_attention/v3/kernel.py:1568` |
| `LogitsProcessor` (`qwen2.py:567`) via `compute_logits` (`qwen2.py:588`) | **Wrapped + jitted.** `lm_head` matmul over hidden states; runs as a separate jitted JAX function. | `compute_logits_fn` `runner/tpu_runner.py:902`; jit `models/vllm/vllm_model_wrapper.py:527` |

> Key takeaway for adding models: **structural** layers (embedding / linear /
> attention) are patched; **pointwise** layers (norm / activation / standard RoPE) are
> traced. Qwen2 needs zero new patches because vLLM already builds it from layers
> tpu-inference already patches.

---

## 3. Walk each submodule

### 3.1 `embed_tokens` — `VocabParallelEmbedding`

The class is replaced at construction by `VllmVocabParallelEmbedding`
(`layers/vllm/custom_ops/embedding.py:23`). Its `__init__` swaps in the TPU quant
method so weights get processed/sharded as `jax.Array`s:

```python
# embedding.py:28
if isinstance(self.quant_method, UnquantizedEmbeddingMethod):
    mesh = get_current_vllm_config().quant_config.mesh
    self.quant_method = VllmUnquantizedEmbeddingMethod(mesh)
```

`forward()` just calls `super().forward(input_)` — the gather runs against the sharded
JAX weight under the torchax env. The vocab dimension is sharded on the model axis
(`PartitionSpec(ShardingAxisName.MLP_TENSOR, None)` in the embedding method). If
`config.tie_word_embeddings`, `lm_head` *is* this same module (`qwen2.py:556`).

### 3.2 `input_layernorm` / `post_attention_layernorm` / `norm` — `RMSNorm`

**Not patched.** Confirmed: the only `register_oot` patches in `layers/vllm/` are
embedding, linear, fused_moe, MLA/GDN attention, and the *Deepseek* RoPE — `RMSNorm`
appears nowhere as a patch target. Qwen2's `RMSNorm` runs vLLM's standard
`forward_native` (the fused-residual form, `qwen2.py:303,310`), and torchax lowers the
`x * rsqrt(mean(x²)+eps) * weight` graph straight to JAX/XLA. No Pallas kernel. This is
fine: XLA fuses it well, and correctness comes for free from the torch→JAX op coverage.

### 3.3 `rotary_emb` — `get_rope(...)` → `RotaryEmbedding`

`Qwen2Attention.__init__` builds RoPE via `get_rope(...)` (`qwen2.py:181`). For Qwen2's
config this returns the **NEOX-style `RotaryEmbedding`**
(`vllm/.../rotary_embedding/base.py:118`). In `forward` it's applied right before
attention:

```python
# qwen2.py:233
q, k = self.rotary_emb(positions, q, k)
```

There **is** a TPU RoPE custom op at `layers/vllm/custom_ops/rope.py`, but it only
patches `DeepseekScalingRotaryEmbedding` (`rope.py:32`) and only overrides the
**non-NEOX** path (`rope.py:49` returns `super()` for NEOX). So for **plain Qwen2 RoPE
this file does not apply** — the standard `RotaryEmbedding.forward_native` is traced by
torchax to JAX. (The `rope.py` op exists for models like DeepSeek; cite it when you get
there, but don't attribute Qwen2's RoPE to it.)

### 3.4 QKV / O / gate_up / down — the Linear layers

This is the most important patched path. The vLLM wrappers are thin
(`layers/vllm/custom_ops/linear.py:22–49`): `VllmRowParallelLinear` /
`VllmColumnParallelLinear` just call `super().forward(input_)`. The real work is in the
**quant method** that vLLM's `LinearBase.forward` dispatches to. For unquantized Qwen2:

```
VllmRowParallelLinear.forward(x)        linear.py:22  → super().forward
  └ self.quant_method.apply(self, x, bias)
      VllmUnquantizedLinearMethod.apply  layers/vllm/quantization/unquantized.py:242
        ├ x_jax = jax_view(x)            # torch → JAX (same TPU buffer)
        ├ _apply_fused / _apply_split    layers/common/quantization/unquantized.py:41 / :69
        │   └ jnp.einsum("...n,pn->...p", x_jax, weight_jax)   :60 / :89
        └ return torch_view(out_jax)     # JAX → torch
```

The matmul is a single `jnp.einsum`; sharding is **not** manual — the weight is a
`NamedSharding`'d `jax.Array`, so XLA inserts the collective (all-gather / reduce-
scatter) automatically. The PartitionSpecs come from `VllmQuantLinearConfig`
(`layers/vllm/quantization/configs.py`) keyed off `ShardingAxisName`
(`layers/common/sharding.py`):

| Qwen2 layer | vLLM base | weight PartitionSpec (concept) | sharded dim |
|---|---|---|---|
| `qkv_proj`, `gate_up_proj` | Column/Merged-Column | `P(ATTN_HEAD, None)` | output (heads / intermediate) |
| `o_proj`, `down_proj` | Row | `P(None, ATTN_HEAD)` | input (heads / intermediate) |

(`ATTN_HEAD` is the `model` mesh axis in the default 2D mesh.) The `mesh` itself is
threaded onto the quant config at setup (`quant_config.mesh`), originating from the
runner's device mesh.

> Weight processing (the `t2j` + shard step) happens in `process_weights_after_loading`
> of the same method (`layers/vllm/quantization/unquantized.py`) — see §4.

### 3.5 `attn` — the attention call

After QKV + RoPE, `Qwen2Attention.forward` calls the vLLM `Attention` wrapper:

```python
# qwen2.py:234
attn_output = self.attn(q, k, v)
```

`Attention` is **not** subclassed; instead its `.impl` is the TPU backend. The backend
is chosen by `TpuPlatform.get_attn_backend_cls` (`platforms/tpu_platform.py`). For a
plain dense decoder (Qwen2, `use_mla=False`) it returns the **FLASH_ATTN** path, which
resolves to `PallasAttentionBackend` (`layers/vllm/backends/flash_attn.py:32`). The
chain:

```
Attention.forward(q,k,v)                       vllm .../attention/attention.py
  └ self.impl = PallasAttentionBackendImpl     flash_attn.py:get_impl_cls
      .forward(...)                            flash_attn.py
        └ attention(kv_cache,q,k,v,attn_metadata,mesh,...)
            layers/common/attention_interface.py:406
          └ sharded_ragged_paged_attention(...)
              layers/common/attention_interface.py:326
            └ ragged_paged_attention(...)      # Pallas TPU kernel
                kernels/ragged_paged_attention/v3/kernel.py:1568
```

The KV cache layout, block tables, and per-request lengths arrive in an
`AttentionMetadata` (`layers/common/attention_metadata.py`): `block_tables`,
`seq_lens`, `query_start_loc`, `request_distribution`. This metadata is *not* a model
argument — it's injected via vLLM's forward context (see §5). The same kernel handles
mixed prefill+decode batches, which is why one decode step and one prefill step take
the same path. (Full kernel/backend internals: **doc 06 — attention & kernels**.)

### 3.6 `mlp` — `Qwen2MLP`

```python
# qwen2.py:113
def forward(self, x):
    gate_up, _ = self.gate_up_proj(x)   # patched linear → einsum (§3.4)
    x = self.act_fn(gate_up)            # SiluAndMul, traced
    x, _ = self.down_proj(x)            # patched linear → einsum (§3.4)
    return x
```

`gate_up_proj` (fused, column-parallel) and `down_proj` (row-parallel) ride the linear
path from §3.4. `SiluAndMul` is traced. That's the whole MLP — two sharded matmuls and
one pointwise op.

### 3.7 LM head + logits

After all layers and the final `norm`, `Qwen2Model.forward` returns `hidden_states`
(`qwen2.py:459`). `Qwen2ForCausalLM.compute_logits` (`qwen2.py:588`) runs the
`logits_processor` against `lm_head`. On TPU this is invoked as a **separate jitted
function**, `compute_logits_fn` (`runner/tpu_runner.py:902`), jitted in
`models/vllm/vllm_model_wrapper.py:527`. Sampling then happens in JAX (`sample(...)`,
`runner/tpu_runner.py:949`).

---

## 4. How weights load (concretely, for Qwen2)

The torchax route loads weights with vLLM's *own* loader and then shards the resulting
tensors to TPU. Ordered chain:

```
get_model()                              model_loader.py:463
 └ case "vllm" → get_vllm_model(...)     model_loader.py:502 → :412
     └ VllmModelWrapper.load_weights()   models/vllm/vllm_model_wrapper.py:169
         ├ vllm_get_model(...)           # instantiate PyTorch Qwen2ForCausalLM, load HF weights on CPU
         │    └ Qwen2ForCausalLM.load_weights → AutoWeightsLoader   qwen2.py:595
         │         └ Qwen2Model.load_weights   qwen2.py:461  (stacked QKV/gate_up mapping)
         ├ self.model = _VllmRunner(vllm_model)            vllm_model_wrapper.py:287
         └ params_and_buffers = shard_model_to_tpu(...)    vllm_model_wrapper.py:288
              # each CPU torch tensor → t2j → general_device_put(sharding) → torch_view
         return jax_view(params_and_buffers)               vllm_model_wrapper.py:316
```

Two things worth internalizing:

1. **The model's own `load_weights` does the HF-name remapping.** `Qwen2Model.load_weights`
   (`qwen2.py:461`) carries the `stacked_params_mapping` that fuses `q/k/v_proj` →
   `qkv_proj` and `gate_proj/up_proj` → `gate_up_proj` (`qwen2.py:462–469`). This is
   ordinary vLLM logic — torchax doesn't change it. A new torchax model gets this for
   free if its vLLM class already defines the mapping.
2. **Sharding is applied as a `t2j` + device-put per tensor**, using the same
   PartitionSpecs the quant method expects (§3.4). After this, every parameter is a
   sharded `jax.Array` viewed as a torch tensor (`torch_view`). (Full loading details:
   **doc 04 — weight loading**.)

---

## 5. A single decode step, end to end

The runner calls a **jitted** step function. The wrapped torch model is invoked inside
it via `torch.func.functional_call`, with the sharded params as the parameter dict and
all the TPU plumbing established by context managers:

```
TPURunner step:
  model_fn(state, kv_caches, input_ids, attn_metadata,
           inputs_embeds, input_positions, layer→kv index, ...)   runner/tpu_runner.py:862
   │  (model_fn = jitted step_fun, models/vllm/vllm_model_wrapper.py:318)
   ▼
  step_fun(...):                                   vllm_model_wrapper.py:340
    with torchax.default_env(),                    # torch ops → JAX
         set_vllm_model_wrapper_context(kv_caches, mesh, layer→kv index),  # KV cache plumbing
         set_forward_context(attn_metadata=attn_metadata, vllm_config):    # injects attn metadata
      output = torch.func.functional_call(         vllm_model_wrapper.py:368
          self.model,                              # _VllmRunner(Qwen2ForCausalLM)
          torch_view(params_and_buffers),          # sharded jax.Array → torch view
          kwargs={"input_ids": ..., "positions": ..., "inputs_embeds": ...})
      new_kv_caches = get_vllm_model_wrapper_context().kv_caches   vllm_model_wrapper.py:381
    return new_kv_caches, jax_view(output), aux
```

What the three context managers buy you (this is the crux of the whole route):

- `torchax.default_env()` — makes every torch op inside Qwen2's forward dispatch to
  JAX. This is why `RMSNorm`, `SiluAndMul`, RoPE, residual adds "just work."
- `set_vllm_model_wrapper_context(...)` — stashes the KV caches and the
  layer→KV-cache-index map so the patched attention can read/write the right cache
  slice. Updated caches are pulled back out at `vllm_model_wrapper.py:381` (note
  `@jax.jit(donate_argnames=("kv_caches",))` at `:320` — caches are donated and
  returned, not mutated in place).
- `set_forward_context(attn_metadata=...)` — vLLM's standard mechanism; the
  `Attention` layer reads `attn_metadata` from the forward context (not from a
  function arg), which is how block tables / seq lens reach the Pallas kernel (§3.5).

Then the **whole thing is `jax.jit`-compiled** (`vllm_model_wrapper.py:320`), so the
torch trace is captured once and replayed as a fused XLA program. Logits and sampling
run as separate jitted steps (§3.7).

### The decode step as one picture

```
input_ids, positions ──► embed_tokens (patched gather)
        │
        ▼  for each Qwen2DecoderLayer:
   ┌───────────────────────────────────────────────────────────┐
   │ residual = h                                               │
   │ h = input_layernorm(h)            [RMSNorm, traced]        │
   │ qkv = qkv_proj(h)                 [linear, einsum]         │
   │ q,k = rotary_emb(pos, q, k)       [RoPE, traced]           │
   │ a = attn(q,k,v)  ── PallasAttentionBackendImpl             │
   │        └► ragged_paged_attention (Pallas kernel, KV cache) │
   │ h = o_proj(a)                     [linear, einsum]         │
   │ h, residual = post_attention_layernorm(h, residual)       │
   │ gate_up = gate_up_proj(h)         [linear, einsum]         │
   │ h = SiluAndMul(gate_up)           [activation, traced]     │
   │ h = down_proj(h)                  [linear, einsum]         │
   │ h = h + residual                                           │
   └───────────────────────────────────────────────────────────┘
        │
        ▼  norm (RMSNorm, traced)
   hidden_states ──► compute_logits (lm_head matmul, jitted) ──► sample (JAX)
```

Everything inside the box runs as one jitted XLA program over sharded `jax.Array`s on
the TPU mesh.

---

## 6. What this means for adding a new torchax model

Qwen2 demonstrates the happy path. To bring up a new dense model on the torchax route:

1. **Reuse vLLM's PyTorch class if it exists** — you write *no model code*. Set
   `MODEL_IMPL_TYPE=vllm` (since `auto` only auto-selects torchax for the few archs in
   `_VLLM_PREFERRED_ARCHITECTURES`, `model_loader.py:51`).
2. **Your model is "supported" if it's built from already-patched layers** — the
   embedding, the linear variants, and the `Attention` wrapper. Pointwise math
   (norms, activations, NEOX RoPE) needs nothing; torchax traces it.
3. **You only write a patch when a layer is structural and unpatched** — e.g. a new
   attention variant (cf. MLA / GDN in `custom_ops/`) or a non-NEOX RoPE
   (cf. `custom_ops/rope.py`). For DeepSeek-v4 the work concentrates in the MLA
   attention + RoPE + (if MoE) the fused-MoE path, *not* in the dense scaffolding,
   which Qwen2 already proves out.
4. **Weight name remapping comes from the vLLM class's `load_weights`**
   (`qwen2.py:461`) — if vLLM already handles the checkpoint, sharding is automatic via
   `shard_model_to_tpu`.

In short: the dense skeleton is solved. New work lives in the patched structural
layers, and Qwen2 is the reference for how those slot together.

---

## Most important anchors

- `vllm/.../models/qwen2.py:527` — `Qwen2ForCausalLM` (the class that runs).
- `qwen2.py:120,159,168,181,192,234` — `Qwen2Attention`: QKV/O proj, RoPE, attn call.
- `qwen2.py:83,93,100,111,113` — `Qwen2MLP`: gate_up/down/SiluAndMul.
- `qwen2.py:461` — `Qwen2Model.load_weights` (stacked QKV/gate_up mapping).
- `model_loader.py:463,502,51,563` — dispatch + the `auto`→`flax_nnx` gotcha for Qwen2.
- `layers/vllm/custom_ops/linear.py:22,32` — patched linear wrappers.
- `layers/vllm/quantization/unquantized.py:242` — `apply` (matmul entry).
- `layers/common/quantization/unquantized.py:60,89` — the `jnp.einsum` matmul.
- `layers/vllm/custom_ops/embedding.py:23,41` — patched embedding / lm_head.
- `layers/vllm/custom_ops/rope.py:32` — Deepseek-only RoPE op (NOT Qwen2's path).
- `layers/vllm/backends/flash_attn.py:32` — `PallasAttentionBackend`.
- `kernels/ragged_paged_attention/v3/kernel.py:1568` — the attention Pallas kernel.
- `models/vllm/vllm_model_wrapper.py:169,287,288,316,318,340,368,527` — load + jitted step.
- `runner/tpu_runner.py:862,902,949` — forward call, compute_logits, sample.
