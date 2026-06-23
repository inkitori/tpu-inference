# 03 — Layer Patching & Custom Ops: how tpu-inference hooks into vLLM's PyTorch layers

> Scope: the **general mechanism** by which `tpu_inference` overrides vLLM's PyTorch
> `nn.Module` layer classes and individual torch ops on the **torchax route**
> (`MODEL_IMPL_TYPE=vllm`), for **dense models**. Quantization internals (`fp8`,
> `awq`, `mxfp4`, `mlx`, `compressed_tensors`) and **MLA**/**GDN** kernel details are
> owned by sibling docs; this doc shows *where they plug in*, not how their kernels work.

All `path:line` anchors below are verified against the `hy3` branch.
- tpu-inference: `/home/enyouki/tpu-inference` (package `tpu_inference/`)
- vLLM (editable): `/home/enyouki/vllm/vllm`
- torchax (installed): `.venv/.../site-packages/torchax/`

---

## 0. TL;DR — there are exactly **four** patch mechanisms

tpu-inference does **not** edit vLLM. It plugs into four extension points, all
fired as **import side-effects** when the `tpu_inference.layers.vllm` package is
imported:

| # | Mechanism | What it overrides | vLLM-side hook | TPU-side example |
|---|-----------|-------------------|----------------|------------------|
| 1 | **`@Base.register_oot`** | a whole vLLM `nn.Module` **layer class** (swapped in at instantiation) | `CustomOp`/`PluggableLayer.__new__` consults `op_registry_oot` | `VllmFusedMoE`, `VllmRowParallelLinear`, `VllmVocabParallelEmbedding`, `VllmDeepseekScalingRotaryEmbedding`, `VllmGatedDeltaNetAttention`, `VllmMultiHeadLatentAttentionWrapper` |
| 2 | **`@register_function(torch_fn)`** (torchax) | a single **torch function/op** → JAX impl | torchax `XLAFunctionMode.__torch_function__` → `Environment.dispatch` | `torch.nn.functional.scaled_dot_product_attention` → sharded flash attn |
| 3 | **`@register_backend(Enum)`** (vLLM) | the **attention backend** object selected for the model | `TpuPlatform.get_attn_backend_cls` returns the enum path; vLLM looks it up in its backend registry | `PallasAttentionBackend` (dense), `PallasMLAttentionBackend` (MLA) |
| 4 | **quant-method override** (`process_weights_after_loading` / `apply`) | per-layer **weight processing** + the **matmul/attention compute** | vLLM calls `quant_method.process_weights_after_loading(...)` after load; `forward` calls `quant_method.apply(...)` | `VllmUnquantizedLinearMethod`, `VllmUnquantizedFusedMoEMethod`, … |

Plus one **defensive shim**:

| 5 | **`torch.ops._C` dummy ops** | stubs vLLM's CUDA `_C` kernel namespace so importing CUDA-oriented vLLM code doesn't crash on TPU | `torch.library.define` + no-op impl | `_C::rms_norm`, `_C::rotary_embedding`, `_C::*_fp8_quant`, … |

The remainder of the doc walks each one with code.

---

## 1. Activation: how the patches get loaded (`register_layers` is empty!)

### 1.1 The entry point

`setup.py` declares a `vllm.general_plugins` entry point
(`/home/enyouki/tpu-inference/setup.py:95-99`):

```python
entry_points={
    "vllm.general_plugins": [
        "register_layers = tpu_inference.layers.vllm:register_layers",
    ],
},
```

### 1.2 `register_layers` does nothing — the imports do everything

`tpu_inference/layers/vllm/__init__.py:14-22`:

```python
from tpu_inference.layers.vllm import backends as backends
from tpu_inference.layers.vllm import custom_ops as custom_ops
from tpu_inference.layers.vllm import ops as ops
from tpu_inference.layers.vllm import quantization as quantization

# NOTE: this empty function exists for an entry_points target for vllm plugin.
def register_layers():
    pass
```

The body is `pass`. The real work is the four **top-level imports**: importing the
package transitively imports every `custom_ops/*.py`, `backends/*.py`, `ops/*.py`,
and `quantization/*.py` module, and **running those modules executes the
decorators** (`@X.register_oot`, `@register_function(...)`, `@register_backend(...)`).
Registration is a pure import side-effect.

### 1.3 Where vLLM triggers it

vLLM loads general plugins at engine-args construction:

```
EngineArgs.__post_init__                      vllm/engine/arg_utils.py:718-720
  → load_general_plugins()                    vllm/plugins/__init__.py:69
    → load_plugins_by_group("vllm.general_plugins")   vllm/plugins/__init__.py:28
        func = plugin.load()                  vllm/plugins/__init__.py:61  ← IMPORTS tpu_inference.layers.vllm (fires all decorators)
      for func in plugins.values(): func()    vllm/plugins/__init__.py:81-82  ← calls the no-op register_layers()
```

`plugin.load()` is the load-bearing line: resolving the entry-point target imports
the `tpu_inference.layers.vllm` package, which is when every decorator runs.

```
setup.py entry_point ──► vLLM load_general_plugins ──► import tpu_inference.layers.vllm
                                                          ├─ custom_ops/*  → @register_oot     (mech 1)
                                                          ├─ ops/*         → @register_function (mech 2)
                                                          ├─ backends/*    → @register_backend  (mech 3)
                                                          └─ quantization/* → quant configs/methods (mech 4)
```

---

## 2. Mechanism 1 — `register_oot`: swapping an entire vLLM layer class

This is vLLM's **out-of-tree (OOT) layer** facility. It is **not** torchax.

### 2.1 The TPU-side usage (what a patch looks like)

Every TPU layer subclasses the vLLM base and decorates with `@Base.register_oot`.
The dense-model set (all in `tpu_inference/layers/vllm/custom_ops/`):

| vLLM base class | TPU subclass | file:line |
|---|---|---|
| `RowParallelLinear` | `VllmRowParallelLinear` | `custom_ops/linear.py:22` |
| `ColumnParallelLinear` | `VllmColumnParallelLinear` | `custom_ops/linear.py:32` |
| `ReplicatedLinear` | `VllmReplicatedLinear` | `custom_ops/linear.py:42` |
| `VocabParallelEmbedding` | `VllmVocabParallelEmbedding` | `custom_ops/embedding.py:23` |
| `ParallelLMHead` | `VllmParallelLMHead` | `custom_ops/embedding.py:41` |
| `FusedMoE` | `VllmFusedMoE` | `custom_ops/fused_moe.py:19` |
| `DeepseekScalingRotaryEmbedding` | `VllmDeepseekScalingRotaryEmbedding` | `custom_ops/rope.py:32` |
| `GatedDeltaNetAttention` | `VllmGatedDeltaNetAttention` | `custom_ops/gdn_attention_op.py:148` (GDN — sibling doc) |
| `MultiHeadLatentAttentionWrapper` | `VllmMultiHeadLatentAttentionWrapper` | `custom_ops/mla_attention.py:153` (MLA — sibling doc) |

Note how thin most are. `linear.py` and `fused_moe.py` overrides do nothing but
call `super().forward(...)`:

```python
@RowParallelLinear.register_oot
class VllmRowParallelLinear(RowParallelLinear):
    def forward(self, input_):
        return super().forward(input_)
```

The override exists **purely to register the class**. The TPU behavior comes not
from `forward` but from the `quant_method` swapped onto the layer (mechanism 4,
§5) — `forward` → `self.quant_method.apply(...)` is where torch→JAX happens.

`embedding.py` is the one dense case that does real work in `__init__`: it replaces
the layer's `quant_method` with the TPU one so weight processing/sharding fires
(`custom_ops/embedding.py:28-35`):

```python
if isinstance(self.quant_method, UnquantizedEmbeddingMethod):
    vllm_config = get_current_vllm_config()
    mesh = vllm_config.quant_config.mesh
    self.quant_method = VllmUnquantizedEmbeddingMethod(mesh)
```

### 2.2 The vLLM-side mechanism (how the swap happens)

Everything lives in **`/home/enyouki/vllm/vllm/model_executor/custom_op.py`**.
There are two base classes that provide `register_oot`, both backed by the **same**
module-global registry:

```python
op_registry: dict[str, ...] = {}       # custom_op.py:21  in-tree ops
op_registry_oot: dict[str, ...] = {}   # custom_op.py:22  OOT overrides, keyed by BASE-class name
```

`register_oot` (e.g. `PluggableLayer.register_oot`, `custom_op.py:84`) stores the
subclass keyed by **`cls.__name__`** — the name of the *base* the decorator was
invoked on (`custom_op.py:86-89`):

```python
def decorator(layer_cls):
    reg_name = name if name is not None else cls.__name__   # e.g. "FusedMoE"
    op_registry_oot[reg_name] = layer_cls                   # "FusedMoE" -> Vllm subclass
    return layer_cls
```

The substitution happens in **`__new__`** of the base class — i.e. at the moment
the model does `FusedMoE(...)`, allocation returns the OOT subclass instead
(`PluggableLayer.__new__`, `custom_op.py:47-66`; `CustomOp.__new__`, `:109-128`):

```python
layer_class_name = cls.__name__
layer_cls_to_instantiate = op_registry_oot.get(layer_class_name, cls)
return super().__new__(layer_cls_to_instantiate)
```

So **unmodified vLLM model code** (e.g. `Qwen3MoeForCausalLM`) that constructs
`FusedMoE(...)` transparently gets a `VllmFusedMoE` instance, and its `__init__`
runs. The two base classes:

- **`PluggableLayer`** (`custom_op.py:84`) — class-swap only. Backs
  `RowParallelLinear` (→ `LinearBase`), `VocabParallelEmbedding`, `FusedMoE`.
- **`CustomOp`** (`custom_op.py:332`) — class-swap **plus** per-platform
  `forward_*` dispatch (`forward_cuda`/`forward_tpu`/…). Backs
  `DeepseekScalingRotaryEmbedding` (→ `RotaryEmbeddingBase`).

> **dispatch_key="XLA"** (`TpuPlatform`, `platforms/tpu_platform.py:89`) is
> *orthogonal* to OOT. It is a `torch.library` dispatch key for low-level custom-op
> kernel registration; the OOT swap keys only on class `__name__`. Don't conflate them.

### 2.3 RoPE — the one OOT class with a real algorithmic override

`custom_ops/rope.py` is the cleanest example of OOT changing *behavior* (not just
registering). `VllmDeepseekScalingRotaryEmbedding.forward_native` (`rope.py:41-76`)
replaces the strided-slice GPT-J rotation with a TPU-friendly reshape/flip
(`rotate_gptj_tpu`, `rope.py:20-29`) that avoids strided slicing + stacking
(which lower poorly on XLA). For the neox style it just defers to `super()`. This
is a pure-PyTorch override — it is later *lowered* op-by-op to JAX by torchax when
it runs on torchax tensors (see §3).

---

## 3. Mechanism 2 — torchax `register_function`: lowering a torch op to a JAX kernel

This is the **torchax** route for replacing **one torch function** with a hand-written
JAX implementation. Worked example: `tpu_inference/layers/vllm/ops/scaled_dot_product_attention.py`.

### 3.1 The TPU-side usage

`ops/scaled_dot_product_attention.py:25-35`:

```python
from torchax.ops.jtorch import register_function
from tpu_inference.layers.common.attention_interface import sharded_flash_attention

@register_function(torch.nn.functional.scaled_dot_product_attention)
def scaled_dot_product_attention(query, key, value, attn_mask=None, ...):
    mesh = jax.sharding.get_abstract_mesh()
    ...                                   # query/key/value arrive as jax.Array
    attn_fn = sharded_flash_attention(mesh, causal=is_causal, sm_scale=scale, ...)
    out = attn_fn(query, key, value, attention_bias, None)   # Pallas flash-attn kernel
    return out                            # jax.Array, auto-rewrapped to torchax Tensor
```

The same file also lowers `torch.ops.vllm.torch_sdpa_wrapper` (the ViT SDPA path)
the same way (`:91`). Note the body uses `jnp.pad`, `jnp.where`, `.shape` etc.
directly on the arguments — **inside a registered function the args are already
`jax.Array`, not torch tensors**, and the return value is a `jax.Array`.

### 3.2 The torchax-side mechanism

1. **Registration** — `register_function` (`torchax/ops/jtorch.py:34`) curries
   `register_torch_function_op` (`torchax/ops/ops_registry.py:57`), which wraps the
   impl in an `Operator` and stores it keyed by the torch callable in a global dict:
   ```python
   all_torch_functions: dict[TorchCallable, Operator] = {}   # ops_registry.py:31
   all_torch_functions[torch_func] = Operator(torch_func, impl, is_jax_function=True, ...)
   ```
2. **Env build** — when a torchax `Environment` is created it merges all registered
   ops into per-env tables, splitting on `is_jax_function`
   (`Environment.load_ops`, `torchax/tensor.py:419-428`): JAX-functions go into
   `self._ops`.
3. **Interception** — under the active torchax env, torch installs two modes
   (`tensor.py:619-622`). `XLAFunctionMode.__torch_function__`
   (`tensor.py:232-254`) intercepts high-level torch **functions** like
   `F.scaled_dot_product_attention` and routes to `self.env.dispatch(...)`;
   `XLADispatchMode` (`tensor.py:261-277`) does the same for decomposed **aten** ops.
4. **Dispatch + unwrap/rewrap** — `Environment.dispatch` (`tensor.py:541-617`)
   looks the op up in `self._ops`, then:
   ```python
   args, kwargs = self.t2j_iso((args, kwargs))   # tensor.py:591  torchax Tensor -> jax.Array
   res = op.func(*args, **kwargs)                 # tensor.py:602  YOUR jnp impl runs
   res = self.j2t_iso(res)                         # tensor.py:610  jax.Array -> torchax Tensor
   ```
   `t2j_iso`/`j2t_iso` (`tensor.py:653-693`) are zero-copy: they pull the inner
   `_elem` jax.Array out / wrap a jax.Array back in a `torchax.Tensor`.

So a model calling `F.scaled_dot_product_attention(q,k,v)` on torchax tensors →
`XLAFunctionMode` → `env.dispatch` → finds the `Operator` → runs the Pallas-backed
JAX flash-attn → result rewrapped. **Transparent to the model code.**

### 3.3 How a patched op lowers to a kernel (the full chain)

```
 vLLM PyTorch model forward (unmodified)
   │   q,k,v are torchax.Tensor  (jax.Array under the hood)
   ▼
 torch.nn.functional.scaled_dot_product_attention(q, k, v, ...)
   │   intercepted by XLAFunctionMode.__torch_function__   (torchax/tensor.py:232)
   ▼
 Environment.dispatch(func, ...)                            (torchax/tensor.py:541)
   │   lookup all_torch_functions / self._ops  →  Operator(is_jax_function=True)
   │   t2j_iso: torchax.Tensor → jax.Array                  (torchax/tensor.py:591)
   ▼
 tpu_inference ops/scaled_dot_product_attention.py:26  (the @register_function impl)
   │   pad to 128, build attention_bias
   ▼
 sharded_flash_attention(mesh, ...)   (layers/common/attention_interface.py)
   │   shard_map over the JAX mesh
   ▼
 Pallas flash-attention kernel  (tpu_inference/kernels/…)  →  jax.Array out
   │   j2t_iso: jax.Array → torchax.Tensor                  (torchax/tensor.py:610)
   ▼
 back into the model forward as a torch.Tensor
```

The exact same lowering applies to **every** torch op in the model: ordinary ops
(`matmul`, `add`, RoPE's reshape/flip) are decomposed and dispatched to torchax's
built-in aten lowerings; the ops tpu-inference cares about performance-wise are the
ones it explicitly `register_function`s (SDPA) or routes through a quant-method
`apply` (linear/MoE — §5).

### 3.4 `register_function` vs `register_oot` (don't confuse them)

- **`register_function`** (torchax) overrides *what one tensor operation computes*:
  a leaf torch function → a JAX kernel. Op-granular, library-level, invisible to
  the model. Lives in torchax's `all_torch_functions`.
- **`register_oot`** (vLLM) overrides *which `nn.Module` class a whole layer
  instantiates*: e.g. `FusedMoE` → `VllmFusedMoE`. Module-granular, vLLM-level.
  Lives in vLLM's `op_registry_oot`.

They compose: an OOT layer's PyTorch `forward` is itself lowered op-by-op (and via
its quant-method `apply`) by torchax.

---

## 4. Mechanism 3 — attention backends (`register_backend` + platform selection)

`tpu_inference/layers/vllm/backends/` provides the TPU **AttentionBackend**
implementations. `backends/__init__.py:14-15` imports both `flash_attn` (dense) and
`flash_attn_mla` (MLA — *sibling doc owns the kernel details*).

### 4.1 Registration

`backends/flash_attn.py:32-33`:

```python
@register_backend(AttentionBackendEnum.FLASH_ATTN)
class PallasAttentionBackend(AttentionBackend):
    @staticmethod
    def get_impl_cls() -> type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl
    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, ...):
        padded_head_size = cdiv(head_size, 128) * 128   # TPU needs head_size %128==0
        return (num_blocks, block_size, num_kv_heads * 2, padded_head_size)
    ...
```

`register_backend(AttentionBackendEnum.FLASH_ATTN)`
(`vllm.v1.attention.backends.registry`) inserts the class into vLLM's backend
registry under that enum. The MLA backend does the same with
`@register_backend(AttentionBackendEnum.FLASH_ATTN_MLA)` (`backends/flash_attn_mla.py:33`).

### 4.2 Selection — the platform chooses the enum

`TpuPlatform.get_attn_backend_cls` (`platforms/tpu_platform.py:114-128`) maps the
model to one of two TPU backends and returns its registry path:

```python
use_mla = attn_selector_config.use_mla
if use_mla:
    selected_backend = AttentionBackendEnum.FLASH_ATTN_MLA
elif selected_backend != AttentionBackendEnum.FLASH_ATTN:
    selected_backend = AttentionBackendEnum.FLASH_ATTN     # force FLASH_ATTN on TPU
return selected_backend.get_path()
```

So MLA models → `PallasMLAttentionBackend`; everything else is forced to
`PallasAttentionBackend`. vLLM then constructs the backend and its
`get_impl_cls()` impl per attention layer.

### 4.3 The backend interface it implements

`PallasAttentionBackend(AttentionBackend)` provides the static descriptors vLLM
needs: `get_name`, `get_impl_cls`, `get_kv_cache_shape` (note the
`(num_blocks, block_size, num_kv_heads*2, padded_head_size)` TPU layout — K and V
packed into one tensor), plus TPU-specific page/SMEM sizing
(`get_min_page_size`/`get_page_size`/`get_max_num_seqs`, `flash_attn.py:67-99`).
`swap_blocks` raises — block swap is unused on TPU (`flash_attn.py:55-61`).

`PallasAttentionBackendImpl(AttentionImpl)` (`flash_attn.py:102`) is where compute
crosses into JAX. Its `forward` (`flash_attn.py:152-218`):

1. Ignores vLLM's empty `kv_cache` arg; pulls the real KV cache from the
   **vLLM-model-wrapper context** by layer name
   (`get_vllm_model_wrapper_context().kv_caches[idx]`, `flash_attn.py:176-179`).
2. `jax_view`s q/k/v (torchax→jax) (`:183`).
3. Calls the jitted `_jax_attn_func` (`flash_attn.py:221-282`,
   `@jax.jit` with the KV cache **donated**), which reshapes to TPU convention and
   calls `attention(...)` from `layers/common/attention_interface.py` (→ ragged
   paged-attention Pallas kernel).
4. Writes the new KV cache back into the wrapper context and `torch_view`s the
   output (`:213-215`).

`process_weights_after_loading` here (`flash_attn.py:145-150`) just moves attention
sinks to JAX float32 — the AttentionImpl variant of the §5 hook.

> Metadata builders / KV-cache-spec wiring and the MLA backend internals are owned
> by sibling docs. The thing to remember: the backend is a **registry-selected
> object**, picked by `TpuPlatform.get_attn_backend_cls`, whose `Impl.forward`
> `jax_view`s into a jitted JAX/Pallas kernel and stashes KV cache in the wrapper
> context (not in vLLM's tensor).

---

## 5. Mechanism 4 — quantization methods & weight post-processing

This is where the **dense linear / MoE compute actually lives** (the OOT `forward`
overrides in §2 just delegate to `self.quant_method`). `quantization/__init__.py`
selects a TPU `QuantizationConfig` per checkpoint
(`get_tpu_quantization_config`, `quantization/__init__.py:36-64` — maps
`None→VllmUnquantizedConfig`, `fp8→VllmFp8Config`, `awq`, `mxfp4`, `mlx`,
`compressed-tensors`; *sibling doc owns the quantized ones*). The dense/unquantized
path is `quantization/unquantized.py`.

### 5.1 `process_weights_after_loading` — the load-time sharding hook

vLLM calls this hook on every module after weights load
(`/home/enyouki/vllm/vllm/model_executor/model_loader/utils.py:104-111`):

```python
for _, module in model.named_modules():
    quant_method = getattr(module, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        with device_loading_context(module, target_device):
            quant_method.process_weights_after_loading(module)
```

tpu-inference's quant methods override it to **convert torch CPU weights into
sharded torchax/JAX tensors on the mesh** — this is the *main per-tensor
sharding+quant path*. Representative overrides
(`quantization/unquantized.py`):

- `VllmUnquantizedLinearMethod.process_weights_after_loading` — `:184`. Loads the
  weight onto the mesh (`NamedSharding`), **frees the CPU copy**
  (`layer.weight.untyped_storage().resize_(0)`, `:195`), runs a jitted
  `process_linear_weights` (fuse/reorder for sharding, `:208-223`) +
  `shard_linear_weights`, then reinstalls `layer.weight` as a torchax `Parameter`
  (`:234-240`).
- `VllmUnquantizedEmbeddingMethod.process_weights_after_loading` — `:138`.
- `VllmUnquantizedFusedMoEMethod.process_weights_after_loading` — `:318`
  (expert-sharded `w13`/`w2`).

A `maybe_process_weights` helper (`:162-182`) processes a layer as soon as all its
constituent shards (e.g. q/k/v of a fused QKV) have loaded.

### 5.2 `apply` — the forward-time torch→JAX bridge

OOT linear `forward` → `super().forward` → `self.quant_method.apply(...)`. For the
dense path, `VllmUnquantizedLinearMethod.apply` (`unquantized.py:242-274`) is the
torch→JAX boundary:

```python
def apply(self, layer, x, bias=None):
    with jax.named_scope(layer._get_name()):
        x.shard_(NamedSharding(self.linear_config.mesh, in_sharding))   # input sharding
        x_jax = jax_view(x)                                             # torchax -> jax
        weight_jax = jax_view(layer.weight)
        out_jax = self._apply_fused(x_jax, weight_jax, bias_jax)        # JAX matmul
        out = torch_view(out_jax)                                       # jax -> torchax
        out.shard_(NamedSharding(self.linear_config.mesh, out_sharding))
    return out
```

So even the "do-nothing" `VllmRowParallelLinear.forward` (§2.1) ends up running a
sharded JAX matmul, because the *quant method*, not the module, owns the compute.
The MoE analogue is `vllm_moe_apply` in `interface/moe.py` (§6).

> The MLX/4-bit and fp8/awq/mxfp4/compressed-tensors `apply` + weight processing
> (keep-4bit, dequant-in-forward, group scales) are owned by the quantization
> sibling doc. The *mechanism* is identical: an overridden `quant_method` whose
> `process_weights_after_loading` shards/quantizes and whose `apply` `jax_view`s
> into a JAX/Pallas kernel.

---

## 6. `interface/` and `ops/` — the shared helpers

### 6.1 `layers/vllm/ops/` — torch-function lowerings (mechanism 2)

Only `scaled_dot_product_attention.py` (covered in §3). `ops/__init__.py:15-16`
imports it so the `@register_function` decorators fire. This is the home for
"lower a specific torch function to a JAX kernel" patches.

### 6.2 `layers/vllm/interface/moe.py` — the MoE compute bridge

`interface/moe.py` provides two helpers used by the MoE quant methods (it is *not*
a patch itself; it's shared logic the §5 MoE methods call):

- `select_moe_backend_from_fused_moe_config` (`interface/moe.py:29-58`) — picks a
  `MoEBackend` enum (`FUSED_MOE` EP kernel / `GMM_EP` / `GMM_TP`) from the
  `FusedMoEConfig`, gated by `envs.USE_MOE_EP_KERNEL` and `moe.use_ep`.
- `vllm_moe_apply` (`interface/moe.py:61-111`) — the torch→JAX MoE forward bridge.
  It `jax_view`s `x` and `router_logits`, plumbs DeepSeek-V3-style routing extras
  (`e_score_correction_bias`, `routed_scaling_factor`) — moving the bias onto the
  JAX device when it escaped torchax reparametrization (`:86-97`) — and calls
  `moe_apply(...)` from `layers/common/moe.py` (the GMM/fused-MoE Pallas path),
  `torch_view`ing the result.

This is the MoE analogue of §5.2's linear `apply`: the OOT `VllmFusedMoE.forward`
delegates to a quant method that calls `vllm_moe_apply`, which crosses into JAX.

---

## 7. `process_weights/` — `cleanup_sharding.py` (LoRA + replicated fallback)

`process_weights/__init__.py` is empty. The real file is
`process_weights/cleanup_sharding.py`. Despite the directory name, its job at
load time is **LoRA sharding + a replicated catch-all**, *not* the main weight
sharding (that's §5's `process_weights_after_loading`).

`shard_model_to_tpu(model, mesh)` (`cleanup_sharding.py:41-67`):

1. `_shard_module_to_tpu` (`:194-200`) walks modules and applies a LoRA-specific
   sharding func matched by exact type from `MODULE_TYPE_TO_SHARDING_FUNC`
   (`:182-191`) — column/row/QKV/merged LoRA layers get their `lora_a_stacked`/
   `lora_b_stacked` sharded (`:139-159` etc.).
2. Then `tree_map_only(_tensor_is_in_cpu, _shard_tensor_to_tpu_replicated, ...)`
   (`:62-65`) **replicates onto all chips any tensor still on CPU** — i.e. anything
   the per-tensor quant-method path didn't already place on the mesh.

`_convert_to_torchax_and_shard` (`:94-115`) does the actual torch→torchax move
(`t2j`/`jax_view` → `general_device_put(tensor, sharding)`), with a Pathways
dummy-load branch (`create_dummy_weights_on_tpu`).

**Caller:** single call site —
`VllmModelWrapper.load_weights()` at
`tpu_inference/models/vllm/vllm_model_wrapper.py:288`:

```python
self.model = _VllmRunner(vllm_model)
params_and_buffers = shard_model_to_tpu(self.model, self.mesh)
```

So the load order is: (a) vLLM loads weights and fires per-layer
`process_weights_after_loading` (main sharding, §5) → (b) `shard_model_to_tpu`
shards LoRA + replicates leftovers, and returns the params/buffers dict used for
`torch.func.functional_call` on the torchax-wrapped model. `update_lora`
(`:70-79`) is a separate inference-time helper (caller:
`runner/lora_utils.py:59`).

---

## 8. `torch.ops._C` dummy shims (defensive, in `tpu_platform.py`)

At the **top of** `tpu_inference/platforms/tpu_platform.py:21-61`, before the
platform class, there's a guarded block:

```python
try:
    import vllm._C  # noqa: F401
except ImportError:
    if not hasattr(torch.ops, "_C"):
        torch.library.define("_C::dummy", "() -> ()")
    def _register_dummy(name, schema):
        if not hasattr(torch.ops._C, name):
            torch.library.define(f"_C::{name}", schema)
            torch.library.impl(f"_C::{name}", "default", lambda *a, **k: None)
    _register_dummy("rms_norm", "(Tensor input, Tensor weight, float epsilon) -> Tensor")
    _register_dummy("fused_add_rms_norm", "(...) -> (Tensor, Tensor)")
    _register_dummy("rotary_embedding", "(...) -> ()")
    _register_dummy("static_scaled_fp8_quant", ...)
    _register_dummy("dynamic_scaled_fp8_quant", ...)
    _register_dummy("dynamic_per_token_scaled_fp8_quant", ...)
    _register_dummy("silu_and_mul", ...)
    _register_dummy("rms_norm_static_fp8_quant", ...)
    _register_dummy("fused_add_rms_norm_static_fp8_quant", ...)
    _register_dummy("rms_norm_dynamic_per_token_quant", ...)
```

**Why it exists.** `vllm._C` is vLLM's compiled **CUDA** extension. On TPU it isn't
built, so `import vllm._C` fails and `torch.ops._C` does not exist. vLLM's pure-Python
layers reference `torch.ops._C.rms_norm`, `torch.ops._C.rotary_embedding`,
`torch.ops._C.*_fp8_quant`, `silu_and_mul`, etc. **at import / class-definition
time** (e.g. Sequence-Parallelism init — see the file's TODO at `:16-19`). Without
these stubs, simply *importing* the relevant vLLM modules would `AttributeError`
before any TPU code could redirect them.

**What they do.** Each stub `torch.library.define`s the op with the schema vLLM
expects and registers a no-op `"default"` impl returning `None`. They are
**never executed for real numerics** on the hot path — RMSNorm/RoPE/SiLU/fp8-quant
on TPU run through the torchax lowerings, the OOT RoPE class, and the quant
methods. The stubs exist only to satisfy attribute lookups so the imports don't
crash. (The `hasattr` guards make them no-ops if a real `_C` is ever present.)

Stubbed ops: `rms_norm`, `fused_add_rms_norm`, `rotary_embedding`,
`static_scaled_fp8_quant`, `dynamic_scaled_fp8_quant`,
`dynamic_per_token_scaled_fp8_quant`, `silu_and_mul`,
`rms_norm_static_fp8_quant`, `fused_add_rms_norm_static_fp8_quant`,
`rms_norm_dynamic_per_token_quant`.

---

## 9. Putting it together — one dense model's patch surface

For a dense model like Qwen3 on the torchax route, the layers it constructs are
silently redirected as follows:

```
vLLM model __init__ constructs:                  TPU actually instantiates / runs:
  RowParallelLinear(...)        ── register_oot ─► VllmRowParallelLinear
      .forward → quant_method.apply ──────────────► VllmUnquantizedLinearMethod.apply  → jax_view → JAX matmul
  VocabParallelEmbedding(...)   ── register_oot ─► VllmVocabParallelEmbedding (+ TPU quant_method)
  FusedMoE(...)                 ── register_oot ─► VllmFusedMoE
      .forward → quant_method ─────────────────────► vllm_moe_apply → moe_apply (GMM/fused Pallas)
  *RotaryEmbedding              ── register_oot ─► Vllm*RotaryEmbedding (TPU rotate)
  F.scaled_dot_product_attention ─ register_function ─► JAX sharded flash attn   (ViT/encoder paths)
  Attention(...)                ── platform select ─► PallasAttentionBackend(Impl) → _jax_attn_func (ragged paged attn)
  torch.ops._C.rms_norm / etc.  ── dummy shim ───► no-op stub (real RMSNorm via torchax lowering)

Weight loading:
  per-layer process_weights_after_loading  →  shard+quant onto JAX mesh (frees CPU copy)
  shard_model_to_tpu                        →  LoRA sharding + replicate any CPU leftovers
```

To **add a new dense model**: usually *nothing* in this directory needs touching —
vLLM's model class already builds `RowParallelLinear`/`FusedMoE`/etc., and the OOT
swaps + quant methods + backend kick in automatically. You add patches here only
when the model introduces a **new layer class** (subclass + `@register_oot`), a
**new performance-critical torch op** (`@register_function`), or a **new attention
shape** (new `@register_backend`). To **add a new quantization**, add a
`VllmQuantConfig` + method under `quantization/` (sibling doc).

---

## Appendix — key anchors

**Activation**
- `setup.py:95-99` — `vllm.general_plugins` entry point
- `tpu_inference/layers/vllm/__init__.py:14-22` — imports fire decorators; `register_layers` is `pass`
- `vllm/plugins/__init__.py:61,81-82` + `vllm/engine/arg_utils.py:718-720` — vLLM loads/calls plugins

**Mechanism 1 — register_oot**
- `vllm/model_executor/custom_op.py:21-22` (`op_registry_oot`), `:84-89` (store), `:47-66`/`:109-128` (`__new__` swap)
- `custom_ops/linear.py:22,32,42`, `custom_ops/embedding.py:23,41`, `custom_ops/fused_moe.py:19`, `custom_ops/rope.py:32`
- `custom_ops/rope.py:20-29,41-76` — TPU GPT-J rotation

**Mechanism 2 — register_function (torchax)**
- `ops/scaled_dot_product_attention.py:25-35,91`
- `torchax/ops/jtorch.py:34`, `torchax/ops/ops_registry.py:31,57`
- `torchax/tensor.py:232-254` (`XLAFunctionMode`), `:541-617` (`dispatch`), `:591,610` (iso unwrap/rewrap)

**Mechanism 3 — backends**
- `backends/flash_attn.py:32` (`@register_backend`), `:152-218` (`forward`), `:221-282` (`_jax_attn_func`)
- `platforms/tpu_platform.py:114-128` — `get_attn_backend_cls` selects FLASH_ATTN / FLASH_ATTN_MLA

**Mechanism 4 — quant methods / weight processing**
- `vllm/model_executor/model_loader/utils.py:104-111` — vLLM calls `process_weights_after_loading`
- `quantization/__init__.py:36-64` — `get_tpu_quantization_config`
- `quantization/unquantized.py:184` (linear pwal), `:242-274` (`apply`), `:138` (embedding), `:318` (MoE)
- `interface/moe.py:29-58` (backend select), `:61-111` (`vllm_moe_apply`)

**Mechanism 5 — `_C` dummy shims**
- `platforms/tpu_platform.py:21-61`

**process_weights**
- `process_weights/cleanup_sharding.py:41-67` (`shard_model_to_tpu`), `:182-200` (LoRA), `:94-115` (convert)
- caller: `models/vllm/vllm_model_wrapper.py:288`
