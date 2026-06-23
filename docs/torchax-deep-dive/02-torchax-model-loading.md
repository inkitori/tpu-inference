# 02 — Torchax Model Loading & the `VllmModelWrapper` (the `vllm` route)

> **Scope.** How a vLLM **PyTorch** `nn.Module` model is instantiated under **torchax**
> and turned into a JIT-compilable JAX function that the tpu-inference runner can call
> on multi-chip TPU. This is the `MODEL_IMPL_TYPE=vllm` path. The pure-JAX/Flax route
> (`flax_nnx`) is described elsewhere; here it is only contrast.
>
> **Audience.** A strong engineer new to this codebase who wants to add new models,
> new quantizations, and eventually DeepSeek-v4 via the torchax route.
>
> All `path:line` anchors are into `/home/enyouki/tpu-inference` unless prefixed with
> `vllm/` (= `/home/enyouki/vllm/vllm`) or `torchax/`
> (= `/home/enyouki/vllm_env/lib/python3.12/site-packages/torchax`).

---

## 0. TL;DR mental model

A vLLM PyTorch model never "runs as PyTorch" on TPU. Instead:

1. The vLLM `nn.Module` is **built on CPU** (via vLLM's own loader) and its **weights
   are loaded + sharded into `jax.Array`s** wrapped as **torchax tensors**.
2. The module's parameters/buffers are **pulled out** into a flat pytree (`params_and_buffers`).
3. At run time, tpu-inference calls
   `torch.func.functional_call(model, params_and_buffers, kwargs)` **inside a
   `torchax.default_env()` context**. Under that env, every torch op the module
   executes is intercepted and **re-dispatched as a JAX op** on the underlying
   `jax.Array`s. The whole call is wrapped in `@jax.jit`, so it compiles to one XLA
   program with SPMD sharding from the mesh.

So torchax is the bridge: **torch op → JAX op**, **`torch.Tensor` ⇄ `jax.Array`**, and
the wrapper just arranges params/inputs/outputs to cross that bridge cleanly.

---

## 1. Entry: `get_model` → `get_vllm_model`

`get_model` (`tpu_inference/models/common/model_loader.py:463`) dispatches on
`envs.MODEL_IMPL_TYPE`:

- `flax_nnx` → `get_flax_model` (pure JAX; `model_loader.py:283`).
- `vllm` → `get_vllm_model` (`model_loader.py:412`) — **our path**.
- `auto` → `resolve_model_architecture` (`model_loader.py:508`) picks one; `vllm` is
  chosen for arches in `_VLLM_PREFERRED_ARCHITECTURES`
  (`GptOssForCausalLM`, `Qwen3MoeForCausalLM`; `model_loader.py:51`). Note `flax_nnx`
  also **falls back** to `get_vllm_model` if the arch isn't registered in the JAX
  registry, or for PP-disabled models (`model_loader.py:483-501`).

`get_vllm_model` (`model_loader.py:412-460`) end to end:

```python
def get_vllm_model(vllm_config, rng, mesh, is_draft_model=False) -> ModelInterface:
    model_dtype = to_torch_dtype(vllm_config.model_config.dtype)   # JAX dtype -> torch dtype
    vllm_config.model_config.dtype = model_dtype                    # vLLM build needs a torch dtype
    from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper

    model = VllmModelWrapper(vllm_config, rng, mesh, is_draft_model)
    params, lora_manager = model.load_weights()                    # build + load + shard
    jit_model            = model.jit_step_func()                   # the jit-able forward
    compute_logits_fn    = model.jit_compute_logits_func()
    pooler_fn            = model.build_pooler_func()
    combine_hidden_states_fn = model.jit_combine_hidden_states_func()
    multimodal_fns = MultiModalInterface(...)                       # embed_multimodal / embed_input_ids
    return ModelInterface(
        model_fn=jit_model, compute_logits_fn=compute_logits_fn, ...,
        state=params,           # the param pytree the runner threads back in on every call
        lora_manager=lora_manager, model=model)
```

Key observations:

- It **converts the model dtype to a torch dtype** before building (`model_loader.py:418`)
  — vLLM constructs PyTorch modules, so it needs `torch.bfloat16`, not `jnp.bfloat16`.
  (`to_torch_dtype` round-trips through JAX: `tpu_inference/utils.py:63`.)
- It returns a `ModelInterface` (the same dataclass the Flax route returns), so the
  runner is **agnostic** to which route produced it. The "model" is now a set of
  **jit-able JAX functions** (`model_fn`, `compute_logits_fn`, …) plus a **`state`
  pytree** (the params) the runner passes on every invocation.
- `state=params` is a **JAX pytree** (`jax_view`'d torchax tensors) — see §4/§5.

---

## 2. `VllmModelWrapper.load_weights` — build, load, shard

`VllmModelWrapper.__init__` (`tpu_inference/models/vllm/vllm_model_wrapper.py:117-152`):

- Computes the **TPU quantization config** and stashes it on `vllm_config.quant_config`
  via `get_tpu_quantization_config(vllm_config, mesh)`
  (`tpu_inference/layers/vllm/quantization/__init__.py:36`). This maps
  `model_config.quantization` (`None`, `fp8`, `awq`, `mxfp4`, `compressed-tensors`,
  `mlx`) to a tpu-inference `Vllm*Config` whose `get_quant_method` returns the
  **TPU-aware quant method** for each layer (e.g. `VllmUnquantizedLinearMethod`,
  `quantization/__init__.py:46-64`). **This is the hook through which all per-layer
  sharding + JAX op selection enters** (see §4).
- **MLX auto-wiring** (`vllm_model_wrapper.py:139-150`): if the checkpoint is MLX-quantized
  and `load_format=="auto"`, it flips `load_format` to `"tpu_streaming_loader"` so the
  MLX weight transform runs. *(MLX details deferred — see §6.)*
- `_apply_pp_patch` (`vllm_model_wrapper.py:154-167`): monkeypatches vLLM's
  `get_pp_group` to tpu-inference's JAX pipeline-parallel group.

`VllmModelWrapper.load_weights` (`vllm_model_wrapper.py:169-316`) — the heart:

1. **Build on CPU.** It deep-copies the config, sets
   `vllm_config_for_load.device_config.device = "cpu"` (`:196`), disables vLLM expert
   parallelism (tpu-inference does its own sharding, `:208`), and resolves the dummy
   load format if needed (`:210-227`, see §5). Loading happens under three nested
   contexts (`:238-239`):
   - `jax.default_device(cpu)` so torchax tensors created during load land on CPU
     (skipped under Pathways, `:232-234`);
   - a `patch("torch._sync", return_value=None)` for the dummy path (`:226-227`);
   - `set_current_vllm_config(self.vllm_config)`.
2. **Call into vLLM's loader** (`:265-266`):
   ```python
   vllm_model = vllm_get_model(vllm_config=vllm_config_for_load,
                               model_config=model_config_for_load)
   ```
   This is `vllm.model_executor.model_loader.get_model` — it resolves the **model class
   from vLLM's `ModelRegistry`** by architecture, builds the `nn.Module`, loads/initializes
   weights, and runs `process_weights_after_loading`. See §3.
3. **LoRA** (optional, `:267-275`): replace layers with LoRA variants under
   `torchax.default_env()`; `replace_set_lora` wraps `set_lora/reset_lora` so they run
   under the env (`:631-652`).
4. **Wrap** in `_VllmRunner` (`:287`), a thin `nn.Module` (`vllm_model_wrapper.py:69-107`)
   that routes `forward(**kwargs)` to one of: `compute_hidden_state` (the main forward),
   `compute_logits` (when `hidden_state` kwarg present), or an arbitrary
   `call_method` (used for multimodal/embedding/`combine_hidden_states`). This is the
   single `nn.Module` that `functional_call` targets.
5. **Shard to TPU** (`:288`):
   ```python
   params_and_buffers = shard_model_to_tpu(self.model, self.mesh)
   ```
   Converts every still-on-CPU param/buffer to a sharded torchax tensor (see §4).
6. **Return a JAX pytree** (`:316`):
   ```python
   return jax_view(params_and_buffers), lora_manager
   ```
   `jax_view` strips the torchax wrapper, yielding a pytree of raw `jax.Array`s — this
   is the `state` the runner threads into every jitted call.

> **Why build on CPU then move?** Two-phase loading: build the structure, stream each
> layer's real weights in, shard + free CPU memory per layer. Avoids holding a full
> random-init copy and avoids OOM on huge MoE layers. (See the comments at
> `model_loader.py:218-227` for the analogous reasoning in the Flax path.)

---

## 3. How the vLLM model **class** is chosen & built (vLLM side)

`vllm_get_model` = `vllm/model_executor/model_loader/__init__.py:get_model`:

```
get_model(vllm_config, model_config)                      vllm/.../model_loader/__init__.py
  └─ get_model_loader(load_config)                        # dispatch by load_format string
        → _LOAD_FORMAT_TO_MODEL_LOADER[load_format](load_config)
  └─ loader.load_model(vllm_config, model_config)         vllm/.../base_loader.py
        ├─ initialize_model(vllm_config, model_config)    vllm/.../model_loader/utils.py
        │     └─ get_model_architecture(model_config)
        │           └─ model_config.registry.resolve_model_cls(architectures)
        │                 → _try_load_model_cls(arch)      vllm/.../models/registry.py
        │                   (lazy-imports "module:Class" from ModelRegistry)
        │     model = model_class(vllm_config=..., prefix=...)
        ├─ self.load_weights(model, model_config)         # stream (name, tensor) pairs
        └─ process_weights_after_loading(model, ...)      # per-submodule quant hook
```

Anchors:

- **Loader dispatch** by load-format string: `get_model_loader` looks up
  `_LOAD_FORMAT_TO_MODEL_LOADER` (`vllm/model_executor/model_loader/__init__.py`).
  Custom loaders are added with the `@register_model_loader("name")` decorator.
- **Architecture → class:** `get_model_architecture` →
  `model_config.registry.resolve_model_cls(model_config.hf_config.architectures)`
  (`vllm/model_executor/model_loader/utils.py`), which walks vLLM's `ModelRegistry`
  (`vllm/model_executor/models/registry.py`) and **lazy-imports** the `"module:Class"`
  string for e.g. `Qwen3MoeForCausalLM`. **This is the standard vLLM PyTorch class** —
  tpu-inference does *not* re-implement it; it only patches its **layers** (see below).
- **Config plumbing:** the resolved class is instantiated with `vllm_config=` and
  `prefix=` under `set_current_vllm_config(...)` (`utils.py:initialize_model`). The
  `quant_config` we set in §2 rides on `vllm_config`, so when layers call
  `quant_config.get_quant_method(layer, prefix)` during construction, they get the
  **tpu-inference TPU quant method** instead of vLLM's default.
- **`process_weights_after_loading`** (`vllm/model_executor/model_loader/utils.py`)
  loops over submodules and calls `quant_method.process_weights_after_loading(module)`
  on each — **this is where tpu-inference's per-layer sharding actually fires** (§4).

> **Layer patching (cross-cutting, from SHARED_CONTEXT).** `setup.py` registers a
> `vllm.general_plugins` entry point `register_layers`, which monkeypatches vLLM layer
> classes (linear, MoE, attention, embedding, rmsnorm…) with TPU implementations from
> `tpu_inference/layers/vllm/`. So although the **model graph** is vLLM's, the **leaf
> ops** are tpu-inference's JAX/Pallas implementations. The wrapper here is the
> *outer* bridge; those patched layers are the *inner* bridge.

The loaders tpu-inference itself registers (in `vllm_model_loader.py` and elsewhere):

| Loader (by `load_format`)    | Class                              | Defined at |
|------------------------------|------------------------------------|------------|
| `tpu_streaming_loader`       | `IncrementalModelLoader`           | `tpu_inference/models/vllm/vllm_model_loader.py:72` |
| `runai_streamer`             | `RunaiIncrementalModelLoader`      | `vllm_model_loader.py:134` |
| `pathways_dummy`             | `PathwaysDummyModelLoader`         | `tpu_inference/models/common/pathways_dummy_loader.py:143` |
| `jax_dummy`                  | `JaxDummyModelLoader` (Flax route) | `tpu_inference/models/jax/utils/weight_utils.py:926` |

`IncrementalModelLoader` (subclass of vLLM `DefaultModelLoader`,
`vllm_model_loader.py:72-131`) is the production real-weight path for the torchax
route. Its `load_model` (`:111-131`):

```python
with set_default_torch_dtype(model_config.dtype), target_device:  # target_device == cpu here
    model = initialize_model(vllm_config, model_config)           # build nn.Module
attach_incremental_weight_loader(model)                            # wrap each param's weight_loader
self.load_weights(model, model_config)                            # stream weights in
process_weights_after_loading(model, model_config, target_device) # per-layer shard
```

`attach_incremental_weight_loader` (`vllm_model_loader.py:32-69`) wraps every
parameter's `weight_loader` so that, **as soon as a module's last weight is loaded**,
it calls `quant_method.maybe_process_weights(layer, ...)` — which shards that layer's
weights to TPU and frees CPU memory immediately, instead of waiting for the whole model.
`get_all_weights` (`:88-109`) is the hook where the **MLX transform** is applied to the
`(name, tensor)` stream (§6).

---

## 4. Weight loading & sharding — where tensors become sharded `jax.Array`s

There are **two complementary sharding mechanisms**, and it's important to keep them
straight:

### 4a. Per-layer tensor/expert-parallel sharding (the real sharding)

Done by the **quant method's `process_weights_after_loading` / `maybe_process_weights`**,
fired either incrementally (`attach_incremental_weight_loader`) or in vLLM's
`process_weights_after_loading` loop. Example — unquantized linear
(`tpu_inference/layers/vllm/quantization/unquantized.py:184-240`):

```python
def process_weights_after_loading(self, layer):
    if not _tensor_is_in_cpu(layer.weight): return        # idempotent
    weight_sharding = NamedSharding(self.linear_config.mesh,
                                    self.linear_config.weight_sharding)   # e.g. P('model', None)
    weight = _load_weight_for_layer(layer, "weight", weight_sharding)     # torch -> jax.Array
    layer.weight.untyped_storage().resize_(0); delattr(layer, 'weight')   # free CPU
    ...
    weights = torch_view(shard_linear_weights(weights, mesh=..., weight_p_spec=..., ...))
    layer.weight = Parameter(weights.weight, requires_grad=False)         # now a sharded torchax param
```

- `_load_weight_for_layer` (`unquantized.py:63-90`) turns the CPU torch tensor into a
  `jax.Array` via `t2j(tensor)` (or, under Pathways, generates it directly on-device).
- `shard_linear_weights` / `shard_moe_weights` apply the **column/row/expert** partition
  specs across the mesh. MoE uses `P(ShardingAxisName.EXPERT)`
  (`unquantized.py:326`, `VllmUnquantizedFusedMoEMethod.process_weights_after_loading`),
  and calls `jax.effects_barrier()` between layers (`:368`) to bound peak HBM.
- The result is re-wrapped with `torch_view(...)` and stored back as a `Parameter`
  whose storage is now a **sharded `jax.Array`** (a torchax tensor).

This is the contract a **new quantization** must satisfy: subclass
`VllmQuantizationMethod` (`tpu_inference/layers/vllm/quantization/base.py:20`) and
implement `maybe_process_weights` (and the `apply` that does the JAX matmul). The
abstract base is tiny — one method:

```python
class VllmQuantizationMethod(ABC):
    @abstractmethod
    def maybe_process_weights(self, layer, param_name, args, kwargs): ...
```

### 4b. Residual replicated sharding + torchax conversion (`shard_model_to_tpu`)

`shard_model_to_tpu` (`tpu_inference/layers/vllm/process_weights/cleanup_sharding.py:41-67`)
runs **after** the per-layer pass and mops up everything still on CPU:

```python
def shard_model_to_tpu(model, mesh):
    with jax.default_device(jax.devices("cpu")[0]):
        _shard_module_to_tpu(model, mesh)                 # LoRA-specific sharding (table-driven)
        params, buffers = _extract_all_params_buffers(model)
        params, buffers = pytree.tree_map_only(           # everything *still on CPU* ...
            _tensor_is_in_cpu,
            lambda t: _shard_tensor_to_tpu_replicated(t, mesh),   # ... gets replicated P()
            (params, buffers))
        return {**params, **buffers}
```

- `_tensor_is_in_cpu` (`:86-91`): true if the tensor is **not** yet a torchax tensor, or
  is a torchax tensor still on the CPU device. So already-sharded params from §4a are
  left untouched; norms/embeddings/scalars that no quant method handled get
  **replicated** across all chips (`P()`, `:118-120`).
- `_convert_to_torchax_and_shard` (`:94-115`) is the canonical **torch → sharded jax**
  conversion: `t2j(tensor)` then `general_device_put(tensor, sharding)`, wrapped with
  `torch_view`. `general_device_put`
  (`tpu_inference/layers/common/utils.py:106-146`) handles single-host
  (`jax.device_put`) vs Ray multi-host (`jax.make_array_from_callback`) placement.
- `_shard_module_to_tpu` (`:194-200`) is table-driven via
  `MODULE_TYPE_TO_SHARDING_FUNC` (`:182-191`) and currently only special-cases **LoRA**
  layers (`lora_a` replicated, `lora_b` column-sharded `P(None,None,'model',None)`).

After this, `jax_view(params_and_buffers)` (§2 step 6) returns a pytree of raw
`jax.Array`s — that pytree **is** the model state.

### 4c. Dummy / random load path (profiling & compile warm-up)

Used for memory profiling and to compile the graph before real weights exist
(`load_format == "dummy"`). Resolution in `load_weights`
(`vllm_model_wrapper.py:210-227`):

```python
use_random_weights = (load_format == "dummy")
use_pathways_dummy  = use_random_weights and vllm_envs.VLLM_TPU_USING_PATHWAYS
if use_pathways_dummy:
    load_format = "pathways_dummy"     # generate random weights *directly on TPU*
elif use_random_weights:
    ...                                # vLLM DummyModelLoader fills random CPU tensors
load_context = patch("torch._sync", return_value=None) if (use_random_weights and not use_pathways_dummy) else nullcontext()
```

Two sub-paths:

- **Plain dummy** → vLLM's `DummyModelLoader` (`load_format=="dummy"`): builds the
  module and fills params with small uniform random values
  (`initialize_dummy_weights`, range `[-1e-3, 1e-3]`) — **the checkpoint is never read**.
  Those random CPU tensors then flow through §4a/§4b to TPU exactly like real ones.
- **Pathways dummy** → tpu-inference's `PathwaysDummyModelLoader`
  (`tpu_inference/models/common/pathways_dummy_loader.py:143`): `load_weights` is a
  no-op; weights are created **directly on the TPU mesh** by
  `create_dummy_weights_on_tpu(sharding, shape, dtype)` (`pathways_dummy_loader.py:55`),
  invoked from `_load_weight_for_layer` / `_convert_to_torchax_and_shard` when
  `is_pathways_dummy_load()` is true (`unquantized.py:75-85`,
  `cleanup_sharding.py:96-110`). This avoids ever materializing a full unsharded copy on
  one device — important for large MoE.

> The Flax route has its own `jax_dummy` loader (`weight_utils.py:926`); the torchax
> route uses `dummy`/`pathways_dummy` as above.

---

## 5. The torchax bridge — what `default_env` / `jax_view` / `torch_view` actually do

torchax lives at
`/home/enyouki/vllm_env/lib/python3.12/site-packages/torchax`. The pieces this code
uses:

### `torchax.default_env()` — the dispatch env
`torchax/__init__.py:60` returns a singleton `Environment`
(`torchax/tensor.py`). Entering it (`with torchax.default_env():`) activates **two**
dispatch modes (`tensor.py:619-623`, `Environment.enable_torch_modes`):

- `XLAFunctionMode` (a `torch.overrides.TorchFunctionMode`) — intercepts high-level
  torch *function* calls via `__torch_function__` (`tensor.py:232-254`).
- `XLADispatchMode` (a `TorchDispatchMode`) — intercepts low-level *aten* ops via
  `__torch_dispatch__`, filtered to namespaces `aten/_c10d_functional/torchvision/xla`
  (`tensor.py:257-277`).

Both funnel into `env.dispatch(func, types, args, kwargs)` (`tensor.py:541-617`), which:
looks up a registered **JAX implementation** of the op → converts torch tensor args to
`jax.Array` (`t2j_iso`) → calls the JAX op → wraps results back to torchax `Tensor`
(`j2t_iso`). **This is precisely how a torch op becomes a JAX op.**

### `torchax.tensor.Tensor` — torch.Tensor backed by jax.Array
A `torch.Tensor` subclass (wrapper subclass with `device="meta"`) that stores the real
data in `self._elem: jax.Array` (`tensor.py:~55-80`). `.jax()` returns `_elem`;
`.jax_device` returns `_elem.device` (used by `_tensor_is_in_cpu`). `.to(device="jax")`
moves a plain torch tensor onto a torchax/JAX tensor.

### `jax_view` / `torch_view` — crossing the boundary (`torchax/interop.py`)
Both are `pytree.tree_map` over leaves (`interop.py:211-227`):

```python
def _jax_view(t):
    if isinstance(t, torch.Tensor):  return t.jax()                 # torchax.Tensor -> jax.Array
    if isinstance(t, torch.dtype):   return mappings.t2j_dtype(t)
    if callable(t):                  return functools.partial(call_torch, t)
    return t
def _torch_view(t):
    if isinstance(t, jax.Array):     return tensor.Tensor(t, default_env())  # jax.Array -> torchax.Tensor
    if isinstance(t, jnp.dtype):     return mappings.j2t_dtype(t)
    if callable(t):                  return functools.partial(call_jax, t)
    return t
```

So `jax_view(params)` = "give me the raw `jax.Array`s" (what JAX/jit consumes);
`torch_view(arr)` = "wrap this `jax.Array` as a torch tensor" (what `functional_call`
needs). Low-level conversions: `t2j`/`j2t` and `TORCH_DTYPE_TO_JAX` in
`torchax/ops/mappings.py` (DLPack zero-copy with numpy fallback). Note tpu-inference
ships its own faster `t2j` (`tpu_inference/utils.py:79`) that **bit-casts** bf16/fp8
instead of going through float32.

> `torchax.extract_jax(mod)` (`torchax/__init__.py:68`) is the canonical
> "module → (states, jax_func)" helper and shows the same pattern the wrapper uses
> by hand. tpu-inference does **not** call `extract_jax`; it rolls its own
> `functional_call` inside `jax.jit` (§6) for finer control over sharding, the KV-cache
> context, and the forward-context.

### How `functional_call` executes as JAX
`torch.func.functional_call(module, params_and_buffers, kwargs)` runs the module's
`forward` while **substituting** the given params/buffers for the module's own. When:
(a) the substituted params are torchax `Tensor`s (backed by sharded `jax.Array`s), and
(b) the call is inside `torchax.default_env()`, then every torch op in `forward`
dispatches through `env.dispatch` to a JAX op operating on those `jax.Array`s. Wrapping
the whole thing in `@jax.jit` traces it into **one XLA program**, and the params'
`NamedSharding` drives SPMD partitioning across the mesh.

---

## 6. The jit-able forward — `jit_step_func` and friends

`jit_step_func` (`vllm_model_wrapper.py:318-445`) returns the `model_fn` the runner
calls. The core `step_fun` (`:340-393`):

```python
@jax.jit(donate_argnames=("kv_caches",),
         out_shardings=(None, NamedSharding(mesh, P(ATTN_DATA, None)), None),
         compiler_options={...all_gather/reduce_scatter collective-matmul modes...},
         static_argnames=("layer_name_to_kvcache_index","is_first_rank","is_last_rank"))
def step_fun(params_and_buffers, kv_caches, input_ids, attn_metadata, input_embeds,
             input_positions, layer_name_to_kvcache_index, lora_metadata,
             intermediate_tensors=None, is_first_rank=True, is_last_rank=True, *args):
    with torchax.default_env(), \
         set_vllm_model_wrapper_context(kv_caches=kv_caches, mesh=self.mesh,
                                        layer_name_to_kvcache_index=...), \
         set_forward_context(attn_metadata=attn_metadata, vllm_config=self.vllm_config):
        output_from_torch = torch.func.functional_call(
            self.model,                       # the _VllmRunner nn.Module
            torch_view(params_and_buffers),   # jax.Array pytree -> torchax tensors
            kwargs={"input_ids": torch_view(input_ids),
                    "positions": torch_view(input_positions),
                    "intermediate_tensors": intermediate_tensors,
                    "inputs_embeds": torch_view(input_embeds)},
            tie_weights=False)
        new_kv_caches = get_vllm_model_wrapper_context().kv_caches
    output = jax_view(output_from_torch)      # torchax tensor -> jax.Array
    return new_kv_caches, output, aux_hidden_states
```

The callable's contract / threading:

- **Signature:** `(params_and_buffers, kv_caches, input_ids, attn_metadata,
  input_embeds, input_positions, layer_name_to_kvcache_index, lora_metadata, …)` — all
  **JAX** values (the runner lives in JAX-land). `params_and_buffers` is the `state`
  pytree returned by `load_weights` (§2). `kv_caches` is **donated** (in-place update).
- **Boundary crossing:** inputs are `torch_view`'d on entry, outputs `jax_view`'d on
  exit. Inside, everything is a torchax tensor and dispatches to JAX (§5).
- **KV cache plumbing:** `set_vllm_model_wrapper_context(kv_caches, mesh,
  layer_name_to_kvcache_index)` (`vllm_model_wrapper_context.py:41`) stashes the caches
  in a module-global; the patched attention layers read/write them and the **updated**
  caches are pulled back out via `get_vllm_model_wrapper_context().kv_caches`
  (`vllm_model_wrapper.py:381-382`). This is how a functional JAX program threads
  mutable KV cache without it being an explicit return of every layer.
- **Forward context:** `set_forward_context(attn_metadata, vllm_config)` is vLLM's own
  context that layers read (attn metadata, etc.).
- **PP / Eagle3:** non-first-rank receives `intermediate_tensors`; non-last-rank returns
  `JaxIntermediateTensors`; eagle3 splits out `aux_hidden_states` (`:386-392`).

Sibling jitted functions, all same pattern (env + `functional_call` + view-crossing):

- `jit_compute_logits_func` (`:527-557`) — calls `_VllmRunner` with `hidden_state=` →
  `compute_logits`; out-sharded `P(MLP_DATA, MLP_TENSOR)`.
- `jit_combine_hidden_states_func` (`:559-582`) — eagle3 hidden-state combine.
- `wrap_embed_multimodal_func` / `wrap_embed_input_ids_func` (`:447-525`) — **not**
  `jax.jit`ed (dynamic shapes); still cross the boundary via `torch.func.functional_call`
  inside `default_env()`, routed through `_VllmRunner`'s `call_method` branch.
- `build_pooler_func` (`:584-607`) — runs the pooler on CPU (`.to('cpu')`) under the env.
- For draft models, `draft_step_fun` (`:411-443`) is returned instead.

### MLX hook point (deferred)
`mlx_weight_transform.py` (owned by another agent) plugs in at **one place**:
`IncrementalModelLoader.get_all_weights` (`vllm_model_loader.py:88-109`) calls
`transform_mlx_weights(weights, group_size, bits, num_experts)` to rewrite the
`(name, tensor)` stream before `model.load_weights` consumes it (un-stack `switch_mlp`,
dequant MLX-only embed/lm_head, rename to vLLM's `Qwen3MoeForCausalLM` layout). The
auto-selection of the `tpu_streaming_loader` and the loud guard live in
`VllmModelWrapper.__init__`/`load_weights` (`:139-150`, `:253-264`). MLX is detected by
`is_mlx_quantized` (`tpu_inference/layers/vllm/quantization/mlx.py:54`) and routed to
`VllmMLXConfig` in `get_tpu_quantization_config`.

---

## 7. End-to-end diagram

```
get_model(MODEL_IMPL_TYPE="vllm")            tpu_inference/models/common/model_loader.py:463
   └─ get_vllm_model(vllm_config, rng, mesh)                                          :412
        │  dtype: jax -> torch  (to_torch_dtype)                                      :418
        ▼
      VllmModelWrapper(...)                  tpu_inference/models/vllm/vllm_model_wrapper.py:117
        ├─ get_tpu_quantization_config()  -> vllm_config.quant_config  (per-layer TPU quant methods)
        ├─ (MLX) auto-select tpu_streaming_loader
        ▼
      .load_weights()                                                                 :169
        ├─ build on CPU:  vllm_get_model(vllm_config_for_load)  ───────────────► vLLM side
        │     ├─ get_model_loader[load_format]    (Incremental/Runai/Dummy/Pathways)
        │     ├─ initialize_model -> registry.resolve_model_cls(arch) -> nn.Module
        │     ├─ load_weights  (stream (name,tensor); MLX transform in get_all_weights)
        │     └─ process_weights_after_loading / maybe_process_weights
        │            └─ §4a  PER-LAYER shard: t2j + shard_linear/moe_weights (P('model')/P(EXPERT))
        ├─ wrap in _VllmRunner (nn.Module: forward -> hidden / logits / call_method)  :287
        ├─ shard_model_to_tpu(model, mesh)   §4b  residual REPLICATED shard + torchax :288
        │     (cleanup_sharding.py: _convert_to_torchax_and_shard = t2j + general_device_put)
        └─ return jax_view(params_and_buffers)   ──►  state = pytree of jax.Array      :316
        ▼
      .jit_step_func() -> step_fun  (@jax.jit)                                         :318
        ┌──────────────────────── per call, in JAX-land ────────────────────────────┐
        │ torch_view(inputs) ──► with torchax.default_env():                          │
        │                          torch.func.functional_call(_VllmRunner,            │
        │                              torch_view(params_and_buffers), kwargs=...)     │
        │                          ▲ every torch op -> env.dispatch -> JAX op          │
        │                          │ (patched tpu_inference layers do JAX/Pallas)      │
        │                        KV cache via set_vllm_model_wrapper_context           │
        │                     ──► jax_view(output)                                     │
        └────────────────────────────────────────────────────────────────────────────┘
        ▼
   ModelInterface(model_fn=step_fun, compute_logits_fn=..., state=params, model=wrapper)
        ▼
   tpu_inference runner calls model_fn(state, kv_caches, input_ids, ...)  every step
```

---

## 8. Pointers for the end goal

- **Add a new model (torchax route):** usually *nothing* in this file changes. The model
  graph comes from vLLM's `ModelRegistry`; you (a) ensure the arch is in
  `_VLLM_PREFERRED_ARCHITECTURES` or load with `MODEL_IMPL_TYPE=vllm`, and (b) make sure
  every leaf layer it uses has a TPU patch in `tpu_inference/layers/vllm/` (registered
  by `register_layers`). New layer types → new patches + sharding in their quant
  method's `process_weights_after_loading`.
- **Add a new quantization:** implement a `Vllm*Config` (returning a quant method per
  layer) and a quant method subclassing `VllmQuantizationMethod`
  (`quantization/base.py:20`) with `maybe_process_weights` (shard at load) + `apply`
  (the JAX matmul). Register it in the `method_to_config` map
  (`quantization/__init__.py:46`). The wrapper machinery is untouched.
- **DeepSeek-v4:** its PyTorch class would come from vLLM's registry; the work is the
  per-layer TPU patches (MLA attention, MoE, any new quant) under
  `tpu_inference/layers/vllm/` and their sharding in `process_weights_after_loading`.
  The `VllmModelWrapper` forward and the torchax bridge are model-agnostic and should
  not need changes.
```
