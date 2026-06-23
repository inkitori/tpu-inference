# 08 — The Quantization Framework (torchax route)

> **Audience:** a strong engineer, new to this codebase, who wants to **add a new
> quantization** to the torchax route (`MODEL_IMPL_TYPE=vllm`).
> **Scope:** how vLLM's quantization abstractions get TPU implementations, how a
> quant *name* maps to a TPU config/method, the weight-processing pipeline, the
> common utilities, and the quantized-matmul Pallas kernel. MLX is treated as one
> example among many — see `07-hy3-mlx-4bit.md` for the MLX deep-dive.

All paths are absolute under `/home/enyouki/tpu-inference/`. vLLM upstream lives at
`/home/enyouki/vllm/vllm/` (read freely; it is the editable install).

---

## 0. The big picture

vLLM's quantization stack is a small set of abstract base classes:

| vLLM base class | File | Role |
|---|---|---|
| `QuantizationConfig` | `vllm/.../quantization/base_config.py` | parses the checkpoint's quant config; factory: `get_quant_method(layer, prefix)` |
| `QuantizeMethodBase` | same | abstract method base; `create_weights` + `process_weights_after_loading` + `apply` |
| `LinearMethodBase` | `vllm/.../layers/linear.py` | per-Linear-layer method (subclass of `QuantizeMethodBase`) |
| `FusedMoEMethodBase` | `vllm/.../fused_moe/...` | per-FusedMoE-layer method |

tpu-inference does **not** rewrite this machinery. Instead, for each quant it
**subclasses the upstream vLLM config + method classes** and overrides exactly the
hooks that turn checkpoint torch tensors into sharded **`jax.Array`s** and that run
the forward pass as a **JAX/Pallas** kernel under torchax. The wiring is:

```
                                 register via @register_quantization_config(NAME)
  tpu_inference.layers.vllm  ───────────────────────────────────────────────┐
  (import side-effect)                                                       │
                                                                            ▼
  LLM(quantization=NAME)                                vLLM global registry:
     │                                       _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[NAME] = TpuCfg
     ▼
  ModelConfig._verify_quantization()  ── gate ──►  TpuPlatform.supported_quantization
     │                                              (vllm/platforms/interface.py:650)
     ▼
  VllmModelWrapper.__init__
     └─ get_tpu_quantization_config(vllm_config, mesh)        [quantization/__init__.py:36]
          ├─ method_to_config[name] → VllmXxxConfig           (the TPU selector dict)
          ├─ VllmXxxConfig.set_configs(vllm_config, mesh)     (stashes mesh on the class)
          └─ VllmConfig.get_quantization_config(...)          → instantiates TpuCfg
     │
     ▼   at layer-build time, per nn.Module:
  TpuCfg.get_quant_method(layer, prefix)
     ├─ LinearBase  → VllmXxxLinearMethod        (or VllmUnquantizedLinearMethod if skipped)
     ├─ FusedMoE    → VllmXxxMoEMethod           (or VllmUnquantizedFusedMoEMethod if skipped)
     └─ Attention   → usually None
     │
     ▼   after weights load:
  method.process_weights_after_loading(layer)    → t2j → process_* → shard_* → re-attach
     │
     ▼   at forward:
  Linear: method.apply(layer, x, bias)           → einsum / sharded_quantized_matmul
  MoE:    method.apply_monolithic(layer, x, router_logits, input_ids)  → moe_apply kernel
```

There are **two registries** working together (this trips people up):

1. **The TPU selector** — a plain dict `method_to_config` in
   `tpu_inference/layers/vllm/quantization/__init__.py:46`. This is what tpu-inference
   checks to decide *which TPU config class* to use. If a name is not in it →
   `NotImplementedError`.
2. **The vLLM global registry** — `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG` in
   `vllm/model_executor/layers/quantization/__init__.py`. Each TPU config registers
   itself here via the `@register_quantization_config(NAME)` decorator; vLLM's
   `get_quantization_config(name)` merges it over its built-ins so the **TPU class
   wins** for that name.

---

## 1. The override mechanism

### 1.1 The name constants

The canonical quant-name strings live in one tiny file
`tpu_inference/layers/common/quant_methods.py:15-20` — six string constants, **no
enums, no helpers**:

```python
UNQUANTIZED = "unquantized"
MXFP4 = "gpt_oss_mxfp4"          # note: the name is NOT "mxfp4"
AWQ = "awq"
COMPRESSED_TENSORS = "compressed-tensors"
FP8 = "fp8"
MLX = "mlx"
```

### 1.2 The TPU selector dict (`get_tpu_quantization_config`)

`tpu_inference/layers/vllm/quantization/__init__.py:36-64` — the entry point, called
from `tpu_inference/models/vllm/vllm_model_wrapper.py:127`:

```python
def get_tpu_quantization_config(vllm_config, mesh) -> QuantizationConfig:
    model_config = copy.deepcopy(vllm_config.model_config)
    # MLX checkpoints carry quantization_config (group_size+bits) but no
    # quant_method key, so vLLM leaves model_config.quantization unset.
    if model_config.quantization is None and is_mlx_quantized(model_config.hf_config):
        model_config.quantization = quant_methods.MLX
    # TODO(kyuyeunk): Add support for "tpu_int8".
    method_to_config = {
        None: VllmUnquantizedConfig,
        quant_methods.COMPRESSED_TENSORS: VllmCompressedTensorsConfig,
        quant_methods.AWQ: VllmAWQConfig,
        quant_methods.FP8: VllmFp8Config,
        quant_methods.MXFP4: VllmMxfp4Config,        # "gpt_oss_mxfp4"
        quant_methods.MLX: VllmMLXConfig,
    }
    if model_config.quantization not in method_to_config:
        raise NotImplementedError(...)
    quant_config = method_to_config[model_config.quantization]
    assert issubclass(quant_config, VllmQuantConfig)
    quant_config.set_configs(vllm_config, mesh)               # stash mesh on the class
    model_config.quantization = quant_config.get_name()
    return VllmConfig.get_quantization_config(model_config, vllm_config.load_config)
```

Key facts:

- **`None` → `VllmUnquantizedConfig`.** Unquantized models still go through this
  framework — they get TPU-sharded einsum linears and `moe_apply`.
- **MLX is special-cased** *before* the dict (lines 42-44): MLX checkpoints have no
  `quant_method` key, so `is_mlx_quantized(hf_config)` forces the name.
- **`tpu_int8` is in the platform list but NOT in the dict** — there is a
  `# TODO` at line 45, so it currently raises `NotImplementedError`. The two
  allow-lists (§1.5) are deliberately *not* identical.
- The function **does not instantiate the class itself**; it delegates to vLLM's
  `VllmConfig.get_quantization_config`, which pulls the TPU class out of the global
  registry (§1.4). tpu-inference only *registers* and *selects*.

### 1.3 `set_configs` — how the mesh reaches the methods

`get_quant_method` (called per layer, deep inside vLLM) has no access to the JAX
mesh. The framework threads it via **class attributes** on `VllmQuantConfig`
(`tpu_inference/layers/vllm/quantization/configs.py:111-129`):

```python
class VllmQuantConfig:
    vllm_config: VllmConfig
    mesh: Mesh
    @classmethod
    def set_configs(cls, vllm_config, mesh):     # called in get_tpu_quantization_config
        cls.vllm_config = vllm_config
        cls.mesh = mesh
    def get_linear_config(self, layer) -> VllmQuantLinearConfig:
        return VllmQuantLinearConfig(self.vllm_config, self.mesh, layer)
    def get_moe_config(self, layer) -> FusedMoEConfig:
        moe_config = layer.moe_config
        moe_config.moe_parallel_config.use_ep = \
            self.vllm_config.parallel_config.enable_expert_parallel
        return moe_config
```

> ⚠️ **Caveat:** `set_configs` writes **class-level** attributes, so the most recent
> `set_configs` call wins per config class. Fine for single-model serving (the only
> use case today).

`VllmQuantLinearConfig` (`configs.py:41-108`, extends the common
`QuantLinearConfig`) is the per-Linear-layer descriptor every linear method reads.
It computes the sharding from the layer type:

- `RowParallelLinear` → `weight_sharding = P(None, ATTN_HEAD)` (input-sharded → needs all-reduce).
- `ColumnParallelLinear` → `weight_sharding = P(ATTN_HEAD, None)` (output-sharded).
- `ReplicatedLinear` → `P(None, None)`.
- `n_shards`, `tp_size`, `num_proj` (3 for QKV, len(output_sizes) for merged-column),
  `fuse_matmuls` (from `get_model_matmul_fusion_assignment`), `output_sizes`,
  `bias_sharding`, plus the fp8-only `enable_quantized_matmul_kernel`,
  `requant_block_size`, `requant_weight_dtype`.

### 1.4 Registration: the import side-effect

The vLLM plugin entry point `register_layers` (wired in `setup.py:96-97` under
`vllm.general_plugins`) is an **empty no-op**
(`tpu_inference/layers/vllm/__init__.py:21-22`). The real work is the **import**:
`tpu_inference/layers/vllm/__init__.py:17` imports the `quantization` package, whose
`__init__.py` imports every config module (`awq`, `fp8`, `mlx`, `mxfp4`,
`compressed_tensors`, `unquantized`). Each module runs its
`@register_quantization_config(NAME)` decorator at import, which (in
`vllm/model_executor/layers/quantization/__init__.py:59-106`):

1. appends `NAME` to `QUANTIZATION_METHODS`,
2. appends `NAME` to `current_platform.supported_quantization`,
3. stores `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG[NAME] = cls`.

Then `get_quantization_config(name)` does
`method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)` (line 186), so the TPU
override wins over vLLM's built-in class of the same name.

Registration sites:

| NAME | Decorator site | Config class |
|---|---|---|
| `unquantized` | `unquantized.py:93` | `VllmUnquantizedConfig` |
| `compressed-tensors` | `compressed_tensors/compressed_tensors.py:50` | `VllmCompressedTensorsConfig` |
| `awq` | `awq.py:62` | `VllmAWQConfig` |
| `fp8` | `fp8.py:55` | `VllmFp8Config` |
| `gpt_oss_mxfp4` | `mxfp4.py:64` | `VllmMxfp4Config` |
| `mlx` | `mlx.py:64` | `VllmMLXConfig` |

> **Why subclass upstream?** e.g. `VllmFp8Config(vllm_fp8.Fp8Config, VllmQuantConfig)`
> inherits all the checkpoint-parsing (`from_config`, `ignored_layers`,
> `is_checkpoint_fp8_serialized`) and only overrides `get_name` + `get_quant_method`.
> You get vLLM's config parsing for free.

### 1.5 The platform gate (`supported_quantization`)

`tpu_inference/platforms/tpu_platform.py:94-96`:

```python
supported_quantization: list[str] = [
    "tpu_int8", "compressed-tensors", "awq", "fp8", "gpt_oss_mxfp4", "mlx"
]
```

Enforced **in vLLM upstream**, not in tpu-inference:
`ModelConfig._verify_quantization()` (`vllm/config/model.py:1022-1028`) checks the
name ∈ `QUANTIZATION_METHODS`, then calls `current_platform.verify_quantization`
(`vllm/platforms/interface.py:650-657`), which raises
`ValueError("{quant} quantization is currently not supported in {device}.")` if the
name ∉ `supported_quantization`.

So there are **two independent gates**: the platform list (vLLM-side `ValueError`)
and the TPU selector dict (`NotImplementedError`). `tpu_int8` passes the first and
fails the second.

### 1.6 `get_quant_method` — Linear vs MoE dispatch

Every TPU config uses Python structural pattern-matching on the layer type. The FP8
config is the canonical shape (`fp8.py:62-93`):

```python
def get_quant_method(self, layer, prefix):
    match layer:
        case vllm_linear.LinearBase():
            linear_config = self.get_linear_config(layer)
            if is_layer_skipped(prefix, self.ignored_layers, ...):
                return VllmUnquantizedLinearMethod(linear_config)     # ← skipped layer
            return VllmFp8LinearMethod(self, linear_config)
        case FusedMoE():
            if is_layer_skipped(...):
                return VllmUnquantizedFusedMoEMethod(layer.moe_config)
            if self.is_checkpoint_fp8_serialized:
                layer.moe_config = self.get_moe_config(layer)
                return VllmFp8MoEMethod(self, layer, self.mesh)
            else:
                raise NotImplementedError("FP8OnelineMoEMethod is not supported.")
        case Attention():
            return None     # FP8 KV-cache not implemented
        case _:
            return None
```

Uniform pattern across configs:
- `LinearBase` → `VllmXxxLinearMethod`, **or** `VllmUnquantizedLinearMethod` for
  skipped/ignored layers.
- `FusedMoE` → `VllmXxxMoEMethod`, **or** `VllmUnquantizedFusedMoEMethod` if skipped.
- `Attention` → usually `None`.
- The unquantized config additionally handles `VocabParallelEmbedding` →
  `VllmUnquantizedEmbeddingMethod` (`unquantized.py:127`).
- **mxfp4 is MoE-only:** a `LinearBase` returns `VllmUnquantizedLinearMethod`
  (`mxfp4.py:82-85`); only `FusedMoE` gets the real method.

---

## 2. Worked examples — the common contract

### 2.1 Class hierarchy

| Quant | Config (base) | Linear method (base) | MoE method (base) |
|---|---|---|---|
| **unquantized** | `VllmUnquantizedConfig(QuantizationConfig, VllmQuantConfig)` `unquantized.py:93` | `VllmUnquantizedLinearMethod(UnquantizedLinearMethod×2, VllmQuantizationMethod)` `:155` | `VllmUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod, VllmQuantizationMethod)` `:277` |
| **fp8** | `VllmFp8Config(vllm_fp8.Fp8Config, VllmQuantConfig)` `fp8.py:56` | `VllmFp8LinearMethod(vllm_fp8.Fp8LinearMethod, common_fp8.Fp8LinearMethod)` `:96` | `VllmFp8MoEMethod(vllm_fp8.Fp8MoEMethod)` `:231` |
| **awq** | `VllmAWQConfig(AWQConfig, VllmQuantConfig)` `awq.py:63` | `VllmAWQLinearMethod(AWQLinearMethod)` `:91` | `VllmAWQMoEMethod(FusedMoEMethodBase)` `:235` |
| **mxfp4** | `VllmMxfp4Config(Mxfp4Config, VllmQuantConfig)` `mxfp4.py:65` | *(none → unquantized fallback)* | `VllmMxfp4MoEMethod(Mxfp4MoEMethod)` `:95` |
| **compressed-tensors** | `VllmCompressedTensorsConfig(CompressedTensorsConfig, VllmQuantConfig)` `compressed_tensors.py:51` | upstream `CompressedTensorsLinearMethod` + a TPU **scheme** | `VllmCompressedTensorsMoEMethod` factory → per-scheme |

Compressed-tensors is itself a *family*: the upstream `CompressedTensorsLinearMethod`
dispatches by **scheme**, and tpu-inference overrides the schemes:

- Linear schemes (subclass the upstream same-name scheme):
  `VllmCompressedTensorsW8A8Fp8` `schemes/compressed_tensors_w8a8_fp8.py:82`,
  `VllmCompressedTensorsW8A8Int8` `schemes/compressed_tensors_w8a8_int8.py:49`,
  `VllmCompressedTensorsW4A8Fp8` `schemes/compressed_tensors_w4a8_fp8.py:51`.
- MoE methods (chosen by `get_moe_method`, `compressed_tensors_moe.py:31`):
  `VllmCompressedTensorsW8A8Fp8MoEMethod` `compressed_tensors_moe/...w8a8_fp8.py:39`,
  `VllmCompressedTensorsW4A8MoEMethod` `compressed_tensors_moe/...w4a8.py:45`.

### 2.2 The common contract (method → required overrides)

This is the table to internalize. **bold = always required; ○ = override only when
shapes/behavior differ from upstream; — = inherited / N/A.**

#### Linear path

| Quant | `create_weights` | `process_weights_after_loading` | forward |
|---|---|---|---|
| **unquantized** | — | **`:184`** t2j → `process_linear_weights` → `shard_linear_weights` → re-attach | **`apply(layer, x, bias=None)`** `:242` einsum `"...n,pn->...p"` |
| **fp8** | ○ `:125` (force `use_marlin=True`) | **`:145`** asserts block-quant; t2j weight + `weight_scale_inv`; `process_blockwise_fp8_linear_weights` | **`apply(layer, x, bias=None)`** `:199` → `sharded_quantized_matmul` |
| **awq** | — | **`:98`** t2j `qweight/scales/qzeros`; `awq_u32_unpack_u4`; `process_linear_weights(transposed=False)` | **`apply(layer, x, bias=None)`** `:168` dequant `(q-z)*s` then einsum |
| **mxfp4** | — | *no linear path* | inherits unquantized `apply` |
| **CT W8A8-fp8** | ○ (stub kernel) | **`apply_weights`** `:97` per-tensor dequant/requant to `float8_e4m3fn` | **`apply_weights(layer, x, bias)`** `:185` |
| **CT W8A8-int8** | ○ **`:58`** (int8 `ModelWeightParameter` + scale params) | **`:114`** `convert_to_channelwise` | **`apply_weights(layer, x, bias)`** `:179` → `sharded_quantized_matmul` |
| **CT W4A8-fp8** | ○ **`:92`** (`PackedvLLMParameter` int32) | **`:157`** unpack int4, reshape scale `(blocks,1,out)` | **`apply_weights(layer, x, bias)`** `:235` → `sharded_quantized_matmul(x_q_dtype=float8_e4m3fn)` |

> **Forward method name differs by lineage:** direct configs override **`apply`**;
> compressed-tensors schemes override **`apply_weights`**, because the upstream
> `CompressedTensorsLinearMethod.apply` (`vllm/.../compressed_tensors.py:941`) calls
> `scheme.apply_weights(layer, x, bias=bias)`.

Verbatim Linear forward signatures:
```python
def apply(self, layer, x, bias: Optional[torch.Tensor] = None) -> torch.Tensor      # direct
def apply_weights(self, layer, x, bias: Optional[torch.Tensor]) -> torch.Tensor      # CT schemes
```

#### MoE path

The TPU MoE integration is uniform: **set `is_monolithic=True` and implement
`apply_monolithic`.** None of the TPU MoE methods override `apply`.

| Quant | `create_weights` | `process_weights_after_loading` | `apply_monolithic` |
|---|---|---|---|
| **unquantized** | — | **`:318`** `process_unquantized_moe_weights` → `shard_moe_weights` | **`:370`** |
| **fp8** | — | **`:257`** dequant→requant via `process_fp8_moe_weights` | **`:302`** |
| **awq** | ○ **`:259`** (int32 packed + group scales/zeros) | **`:377`** unpack, dequant, `process_moe_weights` | **`:482`** |
| **mxfp4** | — | **`:131`** dequant mxfp4→fp4, `quantize_moe_weights`, `process_moe_weights` | **`:202`** |
| **CT W8A8-fp8** | — | **`:61`** | **`:147`** |
| **CT W4A8** | ○ **`:88`** (int32 packed) | **`:189`** unpack int4 | **`:291`** |

Verbatim MoE forward signature (identical across all TPU MoE methods):
```python
def apply_monolithic(
    self,
    layer: FusedMoE,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
```

**Where `apply_monolithic` is actually called** — the dispatch lives in vLLM
upstream, not tpu-inference. `MoERunner._apply_quant_method` reads
`self.quant_method.is_monolithic`
(`vllm/model_executor/layers/fused_moe/runner/moe_runner.py:454`) and, when `True`,
calls `apply_monolithic(layer, x, router_logits, input_ids)` (line 455), bypassing
`router.select_experts` and the base `apply`. Each TPU `apply_monolithic` builds a
`FusedMoEWeights` from `jax_view(layer.w13_weight)` etc. and returns
`vllm_moe_apply(layer, weights, quant_method_instance=self, x, router_logits)`
(`interface/moe.py:61`), which runs gating/topk **inside** the TPU kernel.

### 2.3 The minimal contract, stated plainly

To make a layer work on TPU, a quant **method** must:

- **Linear:** implement `process_weights_after_loading(layer)` (always) — convert,
  process, shard, re-attach. Implement `apply` (or `apply_weights` for a CT scheme)
  to run the matmul as JAX. Override `create_weights` **only** if the parameter
  shapes/dtypes differ from upstream (int8/int4/packed).
- **MoE:** set `@property is_monolithic -> True`, implement
  `process_weights_after_loading(layer)` and `apply_monolithic(...)`. Optionally
  `create_weights` and `get_fused_moe_quant_config`.

The unquantized linear/MoE methods additionally implement the
`VllmQuantizationMethod` ABC (`base.py:20`), whose single abstract method
`maybe_process_weights(self, layer, param_name, args, kwargs)` (`base.py:23`) is the
incremental "all shards loaded yet?" gate — relevant for sharded checkpoint loads.

---

## 3. Weight processing — torch checkpoint → sharded jax.Array

This is the heart of "process_weights_after_loading." Two layers:

- **`tpu_inference/layers/common/process_weights/`** — the pytrees + the transform
  primitives (`process_linear_weights`, `shard_linear_weights`, `process_moe_weights`,
  `shard_moe_weights`, `quantize_moe_weights`, `process_fp8_moe_weights`).
- The per-quant hooks under `layers/vllm/quantization/` call these. (There is **no**
  generic wrapper in `layers/vllm/process_weights/` — that directory holds
  `cleanup_sharding.py`, a separate model-wide LoRA + move-to-TPU pass, §3.5.)

### 3.1 The pytrees

`LinearWeights` (`common/process_weights/linear_weights.py:34`, registered via
`@jax.tree_util.register_dataclass`):

```python
@dataclass
class LinearWeights:
    weight:       jax.Array | Tensor | list[...]
    weight_scale: jax.Array | Tensor | list[...] | None
    zero_point:   jax.Array | Tensor | list[...] | None
    bias:         jax.Array | Tensor | list[...] | None
```

Fields hold a **single array** (fused matmul) or a **list** (unfused/split — e.g.
separate q/k/v). `Tensor` = `torchax.tensor.Tensor`, so leaves can be torchax or
jax.

`FusedMoEWeights` (`common/process_weights/moe_weights.py:38`, same registration):

```python
@dataclass
class FusedMoEWeights:
    w13_weight, w13_weight_scale, w13_bias
    w2_weight,  w2_weight_scale,  w2_bias
    w13_groupbias = None    # MLX affine: w = scale*q + groupbias
    w2_groupbias  = None
```

The two `*_groupbias` fields ride **the same rails as `*_weight_scale`** (NOT the
per-channel `*_bias` rails) — they are part of weight reconstruction in the k-loop,
so they must land on every shard exactly like scale.

`to_parameter_list(tensor_list)` (`linear_weights.py:71`) wraps a list of sharded
tensors into a `torch.nn.ParameterList` of `requires_grad=False` params — how the
**list-valued** (split) results get stored back on the layer.

### 3.2 Linear transforms

`process_linear_weights(...)` (`linear_weights.py:82`):
```python
def process_linear_weights(weights, fused=False, output_sizes=None,
                           reorder_size=None, transposed=True,
                           per_tensor=False) -> LinearWeights:
```
- `dim = 0 if transposed else -1` — which axis is the output dim.
- **fused branch:** `reorder_concatenated_tensor_for_sharding(weight, output_sizes,
  reorder_size, dim)` permutes the concatenated weight so each TP shard locally holds
  the per-shard slice of every constituent (q/k/v or gate/up) → **no collectives** on
  the later split. Same reorder applied to scale (skipped if `per_tensor`),
  zero_point, bias.
- **unfused branch:** `jax.lax.slice_in_dim` chops the concat into a **list** of
  sub-tensors (→ stored via `to_parameter_list`).
- `per_tensor=True` leaves the (scalar) `weight_scale` untouched.

`shard_linear_weights(...)` (`linear_weights.py:144`):
```python
def shard_linear_weights(weights, mesh, weight_p_spec, bias_p_spec,
                         transposed=True, per_tensor=False) -> LinearWeights:
```
- For `not transposed`, reverses the weight spec and re-derives the bias spec.
- **3-D weight_scale (block-quant)** gets a dedicated spec `P(in_axis, None,
  out_axis)` where the block axis can only be sharded if `num_blocks > 1`.
- Otherwise scale is replicated (`per_tensor`) or shares the bias sharding.
- `general_device_put`s each non-None field onto its sharding.

### 3.3 MoE transforms

`process_moe_weights(...)` (`moe_weights.py:217`):
```python
def process_moe_weights(weights, moe_backend, w13_reorder_size=None,
                        w13_interleave=False) -> FusedMoEWeights:
```
Pipeline:
1. **w13 un-interleave** (`w13_interleave=True`): de-interleaves checkpoints where
   even idx = w1, odd idx = w3 into contiguous `[w1 | w3]`. Triggered iff activation
   is `MoEActivation.SWIGLUOAI` (gpt-oss-style); plain SWIGLU stores them contiguous.
2. transpose the contracting dim to rightmost (`swapaxes(1,2)`).
3. reshape scale/groupbias/bias to `[E, num_blocks, 1, N]`.
4. **`moe_backend` match** — changes the layout & padding:
   - `FUSED_MOE`: asserts no groupbias; reshapes to `(E, hidden, 2, intermediate)`,
     pads hidden/intermediate to multiples of 256.
   - `GMM_TP`: requires `w13_reorder_size`; `process_w13_for_gmm` (split/pad/concat +
     `reorder_concatenated_tensor_for_sharding`) so each TP shard holds w1 and w3.
   - `GMM_EP`: `reorder_size=1`, no reorder.

`w13_reorder_size` = `get_mesh_shape_product(mesh, ShardingAxisName.MLP_TENSOR)` —
only meaningful for `GMM_TP`.

`shard_moe_weights(...)` (`moe_weights.py:466`):
- **`FUSED_MOE` / `GMM_EP`** → everything sharded on the **expert** axis
  (`P(ShardingAxisName.EXPERT)`) = expert-parallel.
- **`GMM_TP`** → tensor-parallel on `MLP_TENSOR`: `w13_weight` on
  `P(None, None, MLP_TENSOR)`, `w2_weight` on `P(None, MLP_TENSOR, None)`; scales /
  groupbiases get matching 4-D specs (replicated when the block dim == 1).

`quantize_moe_weights(weights, dtype, block_size)` (`moe_weights.py:76`) — pads
contracting dims to `align_to(block_size)` then `quantize_tensor` each. Asserts the
input isn't already quantized.

`process_fp8_moe_weights(weights, moe_backend, mesh, activation,
weight_block_size=None)` (`moe_weights.py:569`, `@jax.jit`) — **dequantizes** the fp8
block-quantized w13/w2 to fp32, then **re-quantizes** (env-overridable
`MOE_REQUANTIZE_WEIGHT_DTYPE` / `MOE_REQUANTIZE_BLOCK_SIZE`) and tail-calls
`process_moe_weights`. Does **not** call `shard_moe_weights` — the hook does that.

`process_unquantized_moe_weights(...)` (`common/quantization/unquantized.py:98`) —
the scale-less analogue; just derives `w13_interleave` / `w13_reorder_size` and calls
`process_moe_weights`.

The **MoE backend** itself is picked by `select_moe_backend_from_fused_moe_config`
(`interface/moe.py:29`) and the enum is `MoEBackend` (`common/moe.py:41`):
`FUSED_MOE="fused_moe"`, `GMM_EP="gmm_ep"`, `GMM_TP="gmm_tp"` (+ JAX-path-only
`DENSE_MAT`, `MEGABLX_GMM`). `fused_moe_backends()` returns the first three.

### 3.4 The call sequence (Linear)

```
method.process_weights_after_loading(layer):
   weight = t2j(layer.weight, use_dlpack=False);  delattr(layer, "weight")
   ... (per-quant: unpack/dequant/requant) ...
   w = process_linear_weights(LinearWeights(...), fused=..., output_sizes=...,
                              reorder_size=n_shards, transposed=..., per_tensor=...)
   w = shard_linear_weights(w, mesh, weight_sharding, bias_sharding, ...)
   layer.weight = Parameter(w.weight)            # fused
   #  or  layer.weight = to_parameter_list(w.weight)   # split
```
(MoE is the same shape: build `FusedMoEWeights` → `process_*_moe_weights` →
`shard_moe_weights` → re-attach as `Parameter`.)

`t2j` (`tpu_inference/utils.py:79`, `t2j(t, use_dlpack=False)`) is the torch→jax
bridge; `use_dlpack=False` = **copy-based** conversion (numpy/host copy), not a
zero-copy DLPack hand-off.

### 3.5 The other "process_weights" (cleanup_sharding)

`layers/vllm/process_weights/cleanup_sharding.py` is a **separate model-wide pass**,
not part of the per-layer quant hooks:
- `shard_model_to_tpu(model, mesh)` (`:41`) replicates any still-on-CPU param onto TPU
  and shards LoRA layers.
- `_tensor_is_in_cpu(tensor)` (`:86`) — the predicate that finds tensors still on
  `jax.devices('cpu')[0]`.
- `_convert_to_torchax_and_shard` (`:94`) — where `t2j` + **Pathways dummy-load**
  awareness lives (`create_dummy_weights_on_tpu` when `is_pathways_dummy_load()`).

---

## 4. Common quant utilities

`tpu_inference/layers/common/quantization/`:

| File | Contents |
|---|---|
| `__init__.py` | all low-level quant/dequant + packing math (mxfp4, awq, mlx, int/fp8) |
| `configs.py` | `QuantLinearConfig` (base for `VllmQuantLinearConfig`) |
| `fp8.py` | `Fp8LinearMethod` (common base) + `process_blockwise_fp8_linear_weights` |
| `unquantized.py` | `UnquantizedLinearMethod` (common base) + `process_unquantized_moe_weights` |

### 4.1 `UnquantizedLinearMethod` (the common base)

`unquantized.py:32` — *"shared in both vLLM and jax path."*

- `_apply_fused` (`:41`): single weight matrix. einsum **`"...n,pn->...p"`** —
  `x[..., n=hidden] · W[p=out, n=hidden] → [..., p=out]`, +bias, then
  `slice_sharded_tensor_for_concatenation` + `concat(axis=-1)`.
- `_apply_split` (`:69`): loops over a sequence of weights, same (hard-coded)
  contraction per weight, concat on `axis=-1`.

### 4.2 `Fp8LinearMethod` (the common base)

`fp8.py:32`:
- `_apply_fused` (`:41`) / `_apply_split` (`:56`) call **`sharded_quantized_matmul`**
  (§5) instead of a raw einsum, +bias, concat.
- `process_blockwise_fp8_linear_weights` (`:85`, `@jax.jit`): per output slice,
  `dequantize_tensor(..., weight_block_size)` → `quantize_tensor(requant_weight_dtype,
  ..., requant_block_size)` — i.e. **dequant-then-requant** to the TPU-friendly block
  size — then `process_linear_weights`.

### 4.3 The low-level math (`__init__.py`)

All the per-format primitives live here (no separate `awq.py`/`mxfp4.py`):

| Symbol | Line | Note |
|---|---|---|
| `MXFP4_BLOCK_SIZE = 32` | 21 | mxfp4 block size |
| `awq_u32_unpack_u4` | 73 | AWQ unpack; reverse order `(0,4,1,5,2,6,3,7)` |
| `dequantize_tensor` | 85 | generic per-tensor / per-axis / block dequant |
| `dequantize_tensor_from_mxfp4_packed` | 152 | unpack e2m1 + e8m0→fp32 |
| `quantize_tensor` | 180 | generic quant (int & float dtypes) |
| `static_per_tensor_quantize_tensor` | 255 | static scale |
| `mlx_unpack` / `mlx_dequantize` | 271 / 281 | MLX int4 unpack + affine `q*scale+bias` |
| `quantize_kv` | 291 | static per-tensor K/V quant |

> **Naming gotcha:** the w13 reorder axis is `ShardingAxisName.MLP_TENSOR`
> (`sharding.py:45` base / `:65` 2-D). There is **no `MLP_TP` symbol** anywhere in
> the codebase. The interleave toggle is `MoEActivation.SWIGLUOAI`
> (vLLM `fused_moe/activation.py:19`) — there is no plain `SWIGLU` enum member.

---

## 5. The quantized-matmul Pallas kernel

`tpu_inference/kernels/quantized_matmul/` — what fp8 / int8 / int4 linear methods
dispatch to.

| File | Role |
|---|---|
| `kernel.py` | per-channel kernel `quantized_matmul_kernel` (1-D weight scale) |
| `blockwise_kernel.py` | subchannel/blockwise variant (3-D weight scale) |
| `util.py` | quantize helpers + `xla_quantized_matmul` (pure-JAX reference) |
| `tuned_block_sizes.py` | autotuned tile-size table + lookup |

### 5.1 The sharding wrapper

`sharded_quantized_matmul` (`tpu_inference/layers/common/linear.py:40`):
```python
def sharded_quantized_matmul(x, w_q, w_s, weight_sharding, *,
                             mesh=None, x_q_dtype=None) -> jax.Array:
```
- **Picks the kernel by scale rank** (`enable_quantized_matmul_kernel = len(w_s.shape)
  == 3`, `linear.py:73`): 3-D `w_s` (block-quant) → blockwise Pallas kernel
  (`:96-100`); 1-D `w_s` (per-channel) → pure-JAX `xla_quantized_matmul` (`:102`). A
  NOTE at `linear.py:69-70` explains the per-channel **Pallas** path is disabled:
  *"there have been numeric issues (concerning) NaNs with the kernel and thus we
  disable it for now."* So 1-D scales currently go through XLA, not Pallas. (It's a
  NOTE/structural fallback, not a literal `if False:` — the **blockwise** Pallas kernel
  is still live for 3-D scales.)
- **Infers activation dtype** if `x_q_dtype is None` (`_get_x_q_dtype`, `:27`):
  integer weight → `int8`, float weight → `float8_e4m3fn`.
- **Runs under `jax.shard_map`** (`:107`): if the contracting (`in_axis`) dim is
  sharded, does `jax.lax.psum(output, axis_name=in_axis)` — tensor-parallel
  all-reduce of partial sums.

### 5.2 What the kernel fuses

Inner `matmul_kernel` (`kernel.py:21`), grid `(n_batch, n_out, n_in)`. Per tile:
1. **(optional) dynamic activation quantization** in-kernel (gated by
   `x_q_dtype != x.dtype`). Activation quant is **per-token, symmetric**: the row
   abs-max is precomputed outside the grid (`kernel.py:157`, because each Pallas block
   sees only a slice of K), scale = `abs_max / dtype_max`.
2. **the matmul** via `dot_general(contracting=((1,),(1,)))`, `int32` accumulator for
   int×int else fp32.
3. **in-K accumulation** across input blocks.
4. **dequant on the last K-step:** `acc *= w_scale; acc *= x_scale`, cast back.

**Bias is NOT fused** — added by the caller's `apply`. **Zero-point / asymmetric is
not supported** (raises `NotImplementedError`). Supported dtypes (from the tuned
table): weight/activation `int8` and `float8_e4m3fn`, plus weight `float4_e2m1fn`
(fp4) × `float8_e4m3fn` activation (blockwise kernel, accumulates in bf16).

The **blockwise kernel** (`blockwise_kernel.py:137`) applies both `lhs_scale` and
per-block `rhs_scale` per subtile and accumulates in bf16 — this is the path that
serves W4A8 (int4/fp4 weights × fp8 activations).

### 5.3 Concrete call traces

- **CT W8A8-int8:** `apply_weights` (`compressed_tensors_w8a8_int8.py:179`) →
  `_apply_fused` (`:189`) → `sharded_quantized_matmul(...)` (`:195`). 1-D scale → XLA
  path; `x_q_dtype` inferred `int8`. Bias added at `:202`.
- **FP8:** `Fp8LinearMethod._apply_fused` (`common/quantization/fp8.py:41`) →
  `sharded_quantized_matmul` (`:44`), `x_q_dtype` inferred `float8_e4m3fn`.
- **CT W4A8-fp8:** `_apply_fused` (`compressed_tensors_w4a8_fp8.py:243`) →
  `sharded_quantized_matmul(..., x_q_dtype=jnp.float8_e4m3fn)` (`:249`). 3-D scale →
  **blockwise Pallas** kernel.
- **AWQ does NOT use this kernel** — its `apply` dequantizes
  `(qweight - qzeros) * scales` and runs a plain `jnp.einsum("bd,df->bf")`.

### 5.4 Tuning

`tuned_block_sizes.py` holds `TUNED_BLOCK_SIZES_RAW` keyed by `TunedKey(tpu_version,
n_batch, n_out, n_in, x_q_dtype, w_q_dtype)` → `TunedValue(batch_block, out_block,
in_block, n_lane_multiplier)`. Lookup `get_tuned_block_sizes` (`:666`), default
`TunedValue(128,128,128)`. Per-device VMEM: `{6: 96MiB, 7: 48MiB}` (`:626`).

---

## 6. Checklist — to add a new quantization `X`

Concrete, file-anchored. Replace `X` with your quant name (e.g. `int4`).

### Step 0 — pick the name
- Add `X = "<checkpoint quant_method string>"` to
  `tpu_inference/layers/common/quant_methods.py`. The string must match what the HF
  checkpoint's `quantization_config.quant_method` reports (or detect it explicitly if
  the checkpoint has no such key, MLX-style).

### Step 1 — register & gate
- Add `"X"` to `TpuPlatform.supported_quantization`
  (`tpu_inference/platforms/tpu_platform.py:94`). Otherwise vLLM's
  `verify_quantization` rejects it with `ValueError`.
- Add `quant_methods.X: VllmXConfig` to the `method_to_config` dict in
  `tpu_inference/layers/vllm/quantization/__init__.py:46`, and import `VllmXConfig` at
  the top. Otherwise → `NotImplementedError`.

### Step 2 — the config class
- New file `tpu_inference/layers/vllm/quantization/X.py`:
  ```python
  @register_quantization_config(X)            # ← populates the vLLM registry
  class VllmXConfig(UpstreamXConfig, VllmQuantConfig):
      @classmethod
      def get_name(cls): return X
      def get_quant_method(self, layer, prefix):
          match layer:
              case LinearBase():
                  cfg = self.get_linear_config(layer)
                  if is_layer_skipped(...): return VllmUnquantizedLinearMethod(cfg)
                  return VllmXLinearMethod(self, cfg)
              case FusedMoE():
                  if is_layer_skipped(...): return VllmUnquantizedFusedMoEMethod(layer.moe_config)
                  layer.moe_config = self.get_moe_config(layer)
                  return VllmXMoEMethod(self, layer, self.mesh)
              case _: return None
  ```
  Make sure `layers/vllm/quantization/__init__.py` imports your module so the
  decorator runs (import side-effect = registration).
- Subclass the **upstream vLLM** config to inherit checkpoint parsing
  (`ignored_layers`, etc.).

### Step 3 — the Linear method
- `class VllmXLinearMethod(...)`:
  - **`process_weights_after_loading(self, layer)`** (required): `t2j` each param
    (`use_dlpack=False`), unpack/dequant/requant as needed, build `LinearWeights`,
    call `process_linear_weights(...)` then `shard_linear_weights(...)`, re-attach via
    `Parameter` (fused) or `to_parameter_list` (split).
  - **`apply(self, layer, x, bias=None)`** (or `apply_weights` if you piggyback on
    `CompressedTensorsLinearMethod`): run the matmul. Reuse `_apply_fused` /
    `_apply_split` from `common/quantization/{unquantized,fp8}.py`, or dispatch to
    `sharded_quantized_matmul` (§5) if your format is int8/fp8/int4. **Add bias
    yourself** — the kernel doesn't.
  - **`create_weights(...)`** — override only if your param shapes/dtypes differ from
    upstream (packed int32, etc.); copy the CT W8A8-int8 / W4A8-fp8 shape as a model.

### Step 4 — the MoE method (if your quant supports MoE)
- `class VllmXMoEMethod(FusedMoEMethodBase)`:
  - `@property is_monolithic(self) -> bool: return True`.
  - **`process_weights_after_loading(self, layer)`**: build `FusedMoEWeights`, call
    `process_moe_weights(..., moe_backend, w13_reorder_size, w13_interleave)` (or
    `quantize_moe_weights` first for requant), then `shard_moe_weights(...)`. Compute
    `moe_backend` via `select_moe_backend_from_fused_moe_config`,
    `w13_reorder_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_TENSOR)`,
    `w13_interleave = activation == MoEActivation.SWIGLUOAI`.
  - **`apply_monolithic(self, layer, x, router_logits, input_ids=None)`**: build
    `FusedMoEWeights(jax_view(layer.w13_weight), ...)`, return
    `vllm_moe_apply(layer, weights, quant_method_instance=self, x, router_logits)`.
  - optional `create_weights` / `get_fused_moe_quant_config`.
- ⚠️ If your format uses an affine (scale + bias) reconstruction like MLX, note
  `FUSED_MOE` backend **rejects groupbias** (`moe_weights.py:314` assert) — you must
  run on `GMM_TP`/`GMM_EP`.

### Step 5 — kernel / math (if a new numeric format)
- If int8/fp8/int4 dynamic-activation matmul covers you, reuse
  `sharded_quantized_matmul` — nothing to add.
- If you need new pack/unpack/dequant math, add it to
  `tpu_inference/layers/common/quantization/__init__.py` (e.g. an `x_u32_unpack`
  alongside `awq_u32_unpack_u4`, `mlx_unpack`).
- If you need a genuinely new accumulation pattern, extend
  `kernels/quantized_matmul/` and add tuned entries to `tuned_block_sizes.py`.

### Step 6 — tests
- Follow the existing pattern: there are per-quant tests under the repo's test tree
  (the MLX/W4A8 work added Stage-2 MoE oracle tests). Verify (a) Linear apply matches
  a dequant-then-fp reference, (b) MoE expert weights match at `tp>1` exactly.

### Files you touch, at a glance

| Always | File |
|---|---|
| name constant | `layers/common/quant_methods.py` |
| platform gate | `platforms/tpu_platform.py:94` |
| TPU selector dict + import | `layers/vllm/quantization/__init__.py:46` |
| config + methods | `layers/vllm/quantization/X.py` (new) |
| package import (registration) | `layers/vllm/quantization/__init__.py` (top imports) |
| **maybe** | |
| new numeric math | `layers/common/quantization/__init__.py` |
| new kernel + tuning | `kernels/quantized_matmul/` |

---

## Appendix — most important anchors

- TPU selector dict + MLX special-case: `layers/vllm/quantization/__init__.py:36-64`
- Name constants: `layers/common/quant_methods.py:15-20`
- `set_configs` / `get_linear_config` / `get_moe_config`: `layers/vllm/quantization/configs.py:115-129`
- `VllmQuantLinearConfig` (sharding by layer type): `configs.py:41-108`
- Platform allow-list: `platforms/tpu_platform.py:94-96`; enforcement: vLLM `config/model.py:1022-1028` + `platforms/interface.py:650-657`
- Registration decorator + merge: vLLM `model_executor/layers/quantization/__init__.py:59-106` (`register`) + `:186` (merge)
- `get_quant_method` dispatch (canonical): `fp8.py:62-93`
- Linear forward signature: `apply(layer, x, bias=None)` (e.g. `fp8.py:199`); CT scheme: `apply_weights(layer, x, bias)`
- MoE forward: `apply_monolithic(layer, x, router_logits, input_ids=None)`; dispatch gate vLLM `fused_moe/runner/moe_runner.py:454`
- `LinearWeights`: `common/process_weights/linear_weights.py:34`; `process_linear_weights:82`; `shard_linear_weights:144`
- `FusedMoEWeights`: `common/process_weights/moe_weights.py:38`; `process_moe_weights:217`; `shard_moe_weights:466`; `quantize_moe_weights:76`; `process_fp8_moe_weights:569`
- `MoEBackend` enum: `common/moe.py:41`; `MLP_TENSOR` axis: `sharding.py:45/65`; `SWIGLUOAI`: vLLM `fused_moe/activation.py:19`
- `sharded_quantized_matmul`: `layers/common/linear.py:40`; Pallas kernels: `kernels/quantized_matmul/kernel.py:115`, `blockwise_kernel.py:20`
