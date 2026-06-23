# 00 — Overview & Reading Map: `tpu-inference` × vLLM via the **torchax route**

> **Read this first.** This is the map for the `docs/torchax-deep-dive/` series. It
> explains the two ways a model can run on TPU in this stack, how vLLM hands control to
> `tpu-inference`, the core "patched vs traced" mental model, a directory map, a glossary,
> and an index of docs 01–09. Everything here is synthesized (and cross-checked) from the
> deeper docs; follow the anchors there for proof.
>
> Repos: **tpu-inference** = `/home/enyouki/tpu-inference` (package `tpu_inference/`).
> **vLLM** (editable upstream) = `/home/enyouki/vllm/vllm`. Branch `hy3`.
> All `path:line` anchors are relative to those two trees.

---

## 1. The two model-implementation routes

`tpu_inference/models/common/model_loader.py::get_model()` (`model_loader.py:463`)
dispatches on the env var **`MODEL_IMPL_TYPE`**:

```
                         MODEL_IMPL_TYPE
                                │
        ┌───────────────────────┴───────────────────────┐
     flax_nnx                                          vllm   ◄── "the torchax route"
        │                                                │        (this series)
 get_flax_model                                  get_vllm_model  (model_loader.py:412)
        │                                                │
 pure-JAX/Flax rewrites                      vLLM's OWN PyTorch nn.Module model classes,
 in tpu_inference/models/jax/                wrapped by **torchax** and executed as JAX:
 (deepseek_v3.py, llama3.py, qwen3.py …)     torch ops → JAX ops on TPU.
        │                                                │
 DEFAULT-served, hand-written,               Entry code: tpu_inference/models/vllm/
 battle-tested                               (vllm_model_loader / vllm_model_wrapper / …)
```

- **`flax_nnx`** — hand-written pure-JAX/Flax model rewrites. The default-served path for
  the models that have one (e.g. DeepSeek-V3). *Not* the focus of this series, except as a
  contrast (doc 06).
- **`vllm`** — **the torchax route.** vLLM's upstream PyTorch model classes are run *as JAX*
  via **torchax**: a `torch.Tensor` is backed by a `jax.Array` under a torchax "env", and
  torch ops are lowered to JAX ops. This lets vLLM's PyTorch models run on TPU/XLA without a
  JAX rewrite. This is the whole subject of docs 01–09.
- **`auto`** resolves to one of the two via `resolve_model_architecture()`. ⚠️ **`auto` only
  routes two architectures to the torchax route** — `GptOssForCausalLM` and
  `Qwen3MoeForCausalLM` (the `_VLLM_PREFERRED_ARCHITECTURES` set, `model_loader.py:51`);
  **everything else falls to `flax_nnx`.** So even Qwen2 resolves to `flax_nnx` under `auto`
  — you must set `MODEL_IMPL_TYPE=vllm` explicitly to force the torchax route (doc 05 §0).

**torchax** = the bridge library that makes PyTorch `nn.Module`s run on JAX/XLA. Installed at
`/home/enyouki/vllm_env/lib/python3.12/site-packages/torchax`.

---

## 2. How vLLM hands off to `tpu-inference`

There are exactly **two** integration seams — and they are *different mechanisms* (a common
source of confusion):

```
 vLLM startup
   │
   ├─(A) PLATFORM (hard-coded, NOT a plugin) ───────────────────────────────┐
   │     vllm/platforms/__init__.py:42 returns the literal string           │
   │     "tpu_inference.platforms.tpu_platform.TpuPlatform" (Pathways),      │
   │     or vllm.platforms.tpu.TpuPlatform in-tree under plain libtpu (:54). │
   │     Registered as a builtin probe (builtin_platform_plugins, :204).     │
   │     ⇒ loads tpu_inference/platforms/tpu_platform.py::TpuPlatform        │
   │        — declares device/attention/quant caps, the MLA gate, worker_cls.│
   │                                                                          │
   └─(B) LAYER/OP PATCHING (the ONE setuptools entry point) ─────────────────┘
         setup.py:96-97 registers vllm.general_plugins →
           register_layers = tpu_inference.layers.vllm:register_layers
         vLLM calls every general plugin during init.
         ⚠️ register_layers' BODY IS `pass` (layers/vllm/__init__.py:21-22).
         The real work is the IMPORT SIDE-EFFECTS of that module's top-level
         imports (backends, custom_ops, ops, quantization), which fire the
         @register_oot / @register_function / @register_backend decorators.
```

**Key invariant:** `TpuPlatform` is *hard-coded inside vLLM*, **not** a `platform_plugins`
entry point. The *only* entry point `tpu-inference` declares is `general_plugins →
register_layers`. And `register_layers` is a **no-op** — patching is purely an import
side-effect. (Docs 01 §1–2, 03 §1.)

Downstream of the handoff: worker = `tpu_inference/worker/tpu_worker.py`; model runner =
`tpu_inference/runner/tpu_runner.py`; executors in `tpu_inference/executors/`.

---

## 3. The core mental model: **Patched vs Traced**

When a vLLM PyTorch model runs on the torchax route, its modules fall into two buckets:

| Bucket | What | How it gets to TPU |
|---|---|---|
| **Patched** (structural) | embedding, all linear variants (Row/Column/Merged/Replicated parallel), `FusedMoE`, MLA / GDN attention, Deepseek-*scaling* RoPE | vLLM `@CustomOp.register_oot` **swaps the whole layer class** via base `__new__` keyed on class `__name__` (`custom_op.py` class `:103`, `register_oot` `:332`); quant methods own the real weight sharding |
| **Traced** (pointwise) | RMSNorm, SiluAndMul, **NEOX-style** RoPE, residual adds | *Not patched.* Plain torch → JAX **lowering** by torchax. For SDPA, torchax `@register_function` (`torchax/ops/jtorch.py:34`) lowers one torch fn to a JAX kernel |

**Two distinct patch systems — never conflate them:**
1. vLLM `@register_oot` → replaces an entire **layer class**.
2. torchax `@register_function` → lowers a single **torch function** to JAX.

**Special case — base `Attention`:** it is *neither* patched nor naively traced. It delegates
to its `.impl` backend, which is chosen by **`TpuPlatform.get_attn_backend_cls`**
(`tpu_platform.py:114-128`), **not** via `register_oot`. The backend (e.g. `flash_attn.py`,
`flash_attn_mla.py`) calls the Pallas kernel. (Docs 03–05; esp. 04 §0/§3.1.)

---

## 4. Cross-cutting facts everyone trips on

These hold across the whole series — internalize them before the deep docs:

1. **`register_layers` is a no-op; patching = import side-effects.** (`setup.py:96-97`;
   `layers/vllm/__init__.py:21-22`.)
2. **`TpuPlatform` is hard-coded in vLLM**, not an entry point. (`vllm/platforms/__init__.py:42`.)
3. **KV cache is threaded via a module-global wrapper context keyed by `layer_name`**
   (`set/get_vllm_model_wrapper_context`), **not** vLLM's per-layer kv tensor — the vLLM
   `kv_cache` arg into the backend is *expected* empty (the backend warns on a non-empty
   one, then deletes/ignores it). The `layer_name→index` map is passed
   as a static (hashable) jit arg; caches are donated and returned. (Docs 02 §6, 04 §2–3.)
4. **Real TP/EP sharding lives in the quant methods' `process_weights_after_loading`**
   (e.g. `unquantized.py:184` linear, `:318` MoE): t2j + shard onto the mesh, free the CPU
   copy. `shard_model_to_tpu` (`cleanup_sharding.py:41-67`) is a *separate, later* pass doing
   only LoRA sharding + replicate-leftovers (`P()`) + torchax conversion. (Docs 02 §4, 03 §7,
   08 §3.)
5. **Two quant registries** (doc 08): the TPU selector dict `method_to_config`
   (`quantization/__init__.py:46`, miss → `NotImplementedError`) picks the TPU config class;
   vLLM's global `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG` (via `@register_quantization_config`)
   gets the TPU class *merged over* vLLM's built-in (TPU wins); and the platform allow-list
   `supported_quantization` (`tpu_platform.py:94-96`) is a third independent gate.
6. **MLA hard-requires `NEW_MODEL_DESIGN=1` + DP-attention** else `check_and_update_config`
   raises (`tpu_platform.py:200-207`). `use_mla` forces the `FLASH_ATTN_MLA` backend; compute
   uses `kernels/mla/v2`; **weight absorption (W_UK→query, W_UV→output) happens OUTSIDE the
   kernel** in the custom op. (Doc 06.) *(The MLX 4-bit path, doc 07, has no NEW_MODEL_DESIGN
   gating — different code path.)*
7. **DeepSeek = two asymmetric routes:** torchax runs vLLM's *upstream*
   `DeepseekV3ForCausalLM`; the *default-served* path is the pure-JAX
   `models/jax/deepseek_v3.py`. Both converge on the same v2 Pallas MLA kernel. (Doc 06 §6–7.)

---

## 5. Directory map (orientation)

```
tpu_inference/
  platforms/tpu_platform.py     TpuPlatform: caps, MLA gate, get_attn_backend_cls, _C shims
  worker/tpu_worker.py          worker entry (declared by TpuPlatform.worker_cls)
  runner/tpu_runner.py          TPUModelRunner: init, mesh, KV alloc, execute_model, sampling
  executors/                    multiproc + ray distributed executors
  models/
    common/model_loader.py      get_model() — the MODEL_IMPL_TYPE dispatch
    vllm/                       ◄ torchax route: vllm_model_loader, vllm_model_wrapper,
                                   vllm_model_wrapper_context, mlx_weight_transform
    jax/                        pure-JAX rewrites (deepseek_v3, llama3, qwen3 …)
  layers/
    vllm/                       ◄ THE PATCHES (imported by register_layers):
      backends/                   attn backends: flash_attn.py, flash_attn_mla.py
      custom_ops/                 mla_attention.py, rope.py, gdn_attention_op.py, linear/embedding/fused_moe
      ops/                        scaled_dot_product_attention.py (torchax register_function)
      process_weights/            weight post-processing primitives
      quantization/               fp8.py, awq.py, mxfp4.py, mlx.py, unquantized.py, compressed_tensors/
      interface/
    common/                     shared: sharding.py, linear.py, moe.py, quant_methods.py,
                                attention_interface.py, quantization/
  kernels/                      Pallas: ragged_paged_attention/v3, mla/v1+v2, fused_moe/,
                                megablox/ (gmm), quantized_matmul/, …
  distributed/                  sharding / multi-host
  envs.py                       MODEL_IMPL_TYPE, NEW_MODEL_DESIGN, … (TPU env vars)
```

---

## 6. Glossary

- **torchax route / `vllm` route** — `MODEL_IMPL_TYPE=vllm`: run vLLM's PyTorch model as JAX.
- **torchax** — library lowering torch ops to JAX; `jax_view`/`torch_view` convert between
  views; the "env" is the active lowering context.
- **`register_oot`** — vLLM `CustomOp` "out-of-tree" registration; swaps a whole layer class
  on TPU via base `__new__`.
- **`register_function`** — torchax decorator lowering one torch fn to a JAX implementation.
- **register_layers** — the (no-op) general-plugin entry point whose *import* triggers all the
  above.
- **Patched / Traced** — structural layers swapped via register_oot/quant-method vs pointwise
  ops lowered as plain torch→JAX (see §3).
- **wrapper context** — module-global holding `kv_caches` + `layer_name_to_kvcache_index`;
  how KV is threaded into the jitted forward (§4.3).
- **MLA** — Multi-head Latent Attention (DeepSeek): latent KV (`kv_lora_rank`) + decoupled
  RoPE; on TPU uses `kernels/mla/v2` with external weight absorption.
- **process_weights_after_loading** — the quant-method hook where real t2j + TP/EP sharding
  happens.
- **NEW_MODEL_DESIGN** — env gate required (with DP-attention) for the MLA path.

---

## 7. Index — the deep-dive docs

Read roughly in order; 05 is the best concrete anchor once you've seen 01–04.

| # | Doc | What it covers | Start here if you want… |
|---|---|---|---|
| **00** | this file | routes, handoff, mental model, map, glossary | the big picture |
| **01** | `01-engine-platform-bringup.md` | engine create → TpuPlatform detection → register_layers → executor → worker/runner init → JAX mesh & multi-chip sharding (up to "ready", no forward) | how startup wires up on TPU |
| **02** | `02-torchax-model-loading.md` | `get_vllm_model` → build vLLM nn.Module on CPU → load+shard weights into jax.Arrays → torchax bridge → `jit_step_func` | how a PyTorch model becomes a jitted JAX fn |
| **03** | `03-layer-patching-custom-ops.md` | the four patch mechanisms (register_oot, torchax register_function, register_backend, quant-method) + `_C` dummy shims, for dense models | how layer overriding actually works |
| **04** | `04-forward-pass-pallas.md` | dense decode/prefill: `execute_model` → padded sharded inputs → jitted forward → ragged-paged-attention v3 Pallas kernel → deferred sampling | the runtime forward path |
| **05** | `05-simple-model-trace.md` | **concrete Qwen2 trace** end-to-end; patched-vs-traced layer table; the `auto`→flax_nnx routing gotcha | a worked example to ground everything |
| **06** | `06-deepseek-v3-mla.md` | MLA on TPU: backend selection, custom op + **external weight absorption**, v1/v2 Pallas kernels, latent KV cache; pure-JAX vs torchax contrast; why **v4 ≠ free** | attention beyond dense; the V3/V4 contract |
| **07** | `07-hy3-mlx-4bit.md` | MLX 4-bit MoE checkpoint (Qwen3-30B-A3B-4bit / Hunyuan): keep-4bit + dequant-in-forward, MLX linear/MoE methods, in-kernel int4 dequant in `gmm_v2` | a real quantized model bring-up |
| **08** | `08-quantization-framework.md` | quant name → TPU config/method, the two registries, the weight-processing pipeline, the quantized-matmul Pallas kernel, + **"add a new quant" checklist** | adding a new quantization |
| **09** | `09-adding-deepseek-v4-playbook.md` | concrete playbook to add **DeepSeek V4** (new kernel/op/sharding — *not* a V3 reuse) and the easier **V3.2** path; gap analysis + sequencing | adding a new hard model |

---

## 8. Where to go for the three end goals

- **Add a new model via torchax** → 05 (trace) + 03 (patching) + 02 (loading); 04 for the
  forward path.
- **Add a new quantization** → 08 (framework + checklist) + 07 (a real worked example, MLX).
- **Add DeepSeek V4 (or V3.2)** → 09 (playbook), grounded in 06 (MLA contract) and 08 (quant
  / sharding hooks). Bottom line from 09: **V3.2 already fits** the existing MLA contract
  (runs correctly today, just dense not sparse-accelerated); **V4 does not** — it needs a new
  Pallas kernel, a new custom op, fp8-KV plumbing, and new MoE/sharding work.
