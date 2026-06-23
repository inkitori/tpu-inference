# 01 — Engine creation, TPU platform bring-up, worker & runner init, multi-chip distribution

> Scope: the **startup / initialization path** of a vLLM V1 engine on a multi-chip TPU,
> ending with a *ready* `TPUWorker` + `TPUModelRunner` (mesh built, model loaded, KV cache
> allocated, model warmed up). The **forward pass** (`execute_model`) is intentionally out
> of scope — see the runner/forward doc.
>
> Audience: an engineer new to this codebase who will add new models / quantizations via the
> torchax route. All anchors are `path:line` against the `hy3` branch.
>
> Repos: tpu-inference = `/home/enyouki/tpu-inference`, vLLM (editable) = `/home/enyouki/vllm/vllm`.

---

## 0. The one-paragraph mental model

vLLM is the **control plane** (scheduler, request lifecycle, config plumbing, executor
abstraction). tpu-inference plugs in underneath via three seams: (1) a **Platform** class
selected by string from vLLM's platform detector; (2) a **general-plugin** entry point that
monkeypatches vLLM's PyTorch layers with TPU implementations at import time; (3) a
**worker class** name (`TPUWorker`) that vLLM instantiates in each executor process. The
worker then constructs a `TPUModelRunner`, which builds the **JAX device mesh**, loads the
model, allocates the KV cache, and compiles. On a single-host v6e-8 (8 chips, TP=8) the whole
thing runs in **one process** doing **single-controller SPMD** over all 8 chips — there are no
per-chip processes. Multi-process only appears with pipeline parallelism (one process per PP
stage) or multi-host (one Ray actor per host).

---

## 1. Entry → platform detection → `TpuPlatform`

### 1.1 How vLLM picks the TPU platform

vLLM's platform detector probes registered platform plugins. The TPU probe lives at
`vllm/platforms/__init__.py:36` (`tpu_platform_plugin()`) — a **builtin** plugin
(`builtin_platform_plugins["tpu"]`, line 204) baked into *this* vLLM build — and returns a
**fully-qualified class string**:

```python
# vllm/platforms/__init__.py:40-54
if envs.VLLM_TPU_USING_PATHWAYS:
    return "tpu_inference.platforms.tpu_platform.TpuPlatform"   # Pathways/proxy path
...
    import libtpu                                               # plain-libtpu path
    return "vllm.platforms.tpu.TpuPlatform"
```

Key fact (verified): the hand-off to tpu-inference is **hard-coded inside vLLM**, *not*
registered by tpu-inference's `setup.py` (which only declares the `general_plugins` entry
point — there is no `vllm.platform_plugins` entry here). So this vLLM is a fork/patched build
whose builtin TPU probe points at `tpu_inference...TpuPlatform` under Pathways. The plain
`import libtpu` branch (line 54) returns vLLM's *in-tree* `vllm.platforms.tpu.TpuPlatform`
instead; per SHARED_CONTEXT the canonical hand-off for this repo is the
`tpu_inference...TpuPlatform` string. vLLM then imports that class and uses it as
`current_platform` for every device-specific decision. This is the **primary seam**: vLLM
never imports tpu-inference directly; it discovers it through this string.

### 1.2 What `TpuPlatform` declares / overrides

File: `tpu_inference/platforms/tpu_platform.py`, class `TpuPlatform(Platform)` at line 85.

**Device identity (class attributes, lines 86-96):**

```python
_enum = PlatformEnum.TPU
device_name = "tpu"; device_type = "tpu"
dispatch_key = "XLA"              # torch dispatch key
ray_device_key = "TPU"           # Ray resource name used by the Ray executor
device_control_env_var = "TPU_VISIBLE_CHIPS"
simple_compile_backend = "openxla"
supported_quantization = ["tpu_int8", "compressed-tensors", "awq", "fp8",
                          "gpt_oss_mxfp4", "mlx"]
```

> `supported_quantization` (line 94) is the allow-list a new quantization must be added to.
> `mlx` is present — relevant to the 4-bit MoE work.

**Extra env vars surfaced to vLLM** (`additional_env_vars`, lines 98-112): includes
`NEW_MODEL_DESIGN`, `MODEL_IMPL_TYPE`, `TPU_MULTIHOST_BACKEND`, `VLLM_MLA_DISABLE`,
`TPU_BACKEND_TYPE`, MoE requantize knobs, JAX profiler knobs.

**Attention backend selection** — `get_attn_backend_cls()` (lines 114-128): TPU supports only
two backends. MLA models force `FLASH_ATTN_MLA`; everything else is coerced to `FLASH_ATTN`
(any other request is logged and overridden). Returns the backend's import path string:

```python
# lines 120-128
if use_mla:
    selected_backend = AttentionBackendEnum.FLASH_ATTN_MLA
elif selected_backend != AttentionBackendEnum.FLASH_ATTN:
    selected_backend = AttentionBackendEnum.FLASH_ATTN
return selected_backend.get_path()
```

**`check_and_update_config(vllm_config)` (lines 192-298) — the most important hook.** vLLM
calls this once, early, to let the platform mutate the fully-assembled config. It does, in
order:

1. **Pathways guard** (195-198): if `VLLM_TPU_USING_PATHWAYS`, assert
   `VLLM_ENABLE_V1_MULTIPROCESSING == 0`.
2. **MLA requires NEW_MODEL_DESIGN + DP attention** (200-207): otherwise raises.
3. **Build the sharding config** (208) → `_initialize_sharding_config()` (186-190), which calls
   `ShardingConfigManager.from_vllm_config(vllm_config)` and **stashes it on
   `vllm_config.sharding_config`** — this is how the runner later finds it.
4. **Force compilation mode** (210-217): TPU only supports
   `CompilationMode.DYNAMO_TRACE_ONCE`; backend defaulted to `"openxla"` (219-220).
   *(Note: comment at 215 says this config "is not used by jax" — JAX/XLA does its own
   compilation via the runner's `CompilationManager`.)*
5. **KV-cache block size** (222-248): chosen from the attention backend's `get_page_size()`
   (MLA vs non-MLA), bumped to `min_page_size` to avoid SMEM OOM, and to 256 if batched RPA is
   enabled.
6. **Worker class injection** (252-253): `parallel_config.worker_cls =
   "tpu_inference.worker.tpu_worker.TPUWorker"` — **this is the seam where vLLM is told to
   instantiate our worker.**
7. **Executor selection** (255-277) — see §3.
8. Multimodal forcing (279-284), KV-transfer connector allow-list (286-294), DP scheduler
   patch (296-298).

**Other notable overrides:** `get_device_communicator_cls()` → TPU communicator (329-330);
`get_punica_wrapper()` → TPU LoRA wrapper (166-167); `current_device()` returns `cpu` because
"we'll manually place tensors on the TPU device(s)" (371-381); `is_async_output_supported()` =
False (162); `inference_mode()` = True (182); `use_sync_weight_loader()` = True (360).

---

## 2. `register_layers` general-plugin: when it fires & what it does

### 2.1 The entry point

`setup.py:96-97`:

```python
"vllm.general_plugins": [
    "register_layers = tpu_inference.layers.vllm:register_layers",
],
```

### 2.2 The subtlety — patching is an *import side effect*, not the function body

`tpu_inference/layers/vllm/__init__.py:21-22` — the registered function is **empty**:

```python
def register_layers():
    pass
```

The real work happens at **module import time** (lines 14-17): importing
`tpu_inference.layers.vllm` imports its submodules `backends`, `custom_ops`, `ops`,
`quantization`, and *those* modules monkeypatch vLLM's layer/op/quant classes with TPU
implementations as a side effect of being imported. The entry-point function only needs to
*exist* so vLLM loads (hence imports) the package; calling `func()` is a no-op. **Takeaway for
adding a new layer/op/quant:** wire it into one of those four submodules' import graph; you do
not edit `register_layers` itself.

### 2.3 Who calls general plugins, and when (ordering)

vLLM's loader: `vllm/plugins/__init__.py:69-82` (`load_general_plugins()`), which iterates the
`vllm.general_plugins` group (`load_plugins_by_group`, lines 28-66) and calls each. It is
**idempotent** (guarded by `plugins_loaded`, line 75) and is invoked from several places so it
fires no matter the entry path:

- `vllm/v1/engine/core.py:103-105` — **`EngineCore.__init__`**, the very first lines, *before*
  the executor/worker/model are created.
- `vllm/v1/worker/worker_base.py:237-239` — inside each worker process during
  `init_worker`, so subprocesses re-trigger the patch.
- `vllm/engine/arg_utils.py:718-720` and `:2455` — at arg-parsing time in the frontend.
- `vllm/model_executor/models/registry.py:1355-1357` — before model-registry resolution.

**Ordering guarantee:** because `EngineCore.__init__` calls it at line 103 — and
`worker_base.py` re-calls it in every worker before model load — the layer monkeypatches are
**always installed before `get_model()` runs**. So when the torchax route instantiates vLLM's
PyTorch model classes, they are already the TPU-patched versions.

---

## 3. Executor selection: how worker processes/ranks are spawned

Decision made in `TpuPlatform.check_and_update_config` (`tpu_platform.py:255-277`), driven by
`TPU_MULTIHOST_BACKEND` and `pipeline_parallel_size`:

| Condition | Executor | Process model |
|---|---|---|
| single host, **PP=1** (e.g. v6e-8 TP=8) | `"uni"` (vLLM `UniProcExecutor`) | **1 process**, SPMD over all chips |
| single host, **PP>1** | `tpu_inference.executors.multiproc_executor.MultiprocExecutor` | **1 process per PP stage** |
| `TPU_MULTIHOST_BACKEND=="ray"` | `tpu_inference.executors.ray_distributed_executor.RayDistributedExecutor` | **1 Ray actor per host** |
| unknown backend | `"uni"` (warns) | 1 process |

```python
# tpu_platform.py:256-266
if not multihost_backend:               # single host
    if parallel_config.pipeline_parallel_size == 1:
        parallel_config.distributed_executor_backend = "uni"
    else:
        parallel_config.distributed_executor_backend = MultiprocExecutor
elif multihost_backend == "ray":
    parallel_config.distributed_executor_backend = RayDistributedExecutor
```

### 3.1 `MultiprocExecutor` (single-host PP)

`tpu_inference/executors/multiproc_executor.py:23` subclasses vLLM's `MultiprocExecutor`. Its
docstring states the design: **MPMD for Pipeline Parallelism, SPMD for everything else (TP, CP,
EP, DP).** Key override `_get_parallel_sizes()` (lines 31-37):

```python
self.world_size = self.parallel_config.pipeline_parallel_size  # one proc per PP stage
tp_size = 1                                                     # TP is NOT split across procs
return tp_size, pp_size, pcp_size
```

→ It spawns **`pipeline_parallel_size` processes**; each process drives its PP stage's chips
via **JAX SPMD** internally (so TP within a stage is single-controller, not process-split).
`_post_init_executor()` (39-43) wires PP transfer connections; `_get_output_rank()` returns the
last PP rank (45-46); every rank is a driver (48-49).

### 3.2 `RayDistributedExecutor` (multi-host)

`tpu_inference/executors/ray_distributed_executor.py:72` subclasses vLLM's Ray executor.
Differences (per its docstring + code):

- `_init_executor()` sets `use_ray_spmd_worker = True` (line 107): driver == worker, all ranks
  are remote Ray workers.
- `_initialize_ray_cluster()` (135-230): **one TPU node (all its chips) maps to one placement
  group** — unlike GPU where one device = one PG. For PP>1, each PP rank gets one node's full
  chip allocation (190-196).
- `_init_workers_ray()` (231-463): each Ray actor is created with `ray_device_key` ("TPU")
  resources = chips-per-node (266-282). Rank order follows sorted host/IP; each worker's env
  (`TPU_VISIBLE_CHIPS`) selects its local chips.

> **Critical multi-chip insight (verified):** even under Ray, **TP collectives happen via JAX
> SPMD on the local chips of a single host/actor** — Ray is used for PP and cross-host
> orchestration, not for TP. See worker `init_device` §4.1: each Ray worker is set up as its
> own isolated single-host JAX cluster.

---

## 4. `TPUWorker` init (device/JAX init, runner construction, KV hooks)

File: `tpu_inference/worker/tpu_worker.py`. vLLM instantiates this (the `worker_cls` set in
§1.2 step 6) once per executor process and drives a fixed lifecycle.

### 4.0 Lifecycle order (vLLM calls these in sequence)

```
__init__                      (94)   capture config; no device, no runner
  └ init_device               (167)  JAX/TPU init + **construct TPUModelRunner** (302)
  └ load_model                (456)  → runner.load_model() (457)
  └ get_kv_cache_spec         (488)  → runner.get_kv_cache_spec()
  └ determine_available_memory(340)  probe HBM → #KV blocks
  └ initialize_from_config    (495)  precompile sampling + **runner.initialize_kv_cache** (510)
  └ compile_or_warm_up_model  (459)  → runner.capture_model() (460) — XLA compile/warm-up
```

### 4.1 `__init__` (94-160) and `init_device` (167-332)

`__init__` only **captures config** (vllm_config, parallel/cache/model config via
`super().__init__`), stores `rank`/`local_rank`, builds a `PPConfig`, and sets up the profiler.
**No runner, no device touched here.**

`init_device` does the heavy lifting:

1. **PP / multi-host env setup** (167-228). For Ray multihost (188-193): isolate each host as
   its **own JAX cluster** (`TPU_PROCESS_ADDRESSES=localhost`, `CLOUD_TPU_TASK_ID=0`) so TP
   collectives work on local chips. For single-host PP (197-205): assign per-rank ports from
   `jax_parallel_state.BASE_JAX_PORT` (=5000).
2. **Device resolution** (230-264): if `self.devices` empty, pick from `jax.local_devices()`
   (PP) or `jax.devices()` (non-PP), sliced to `sharding_config.total_devices`.
3. **Fake vLLM distributed init** (268-280): vLLM's own parallel state is initialized as
   **world_size=1, TP=1, PP=1 with a gloo file backend** — i.e. vLLM thinks it is single-chip;
   real device parallelism is expressed entirely through the **JAX mesh**, not vLLM's
   collectives.
4. **JAX PP init** (282-287): `jax_parallel_state.init_pp_distributed_environment(...)` creates
   a `GroupCoordinator` and, only when PP>1, starts a JAX transfer server for inter-stage
   tensor send/recv (`distributed/jax_parallel_state.py:60-71`).
5. **Construct the runner** (302-304):

   ```python
   self.model_runner = TPUModelRunner(self.vllm_config, self.devices,
                                      self.rank, is_first_rank, is_last_rank)
   ```

   The mesh is built **inside** this constructor (§5).

### 4.2 KV / memory hooks

- `determine_available_memory()` (340-392): probes HBM via `utils.hbm_usage_bytes(self.devices)`
  (uses `device.memory_stats()`, or `jax.live_arrays()` under Pathways), applies
  `gpu_memory_utilization`, deducts KV-offload staging buffers if enabled, returns available
  bytes (vLLM converts this to a #blocks).
- `initialize_cache(num_gpu_blocks, num_cpu_blocks)` (162-165): trivial — just records the
  counts on `cache_config`. **No allocation here.**
- `initialize_from_config(kv_cache_config)` (495-511): precompiles sampling funcs (unless eager),
  `ensure_kv_transfer_initialized()`, then **`self.model_runner.initialize_kv_cache(...)`**
  (510) — the real KV allocation.
- `compile_or_warm_up_model()` (459-467): `self.model_runner.capture_model()` (XLA
  compile/warm-up), then re-seeds RNG.

---

## 5. `TPUModelRunner.__init__` (construction only)

File: `tpu_inference/runner/tpu_runner.py`, `TPUModelRunner.__init__` at **line 242**.

What it does, in order:

1. **Store config refs** (250-269): model/cache/lora/load/parallel/scheduler/spec/device
   configs; `self.devices = devices`; `self.dp_size =
   vllm_config.sharding_config.total_dp_size` (266); rank + first/last-rank flags.
2. **`_init_random()`** (271 → 307): seed PRNG, set `self.rng_key`.
3. **`_init_mesh()`** (272 → 314) — **builds the JAX mesh** (see §6). This is the single place
   the device mesh is created.
4. `_init_phased_profiling()` (273), `_init_mm()` (274), `_init_inputs()` (275 → 444:
   allocates **CPU** numpy buffers + `InputBatch`, *not* device arrays),
   `_init_speculative_decoding()` (276).
5. **Manager construction** (278-289): `CompilationManager(self)` (279) — owns XLA
   compilation/warm-up; `SpeculativeDecodingManager` + `StructuredDecodingManager` (last rank,
   280-283); `KVCacheManager(self)` (284); `MultiModalManager`, `PersistentBatchManager`,
   `LoraUtils`.
6. **Empty KV placeholders** (301-302): `self.kv_caches = []`,
   `self.layer_name_to_kvcache_index = {}` — filled later by `initialize_kv_cache`.

**What `__init__` does NOT do (deferred):**

| Action | Deferred to |
|---|---|
| `get_model()` / weight load | `load_model()` (548-603) |
| KV cache device allocation | `initialize_kv_cache()` (619-645) |
| XLA compile / warm-up | `capture_model()` (653-654) |

**`load_model()` (548-603)** is the model-load trigger. It calls `get_model(self.vllm_config,
self.rng_key, self.mesh)` (≈550) — i.e. `tpu_inference/models/common/model_loader.py::get_model`,
which dispatches on `MODEL_IMPL_TYPE` (the torchax route is documented elsewhere). It stores
`self.model_fn`, `self.compute_logits_fn`, `self.state`, `self.model`, etc., and allocates
`self.rng_params_for_sampling` on-device via `device_array(... NamedSharding(self.mesh,
PartitionSpec()))`. Note the mesh from `__init__` is passed straight into the loader, so
**weights are sharded onto the mesh at load time.**

---

## 6. Multi-chip specifics: how the JAX mesh / sharding is built

### 6.1 The two stages: sizes (platform) → physical mesh (runner)

- **Stage A — logical sizes:** `ShardingConfigManager.from_vllm_config()`
  (`layers/common/sharding.py:147-233`), called from the platform in §1.2. It reads
  `additional_config["sharding"]["sharding_strategy"]` plus
  `parallel_config.tensor_parallel_size` / `data_parallel_size`, and computes a
  `ShardingStrategy` with five degrees: `tensor_parallelism`, `data_parallelism`,
  `expert_parallelism`, `attention_data_parallelism` (`attn_dp`),
  `attention_data_expert_parallelism` (`attn_dp_expert`). `total_devices = product of all
  five` (143). When DP-attention is enabled it **derives `attn_dp` from KV-head/packing math**
  and rebalances expert parallelism (171-212) — this is the non-obvious part. It also forces
  vLLM's own `data_parallel_size` back to 1 (227-230) so vLLM doesn't spin up multiple DP
  engines (DP is expressed in the mesh instead).

- **Stage B — physical mesh:** `TPUModelRunner._init_mesh()` (`tpu_runner.py:314-321`) turns
  those sizes into a `jax.sharding.Mesh` over `self.devices`.

### 6.2 Mesh shape and axis names (the TP→mesh mapping)

Two layouts, chosen by `NEW_MODEL_DESIGN`:

**5D "new model" mesh** (`_create_new_model_mesh` 325-336 → `_create_single_slice_mesh`
338-362). Axis names `MESH_AXIS_NAMES = ("data", "attn_dp", "attn_dp_expert", "expert",
"model")` (`sharding.py:29`). The mesh **shape is exactly the five sharding sizes**:

```python
# tpu_runner.py:340-346
mesh_shape = (model_dp_size, attn_dp_size, attn_dp_expert_size, expert_size, tp_size)
return mesh_utils.create_device_mesh(mesh_shape, self.devices,
                                     allow_split_physical_axes=True)
# falls back to np.array(self.devices).reshape(mesh_shape) on non-pow2 device counts (357-362)
```

So **`tp_size` is the last (`"model"`) mesh axis**; tensor-parallel weights are partitioned
along `"model"`. (Logical → mesh-axis mapping for each weight/activation is defined by
`ShardingAxisName*` and `ShardingRulesConfig` in `sharding.py:33-359`, used in the forward/
weight-sharding path — out of scope here.)

**2D mesh** (`_create_2d_mesh` 396-417), used when `NEW_MODEL_DESIGN` is off (the current MoE
kernel requires a 2D mesh). Axes `MESH_AXIS_NAMES_2D = ('data', 'model')` (`sharding.py:30`);
shape `(model_dp_size, tp_size)`. Built with `make_optimized_mesh` (415) or `jax.make_mesh`
(410) when explicit `device_indexes` are given.

**Multi-slice** (`_create_multi_slice_mesh` 364-394, `NUM_SLICES>1`): splits the `data` axis
across slices using `mesh_utils.create_hybrid_device_mesh` (ICI within slice + DCN across
slices).

> **TP=N on a single host → an N-wide `"model"` mesh axis over N JAX devices, all in one
> process.** There is no per-chip process; XLA/SPMD fans the single program across the N chips
> and inserts the collectives. For v6e-8 / TP=8 / PP=1: `mesh_shape = (1,1,1,1,8)` (5D) or
> `(1,8)` (2D).

### 6.3 Single-controller vs multi-host (Pathways)

- **Single host (default, `TPU_MULTIHOST_BACKEND=""`):** **single-controller SPMD.** One Python
  process, `jax.devices()` returns all local chips, one mesh spans them; vLLM's own distributed
  state is faked to world_size=1 (§4.1 step 3). This is the common multi-chip case.
- **Multi-host via Ray (`TPU_MULTIHOST_BACKEND="ray"`):** one Ray actor per host; **each actor
  is its own isolated single-host JAX cluster** (worker `init_device` 188-193). TP runs SPMD on
  each host's local chips; cross-host coordination (PP, KV transfer) is handled above JAX via
  Ray + the `distributed/` connectors. Topology rank order is derived from JAX device
  `process_index`/coords by `distributed/utils.py::get_device_topology_order_id`.
- **Pathways (`VLLM_TPU_USING_PATHWAYS`, i.e. `JAX_PLATFORMS=proxy`):** single-controller proxy
  front-end to a multi-device backend; requires `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  (`tpu_platform.py:195-198`). HBM probing switches to `jax.live_arrays()` accounting.

The `tpu_inference/distributed/` package does **not** call `jax.distributed.initialize()`; it
holds PP coordination (`jax_parallel_state.py`), multi-host KV-transfer connectors
(`tpu_connector*.py`, `kv_transfer.py`, `host_kv_pool*.py`), and topology utilities
(`utils.py`).

---

## 7. Bring-up sequence diagram

```mermaid
sequenceDiagram
    autonumber
    participant CLI as vllm serve / LLM(...)
    participant Det as vllm/platforms/__init__.py
    participant Plat as TpuPlatform
    participant Core as EngineCore.__init__
    participant Plug as load_general_plugins
    participant Exec as Executor (uni / Multiproc / Ray)
    participant W as TPUWorker
    participant R as TPUModelRunner
    participant ML as get_model (torchax route)

    CLI->>Det: detect platform
    Det-->>Plat: "tpu_inference...TpuPlatform" (string)
    Note over Plat: check_and_update_config()
    Plat->>Plat: build sharding_config (ShardingConfigManager)
    Plat->>Plat: worker_cls = TPUWorker; pick executor
    CLI->>Core: construct EngineCore
    Core->>Plug: load_general_plugins() (BEFORE model load)
    Note over Plug: import tpu_inference.layers.vllm.* → monkeypatch vLLM layers/ops/quant
    Core->>Exec: create executor
    Exec->>W: __init__ (capture config only)
    Exec->>W: init_device()
    Note over W: JAX/TPU env, resolve devices,<br/>fake vLLM dist (ws=1), JAX PP init
    W->>R: TPUModelRunner(vllm_config, devices, rank, ...)
    Note over R: _init_mesh() → jax.sharding.Mesh<br/>(tp_size = "model" axis)<br/>+ CompilationManager, KVCacheManager
    Exec->>W: load_model()
    W->>R: load_model()
    R->>ML: get_model(vllm_config, rng_key, mesh)
    ML-->>R: model_fn / state (sharded on mesh)
    Exec->>W: determine_available_memory()  --> #KV blocks
    Exec->>W: initialize_from_config(kv_cache_config)
    W->>R: initialize_kv_cache()  (allocate KV on device)
    Exec->>W: compile_or_warm_up_model()
    W->>R: capture_model() (XLA compile / warm-up)
    Note over W,R: READY
```

ASCII fallback (linear):

```
vllm serve / LLM
  → platforms/__init__.py:42  returns "tpu_inference...TpuPlatform"
  → TpuPlatform.check_and_update_config  (tpu_platform.py:192)
        sharding_config built (208) | worker_cls=TPUWorker (252) | executor picked (255)
  → EngineCore.__init__  (vllm v1/engine/core.py:103) → load_general_plugins()
        import tpu_inference.layers.vllm.*  → monkeypatch vLLM layers   [BEFORE model load]
  → Executor (uni 1-proc | Multiproc 1/PP-stage | Ray 1/host)
  → TPUWorker.__init__ (94)  [config only]
  → TPUWorker.init_device (167)
        JAX env | resolve devices | fake vLLM dist ws=1 (268) | JAX PP init (282)
        → TPUModelRunner(...) (302)
              _init_mesh → jax.sharding.Mesh   (mesh_shape[-1] = tp_size = "model" axis)
              CompilationManager / KVCacheManager / ...
  → load_model (457) → runner.load_model (548) → get_model(vllm_config, rng_key, mesh)
  → determine_available_memory (340)  → #KV blocks
  → initialize_from_config (495) → runner.initialize_kv_cache (510)
  → compile_or_warm_up_model (459) → runner.capture_model (460)   ==> READY
```

---

## 8. Key takeaways for extending the codebase

- **New quantization:** add its name to `TpuPlatform.supported_quantization`
  (`tpu_platform.py:94`) and register the method via the `quantization` submodule of
  `tpu_inference/layers/vllm/` (imported through `register_layers`'s package).
- **New model (torchax route):** nothing in *this* path changes — the model is resolved in
  `get_model` (called from `runner.load_model`, ~tpu_runner.py:550). The mesh is already built
  and handed in. Your model just needs torchax-compatible weight sharding against
  `MESH_AXIS_NAMES`.
- **The mesh is the single source of truth for device parallelism.** vLLM's own distributed
  state is deliberately faked to world_size=1; TP/EP/DP all live in the JAX mesh axes.

## 9. Open questions / not fully verified

- **RESOLVED (was: how is the platform registered?):** It is **not** a setuptools entry point.
  tpu-inference `setup.py` registers only `vllm.general_plugins`
  (`register_layers`). The TPU→tpu-inference hand-off is **hard-coded inside this vLLM build**
  (`vllm/platforms/__init__.py:42`, the builtin `tpu` probe). Under the plain `import libtpu`
  branch the same function returns vLLM's *in-tree* `vllm.platforms.tpu.TpuPlatform` (line 54),
  so confirm which branch your deployment hits (Pathways vs libtpu) if the active platform class
  ever looks wrong.
- `CompilationManager.capture_model()` internals (what gets traced/compiled, dynamo-once vs
  pure JAX jit) are out of scope — belongs in the runner/forward + compilation doc.
- `mesh_utils.create_device_mesh` physical-topology heuristics (when the pow2 fallback at
  `tpu_runner.py:357-362` triggers in practice) not exercised here.
