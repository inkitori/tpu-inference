# 04 — The Forward / Decode Pass: `execute_model` → Pallas Ragged Paged Attention → Sample

> Scope: the **dense (non-MLA) attention** decode/prefill path on the **torchax route**
> (`MODEL_IMPL_TYPE=vllm`). MLA (DeepSeek-style latent attention) is documented separately.
> Audience: a strong engineer new to this repo, who will eventually add new models /
> quantizations / DeepSeek-v4 via torchax.
>
> All `path:line` anchors verified against the `hy3` branch. Paths are relative to the
> repo root `/home/enyouki/tpu-inference` unless noted.

---

## 0. The 30-second map

`execute_model` is **host-side orchestration**. It turns a vLLM `SchedulerOutput`
into padded, sharded device arrays, calls one big jitted function (`self.model_fn`,
the torchax-wrapped vLLM model), extracts logits, and then samples. The only TPU
compute is: (a) `model_fn`, (b) `compute_logits_fn`, (c) `sample`. Everything else is
NumPy on the host shuffling indices into pre-allocated CPU staging buffers.

The attention kernel is reached *inside* `model_fn`: vLLM's `Attention.forward`
delegates to its `.impl` backend, which on TPU is `PallasAttentionBackendImpl.forward`
(selected via the platform's `get_attn_backend_cls`, **not** a `register_oot` patch). That
backend converts torch→JAX and calls the **ragged paged attention v3 Pallas kernel**.

```
┌─────────────────────── HOST (NumPy, vLLM scheduler) ───────────────────────┐
│ TPUModelRunner.execute_model(scheduler_output)                              │
│   └─ _execute_model                                                         │
│        ├─ persistent_batch_manager.update_states()   (add/remove requests) │
│        ├─ _prepare_inputs(scheduler_output)                                 │
│        │    ├─ gather token ids/positions from CPU staging buffers          │
│        │    ├─ build query_start_loc / seq_lens / block_tables (NumPy)      │
│        │    ├─ pad to a compiled bucket   (num_tokens_paddings)             │
│        │    └─ device_array(...)  → host→device transfer + sharding         │
│        │         → returns AttentionMetadata (a JAX pytree)                 │
│        │                                                                    │
└────────┼────────────────────────────────────────────────────────────────────┘
         │  (input_ids, input_positions, attn_metadata, sampling_metadata, …)
┌────────▼────────────────────── TPU (JAX / XLA) ────────────────────────────┐
│   self.model_fn(state, kv_caches, input_ids, attn_metadata, …)             │
│     = jit_step_func()  (torchax-wrapped vLLM nn.Module, jitted)            │
│        └─ … decoder layers …                                               │
│             └─ vLLM Attention.forward  ──.impl backend──►                  │
│                  PallasAttentionBackendImpl.forward                        │
│                    └─ _jax_attn_func (jax.jit)                             │
│                         └─ attention()  (attention_interface.py)          │
│                              └─ sharded_ragged_paged_attention (shard_map) │
│                                   └─ ragged_paged_attention  (v3 kernel)   │
│                                        └─ pl.pallas_call(...)  ◄ KERNEL    │
│             returns (kv_caches[updated], hidden_states)                    │
│                                                                            │
│   hidden_states = _select_from_array_fn(hidden, logits_indices)            │
│   logits        = compute_logits_fn(state, hidden, lora)                   │
└────────┬────────────────────────────────────────────────────────────────────┘
         │  (logits cached in ExecuteModelState; execute_model returns None)
┌────────▼────────────────────────── later call ────────────────────────────┐
│   sample_tokens(grammar_output) → _sample_from_logits                      │
│     └─ sample(rng, mesh, logits, sampling_metadata)  (jax.jit)             │
│          argmax / temp+top-k+top-p → categorical → tokens                  │
│     └─ jax.device_get → write into input_batch.token_ids_cpu              │
│     └─ return ModelRunnerOutput(sampled_token_ids, logprobs, …)           │
└────────────────────────────────────────────────────────────────────────────┘
```

A subtle but important structural fact: **`execute_model` does not sample.** It runs the
model + logits, stashes everything in `self.execute_model_state`, and returns `None`
(`tpu_runner.py:908-920`). Sampling happens in a *separate* `sample_tokens` call
(`tpu_runner.py:673`). This split exists so structured-decoding / grammar bitmasks can be
injected between logits and sampling, and to overlap host work.

---

## 1. `execute_model` and input preparation

### 1.1 Entry point and the runner's state

`TPUModelRunner` is defined at `tpu_inference/runner/tpu_runner.py:240`. The fields that
matter for the forward pass:

- `self.kv_caches: list[jax.Array]` — one sharded array per attention layer (`:301`).
- `self.layer_name_to_kvcache_index: dict[str,int]` — maps a vLLM layer name to its index
  in `self.kv_caches` (`:302`). Passed into `model_fn` so each attention layer can find
  its own cache.
- `self.mesh` — the JAX device mesh (`_init_mesh`, `:272`).
- `self.model_fn`, `self.compute_logits_fn`, `self.state` — populated by `load_model`
  (`:556-562`). **`model_fn` is the torchax-jitted vLLM model** (see §2).
- `self.input_batch` (`InputBatch`) — the persistent host-side batch state.
- `self.execute_model_state: ExecuteModelState | None` (`:298`) — the ephemeral bridge
  between `execute_model` and `sample_tokens`.
- Managers wired in `__init__`: `CompilationManager` (`:279`), `KVCacheManager` (`:284`),
  `PersistentBatchManager` (`:286`).

`execute_model` (`tpu_runner.py:657`) is thin: it sets the mesh and a profiler annotation
and delegates to `_execute_model` (`:669-670`):

```python
with jax.set_mesh(self.mesh), jax.profiler.TraceAnnotation(...):
    output = self._execute_model(scheduler_output, intermediate_tensors)
```

### 1.2 `_execute_model`: scheduler output → model inputs → logits

`_execute_model` (`tpu_runner.py:794`) does, in order:

1. **Update batch state** (`:799`): `persistent_batch_manager.update_states(...)` applies
   the scheduler's adds/removes/finishes to `self.input_batch` and the per-request
   `block_table`. This is where new requests' prompt tokens land in the CPU staging arrays
   and finished requests are evicted. Early-out with `EMPTY_MODEL_RUNNER_OUTPUT` if nothing
   is scheduled (`:801-816`).

2. **Prepare inputs** (`:819-828`): one call to `_prepare_inputs` returns the full tuple
   `(input_ids, input_positions, attn_metadata, sampling_metadata, logits_indices,
   spec_decode_metadata, logits_indices_selector, padded_num_reqs)`.

3. **Multimodal / embeds** (`:831-846`): text-only models leave `inputs_embeds=None` and
   pass `input_ids` straight through (`_get_input_ids_embeds`).

4. **Run the model** (`:854-874`) — the only line that touches TPU compute for the
   backbone (covered in §2).

5. **Select logit rows + compute logits** (`:900-906`):
   ```python
   hidden_states = self._select_from_array_fn(hidden_states, logits_indices)
   logits = self.compute_logits_fn(self.state, hidden_states, lora_metadata)
   ```
   `logits_indices` picks the *last token of each sequence* (and bonus positions for spec
   decode) out of the packed `hidden_states`, so only those rows get an LM-head matmul.
   `_select_from_array_fn` (`:1138`) is a `shard_map` that indexes within each
   data-parallel shard.

6. **Stash & return `None`** (`:908-920`): builds `ExecuteModelState` and returns `None`.

### 1.3 `_prepare_inputs`: where host→device happens

`_prepare_inputs` (`tpu_runner.py:1312`) dispatches to `_prepare_inputs_dp` (data-parallel,
`:1318`) or `_prepare_inputs_non_dp` (`:1662`). Focus on the non-DP path; DP is the same
idea replicated per rank.

The key moves (`_prepare_inputs_non_dp`, roughly `:1672-1903`):

- **Pad the token count** to a compiled bucket via `runner_utils.get_padded_token_len(...)`
  — this is the single most important step for avoiding recompilation (§3).
- **Allocate views** into a reused host buffer (`self.device_buffer`) for `input_ids`,
  `logits_indices`, `query_start_loc`, `seq_lens` (`:1689-1715`). These are plain NumPy
  views; nothing is on device yet.
- **Gather tokens** flatly from `input_batch.token_ids_cpu` using `np.repeat` /
  `np.take` (`:1730-1776`). The trick: `req_indices = np.repeat(arange[:num_reqs],
  num_scheduled_tokens_per_req)` builds a per-token request id, and positions are
  `num_computed_tokens[req] + within-request offset`.
- **Build attention metadata fields** (NumPy):
  - `query_start_loc[1:] = np.cumsum(num_scheduled_tokens_per_req)` (`:1779+`) — cumulative
    query lengths, i.e. where each sequence's tokens start in the packed buffer.
  - `seq_lens = num_computed_tokens + num_scheduled` — the *total* KV length each sequence
    will have after this step.
  - Block tables copied from `input_batch.block_table[gid].get_cpu_tensor()` into a padded
    `(max_num_reqs, max_blocks_per_req)` view (`:1828-1846`).
- **Transfer to device** (`:1850-1859`): `self.device_buffer.build()` packs all the host
  views into one blob, then a single `device_array(self.mesh, (..., blob), sharding=...)`
  does the host→device copy with the data-parallel sharding
  `NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA))` (`:1668-1669`). Arrays are unpacked
  back out by name. Doing one big transfer instead of N small ones is a deliberate latency
  optimization.

The result is wrapped into the `AttentionMetadata` pytree (see §1.4) and returned.

### 1.4 `AttentionMetadata` — the pytree that flows into the kernel

Defined at `tpu_inference/layers/common/attention_metadata.py:35`, and registered as a JAX
dataclass pytree at `:22-33`:

```python
@functools.partial(jax.tree_util.register_dataclass,
    data_fields=["input_positions","block_tables","seq_lens",
                 "query_start_loc","request_distribution"],
    meta_fields=[],
    drop_fields=["query_start_loc_cpu","seq_lens_cpu"])
@dataclass
class AttentionMetadata(object):
    input_positions: jax.Array        # (padded_total_num_scheduled_tokens,)
    block_tables: jax.Array | None    # (max_num_seqs * max_num_blocks_per_req,)
    seq_lens: jax.Array               # (max_num_seqs,)
    query_start_loc: jax.Array        # (max_num_seqs + 1,)
    request_distribution: jax.Array   # (3,)
    query_start_loc_cpu: Any = field(init=False)   # host copy, NOT traced
    seq_lens_cpu: Any = field(init=False)          # host copy, NOT traced
```

Two design points worth internalizing:

- **It is a pytree.** Because it is `register_dataclass`'d, you can pass the whole struct
  through `jax.jit` / `shard_map` and XLA flattens it to its five arrays. New models add
  attention by consuming these five fields — you never thread them individually.
- **`drop_fields`** hides the `_cpu` mirrors from tracing. The host keeps NumPy copies of
  `query_start_loc`/`seq_lens` for control flow (e.g. discarding partial requests during
  sampling) without those buffers entering the XLA graph and forcing recompiles.

`request_distribution` is the `(3,)` array `(i, j, k)` that the kernel uses to partition
the batch into decode-only / chunked-prefill-only / mixed sequences (see §4).

---

## 2. Invoking the compiled forward (the torchax link, from the runner side)

`load_model` (`tpu_runner.py:548`) calls `get_model(...)` and copies out
`self.model_fn = model.model_fn` (`:556`). On the torchax route, `get_model` →
`get_vllm_model` (`models/common/model_loader.py:412`) sets:

```python
jit_model = model.jit_step_func()        # model_loader.py:430
...
return ModelInterface(model_fn=jit_model, ...)   # :452
```

So **`self.model_fn` is the torchax-wrapped vLLM `nn.Module` compiled with `jax.jit`**
(the wrapper internals — `VllmModelWrapper.jit_step_func` — are documented in the torchax
wrapper doc). From the runner's perspective it's just a jitted JAX callable invoked at
`tpu_runner.py:861-874`:

```python
(self.kv_caches, hidden_states, aux_hidden_states) = self.model_fn(
    self.state,                                       # model weights (pytree)
    self.kv_caches,                                   # list[jax.Array], one per layer
    input_ids,                                        # (padded_num_tokens,)
    attn_metadata,                                    # the AttentionMetadata pytree
    inputs_embeds,                                    # None for text-only
    input_positions,
    tuple(self.layer_name_to_kvcache_index.items()),  # hashable → static arg
    lora_metadata,
    intermediate_tensors,                             # pipeline-parallel input
    self.is_first_rank,
    self.is_last_rank,
)
```

Notes for model authors:

- **KV cache is threaded in and out.** `model_fn` receives `self.kv_caches` and returns the
  updated list — XLA donates and reuses the buffers. The attention layer doesn't mutate a
  global; it returns the new cache (see §3 / §5 handoff).
- **`layer_name_to_kvcache_index` is passed as a `tuple(...).items()`** so it's a *static*
  (hashable) argument; the jit specializes on the layer→cache mapping.
- The whole call is wrapped in `set_forward_context(None, self.vllm_config)` and the
  `maybe_get_kv_connector_output` context (`:854-858`) so vLLM-internal globals
  (e.g. the attention metadata stash the wrapper reads in the backend) are populated.
- `self.maybe_forbid_compile` (`:852`) is a guard that, in steady state, raises if XLA is
  about to recompile — i.e. it asserts you hit a precompiled bucket.

### How the compiled executable is cached (`compilation_manager.py`)

`CompilationManager` (`tpu_inference/runner/compilation_manager.py:47`) does two things:

1. **Enables JAX's persistent compile cache** (`:51-60`) by setting
   `jax_compilation_cache_dir = VLLM_XLA_CACHE_PATH`. XLA keys cache entries on the lowered
   HLO (function + static args + **array shapes/dtypes**). Dynamic array *values* don't
   affect the key; shapes do — hence bucketing.
2. **Warms up every shape bucket** in `capture_model` (`:92`) so the first real request
   doesn't pay compile latency. It loops over `self.runner.num_tokens_paddings`, builds
   dummy `AttentionMetadata` of each size, and calls `model_fn` / `compute_logits_fn` /
   `sample` to force compilation (`_run_compilation`, `:79`, does
   `jax.tree.map(lambda r: r.block_until_ready(), result)`).

**Bucketing** (the anti-recompilation mechanism), from `runner/utils.py`:

- `get_token_paddings(min, max, gap)` (`utils.py:67`): if `padding_gap==0` it's exponential
  doubling (`2×` each step); otherwise exponential up to `gap`, then linear `+gap`. This is
  `self.num_tokens_paddings` (`tpu_runner.py:467`).
- `get_req_paddings(min, max)` (`utils.py:55`) → `self.num_reqs_paddings`, used for sampling
  / logit selection.

So a batch of, say, 173 scheduled tokens is padded up to the next token bucket (e.g. 256);
`block_tables`/`seq_lens` are sized to `max_num_reqs` (constant). All of these are fixed
shapes → one cached executable serves every batch that pads to the same bucket, regardless
of which requests are in it.

---

## 3. Dense attention execution: backend → kernel handoff

### 3.1 The patched attention layer

vLLM's `Attention` class is **not** `register_oot`-patched (unlike the linear /
embedding layers). Instead `Attention.forward` delegates to its `.impl` backend, and on
TPU that backend is `PallasAttentionBackendImpl.forward` in
`tpu_inference/layers/vllm/backends/flash_attn.py:152`. The backend class
`PallasAttentionBackend` is registered for `AttentionBackendEnum.FLASH_ATTN` (`:32`) and is
selected by the platform's `get_attn_backend_cls` (`platforms/tpu_platform.py:115`) for the
dense (`use_mla=False`) path.

`forward` (`:152-218`) is the torch↔JAX boundary:

- The `kv_cache` *argument* from vLLM is required to be **empty** (`:169-172`); the real
  cache is fetched from the wrapper context by layer name:
  ```python
  vllm_model_wrapper_context = get_vllm_model_wrapper_context()
  kv_cache_index = ctx.layer_name_to_kvcache_index[layer.layer_name]   # :177
  kv_cache = ctx.kv_caches[kv_cache_index]                              # :179
  ```
  This is how the runner's `self.kv_caches` reaches the layer despite vLLM's API insisting
  on passing a per-layer tensor.
- **torch → JAX**: `query, key, value = jax_view(query), jax_view(key), jax_view(value)`
  (`:183`). `jax_view` unwraps the torchax tensor to its backing `jax.Array` (zero-copy).
- **Optional KV quantization** (`:185-192`): if `self.kv_cache_quantized_dtype` is set,
  `quantize_kv(...)` quantizes K/V and `k_scale`/`v_scale` are forwarded.
- Calls `_jax_attn_func(...)` (`:196-212`), writes the returned cache back into the context
  (`ctx.kv_caches[kv_cache_index] = new_kv_cache`, `:213`), and converts the output back to
  torch with `torch_view` (`:215`).

`_jax_attn_func` (`:221`, `@jax.jit`, `donate_argnames=("kv_cache")`) reshapes vLLM's flat
`(T, N*H)` q/k/v into `(T, N, H)` / `(T, K, H)` (`:257-259`) and calls `attention(...)`
(`:261`).

### 3.2 `attention` → `sharded_ragged_paged_attention`

`attention` (`tpu_inference/layers/common/attention_interface.py:406`) is a thin adapter:
it defaults `sm_scale = head_dim**-0.5` and unpacks the metadata, then calls
`sharded_ragged_paged_attention` (`:442-458`):

```python
output, kv_cache = sharded_ragged_paged_attention(
    mesh, q, k, v, kv_cache,
    md.seq_lens,             # → kv_lens
    md.block_tables,         # → page_indices
    md.query_start_loc,      # → cu_q_lens
    md.request_distribution, # → distribution (i, j, k)
    sinks, sm_scale=sm_scale, attention_chunk_size=..., q_scale=..., k_scale=..., v_scale=...)
```

This is the precise mapping from `AttentionMetadata` fields to kernel inputs — memorize it:

| `AttentionMetadata` field   | kernel argument | meaning                                      |
|-----------------------------|-----------------|----------------------------------------------|
| `seq_lens`                  | `kv_lens`       | total KV length per sequence                 |
| `block_tables`              | `page_indices`  | flattened (seq, page)→physical page table    |
| `query_start_loc`           | `cu_q_lens`     | cumulative query lengths (ragged boundaries) |
| `request_distribution`      | `distribution`  | `(i,j,k)` decode/prefill/mixed split         |

`sharded_ragged_paged_attention` (`:326`) does the **sharding** and picks the kernel
variant:

- It handles GQA/MQA when `num_kv_heads < tp_size` by replicating KV heads
  (`jnp.repeat(k, factor, axis=1)`, `:348-357`) so heads shard evenly across the
  `ATTN_HEAD` axis.
- It wraps the kernel in `jax.shard_map` with explicit `in_specs` (`:359-403`):
  q/k/v are `P(ATTN_DATA, ATTN_HEAD, None)`, the KV cache is
  `P(ATTN_DATA, None, ATTN_HEAD, None, None)`, and the metadata arrays are
  `P(ATTN_DATA)`. `check_vma=False` trusts these annotations.
- **Variant selection** (`:376-377`): `use_hd64 = q.shape[-1] == 64`; if so it calls
  `ragged_paged_attention_hd64`, else `ragged_paged_attention` (v3). Attention sinks are
  only supported on the hd64 path (`:380-385`). Llama/Qwen-style head_dim=128 takes the
  standard v3 kernel.

### 3.3 The ragged paged attention v3 kernel

Public entry: `tpu_inference/kernels/ragged_paged_attention/v3/kernel.py:1568`.

```python
def ragged_paged_attention(
    queries,      # [max_num_tokens, num_q_heads, head_dim]
    keys,         # [max_num_tokens, num_kv_heads, head_dim]
    values,       # [max_num_tokens, num_kv_heads, head_dim]
    kv_cache,     # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens,      # i32[max_num_seqs]
    page_indices, # i32[max_num_seqs * pages_per_seq]  (flattened block table)
    cu_q_lens,    # i32[max_num_seqs + 1]
    distribution, # i32[3]  (i, j, k)
    *, use_causal_mask=True, sm_scale=1.0, sliding_window=None, soft_cap=None,
       q_scale=None, k_scale=None, v_scale=None, ...): ...
```

**What it computes.** Standard scaled-dot-product attention with a causal mask, but for a
*ragged* batch: many sequences of different query/KV lengths are concatenated along the
token axis, and the KV history lives in a **paged** cache (non-contiguous physical pages).
It uses the FlashAttention-2 online-softmax formulation (running max `m`, running
normalizer `l`, accumulator `acc`) so it never materializes the full `S×S` score matrix.
It also **writes the new K/V into the paged cache in-kernel** and returns the updated cache.

**Ragged = mixed prefill + decode in one launch.** `distribution = (i, j, k)` partitions the
already-sorted sequences (kernel docstring `:1619-1621`):

- `sequences[0:i]` — **decode-only** (`q_len == 1`): no causal mask needed, the single query
  attends to all of its KV history.
- `sequences[i:j]` — **chunked-prefill-only** (a known static query length).
- `sequences[j:k]` — **mixed** (prefill chunk that also produces a decode step); `k` is the
  total sequence count.

The kernel actually runs the inner loop *three times* — once per case — each with the right
static `q_len` and mask settings. The case→sequence-range mapping is `RpaCase.get_range`
(`kernel.py:54-61`), driven by `distribution`. Splitting by case lets each group use a
specialized, well-shaped kernel body (e.g. decode skips the causal mask;
`:339-340`) instead of one branchy mega-kernel.

**Prefill vs decode, concretely:**
- *Prefill*: `q_len > 1`, causal mask applied across the chunk, new K/V appended to the
  sequence's pages.
- *Decode*: `q_len == 1`, the one query attends to the full cached history; only one new
  K/V token is appended.

**KV cache page/block layout.** `get_kv_cache_shape` (`kernel.py:261`):

```python
(total_num_pages,
 page_size,                               # tokens per page (== vLLM block_size)
 align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
 kv_packing,                              # 32 // dtype_bits  (bf16→2, int8→4 … but see note)
 align_to(actual_head_dim, 128))          # head_dim padded to 128-byte tiles
```

- **K and V are interleaved** into one `num_kv_heads * 2` axis (the `*2`), so one page holds
  both K and V for `page_size` tokens.
- **`kv_packing`** splits that axis into `(…//packing, packing)` to pack multiple low-bit
  elements into a machine word for storage/DMA efficiency.
- **`head_dim` is padded to 128** for TPU MXU/VMEM tiling. (The hd64 variant uses a
  differently-ordered shape; `kernel_hd64.py:get_kv_cache_shape`.)
- **`page_indices`** is the flattened block table: for sequence `s`, its `pages_per_seq`
  physical page ids live at `page_indices[s*pages_per_seq : (s+1)*pages_per_seq]`. The
  kernel uses this indirection to gather the right pages — this is the "paging."

**Execution structure (high level).** Inside the Pallas kernel
(`_ragged_paged_attention_kernel`, `:278`, looping per sequence via `_kernel_loop`, `:291`):
SMEM holds the metadata (`kv_lens`, `page_indices`, `cu_q_lens`, `distribution`,
semaphores); VMEM holds **double-buffered** blocks of Q (`bq`), gathered KV (`bkv`), and
output (`bo`), plus the FlashAttention scratch (`l`, `m`, `acc`). Async DMAs prefetch the
next Q/KV block (`_fetch_bq`, `_fetch_bkv`) while the current block computes, and write the
output (`_send_bo`) and updated cache pages (`_update_kv_cache`) back to HBM — all gated by
semaphores. The compute is two steps per KV-head: `flash_attention_step1_qk_softmax`
(Q·Kᵀ, scale, mask, soft-cap, online softmax) and `flash_attention_step2_pv` (P·V into the
accumulator), pipelined across heads to hide latency. `sm_scale`, `sliding_window`,
`soft_cap`, and q/k/v dequant scales are applied here.

For adding a new model you almost never touch the kernel; you only need head_dim ∈ {64,128}
(or pay the padding) and to route through `attention(...)`. The kernel is shared.

---

## 4. KV cache: allocation, paging, mapping to the kernel

### 4.1 Cache tensors (`runner/kv_cache.py`)

`create_kv_caches` (`kv_cache.py:86`) allocates **one sharded array per attention layer**.
Per-layer shape (docstring `:99-101`, and `get_kv_cache_shape` in the kernel) is:

```
(num_blocks, block_size, cdiv(num_kv_heads*2, packing), packing, head_dim)
```

i.e. `total_num_pages = num_blocks`, `page_size = block_size`. Sharding (`:127-130`):

```python
NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None, ShardingAxisName.ATTN_HEAD))
```

so pages shard on the data axis and KV heads on the head axis — matching the `kv_cache_spec`
the kernel's `shard_map` expects (`attention_interface.py:360-361`). Allocation is a jitted
`jnp.empty` with `out_shardings` (`:132-141`). (MLA uses a different shape and
`P(MLP_TENSOR)` sharding; out of scope here.)

`get_kv_cache_shape_with_mesh` (`:49`) is the dispatcher that delegates to the kernel's
`rpa.get_kv_cache_shape` / `rpa_hd64.get_kv_cache_shape` for the dense path, and pads/scales
the head axis for sharding.

### 4.2 Block management (`kv_cache_manager.py`, `continuous_block_pool.py`)

- `KVCacheManager` (`kv_cache_manager.py:56`) builds vLLM `KVCacheSpec`s
  (`_create_attention_spec`, `:74`; `FullAttentionSpec`/`SlidingWindowSpec` for dense) and
  `initialize_kv_cache` (`:548`) computes `num_blocks` from available HBM and the per-layer
  page-size-in-bytes, aligns it to the sharding divisor (`:654-655`), and calls
  `create_kv_caches` (`:718-727`). It also records `layer_name → cache index`
  (`:742-744`) — the same map that flows to the backend.
- **The vLLM scheduler owns logical→physical block allocation.** It hands the runner block
  ids per request, which `PersistentBatchManager.update_states` writes into the
  per-request `BlockTable`.
- `continuous_block_pool.py` (`ContinuousFreeQueue`, `:23`) keeps free blocks as sorted
  contiguous **intervals** (`:36`) and uses best-fit (`popleft_n`, `:117`) so that a
  request's blocks are *contiguous* when possible. Contiguity lets disaggregated KV
  transfer use a single `dynamic_update_slice_in_dim` instead of a scatter — a perf
  optimization, not a correctness requirement for the kernel.

### 4.3 Block table → kernel `page_indices`

`BlockTable` (`runner/block_table.py:13`) keeps a `(max_num_reqs, max_num_blocks_per_req)`
table in both NumPy (`block_table_cpu`) and device (`block_table`) form, plus
`num_blocks_per_row`. `MultiGroupBlockTable` (`:84`) wraps one `BlockTable` per KV-cache
group (e.g. full-attention vs sliding-window layers).

During `_prepare_inputs`, the CPU block table is copied into a padded device array
(`tpu_runner.py:1828-1846`) and stored as `attn_metadata.block_tables`. The kernel reads it
(after flattening to `page_indices`) to translate each sequence's logical block slots into
physical page ids in the cache tensor. So the chain is:

```
scheduler block ids → BlockTable.block_table_cpu → attn_metadata.block_tables (device)
                    → sharded_ragged_paged_attention(page_indices=...) → kernel page gather
```

---

## 5. Sampling and output

### 5.1 The deferred sampling split

`execute_model` returns `None`; the engine later calls `sample_tokens(grammar_output)`
(`tpu_runner.py:673`). It unpacks `self.execute_model_state` (`:683+`), optionally applies a
structured-decoding bitmask to `logits` (grammar-constrained decode), then calls
`_sample_from_logits` (`:922`).

### 5.2 `_sample_from_logits` → `sample`

`_sample_from_logits` (`:922-1136`) splits the RNG (`:940-947`) and, for the non-spec path
(`:949-954`), calls the jitted `sample`:

```python
logits = logits.astype(jnp.float32)
with self.maybe_forbid_compile:
    next_tokens, processed_logits = sample(step_rng, self.mesh, logits, tpu_sampling_metadata)
```

`sample` is in `tpu_inference/layers/jax/sample/sampling.py:67` (`@jax.jit`,
`static_argnames=["mesh"]`):

- Replicates `logits` across the mesh up front (`with_sharding_constraint`, `:83`) so the
  vocab dim isn't sharded during the reduction.
- `greedy_tokens = jnp.argmax(logits, -1)` (`:86`).
- If sampling is on, applies temperature/top-k/top-p transforms then
  `jax.random.categorical(rng, processed_logits)` (`:95`), and selects greedy where
  `temperature < _SAMPLING_EPS` (`1e-5`, `:26`/`:98`).
- Final tokens get `with_sharding_constraint(..., P())` (replicated) before return (`:106`).

`TPUSupportedSamplingMetadata` (`runner/sampling_metadata.py`) carries the per-request
`temperature`/`top_k`/`top_p` device arrays and the `do_sampling`/`logprobs` flags; it is
built in `_prepare_inputs` alongside the rest.

### 5.3 Back to the engine

In `_sample_from_logits` (`:1075+`): `next_tokens = np.asarray(jax.device_get(next_tokens))`
copies tokens to host, reorders by `logits_indices_selector` (DP), clears partial-request
rows, **writes accepted tokens into `input_batch.token_ids_cpu` and
`req_state.output_token_ids`** (so the next step's `_prepare_inputs` sees them), optionally
computes logprobs (`_compute_and_gather_logprobs`, `:1156`), and returns
`ModelRunnerOutput(req_ids, req_id_to_index, sampled_token_ids, logprobs, …)` (`:1127-1136`)
to the engine.

---

## Quick reference — most important anchors

| What | Anchor |
|---|---|
| Runner class / `self.kv_caches` / `layer_name_to_kvcache_index` | `runner/tpu_runner.py:240`, `:301`, `:302` |
| `execute_model` → `_execute_model` | `runner/tpu_runner.py:657`, `:794` |
| `model_fn` invocation (forward) | `runner/tpu_runner.py:861-874` |
| select logits + `compute_logits_fn` | `runner/tpu_runner.py:900-906` |
| defer-sampling stash / return `None` | `runner/tpu_runner.py:908-920` |
| `_prepare_inputs` (non-DP), device transfer | `runner/tpu_runner.py:1662`, `:1850-1859` |
| `model_fn = jit_step_func()` (torchax) | `models/common/model_loader.py:430`, `:452` |
| `AttentionMetadata` pytree | `layers/common/attention_metadata.py:22-49` |
| compile cache + bucketed warmup | `runner/compilation_manager.py:51-60`, `:92`; `runner/utils.py:55`,`:67` |
| patched attention forward (torch↔JAX) | `layers/vllm/backends/flash_attn.py:152-218` |
| `_jax_attn_func` (jit) | `layers/vllm/backends/flash_attn.py:221-282` |
| metadata→kernel mapping | `layers/common/attention_interface.py:442-458` |
| sharding + kernel variant select | `layers/common/attention_interface.py:326-403` |
| kernel public entry + distribution semantics | `kernels/ragged_paged_attention/v3/kernel.py:1568`, `:1619-1621` |
| KV cache shape | `kernels/ragged_paged_attention/v3/kernel.py:261`; `runner/kv_cache.py:99-101` |
| KV cache alloc + sharding | `runner/kv_cache.py:86-142` |
| block table / page_indices | `runner/block_table.py:13-122` |
| `sample` (argmax / categorical) | `layers/jax/sample/sampling.py:67-106` |
| `sample_tokens` / `_sample_from_logits` | `runner/tpu_runner.py:673`, `:922` |

## Open questions / unverified

- The `kv_packing = 32 // dtype_bits` formula in the `kv_cache.py` docstring implies bf16→2,
  but `get_dtype_packing` (kernel) is the source of truth; I did not read its body. The
  exact packing per dtype should be confirmed there if it matters for a new quant.
- `compute_logits_fn` / `_select_from_array_fn` are jitted separately from `model_fn`; I did
  not confirm whether the LM head runs in its own XLA executable or is fused — it's a
  separate jit call at `tpu_runner.py:902`, so almost certainly separate.
- Spec-decode and DP sampling branches in `_sample_from_logits` were only skimmed; the dense
  single-token path is fully traced.
