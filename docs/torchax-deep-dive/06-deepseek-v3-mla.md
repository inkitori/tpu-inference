# DeepSeek-V3 MLA on TPU — torchax route (and the JAX contrast)

> Slice of the *tpu-inference × vLLM torchax route* deep-dive. Covers **Multi-head
> Latent Attention (MLA)** plumbing: what MLA is, how the backend is selected, the
> torchax custom op, the Pallas kernels, the latent KV cache, the vLLM model class
> that runs on the torchax route, and a contrast with the pure-JAX rewrite.
>
> Audience: a strong engineer new to this codebase, working toward adding **DeepSeek
> v4** via torchax. All anchors are `path:line`. vLLM paths live under
> `/home/enyouki/vllm/vllm/`; everything else under `/home/enyouki/tpu-inference/`.
>
> ⚠️ **v4 reality check (read §8 first):** this vLLM checkout *already ships*
> `DeepseekV4ForCausalLM` (`vllm/model_executor/models/deepseek_v4.py`,
> registered `vllm/.../models/registry.py:99`), and its MLA **does not** match the
> `(q_nope, q_pe, kv_c_normed, k_pe)` contract everything below assumes. v4 is a
> different attention algorithm (fused QKV-a, output-LoRA, FlashMLA-*sparse* + SWA,
> hyper-connections, MegaMoE) and is CUDA/SM100-only. The "v4 for free via torchax"
> story is **false** — see §8.

---

## 0. TL;DR / orientation

- **MLA** replaces the standard per-head K/V cache with a single **low-rank latent**
  per token (`kv_lora_rank` wide) plus a small **decoupled-RoPE** key (`qk_rope_head_dim`
  wide). Queries and keys are split into a *nope* part (no RoPE, lives in latent space)
  and a *rope* part. The big saving: the KV cache stores
  `kv_lora_rank + qk_rope_head_dim` floats per token (e.g. 512+64=576) instead of
  `num_kv_heads × head_dim` per token.
- **Two routes exist for DeepSeek-V3 in this repo, and they are not symmetric:**
  - **flax_nnx (pure JAX):** `tpu_inference/models/jax/deepseek_v3.py` — a *native, first-class,
    registered* DeepSeek-V3. This is the **default served path** (see §7). **Mature.**
  - **torchax/vLLM:** runs **vLLM's own** PyTorch `DeepseekV3ForCausalLM`
    (`deepseek_v2.py`) under torchax. There is **no tpu-inference-authored DeepSeek
    model class** on this side — it is reached only as a fallback
    (`MODEL_IMPL_TYPE=vllm`, or pipeline-parallel > 1). The MLA *attention op* on this
    route is `tpu_inference/layers/vllm/custom_ops/mla_attention.py`.
- **Both routes converge on the same Pallas kernel**: `kernels/mla/v2/kernel.py`
  (the v1 kernel survives only for cache-shape helpers + tests).

The single most important structural fact for v4: **weight absorption** (folding the
KV up-projection `W_UK`/`W_UV` into the query path and the output path) happens
*outside* the kernel. On the torchax route it is done explicitly in
`mla_attention.py`. The kernel only ever sees latent-space tensors.

---

## 1. What MLA is, tied to code

DeepSeek MLA projects the hidden state down to compressed latents, then up to
per-head Q/K/V — with RoPE applied only to a small *decoupled* slice.

The canonical dims (DeepSeek-V3): `q_lora_rank=1536`, `kv_lora_rank=512`,
`qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`,
`qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192`. (JAX module constants:
`models/jax/deepseek_v3.py:103-107`. vLLM reads them from HF config in
`deepseek_v2.py:872-878`.)

| Concept | What it is | Where in code (vLLM/torchax route) |
|---|---|---|
| **Q LoRA down/up** | hidden → `q_lora_rank` → `num_heads·qk_head_dim` | down = first slice of `fused_qkv_a_proj` (`deepseek_v2.py:893-898`); up = `q_b_proj` (`deepseek_v2.py:910-916`), with `q_a_layernorm` between (`:909`) |
| **KV LoRA down** | hidden → `kv_lora_rank + qk_rope_head_dim` (the compressed latent + the rope key) | second slice of `fused_qkv_a_proj` (`deepseek_v2.py:893-898`); standalone `kv_a_proj_with_mqa` only when `q_lora_rank is None` (`:900-906`) |
| **KV LoRA up** | `kv_lora_rank` → `num_heads·(qk_nope_head_dim + v_head_dim)` | `kv_b_proj` (`deepseek_v2.py:926-932`). **Not applied in forward** — absorbed (see §3) |
| **Decoupled RoPE** | RoPE applied *only* to `q_pe` (the `qk_rope_head_dim` slice of Q) and the shared `k_pe` | setup `deepseek_v2.py:948-953` (`get_rope`, `is_neox_style=False`, deepseek-yarn); applied `mla.py:153-160` |
| **nope vs rope split** | Q split `[qk_nope_head_dim, qk_rope_head_dim]`; KV latent split `[kv_lora_rank, qk_rope_head_dim]` | `mla.py:133-156` |
| **Compressed-latent KV cache** | cache stores `kv_c_normed` (`kv_lora_rank`) + `k_pe` (`qk_rope_head_dim`) — **one tensor**, MQA-shared across heads | layout §5; written by kernel `kernels/mla/v1/kernel.py:91-94` |

### MLA forward (the torchax wrapper)

`VllmMultiHeadLatentAttentionWrapper.forward` —
`tpu_inference/layers/vllm/custom_ops/mla_attention.py:217-279`:

```
hidden_states
   │  fused_qkv_a_proj            (mla_attention.py:234)
   ├──> q_c            ──(q_a_layernorm,q_b_proj)──> q [T, N·qk_head_dim]   (:239-240)
   └──> kv_lora        ──split──> kv_c, k_pe                                (:249)
                                  kv_c ─(kv_a_layernorm)─> kv_c_normed      (:251)
   q.view(-1,N,qk_head_dim).split([nope, rope]) ──> q_nope, q_pe           (:253-255)
   k_pe.unsqueeze(1)                                                        (:258)
   q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)   # decoupled RoPE        (:261)
   attn_out = self.mla_attn((q_nope, q_pe), kv_c_normed, k_pe, ...)         (:271-277)
   return o_proj(attn_out)                                                  (:279)
```

```
            ┌──────────────────────── hidden_states [T, H] ───────────────────────┐
            │                                                                      │
       fused_qkv_a_proj                                                            │
            │                                                                      │
   ┌────────┴──────────┐                                                           │
 q_c [T,1536]      kv_lora [T, 512+64]                                             │
   │                   │                                                           │
 q_b_proj         split → kv_c[T,512]  k_pe[T,64]                                  │
   │                   │                                                           │
 q [T,N,192]     kv_a_layernorm → kv_c_normed[T,512]                               │
   │                   │                                                           │
 split        ┌────────┘                                                           │
   │          │            ── decoupled RoPE on (q_pe, k_pe) only ──               │
 q_nope[T,N,128]  q_pe[T,N,64]   k_pe[T,1,64]                                      │
   │              │              │                                                 │
   └────► VllmMLAAttention.impl.forward  (mla_attention.py:111) ◄──── kv_cache ────┘
                  │  W_UK absorption: q_nope[T,N,128]·W_UK_T → q_TNA[T,N,512]
                  │  mla_attention() shard_map → v2 Pallas kernel
                  │  W_UV absorption: out[T,N,512]·W_UV → out[T,N,128]
                  ▼
            attn_out [T, N·128]  ── o_proj ──►  [T, H]
```

---

## 2. Backend selection — `use_mla` forces `FLASH_ATTN_MLA`

`TpuPlatform.get_attn_backend_cls` —
`tpu_inference/platforms/tpu_platform.py:114-128`:

```python
use_mla = attn_selector_config.use_mla
if use_mla:
    selected_backend = AttentionBackendEnum.FLASH_ATTN_MLA
...
return selected_backend.get_path()
```

`get_path()` resolves to the registered class
`tpu_inference/layers/vllm/backends/flash_attn_mla.py::PallasMLAttentionBackend`
(registered via `@register_backend(AttentionBackendEnum.FLASH_ATTN_MLA)`,
`flash_attn_mla.py:33`).

**`use_mla` itself** is set by vLLM:
`use_mla = self.is_deepseek_mla and not envs.VLLM_MLA_DISABLE`
(`vllm/config/model.py:1595`). So any DeepSeek-MLA arch on TPU lands on
`FLASH_ATTN_MLA` unless `VLLM_MLA_DISABLE=1` (§5).

**Hard requirements for MLA on TPU** —
`TpuPlatform.check_and_update_config` (`tpu_platform.py:200-207`): MLA requires
`NEW_MODEL_DESIGN=1` **and** DP-attention enabled, else it raises:

```python
if vllm_config.model_config and vllm_config.model_config.use_mla:
    if not envs.NEW_MODEL_DESIGN or not vllm_config.additional_config.get(
            "sharding", {}).get("sharding_strategy", {}).get(
                "enable_dp_attention", False):
        raise ValueError("MLA models require both the NEW_MODEL_DESIGN=1 ... and DP attention ...")
```

Here `envs` is **`tpu_inference.envs`** (not vLLM's), and "DP-attention enabled" means
`additional_config.sharding.sharding_strategy.enable_dp_attention == True`
(passed via `--additional_config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}'`).

**Default block size** for MLA also comes from this backend:
`cache_config.block_size = PallasMLAttentionBackend.get_page_size(vllm_config)`
(`tpu_platform.py:226-230`), and `get_page_size` returns **1024**
(`flash_attn_mla.py:44-46`) — much larger than the dense path's page size, because
each MLA cache slot is one small latent row.

### Backend → impl

`PallasMLAttentionBackend.get_impl_cls()` → `PallasMLAttentionBackendImpl`
(`flash_attn_mla.py:40-42, 49`), an `MLAAttentionImpl` subclass. Its `__init__`
captures the MLA dims (`q_lora_rank, kv_lora_rank, qk_nope_head_dim,
qk_rope_head_dim, qk_head_dim, v_head_dim`, `flash_attn_mla.py:77-82`). The base
class's `forward_mha`/`forward_mqa` are **stubbed to `pass`/no-op** (not raising; `flash_attn_mla.py:84-109`)
— this impl uses a single bespoke `forward` (§3) instead of vLLM's split
prefill/decode MLA interface.

```
get_attn_backend_cls (tpu_platform.py:114)
   └─ FLASH_ATTN_MLA → PallasMLAttentionBackend            (flash_attn_mla.py:33)
        └─ get_impl_cls → PallasMLAttentionBackendImpl      (flash_attn_mla.py:49)
             └─ .forward  (the MLA custom op)               (flash_attn_mla.py:111)
                  └─ mla_attention() shared interface       (attention_interface.py:463)
                       └─ shard_map → mla_ragged_paged_attention (v2)  (kernel.py:1398)
```

---

## 3. The MLA custom op + **weight absorption**

Two files cooperate:

1. **`custom_ops/mla_attention.py`** — the **layer** (`VllmMLAAttention` +
   `VllmMultiHeadLatentAttentionWrapper`), monkeypatched into vLLM via
   `MultiHeadLatentAttentionWrapper.register_oot` (`mla_attention.py:153`) and pulled
   in by `register_layers` (`layers/vllm/custom_ops/__init__.py:20`). It owns the
   *projection graph* (§1 forward) and **the weight-absorption setup**.
2. **`backends/flash_attn_mla.py`** — the **impl** (`PallasMLAttentionBackendImpl`),
   which owns the *math*: it applies the absorbed matrices around the kernel call.

### Weight absorption (the crux)

DeepSeek's `kv_b_proj` would normally up-project the cached latent `kv_c` into full
per-head K (`qk_nope_head_dim`) and V (`v_head_dim`). MLA's trick is to **never
materialize full K/V**: instead fold `W_UK` into the query and `W_UV` into the
output, and run attention directly in the `kv_lora_rank` latent space.

- vLLM's base `MLAAttention.process_weights_after_loading`
  (`vllm/.../attention/mla_attention.py:816-903`) splits `kv_b_proj` into
  `W_UK, W_UV` (`:840`) and stores `W_UK_T = W_UK.permute(1,2,0)` (`:903`),
  `W_UV = W_UV.transpose(0,1)` (`:901`).
- The TPU subclass `VllmMLAAttention.process_weights_after_loading`
  (`mla_attention.py:74-109`) calls `super()`, then **re-quantizes** `W_UK_T`/`W_UV`
  to the KV-cache dtype, shards them on `ATTN_HEAD`
  (`P(ShardingAxisName.ATTN_HEAD)`, `mla_attention.py:84-97`), and **deletes
  `kv_b_proj`'s params** (`:107-109`) — they are no longer needed at runtime.

### The impl forward — absorbed math around the kernel

`PallasMLAttentionBackendImpl.forward` —
`tpu_inference/layers/vllm/backends/flash_attn_mla.py:111-208`:

```python
q_nope, q_pe = q                                   # (:142)
# Absorb W_UK into the query: project q_nope into latent space
q_nope = (einsum("bnp,npl->bnl", q_nope, W_UK_T)   # [T,N,128]·[N,128,512] → [T,N,512]
          * W_UK_T_scale).astype(input_dtype)       # (:151-155)
...
k_pe = k_pe.squeeze(1)                              # (:177)
new_kv_cache, outputs = mla_attention(             # (:178) → shard_map → v2 kernel
    q_nope, q_pe, kv_c_normed, k_pe, kv_cache, ...,
    self.num_heads, self.qk_nope_head_dim, sm_scale=self.scale)
# Absorb W_UV into the output: project latent attention out back to v_head_dim
outputs = outputs.reshape(-1, num_heads, kv_lora_rank)        # (:197)
outputs = (einsum("bnl,nlv->bnv", outputs, W_UV) * W_UV_scale)# [T,N,512]·[N,512,128] (:198-202)
outputs = outputs.reshape(-1, num_heads * v_head_dim)         # (:203)
```

So the kernel receives `q_TNA` (queries already in the `kv_lora_rank=512` latent),
`q_pe` (the rope query), `kv_c_normed` (latent K, == latent V), and `k_pe`
(rope key). It returns the latent attention output, which the impl then lifts to
`v_head_dim` via `W_UV`. **Weight absorption is in the custom op, not in the kernel.**

torchax glue: `jax_view()`/`torch_view()` (`flash_attn_mla.py:143-146, 205`) move
between torch.Tensor (the vLLM module API) and the underlying `jax.Array` the kernel
needs. KV-cache quantization (`quantize_kv`, `static_per_tensor_quantize_tensor`) is
applied to q/kv when `layer.kv_cache_quantized_dtype` is set (`flash_attn_mla.py:158-175`).

### The shared interface → kernel

`mla_attention()` —
`tpu_inference/layers/common/attention_interface.py:463-553` — wraps the v2 kernel in
a `jax.shard_map` (`:544-549`). Sharding specs: q/k on `MLP_TENSOR`, the KV cache on
`MLP_TENSOR`, and metadata (`seq_lens`, `block_tables`, `query_start_loc`,
`request_distribution`) on `ATTN_DATA` (`:502-519`). Block sizes are **hardcoded**
(no autotuner yet, `:522`): `num_kv_pages_per_block=(3,1,1)`,
`num_queries_per_block=(1,16,16)`, `decode_batch_size=4` (`:523-525`). The same
`mla_attention()` is the convergence point for **both** the torchax route and the
JAX route (the JAX model calls it directly from
`models/jax/deepseek_v3.py:696-711`).

---

## 4. The MLA Pallas kernels (v1 vs v2)

Both files define `mla_ragged_paged_attention` and a `get_kv_cache_shape` helper.

### Shared model
The combined latent cache is
`cache_kv: [total_num_pages, align(page_size,kv_packing)//kv_packing, kv_packing,
align(lkv_dim,128) + align(r_dim,128)]` where `lkv_dim=kv_lora_rank`, `r_dim=qk_rope_head_dim`.
The last axis concatenates `kv_c` (latent) and `k_pe` (rope key); the kernel's
*write/validate* side (`update_kv_cache`, `v1/kernel.py:59-71`) aligns **each component
to 128 separately** then concatenates. Query is `ql_nope[T,N,lkv_dim]` + `q_pe[T,N,r_dim]`.
Attention is **MQA over the latent**: one shared latent K/V row read by all query
heads. There is no separate V — `v_i = kv_c` (the value *is* the latent; v1
`kernel.py:217`), so the kernel output width is `lkv_dim` and the up-projection to
`v_head_dim` is external (the `W_UV` absorption in §3).

`get_kv_cache_shape(total_num_pages, page_size, kv_dim, kv_dtype)` is byte-identical in
both (verified): v1 `kernels/mla/v1/kernel.py:32-44`, v2 `kernels/mla/v2/kernel.py:59-71`.

> ⚠️ **Alignment subtlety (latent bug surface for v4).** `get_kv_cache_shape` aligns the
> **raw sum once** — its last dim is `align_to(kv_dim, 128)`, where the caller passes the
> *unaligned* `head_size = kv_lora_rank + qk_rope_head_dim` (= 576 for v3;
> `kv_cache_manager.py:714-715`). But the kernel's *write* side aligns **per-component**:
> `r_dim=align(64,128)=128`, `lkv_dim=align(512,128)=512`, `kv_dim=640`, and then
> hard-asserts `kv_dim == cache_kv_dim` (`v1/kernel.py:71`). For v3 these reconcile **only
> by coincidence** — `align(512+64,128)=640 == align(512,128)+align(64,128)=640`. A v4
> `(kv_lora_rank, qk_rope_head_dim)` that breaks this equality (e.g. either component not
> already 128-aligned in a way that survives the sum) would allocate a mis-sized cache and
> trip the `assert` at runtime. The spec-level `mla_head_size` (`kv_cache_manager.py:362-368`)
> *does* align per-component (=640), so it agrees with the alloc only by the same coincidence.

### v1 — `kernels/mla/v1/kernel.py`
- **Entry** `mla_ragged_paged_attention` (`v1/kernel.py:1092-1124`). Inputs:
  `ql_nope[max_tokens, N, lkv_dim]`, `q_pe[..., r_dim]`, `new_kv_c[max_tokens, lkv_dim]`,
  `new_k_pe[..., r_dim]`, `cache_kv`, `kv_lens`, `page_indices`, `cu_q_lens`,
  `distribution[3]`; kw-only `sm_scale, ..., num_kv_pages_per_block` (single int),
  `num_queries_per_block` (single int), `chunk_prefill_size`, scales.
- **Outputs** `(output[max_tokens, N, lkv_dim], updated_cache_kv)` (`:1120-1124`).
- **KV update is a separate `jax.jit` scatter** (`update_kv_cache`, `v1/kernel.py:47-101`,
  called at `:1189`) — **not fused** into the attention kernel.
- **Grid** `(distribution[2],)` — **one program per sequence**, processed sequentially
  (`:1224`, `dimension_semantics=("arbitrary",)`). Regimes
  (decode / prefill / mixed) dispatched by `@pl.when` on `distribution` thresholds
  *inside* the one kernel (`:994-1004`). FlashAttention-2 is one fused QK→softmax→PV
  function (`:589-664`).

### v2 — `kernels/mla/v2/kernel.py`
- **Entry** `mla_ragged_paged_attention` (`v2/kernel.py:1398-1433`), same positional
  args. Differences: `num_kv_pages_per_block`/`num_queries_per_block` are now **tuples
  `(decode, prefill, mixed)`** (`:1421-1422`); adds `decode_batch_size:int=1` (`:1424`),
  `s_dtype` (`:1425`), `p_same_dtype_as_v` (`:1426`).
- **Three kernel launches**, one per regime: `run_mla_kernel` is called for
  `BATCHED_DECODE`, `DECODE` (remainder), and `MIXED` (`v2/kernel.py:1716, 1737, 1759`),
  each with its own block sizes. Chunked-prefill-only is dropped (TODO `:1756`).
- **Batched decode** is the headline feature: `decode_batch_size` packs several decode
  sequences into one grid step, with cross-sequence software pipelining — FlashAttention
  is split into `flash_attention_step1_qk_softmax` (`:335-399`) and
  `flash_attention_step2_pv` (`:401-422`) so PV of item `b-1` overlaps QK of item `b`
  (`:1205-1256`).
- **Fused KV-cache update**: v1's separate jit scatter is gone — v2 packs and writes
  the cache *inside* the kernel (`_pack_new_kv` `:646-815`, `_update_kv_cache`
  `:817-1060`).

Weight absorption is **external in both** — unchanged.

### Which version is used, and when
- **v2 is the production compute path.** Imported only at
  `layers/common/attention_interface.py:33` and called at `:527`. Both routes reach it.
- **v1 is used only for the cache shape (not its compute, and not its `update_kv_cache`).**
  Imported `as mla` at `runner/kv_cache.py:23`, used **solely** for `mla.get_kv_cache_shape`
  (`kv_cache.py:68`). v1's `update_kv_cache` is *not* called in production (v2 fuses the
  update); v1's *attention* kernel is exercised only by tests
  (`tests/kernels/mla_v1_test.py`, `tests/models/jax/test_deepseek_v3.py:30`).
  **Caveat (real coupling):** production *allocates* with v1's `get_kv_cache_shape` but
  *computes* with v2. The two `get_kv_cache_shape` bodies are byte-identical **today**, but
  it is unguarded: each module defines its *own* `align_to`/`get_dtype_packing` (v1 imports
  from `ragged_paged_attention.v3.util`, `v1/kernel.py:24-25`; v2 defines them locally,
  `v2/kernel.py:41-56`). There is no shared source of truth and no cross-module assertion —
  a divergent edit to v2's copy would silently desync alloc-shape from compute-shape.

### Contrast with dense `ragged_paged_attention`
Dense kernel: `kernels/ragged_paged_attention/` (production = **v3**;
shape `ragged_paged_attention/v3/kernel.py:261-275`).

| | Dense RPA | MLA RPA |
|---|---|---|
| Cache contents | full **K and V** per KV head, packed: `[pages, page, align(num_kv_heads·2,packing)//packing, packing, align(head_dim,128)]` | **one latent** `kv_c` + small `k_pe`, concatenated: `[pages, align(page,packing)//packing, packing, align(kv_lora_rank,128)+align(qk_rope,128)]` |
| Heads | GQA/MQA, full `head_dim` per KV head | MQA over a **single shared latent**; `num_kv_heads` collapses to 1 |
| Head dim | one unified `head_dim` | split nope (`→lkv_dim`) + rope (`→r_dim`); `v_head_dim` handled outside |
| Value | separate V (odd head indices) | **V == the `kv_c` latent** (no separate V) |
| Paging | `page_indices` block table, `cdiv(kv_len, page_size)` | **identical paging mechanics**, different per-page payload |

The mechanics of paging are the same; MLA just shrinks the per-token payload from
`num_kv_heads × head_dim` (K *and* V) to one `kv_lora_rank + qk_rope_head_dim` latent.

---

## 5. KV cache for MLA

The fork on `use_mla` is centralized in `get_kv_cache_shape_with_mesh`
(`runner/kv_cache.py:65-83`):

```python
if use_mla:
    get_kv_cache_shape_fn = mla.get_kv_cache_shape          # v1 helper
    shape = list(get_kv_cache_shape_fn(total_num_pages, block_size,
                                       actual_head_dim, kv_dtype))
else:
    get_kv_cache_shape_fn = rpa_hd64.get_kv_cache_shape if head_dim==64 else rpa.get_kv_cache_shape
    shape = list(get_kv_cache_shape_fn(..., actual_num_kv_heads//model_cnt, actual_head_dim, ...))
    shape[2] *= model_cnt
```

- **One combined latent tensor per layer.** MLA layers get a single `MLAAttentionSpec`
  with `num_kv_heads=1` (`runner/kv_cache_manager.py:80, 87-93, 378-379`). Allocation
  uses `head_size = kv_lora_rank + qk_rope_head_dim`
  (`kv_cache_manager.py:710-728`):

  ```python
  if self.use_mla:
      head_size = text_config.kv_lora_rank + text_config.qk_rope_head_dim
  ...
  kv_cache = create_kv_caches(..., num_kv_heads=layer_spec.num_kv_heads,  # == 1
                              head_size=head_size, use_mla=self.use_mla)[0]
  ```

  This `head_size` is the **raw, unaligned** sum (512+64=576); `get_kv_cache_shape` then
  pads the *last dim* to `align_to(576,128)=640`. The kernel writes latent to
  `[..., :lkv_dim]` and rope to `[..., lkv_dim:]` (`kernels/mla/v1/kernel.py:91-94`), where
  `lkv_dim`/`r_dim` are each 128-aligned at write time — these only line up with the
  alloc-time 640 by coincidence for v3's dims (see §4 alignment subtlety). Contrast the
  dense path: one tensor with a `num_kv_heads·2` head axis (packed K+V).

- **Sharding differs.** MLA cache shards on `MLP_TENSOR`
  (`kv_cache.py:123-130`; `num_blocks` floored to the `MLP_TENSOR` mesh product without
  DP-attention, `kv_cache_manager.py:643-655`); dense shards on
  `(ATTN_DATA, None, ATTN_HEAD)`.

- **`VLLM_MLA_DISABLE`.** Defined and consumed **only in vLLM**
  (`vllm/envs.py:138, 1127`); tpu-inference merely *passes it through* to workers
  (it is listed in `additional_env_vars`, `tpu_platform.py:103`). The gate is
  `vllm/config/model.py:1595`: `use_mla = is_deepseek_mla and not VLLM_MLA_DISABLE`.
  Setting `VLLM_MLA_DISABLE=1` forces `use_mla=False`, flipping **everything above**
  back to the standard per-head K/V path (and `get_head_size` returns the full per-head
  dim instead of the latent, `vllm/config/model.py:54`). Useful as an escape hatch /
  numerical baseline.

### Latent KV layout diagram

```
Standard paged KV (dense), per layer:
  kv_pages[ pages, page_size, ceil(num_kv_heads*2 / packing), packing, align(head_dim,128) ]
            └ K at even head idx, V at odd head idx ┘   (full-dim, per head)

MLA latent KV, per layer (num_kv_heads → 1):
  cache_kv[ pages, ceil(page_size / packing), packing,  align(kv_lora_rank,128) + align(qk_rope,128) ]
                                                         └──── kv_c ────┘ └──── k_pe ────┘
  one latent row per token; kv_c is BOTH the latent K and the latent V.
```

---

## 6. Which vLLM model class runs DeepSeek-V3 (torchax route)

**There is no `deepseek_v3.py` in vLLM.** V2 and V3 share one file:
`vllm/model_executor/models/deepseek_v2.py`.

- `DeepseekV3ForCausalLM` is an **empty subclass**:
  `class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM): pass` (`deepseek_v2.py:1672-1673`).
  Real parent `DeepseekV2ForCausalLM` at `:1322-1665` (composes `DeepseekV2Model`,
  `lm_head`, logits processor).
- **MLA attention module** = `DeepseekV2MLAAttention` (`deepseek_v2.py:843-1030`),
  selected in the decoder layer when `model_config.use_mla` (`:1069-1070`). It does
  **not** call the generic `Attention` layer — it builds an `MLAModules` dataclass
  (`:987-1005`) and wraps it in `MultiHeadLatentAttentionWrapper` (`:1007-1020`). On
  TPU that wrapper is `register_oot`-replaced by
  `VllmMultiHeadLatentAttentionWrapper` (§3), so the *projection graph* runs in vLLM
  PyTorch under torchax while the *attention math* runs in the Pallas v2 kernel.
- The forward math is what §1/§3 traces (vLLM's reference is `mla.py:113-177`; the TPU
  override is `mla_attention.py:217-279` — structurally identical, plus absorption).

### MoE shape (owned by another agent — for reference only)
`DeepseekV2MoE` (`deepseek_v2.py:241-382`): `n_routed_experts` (`:258`),
`n_shared_experts` (`:259`), `GateLinear` (`:269`), shared expert intermediate
`moe_intermediate_size · n_shared_experts` (`:302-312`), routed via `FusedMoE` with
`use_grouped_topk=True`, `num_expert_group=n_group`, `topk_group` (`:314-338`).
DeepSeek-V3 numbers: 256 routed + 1 shared, top-8, 8 groups / top-4 groups,
`first_k_dense_replace=3`.

### V2→V3 config switches (matter for v4)
No `is_v3` bool; everything is `getattr`-gated:
- `routed_scaling_factor` (`deepseek_v2.py:253`).
- **`topk_method == "noaux_tc"`** → creates `gate.e_score_correction_bias`
  (`:274-279`) — the V3 sigmoid-router aux-loss-free bias; the canonical V3 marker.
- `scoring_func` (`:327`) — V3 uses `"sigmoid"`.
- `n_group` / `topk_group` (`:324-325`).
- **`is_v32 = hasattr(config, "index_topk")`** (`:964`) — gates the DeepSeek-V3.2
  sparse `Indexer` (`:597-715`). **This is the newest axis and the most likely place a
  "v4" diverges.**
- MTP / next-token layers via `config.num_nextn_predict_layers`
  (`get_spec_layer_idx_from_weight_name`, `:1682-1693`).

> The CUDA fast paths (`dsv3_fused_a_gemm`, `min_latency_fused_qkv_a_proj`, fp8 indexer
> quant) are all `current_platform.is_cuda()`-guarded and fall back to plain
> `nn.functional.linear`, so the structural graph above is exactly what TPU executes.

---

## 7. Contrast: pure-JAX rewrite vs torchax route

`tpu_inference/models/jax/deepseek_v3.py` is a **native, first-class** DeepSeek-V3 for
the flax_nnx route.

- **Registered** as the DeepSeek-V3 class:
  `_MODEL_REGISTRY["DeepseekV3ForCausalLM"] = DeepseekV3ForCausalLM`
  (`models/common/model_loader.py:84`; class at `models/jax/deepseek_v3.py:1347`).
- **MLA implemented natively** with explicit absorption: `DeepseekV3MLA`
  (`deepseek_v3.py:585`) overrides `kv_b_proj` with `MLAEinsum` (`:478, 593`) that
  splits into `k_up_proj`/`v_up_proj` (`:540-577`); `compute_q_projection` absorbs the
  query into latent space (`q_TNA = k_up_proj(q_nope)`, `:628`);
  `compute_kv_projection` returns the compressed latent (`:633-659`);
  `compute_attention` calls the **same** `mla_attention()` shared interface
  (`:696-711` → v2 kernel); `process_output` applies `v_up_proj` (`:726`). This is the
  exact same absorption strategy as the torchax route — just in JAX/Flax instead of
  torch-under-torchax.
- **MoE native**: 256 routed + 1 shared, sigmoid router with `e_score_correction_bias`
  (`DeepSeekV3Router`, `:969-1042`), `first_k_dense_replace=3` (`:1258`).

**Maturity verdict (matters for choosing the v4 path):**

| | flax_nnx (JAX) | torchax / vLLM |
|---|---|---|
| DeepSeek model class | **native, registered, default** (`model_loader.py:84`) | vLLM's own PyTorch class, **no tpu-inference class**; fallback only |
| Reached when | `auto` default (DeepSeek not in `_VLLM_PREFERRED_ARCHITECTURES`, `model_loader.py:51-54`) | `MODEL_IMPL_TYPE=vllm`, or PP>1 (DeepSeek in `_PP_DISABLED_MODELS`, `:57-58`) |
| MLA attention | native `DeepseekV3MLA`, full quantized weight-load | vLLM graph + `mla_attention.py` absorption op |
| Tests | `tests/models/jax/test_deepseek_v3.py` (~795 lines) | (kernel-level only) |
| Kernel | v2 (shared) | v2 (shared) |

**Both routes converge on the same v2 Pallas kernel and the same absorption strategy.**
The difference is the *model graph* layer: JAX has a hand-written, tested, default-served
DeepSeek-V3; the torchax route leans on vLLM's upstream model class and a thin
TPU attention op.

For **DeepSeek v4** — ⚠️ **the "v4 for free via torchax" story does not hold** (verified
against this vLLM checkout, `v0.20.1rc0-36-g75a7cf2c1`, which *already ships* v4):
- vLLM already has `DeepseekV4ForCausalLM` (`vllm/.../models/deepseek_v4.py`, registered
  `registry.py:99`) with its **own** attention stack — `DeepseekV4MLAModules` +
  `DeepseekV4MultiHeadLatentAttentionWrapper` (`vllm/.../layers/deepseek_v4_attention.py:87, 107`).
  It does **not** build the `MLAModules`/`MultiHeadLatentAttentionWrapper` that
  `mla_attention.py`'s `register_oot` hooks, so the TPU shim would not even attach.
- v4's inner attention `forward(self, q, kv, positions, output)`
  (`deepseek_v4_attention.py:717`) takes a **fused `q` + single `kv`** — *not* the
  `(q_nope, q_pe), kv_c_normed, k_pe` four-tensor contract the v2 kernel assumes. It uses a
  different factorization (`fused_wqa_wkv`, per-head q-norm, output-LoRA `wo_a`/`wo_b`),
  **FlashMLA-sparse + SWA** kernels with an fp8 KV cache, hyper-connections, and MegaMoE — and
  is hard-CUDA/SM100-only (`deepseek_v4.py:519-527`, `deepseek_v4_attention.py:204`).
- **Therefore the torchax route does NOT give v4 for free.** Bringing up v4 means porting a
  new sparse/SWA attention algorithm (new kernel work), not reusing the existing MLA op.
- If the *actual* near-term target is **v3.2** (not v4), that one **does** still fit the
  existing contract — vLLM handles it inside `deepseek_v2.py` via `is_v32`
  (`registry.py:98` maps `DeepseekV32ForCausalLM` → the `deepseek_v2` `DeepseekV3ForCausalLM`),
  with the indexer only *selecting which cached tokens to attend* (still dense on TPU; §8.3).
- The JAX route requires hand-porting the target graph but is the more battle-tested path
  here and is what's served by default for v3.

---

## 8. Cross-cutting facts for the v4 effort

1. **Weight absorption is mandatory and external to the kernel.** v4 must split
   `kv_b_proj` → `W_UK`/`W_UV` and run attention in `kv_lora_rank` latent space. The
   kernel never sees full per-head K/V. (`mla_attention.py:151-203`,
   `vllm/.../mla_attention.py:840-903`.)
2. **The v2 kernel's input contract is fixed:** `(q_TNA in latent, q_pe rope,
   kv_c_normed latent, k_pe rope)`, output in latent width `kv_lora_rank`. **vLLM's shipped
   v4 already breaks this contract** — `DeepseekV4...Wrapper.forward(self, q, kv, ...)`
   (`deepseek_v4_attention.py:717`) passes a fused `q` + single `kv`, plus a different
   latent factorization and FlashMLA-*sparse* attention. So v4 needs **new kernel + new
   custom-op work**, not just a model class. (v3.2's indexer also changes *which* tokens
   are attended, but keeps the four-tensor contract — see #3.)
3. **DeepSeek-V3.2 sparse MLA already has hooks** (`is_v32 = hasattr(config,"index_topk")`,
   `Indexer`, `topk_indices_buffer` in `mla_attention.py:194-197, 263-265`;
   `deepseek_v2.py:964, 597-715`) but the TPU kernel path does **not** implement sparse
   selection — the v2 Pallas kernel is **dense-latent** (zero topk/indexer/sparse logic;
   grep-clean). On TPU the indexer is even *invoked* (`mla_attention.py:264`,
   `_topk_indices = self.indexer(...)`) but its result is **discarded** (leading-underscore
   throwaway, never passed to `self.mla_attn`). A genuinely sparse v3.2/v4 needs the kernel
   to consume topk indices — that gap is unimplemented.
4. **MLA on TPU hard-requires `NEW_MODEL_DESIGN=1` + DP-attention**, else
   `check_and_update_config` raises (`tpu_platform.py:200-207`). Any v4 bring-up must
   set both.
5. **`VLLM_MLA_DISABLE=1`** is the escape hatch to the dense per-head path
   (`vllm/config/model.py:1595`) — useful as a numerical baseline when debugging v4 MLA.
6. **MLA shards on `MLP_TENSOR`** (kernel cache + q/k I/O), not `ATTN_HEAD`/`ATTN_DATA`
   (`attention_interface.py:502-519`, `kv_cache.py:123`); the *absorbed weights*
   `W_UK_T`/`W_UV` shard on `ATTN_HEAD` (`mla_attention.py:84-97`). A TPU v3.x reuse
   inherits this; v4 would need its own scheme (different cache, fp8, SWA).
7. **Block sizes are hardcoded, no autotuner** (`attention_interface.py:522-525`) —
   a known perf TODO, and v4 with different head counts may want re-tuning.

---

## 9. Anchor index (most load-bearing)

- Backend select: `tpu_inference/platforms/tpu_platform.py:114-128` (`get_attn_backend_cls`), `:200-207` (MLA requirements), `:226-230` (block size).
- Backend class: `tpu_inference/layers/vllm/backends/flash_attn_mla.py:33` (register), `:44-46` (page size 1024), `:111-208` (impl forward + absorption).
- Custom op / layer: `tpu_inference/layers/vllm/custom_ops/mla_attention.py:74-109` (absorption setup), `:217-279` (MLA forward graph), `:153` (`register_oot`).
- Shared interface: `tpu_inference/layers/common/attention_interface.py:463-553` (`mla_attention`), `:33` (v2 import), `:523-525` (block sizes).
- Kernels: `tpu_inference/kernels/mla/v2/kernel.py:1398` (entry, production), `:1716/1737/1759` (3 regimes); `tpu_inference/kernels/mla/v1/kernel.py:1092` (entry, tests-only), `:32-44` (`get_kv_cache_shape`, used in prod), `:91-94` (cache write).
- KV cache: `tpu_inference/runner/kv_cache.py:65-83` (`use_mla` fork), `:123-130` (sharding); `tpu_inference/runner/kv_cache_manager.py:710-728` (alloc), `:80/87-93` (`MLAAttentionSpec`, num_kv_heads=1).
- vLLM model: `vllm/model_executor/models/deepseek_v2.py:1672-1673` (`DeepseekV3ForCausalLM`), `:843-1030` (`DeepseekV2MLAAttention`), `:872-878` (dims), `:893-932` (projections), `:964` (`is_v32`), `:274-279` (`noaux_tc` gate bias).
- vLLM MLA layer: `vllm/model_executor/layers/mla.py:113-177` (forward), `vllm/.../attention/mla_attention.py:816-903` (`W_UK`/`W_UV` split).
- vLLM gate: `vllm/config/model.py:1595` (`use_mla`), `:54` (`get_head_size`).
- vLLM **v4** (already shipped, breaks contract): `vllm/model_executor/models/deepseek_v4.py` (`DeepseekV4ForCausalLM`), `vllm/model_executor/layers/deepseek_v4_attention.py:87` (`DeepseekV4MLAModules`), `:107` (`DeepseekV4MultiHeadLatentAttentionWrapper`, new PluggableLayer name), `:717` (`forward(self, q, kv, ...)` — fused contract); registry `vllm/.../models/registry.py:98` (`DeepseekV32`→`deepseek_v2`), `:99` (`DeepseekV4`→`deepseek_v4`). Checkout `v0.20.1rc0-36-g75a7cf2c1`.
- JAX contrast: `tpu_inference/models/jax/deepseek_v3.py:1347` (class), `:585` (`DeepseekV3MLA`), `:696-711` (kernel call); `tpu_inference/models/common/model_loader.py:84` (registry), `:51-58` (route resolution).
