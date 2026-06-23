# 09 — Adding DeepSeek-V4 (and V3.2) on TPU — a concrete playbook (torchax route)

> **Audience:** a strong engineer who has read **doc 06** (DeepSeek-V3 MLA on TPU)
> and **doc 08** (the quantization framework) and now has to actually *land* DeepSeek
> support on the torchax route (`MODEL_IMPL_TYPE=vllm`). This is the last doc in the
> series: it turns "what exists" into "what to build."
>
> **Scope:** two targets, very different cost. **V3.2** (sparse-indexer DeepSeek)
> already fits the existing MLA contract and is a small, well-bounded change. **V4** is
> a *different attention algorithm* — vLLM already ships `DeepseekV4ForCausalLM`, and it
> does **not** reuse the V3 MLA path. Bringing it up on TPU is new-kernel + new-custom-op
> work, not a config flip.
>
> All anchors are `path:line`. vLLM upstream is the editable install at
> `/home/enyouki/vllm/vllm/`; everything else is under `/home/enyouki/tpu-inference/`.
> vLLM checkout: `v0.20.1rc0-36-g75a7cf2c1`. Every V4/V3.2 anchor below was read against
> source.

---

## 1. TL;DR / verdict

- **V3.2 ≈ already fits.** vLLM routes `DeepseekV32ForCausalLM` to the *V3* class in
  `deepseek_v2.py` (`registry.py:98`); the only delta is a sparse **`Indexer`**
  (`deepseek_v2.py:597-715`) gated by `is_v32 = hasattr(config, "index_topk")`
  (`deepseek_v2.py:964`). The attention **shape contract is unchanged** — same
  `(q_nope, q_pe), kv_c_normed, k_pe` four-tensor latent MLA that doc 06 describes. On
  the TPU torchax route it **already runs today, numerically correct, but dense**: the
  TPU custom op invokes the indexer and then **throws the result away**
  (`_topk_indices = self.indexer(...)`, `mla_attention.py:264`, never passed to
  `self.mla_attn`). Supporting V3.2 *properly* (sparse acceleration) means teaching the
  Pallas kernel to consume top-k indices — a real but contained kernel feature. Running
  V3.2 *correctly* needs ~zero new code.

- **V4 = NOT a reuse of V3 MLA.** `DeepseekV4ForCausalLM` is a full new standalone class
  (`deepseek_v4.py:1503`, registered `registry.py:99`) with its **own** attention stack
  in a **new file** `model_executor/layers/deepseek_v4_attention.py`. Its inner
  attention takes a **fused `(q, kv)`** pair and writes an in-place output buffer
  (`forward(self, q, kv, positions, output) -> None`, `deepseek_v4_attention.py:717`) —
  *not* the four-tensor latent contract the TPU v2 Pallas kernel and custom op expect.
  It uses **FlashMLA-sparse + sliding-window attention (SWA)**, a **mandatory fp8 KV
  cache** (`fp8_ds_mla`), a different projection factorization (**fused `fused_wqa_wkv`
  down-proj + two-stage output-LoRA `wo_a`/`wo_b`**, *no* `kv_b_proj`/`W_UK`/`W_UV`
  absorption), **per-head q-norm**, **hyper-connections**, and **MegaMoE**
  (`deep_gemm.fp8_fp4_mega_moe`). It is **hard-gated to CUDA / SM100 (Blackwell)** at
  multiple points. Critically, its wrapper registers as a **new PluggableLayer name**
  (`@PluggableLayer.register("deepseek_v4_multi_head_latent_attention")`,
  `deepseek_v4_attention.py:106`), so the TPU `register_oot` hook — which keys on V3's
  `MultiHeadLatentAttentionWrapper` (`mla.py:34`) — **will not attach to it.**

**Bottom line:** the "V4 for free via torchax" story is false. **V3.2 is the easy win.**
**V4 needs: a new Pallas attention kernel (sparse + SWA + fp8 KV), a new TPU custom op +
backend keyed on V4's layer name, fp8-KV plumbing, MegaMoE/hyper-connection model work,
and new sharding for the fused/LoRA projections.** Plan it as a model bring-up, not a
patch.

---

## 2. Recap: the V3 MLA contract you must satisfy (compressed from doc 06)

This is the contract the **existing** TPU MLA path enforces. Anything you add for V3.2
or V4 either satisfies it (V3.2) or replaces it (V4). Six load-bearing facts:

1. **Backend selection is forced by `use_mla`.** `TpuPlatform.get_attn_backend_cls`
   (`tpu_platform.py:114-128`): `if use_mla: selected_backend =
   AttentionBackendEnum.FLASH_ATTN_MLA` → `get_path()` →
   `flash_attn_mla.py::PallasMLAttentionBackend` (registered `flash_attn_mla.py:33`).
   `use_mla` is set by vLLM (`is_deepseek_mla and not VLLM_MLA_DISABLE`,
   `vllm/config/model.py:1595`).

2. **The custom op is monkeypatched by class name.** `VllmMultiHeadLatentAttentionWrapper`
   is attached via `MultiHeadLatentAttentionWrapper.register_oot`
   (`mla_attention.py:153`) — vLLM's OOT facility keyed on the **base class name**
   (`vllm/model_executor/custom_op.py:84-89`, looked up in `__new__`). It only fires for
   the V3 wrapper. (Doc 03.)

3. **The inner forward is the four-tensor latent contract.** The TPU op calls
   `self.mla_attn((q_nope, q_pe), kv_c_normed, k_pe, output_shape=...)`
   (`mla_attention.py:271-277`). Down in the impl this becomes the v2 kernel's
   `(q_TNA latent, q_pe rope, kv_c_normed latent, k_pe rope)` → output in latent width
   `kv_lora_rank`.

4. **External weight absorption.** `W_UK` is folded into the query and `W_UV` into the
   output, **outside the kernel** — set up in
   `VllmMLAAttention.process_weights_after_loading` (`mla_attention.py:74-109`, which
   shards `W_UK_T`/`W_UV` on `ATTN_HEAD` and **deletes `kv_b_proj`**), applied around the
   kernel call in `PallasMLAttentionBackendImpl.forward`
   (`flash_attn_mla.py:111-208`). The kernel never sees full per-head K/V.

5. **The v2 Pallas kernel is the compute path** (`kernels/mla/v2/kernel.py:1398`); v1 is
   used only for `get_kv_cache_shape` (`kv_cache.py:68`). Cache is **one combined latent
   tensor** per layer, `num_kv_heads=1`, sharded on `MLP_TENSOR`
   (`kv_cache.py:123-130`, `kv_cache_manager.py:710-728`). The kernel signature is **dense
   latent** — *no* topk/sparse/index argument anywhere (`attention_interface.py:463`,
   verified grep-clean).

6. **The hard gate.** MLA on TPU **requires `NEW_MODEL_DESIGN=1` AND DP-attention**, else
   `check_and_update_config` raises (`tpu_platform.py:200-207`). MLA block size defaults
   to **1024** (`tpu_platform.py:226-230` → `flash_attn_mla.py:44-46`).

```
       vLLM model graph (torch, under torchax)
   DeepseekV2MLAAttention (deepseek_v2.py:843)
        builds MLAModules → MultiHeadLatentAttentionWrapper   (mla.py:34, PluggableLayer "multi_head_latent_attention")
                                   │  register_oot  (mla_attention.py:153)  ← keys on THIS class name
                                   ▼
        VllmMultiHeadLatentAttentionWrapper.forward  (mla_attention.py:217)
            fused_qkv_a_proj → q_nope, q_pe, kv_c_normed, k_pe   (4-tensor contract)
                                   │
                                   ▼  self.mla_attn(...)
        PallasMLAttentionBackendImpl.forward  (flash_attn_mla.py:111)
            W_UK absorb → mla_attention() shard_map → v2 kernel → W_UV absorb
                                   ▼
                 kernels/mla/v2/kernel.py:1398   (dense latent, MQA, no sparsity)
```

---

## 3. Why V4 does NOT fit — gap analysis

Each row: **V3 MLA contract item → what V4 does instead → consequence for the TPU path.**
Anchors are in `vllm/model_executor/` unless noted.

| # | V3 contract (doc 06) | What V4 does | TPU consequence |
|---|---|---|---|
| 1 | OOT hook keyed on `MultiHeadLatentAttentionWrapper` (`mla.py:34`, registered `"multi_head_latent_attention"`) | New wrapper class `DeepseekV4MultiHeadLatentAttentionWrapper`, registered `@PluggableLayer.register("deepseek_v4_multi_head_latent_attention")` (`deepseek_v4_attention.py:106-107`) | **TPU `register_oot` never attaches.** The existing TPU MLA op (`mla_attention.py:153`) keys on the V3 class — different class, different registration name. V4 runs vLLM's CUDA wrapper unmodified → instant failure on TPU. |
| 2 | Inner call = 4-tensor latent `((q_nope,q_pe), kv_c_normed, k_pe)` (`mla_attention.py:271`) | Inner call = **fused `(q, kv)`** + in-place buffer: `forward(self, q, kv, positions, output) -> None` (`deepseek_v4_attention.py:717-723`); called `self.mla_attn(q, kv, positions, output=out)` (`deepseek_v4_attention.py:492`) | The v2 kernel and `mla_attention()` interface (`attention_interface.py:463`) take the 4-tensor form. **A new custom op + kernel call signature is required.** |
| 3 | External absorption: split `kv_b_proj`→`W_UK`/`W_UV`, run in `kv_lora_rank` latent, delete `kv_b_proj` (`mla_attention.py:74-109`) | **No `kv_b_proj`, no `W_UK`/`W_UV`** (grep-clean in both V4 files). Different factorization: fused down-proj `fused_wqa_wkv` (`deepseek_v4.py:968`), `wq_b` up-proj (`:977`), two-stage output-LoRA `wo_a`(BMM)→fp8 einsum→`wo_b` (`deepseek_v4_attention.py:307-336`), `o_lora_rank`/`o_groups` | The whole `process_weights_after_loading` absorption logic (doc 06 §3) does not apply. **New weight-processing + sharding for the fused/LoRA projections.** |
| 4 | Dense-latent MQA kernel, no sparsity (`kernels/mla/v2/kernel.py`) | **FlashMLA-sparse**: asserts `issubclass(get_attn_backend(), FlashMLASparseBackend)` (`deepseek_v4_attention.py:672`), `SparseAttnIndexer`/`DeepseekV4Indexer` top-k (`:1114`, `:1029`), compress ratios C4A/C128A (`:794-841`) | TPU has **no sparse MLA kernel**. The v2 kernel takes no index argument. **New Pallas kernel that consumes top-k indices.** |
| 5 | (no per-layer windowing) | **Sliding-window attention**: `DeepseekV4SWACache` (`deepseek_v4_attention.py:232`), `window_size`, decode/prefill `swa_indices`/`swa_lens` (`:810-811`, `:943-953`) | The TPU MLA kernel has no windowing. **The new kernel must implement SWA masking.** |
| 6 | KV cache = combined latent, dtype follows model (bf16) (`kv_cache_manager.py:710-728`) | **fp8 KV cache is mandatory**: `assert kv_cache_dtype.startswith("fp8")` then auto-set to `"fp8_ds_mla"` (`deepseek_v4_attention.py:665-685`) | TPU MLA cache alloc/shape (`runner/kv_cache.py:65-83`, v1 `get_kv_cache_shape`) is not fp8-`ds_mla` aware. **New cache layout + fp8 quant/dequant in the kernel + shape plumbing.** |
| 7 | per-head q split into nope/rope; RoPE on decoupled slice (`mla_attention.py:253-261`) | **Per-head q-norm** `RMSNorm(head_dim, has_weight=False)` (`deepseek_v4_attention.py:215`), applied inside a fused CUDA kernel `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (`:519`) | Extra normalization in the projection graph; the fused CUDA kernel must be reproduced as plain ops (it will not run on TPU). |
| 8 | Standard residual stream | **Hyper-connections**: widened residual `hc_mult*hidden_size`, `hc_pre`/`hc_post` Sinkhorn ops `torch.ops.vllm.mhc_pre`/`mhc_post` (`deepseek_v4.py:1163,1188,1175,1195`) | These are vLLM **custom ops** (`torch.ops.vllm.*`) with no TPU registration. **Either register TPU implementations or rewrite the decoder layer's residual handling.** |
| 9 | `DeepseekV2MoE` + `FusedMoE` (doc 06 §6) — works on TPU via `register_oot` FusedMoE + doc 08 MoE methods | **MegaMoE**: `DeepseekV4MegaMoEExperts` (`deepseek_v4.py:392`), `deep_gemm.fp8_fp4_mega_moe` (`:654`), `scoring_func="sqrtsoftplus"` (`:734`), hash-MoE path | MegaMoE is **not** a vanilla `FusedMoE` — the doc 08 MoE weight-processing/`apply_monolithic` hooks key on `FusedMoE`. **New MoE method (or a graph rewrite back to FusedMoE) is needed.** |
| 10 | runs on TPU (CUDA fast-paths are platform-guarded, doc 06 §6) | **Hard CUDA/SM100 gating**: attention `assert cap is not None, "DeepseekV4 attention requires a CUDA device"` (`deepseek_v4_attention.py:204`); MegaMoE `raise NotImplementedError("DeepSeek V4 MegaMoE requires CUDA.")` + `cap[0] != 10 → "DeepGEMM MegaMoE requires SM100 GPUs."` (`deepseek_v4.py:519,526-527`); unconditional `torch.cuda.Event()`/`torch.cuda.Stream()` (`deepseek_v4_attention.py:229,636`; `deepseek_v4.py:1240`); FlashMLA-sparse imports at module top (`deepseek_v4_attention.py:61-74`) | **Import-time and `__init__`-time CUDA coupling.** Even a fully-implemented TPU op cannot run V4's classes as shipped without removing/guarding these. A TPU bring-up must either fork the classes or patch the CUDA primitives out. |

> **The single structural blocker (row 1):** because V4's wrapper is a *new class under a
> new PluggableLayer name*, the entire existing TPU MLA shim is invisible to it. There is
> nothing to "extend" — you register a fresh OOT override against
> `DeepseekV4MultiHeadLatentAttentionWrapper` and build its forward from scratch.

---

## 4. The V3.2 path — the easy win

### 4.1 What V3.2 changes vs V3

V3.2 (DeepSeek-V3.2-Exp, "sparse MLA") is the *same* MLA attention with one addition: a
**lightning/sparse `Indexer`** that scores tokens and selects a top-k subset to attend,
instead of attending the full prefix.

- Registry: `DeepseekV32ForCausalLM` → **the V3 class** `("deepseek_v2",
  "DeepseekV3ForCausalLM")` (`registry.py:98`). No separate file, no separate model class.
- Gate: `is_v32 = hasattr(config, "index_topk")` — twice:
  `deepseek_v2.py:964` (attention module) and `:1178` (decoder/model scope, where the
  shared `topk_indices_buffer` is allocated, `:1181-1186`).
- The indexer: `class Indexer(nn.Module)` (`deepseek_v2.py:597-715`). `topk_tokens =
  config.index_topk` (`:614`); builds a `SparseAttnIndexer` op (`:658-667`); `forward`
  (`:669-715`) projects/quantizes q to fp8, applies indexer-RoPE, returns
  `self.indexer_op(...)` (`:715`) — the selected top-k token indices.
- Wiring: when `is_v32`, the attention module builds `self.indexer` and
  `self.indexer_rope_emb` (`deepseek_v2.py:966-985`) and threads them into `MLAModules(...
  indexer=..., indexer_rotary_emb=..., is_sparse=self.is_v32,
  topk_indices_buffer=...)` (`:1001-1004`).

**Crucially, the attention *shape contract* is unchanged.** The indexer changes *which
cached tokens are attended*, not the latent tensor layout. So the four-tensor latent MLA
of doc 06 still applies — V3.2 satisfies the contract in §2.

### 4.2 Why it runs on TPU today — but dense

The TPU MLA op already *handles the indexer's existence*, but **discards its output**.
In `VllmMultiHeadLatentAttentionWrapper.forward` (`mla_attention.py:217-279`):

```python
# mla_attention.py:263-265
if self.indexer and self.is_sparse:
    _topk_indices = self.indexer(hidden_states, q_c, positions,
                                 self.indexer_rope_emb)
# ...the result _topk_indices is NEVER read again...
attn_out = self.mla_attn((q_nope, q_pe), kv_c_normed, k_pe,        # :271-277
                         output_shape=(...))                        # no indices passed
```

The indexer is invoked (so it costs compute), the result lands in a leading-underscore
throwaway, and `self.mla_attn` is called with the **full** latent K/V. The shared
`topk_indices_buffer` is stored (`mla_attention.py:197`) but never read either. The v2
Pallas kernel and the `mla_attention()` interface (`attention_interface.py:463`) take
**no** sparse/index argument (verified grep-clean). So:

> **On the TPU torchax route, V3.2 runs numerically correct but effectively dense** — it
> attends every cached token, ignoring the indexer's selection. Correct outputs, no
> sparse speedup, and the indexer's cost is pure overhead.

### 4.3 Task list to support V3.2 *properly* (sparse-accelerated)

The contract doesn't change — only the kernel needs to gain a "gather/mask by top-k
indices" capability. Ordered:

1. **Thread the indices out of the op.** Stop discarding: pass `_topk_indices` (or the
   `topk_indices_buffer`) from `mla_attention.py:264` into `self.mla_attn(...)`. This
   means widening `VllmMLAAttention`/`PallasMLAttentionBackendImpl.forward`
   (`flash_attn_mla.py:111`) and the `mla_attention()` interface
   (`attention_interface.py:463`) with an optional `topk_indices` argument.
2. **Teach the v2 kernel to consume indices.** In `kernels/mla/v2/kernel.py`, when
   indices are present, the per-query attention should read only the selected KV pages
   /rows instead of the full `kv_len`. This is a real kernel change (new gather path or a
   sparse mask over the existing paged read); the paging mechanics are already there
   (doc 06 §4), the selection is the new part.
3. **Validate against dense.** Because the dense path is already correct, you have a
   free oracle: with the same checkpoint, sparse output should match dense within the
   indexer's top-k tolerance (DeepSeek's indexer is designed to be near-lossless).
   Use `VLLM_MLA_DISABLE=1` (`vllm/config/model.py:1595`) for an even-more-baseline
   per-head reference if needed.

If the goal is just **"serve V3.2 correctly,"** steps 1–3 are optional — it already
works. If the goal is **"serve V3.2 fast,"** that's the whole job, and it's
self-contained kernel work (no new model class, no new backend).

> Where the GPU side actually reads the indices (for parity reference): vLLM's sparse MLA
> backend `vllm/v1/attention/backends/mla/indexer.py` (`DeepseekV32IndexerBackend`) and
> `vllm/model_executor/layers/sparse_attn_indexer.py` (`SparseAttnIndexer`) — both
> imported into `deepseek_v2.py:74-75, 90-91`. The TPU kernel would need to mirror that
> selection semantics, not the CUDA implementation.

---

## 5. V4 implementation playbook

V4 is a model bring-up. Below, each workstream is grounded in the extension point the
prior docs established. Treat these as parallelizable once the **load/route** skeleton
(§6 step 1) is in place.

### 5.a New Pallas attention kernel — sparse FlashMLA + SWA + fp8 KV

This is the largest piece. The TPU `kernels/mla/v2/kernel.py` is **dense latent MQA**;
V4 needs a kernel that additionally:

- **Consumes top-k indices** (sparse selection — same capability V3.2 §4.3 step 2 needs;
  build it once, reuse). The indices come from V4's `DeepseekV4Indexer`
  (`deepseek_v4_attention.py:1029`).
- **Applies a sliding window.** V4 carries per-request `swa_indices`/`swa_lens`
  (`deepseek_v4_attention.py:810-811, 943-953`); the kernel must mask to `window_size`.
- **Reads/writes an fp8 KV cache** in the `fp8_ds_mla` layout
  (`deepseek_v4_attention.py:677-685`), with in-kernel dequant on the read side.
- **New shape contract.** V4 passes a fused `(q, kv)` (`deepseek_v4_attention.py:717`),
  not the `(q_nope, q_pe, kv_c_normed, k_pe)` split. Decide whether you re-split inside
  the TPU op (to reuse latent infra) or carry the fused layout into the kernel. Note the
  **alignment subtlety** doc 06 §4 flagged (`get_kv_cache_shape` aligns the raw sum
  once; the write side aligns per-component) — V4's different head dims may break the
  v3 coincidence, so the alloc-shape vs compute-shape reconciliation must be re-derived,
  not inherited.

Realistic recommendation: **fork `kernels/mla/v2/kernel.py` into a `kernels/mla/v4/`**
rather than overloading v2 with flags. Keep v2 dense-clean for V3/V3.2.

### 5.b New TPU custom op — keyed on V4's layer name

The existing op (`mla_attention.py`) will not attach (§3 row 1). You write a parallel one:

- **Register against the V4 wrapper.** The shipped V4 wrapper is
  `DeepseekV4MultiHeadLatentAttentionWrapper`, a `PluggableLayer` registered
  `"deepseek_v4_multi_head_latent_attention"` (`deepseek_v4_attention.py:106-107`).
  Use the same OOT mechanism doc 03 describes
  (`@DeepseekV4MultiHeadLatentAttentionWrapper.register_oot`, importing the V4 class),
  and add the module to `tpu_inference/layers/vllm/__init__.py` so the decorator fires at
  import (the `register_layers` no-op + import side-effect pattern, doc 03 / doc 08 §1.4).
- **Translate the fused forward.** Your op's `forward` must accept V4's outer signature
  (`positions, hidden_states, ...`, matching `deepseek_v4_attention.py:282`), run the
  projection graph as plain torch-under-torchax (replacing the fused CUDA kernels — q-norm,
  RoPE, kv-rope, fp8 quant/insert at `:519` — with explicit ops), and call into the new
  V4 kernel (§5.a).
- **Absorption — likely gone.** V4 has no `kv_b_proj`/`W_UK`/`W_UV` (§3 row 3). The doc 06
  external-absorption setup does **not** carry over. Instead you process the fused
  `fused_wqa_wkv` / `wq_b` / `wo_a` / `wo_b` projections (§5.f). Decide whether any latent
  pre-multiply is still worth doing for the kernel's sake — but it is no longer the
  `kv_b_proj` split.

### 5.c Backend selection

Doc 06 §2: `use_mla` forces `FLASH_ATTN_MLA` in `get_attn_backend_cls`
(`tpu_platform.py:114-128`). For V4 you need TPU to pick a **new** backend:

- V4 sets `use_mla` (it *is* a DeepSeek-MLA arch, so `vllm/config/model.py:1595` still
  fires) — meaning with the code as-is it would land on `PallasMLAttentionBackend`, the
  **V3** backend, which is wrong for V4.
- Add a branch in `get_attn_backend_cls` (`tpu_platform.py:114-128`) that detects V4
  (e.g. via the model arch / `hf_config`) and returns a new
  `AttentionBackendEnum.FLASH_ATTN_MLA_V4` resolving to a `PallasMLAV4Backend`
  (registered with `@register_backend(...)`, mirroring `flash_attn_mla.py:33` and the
  `backends/__init__.py` import). Its impl forward calls the §5.a kernel.
- Re-derive the **page size** (V3 uses 1024, `flash_attn_mla.py:44-46`,
  `tpu_platform.py:226-230`) for V4's fp8/SWA cache.

### 5.d fp8 KV cache plumbing

Doc 04's KV mechanism: the real cache is threaded via the **wrapper context** by layer
name (`get_vllm_model_wrapper_context()` → `ctx.layer_name_to_kvcache_index[layer_name]`
→ `ctx.kv_caches[idx]`; write-back at `flash_attn.py:213`), and the MLA cache shape/alloc
forks on `use_mla` in `runner/kv_cache.py:65-83` / `kv_cache_manager.py:710-728`. For V4:

- The cache dtype is **fp8 mandatory** (`fp8_ds_mla`, `deepseek_v4_attention.py:677-685`).
  The existing MLA alloc uses the model dtype; add an fp8 path to
  `get_kv_cache_shape_with_mesh` / `MLAAttentionSpec` for the V4 layout.
- There is precedent for KV-quant scale threading on the dense path
  (`quantize_kv`, `k_scale`/`v_scale` forwarded, `flash_attn.py:185-192` →
  `attention_interface.py:442-458`) and on MLA (`flash_attn_mla.py:158-175`) — reuse that
  machinery, but the `fp8_ds_mla` layout differs from a plain per-tensor fp8 cache, so the
  pack/shape must be defined to match what the §5.a kernel reads.

### 5.e MegaMoE + hyper-connections

- **MegaMoE is not a vanilla `FusedMoE`.** Doc 08's MoE integration
  (`is_monolithic` + `apply_monolithic`, weight-processing via `process_moe_weights` /
  `shard_moe_weights`) keys on the `FusedMoE` layer class (doc 08 §1.6, §2.3). V4 uses
  `DeepseekV4MegaMoEExperts` (`deepseek_v4.py:392`) → `deep_gemm.fp8_fp4_mega_moe`
  (`:654`), `sqrtsoftplus` routing (`:734`), hash-MoE, and is **SM100-hard-gated**
  (`_check_runtime_supported`, `deepseek_v4.py:518-527`). Two options: (i) rewrite the
  decoder to route through vLLM's standard `FusedMoE` (then the existing TPU MoE methods
  apply, modulo the new fp8-fp4 numeric format → doc 08 §5 new kernel/math), or
  (ii) write a dedicated MegaMoE TPU method/kernel. (i) is the lower-risk first step.
- **Hyper-connections** are `torch.ops.vllm.mhc_pre`/`mhc_post` Sinkhorn custom ops
  (`deepseek_v4.py:1175,1195`) with a widened residual (`hc_mult*hidden_size`,
  `:1114-1116`). These have **no TPU registration**. Either register TPU implementations
  of the two custom ops, or rewrite `hc_pre`/`hc_post` (`deepseek_v4.py:1163,1188`) as
  plain ops in a forked decoder layer. Not numerically trivial — the Sinkhorn
  normalization needs a faithful port.

### 5.f Sharding — new `process_weights_after_loading` for the fused/LoRA projections

Doc 08 §3 is the weight-processing pipeline (`t2j` → `process_*` → `shard_*` →
re-attach). The V3 absorption sharding (`mla_attention.py:74-109`, `W_UK_T`/`W_UV` on
`ATTN_HEAD`) does not apply. New work:

- `fused_wqa_wkv` is a `MergedColumnParallelLinear([q_lora_rank, head_dim])`
  (`deepseek_v4.py:968`) — a merged-column linear, so it wants the
  `reorder_concatenated_tensor_for_sharding` fused-split treatment (doc 08 §3.2,
  `process_linear_weights(fused=True, output_sizes=[q_lora_rank, head_dim])`).
- `wq_b` up-proj (`deepseek_v4.py:977`), `kv_norm`, and the **two-stage output-LoRA**
  `wo_a` (BMM, `is_bmm`, `deepseek_v4.py:987`) → `wo_b` (`:997`) each need a sharding
  spec. `wo_a`'s BMM/grouped (`o_groups`) structure is not a standard Linear — it needs a
  custom `process_weights_after_loading` (a `VllmQuantLinearConfig`-style descriptor,
  doc 08 §1.3) deciding which axis is the head/group axis.
- If V4 ships fp8/fp4 weights, the matmuls route through `sharded_quantized_matmul`
  (doc 08 §5) — but `fp8_fp4_mega_moe` is a *new numeric format* for the MoE; the
  blockwise kernel handles fp4×fp8 (`blockwise_kernel.py`, doc 08 §5.2) so check whether
  it covers MegaMoE's layout before assuming reuse.

### 5.g The NEW_MODEL_DESIGN + DP-attention gate, and config plumbing

V4 is MLA, so the existing hard gate fires: `check_and_update_config`
(`tpu_platform.py:200-207`) requires `NEW_MODEL_DESIGN=1` **and**
`additional_config.sharding.sharding_strategy.enable_dp_attention=true`. Any V4 launch
must set both (doc 06 §2). Additionally:

- V4's config carries extra fields (`index_topk` for the indexer, `window_size`,
  `hc_mult`/`hc_sinkhorn_iters`/`hc_eps`, MegaMoE knobs). Confirm these survive vLLM
  config parsing on the TPU path and are reachable from your custom op / kernel.
- The CUDA/SM100 asserts (§3 row 10) must be neutralized for the TPU classes you fork —
  `check_and_update_config` is a natural place to fail fast with a clear TPU-specific
  error if a V4 prerequisite (e.g. fp8 KV) is mis-set, rather than hitting a raw CUDA
  assert deep in the layer.

---

## 6. Effort & sequencing

A realistic incremental plan. **Validate at each stage before moving on.** "Stub" = make
it load/return-something so the next layer can be exercised.

1. **Route + load (no correct numerics yet).** Get `DeepseekV4ForCausalLM` to *route* to
   the torchax path and *load weights* on TPU. This means: fork the V4 classes (or guard
   their CUDA `__init__`/import coupling — §3 row 10), register a placeholder TPU OOT
   wrapper against `DeepseekV4MultiHeadLatentAttentionWrapper` (§5.b), and stub the
   attention forward (return zeros of the right shape). Stub MegaMoE → a dense fallback or
   vanilla `FusedMoE` (§5.e option i). **Goal: the model builds, shards, and runs a forward
   without crashing.** This shakes out config plumbing (§5.g) and the registration wiring
   first — the cheapest place to find the structural blockers.

2. **Attention numerics (dense, no sparsity, no SWA).** Implement the new custom op's
   projection graph (§5.b) and a **dense** V4 kernel first (ignore top-k and window —
   attend everything, like V3.2 runs today). Validate against a reference (an
   fp32/bf16 eager run of V4's math, or `VLLM_MLA_DISABLE=1`-style per-head baseline).
   This isolates the **projection/factorization** correctness (fused down-proj,
   output-LoRA, per-head q-norm) from the kernel's sparsity.

3. **fp8 KV cache + SWA.** Add the `fp8_ds_mla` cache layout and dequant (§5.d), then
   sliding-window masking (§5.a). Validate windowed output vs a dense-but-masked
   reference.

4. **Sparse selection.** Wire the indexer's top-k through to the kernel (§5.a) — this is
   the same capability V3.2 needs (§4.3), so **do V3.2's sparse kernel first** as a
   simpler proving ground if both are on the roadmap. Validate sparse ≈ dense within the
   indexer's tolerance.

5. **MegaMoE + hyper-connections.** Replace the stubs from step 1 with real
   implementations (§5.e). MegaMoE's fp8-fp4 format may need new quant math/kernel
   (doc 08 §5); hyper-connections need a faithful Sinkhorn port. Validate MoE expert
   outputs and end-to-end perplexity.

6. **Perf.** Block-size tuning (the MLA path has **hardcoded** block sizes, no autotuner —
   `attention_interface.py:522-525`, doc 06 §8), sharding tuning for the new projections,
   and the sparse kernel's gather efficiency.

**Riskiest unknowns (call these out early):**

- **The sparse + SWA + fp8 Pallas kernel (§5.a)** is the long pole and the least
  reusable — it is genuinely new kernel engineering, not a port.
- **Hyper-connections' Sinkhorn ops** are unusual; a wrong port is a silent accuracy bug,
  not a crash. Build a tight numeric oracle.
- **MegaMoE's fp8-fp4 numeric format** may not be covered by the existing blockwise
  quantized-matmul kernel (doc 08 §5.2) — verify before assuming reuse.
- **Cache alignment** (doc 06 §4 subtlety): V4's head dims may break the v3 coincidence
  between alloc-shape and per-component write-shape; re-derive, don't inherit.
- **fp8_ds_mla layout** is a DeepSeek-specific cache packing; matching the GPU semantics
  exactly (so a converted checkpoint round-trips) needs care.

> **If the real near-term target is V3.2, not V4:** stop at §4. V3.2 already serves
> correctly; the only work is the sparse kernel (§4.3 / step 4 above), which is also the
> reusable core of V4's §5.a. Sequencing-wise, **V3.2-sparse is the natural first
> milestone of a V4 effort.**

---

## 7. Anchor index (verified)

### V4 — vLLM (`/home/enyouki/vllm/vllm/`)
- Model class: `model_executor/models/deepseek_v4.py:1503` (`DeepseekV4ForCausalLM`, full standalone `nn.Module`); registry `model_executor/models/registry.py:99`.
- Attention file: `model_executor/layers/deepseek_v4_attention.py`.
  - Dataclass `DeepseekV4MLAModules` `:86-102` (fields: `fused_wqa_wkv`, `q_norm`, `wq_b`, `kv_norm`, `wo_a`, `wo_b`, `indexer`, `topk_indices_buffer`, `aux_stream_list`).
  - Wrapper `DeepseekV4MultiHeadLatentAttentionWrapper` `:107`, registered `@PluggableLayer.register("deepseek_v4_multi_head_latent_attention")` `:106`.
  - **Inner forward (fused contract):** `def forward(self, q, kv, positions, output) -> None` `:717-723`; called `self.mla_attn(q, kv, positions, output=out)` `:492`.
  - Outer wrapper forward `(positions, hidden_states, ...)` `:282`.
  - CUDA cap assert `:204` (`"DeepseekV4 attention requires a CUDA device"`), einsum recipe by cap `:205-206`.
  - fp8 KV mandatory + `fp8_ds_mla` auto-convert `:665-685`; FlashMLASparse backend assert `:672`.
  - SWA: `DeepseekV4SWACache` (imported from `vllm/v1/attention/backends/mla/sparse_swa.py` at `:70`, used `:232`); `swa_indices`/`swa_lens` `:810-811, 943-953`.
  - Sparse: `DeepseekV4Indexer` (class def `:1029`), `SparseAttnIndexer` (imported `:19`, used `:1114`); compress ratios `:794-841`.
  - Per-head q-norm `:215`; fused qnorm/rope/kv/quant kernel `:519`.
  - Output-LoRA einsum `:307-336`; FlashMLA-sparse imports `:61-74`; `torch.cuda.Event()` `:229, 636`.
- MegaMoE: `model_executor/models/deepseek_v4.py:392` (`DeepseekV4MegaMoEExperts`), `:707` (`DeepseekV4MoE`), `deep_gemm.fp8_fp4_mega_moe` `:654`, `sqrtsoftplus` `:734`; CUDA/SM100 guard `_check_runtime_supported` `:518-527`.
- Hyper-connections: `:1091` (`DeepseekV4DecoderLayer`), `hc_mult/hc_sinkhorn_iters/hc_eps` `:1114-1116`, `hc_pre`/`hc_post` `:1163, 1188`, ops `:1175, 1195`; unconditional `torch.cuda.Stream()` in model `__init__` `:1240`.
- Fused/LoRA projections: `fused_wqa_wkv` `:968`, `wq_b` `:977`, `wo_a` (BMM) `:987`, `wo_b` `:997`.
- **No** `kv_b_proj`/`W_UK`/`W_UV` anywhere in either V4 file (grep-clean).

### V3.2 — vLLM
- Registry: `model_executor/models/registry.py:98` (`DeepseekV32ForCausalLM` → `deepseek_v2`/`DeepseekV3ForCausalLM`).
- `is_v32 = hasattr(config, "index_topk")`: `model_executor/models/deepseek_v2.py:964` (attention), `:1178` (model/decoder).
- `Indexer` class `deepseek_v2.py:597-715`; `index_topk` read `:614`; `SparseAttnIndexer` build `:658-667`; forward returns indices `:715`.
- Indexer wiring: build `:966-985`; `MLAModules(... indexer, is_sparse, topk_indices_buffer)` `:1001-1004`; buffer alloc `:1181-1186`.
- GPU consumers (reference): `vllm/v1/attention/backends/mla/indexer.py` (`DeepseekV32IndexerBackend`), `vllm/model_executor/layers/sparse_attn_indexer.py` (`SparseAttnIndexer`); imported `deepseek_v2.py:74-75, 90-91`.

### V3 wrapper + TPU side (`/home/enyouki/tpu-inference/`)
- V3 wrapper (what TPU keys on): `vllm/model_executor/layers/mla.py:34` (`MultiHeadLatentAttentionWrapper`, `@PluggableLayer.register("multi_head_latent_attention")`).
- TPU MLA op: `tpu_inference/layers/vllm/custom_ops/mla_attention.py:153` (`register_oot`), `:217-279` (forward), **`:263-265` (indexer invoked → `_topk_indices` discarded, never passed to `self.mla_attn`)**, `:271-277` (4-tensor `mla_attn` call), `:74-109` (W_UK/W_UV absorption setup), `:197` (`topk_indices_buffer` stored, never read).
- TPU MLA impl/backend: `tpu_inference/layers/vllm/backends/flash_attn_mla.py:33` (register), `:44-46` (page size 1024), `:111-208` (impl forward + absorption), `:158-175` (KV quant).
- Shared interface (no sparse arg): `tpu_inference/layers/common/attention_interface.py:463` (`mla_attention()`); v2 kernel `tpu_inference/kernels/mla/v2/kernel.py:1398` (grep-clean of topk/sparse/index).
- Backend selection: `tpu_inference/platforms/tpu_platform.py:114-128` (`get_attn_backend_cls`, `use_mla → FLASH_ATTN_MLA`), `:200-207` (NEW_MODEL_DESIGN + DP-attention gate), `:226-230` (block size).
- KV threading (doc 04): `tpu_inference/layers/vllm/backends/flash_attn.py:177-179, 213` (wrapper-context lookup + write-back), `:185-192` (kv quant scales); MLA cache fork `tpu_inference/runner/kv_cache.py:65-83`, alloc `tpu_inference/runner/kv_cache_manager.py:710-728`.
- register_oot / register_layers mechanism (doc 03): `vllm/model_executor/custom_op.py:84-89` (OOT store keyed on class name); import-side-effect registration `tpu_inference/layers/vllm/__init__.py:14-22`.
- Quant/MoE hooks (doc 08): `process_weights_after_loading` pipeline `tpu_inference/layers/common/process_weights/` (§3); `sharded_quantized_matmul` `tpu_inference/layers/common/linear.py:40`; MoE `apply_monolithic` dispatch `vllm/model_executor/layers/fused_moe/runner/moe_runner.py:454`.

---

*End of the torchax deep-dive series. Docs 01–08 describe what exists; this doc describes
how to extend it to the next DeepSeek generation. The recurring lesson: the torchax route
gives you vLLM's model graph for free **only when the attention layer matches the TPU MLA
contract**. V3.2 matches it; V4 redefines it.*
