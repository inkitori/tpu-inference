# GLM 5.2 (DSA) on JAX/TPU — Core (shared invariants)

**What this is:** every fact that more than one phase depends on, stated **once**. Phase work-orders
(`phases/phase-*.md`) reference these anchors by id (e.g. "core §B7") instead of restating them.
A phase implementer loads **this file + their one phase file** — nothing else.

**Authority:** architecture facts derived from real source (transformers `glm_moe_dsa`, vLLM
`deepseek_v2.py`, the `claude-deepseek-v4` fork) and verified against **transformers 5.12.1**.
The real `config.json` is authoritative and overrides defaults; **read everything from `hf_config`**.
Re-locate every cited symbol by name — line numbers drift. Provenance / "why" lives in `history.md`.

---

## §A — Architecture invariants

Decoder-only transformer; **every layer is MLA + DSA**; FFN is **MoE** after a few leading dense layers.

- **§A1 — Layer composition.** Each layer: MLA attention + DSA sparse selection; FFN is MoE after the
  first `first_k_dense_replace` dense layers (§B4).
- **§A2 — MLA.** Cache one small **latent** (`kv_lora_rank=512` + shared `qk_rope_head_dim=64` rope key
  = 576 numbers); `kv_b_proj` up-projects to per-head K/V at attention time. Query path compressed
  (`q_a_proj → q_lora_rank → q_a_layernorm → q_b_proj`). RoPE (interleaved, §D) applies only to the
  64-dim rope slice. Split **nope-first** `[qk_nope=192, qk_rope=64]`. Softmax scale
  `qk_head_dim**-0.5 = 256**-0.5` (no mscale under default rope). Only `q_a_layernorm` +
  `kv_a_layernorm`; **no per-head QK-norm**.
- **§A3 — DSA (lightning indexer).** Cheap auxiliary scorer ranks past tokens, selects top-`index_topk`
  =2048; MLA then runs only over the selected set (additive `-inf` mask + causal mask). GLM/V3.2 form:
  per-head dot-product, **`relu` on scores**, weighted head-sum, **no softmax**; `k_norm` = LayerNorm.
  Indexer splits **rope-first** `[qk_rope_head_dim, index_head_dim − qk_rope_head_dim] = [64, 64]` (the
  second chunk is the non-rope "pass" part, **not** a rope slice → any config must keep
  `index_head_dim ≥ qk_rope_head_dim`). Reuses the MLA query LoRA residual
  `q_a_layernorm(q_a_proj(x))`. `indexer_types[i]=="shared"` layers reuse the previous layer's top-k.
- **§A4 — MoE.** 256 expert FFNs (`moe_intermediate_size=2048`) + 1 always-on shared expert; router
  picks **top-8-of-256**. **sigmoid** scoring; `e_score_correction_bias` added **only for selection**
  (gathered weights use bias-free sigmoid scores); `norm_topk_prob` renormalizes; routed output ×
  `routed_scaling_factor=2.5`; shared expert added **unscaled**. First `first_k_dense_replace`=3 layers
  are plain dense SwiGLU.
- **§A5 — THE dense==sparse safety property (drives the whole validation ladder).** For any sequence of
  length ≤ `index_topk` (2048), `topk = min(index_topk, total_len)` selects **every** valid token ⇒ DSA
  and dense attention produce the **identical** result. Sparsity only changes behavior past 2048 tokens.
  Checkable **on TPU at fp32 (≈1e-6)** with no HF dependency — isolates the entire sparse-plumbing bug
  class. *(Referenced by phases 1a, 1c, 2, 3 and the §I9 continuous-batching matrix.)*

---

## §B — Exact config (`GlmMoeDsaConfig`)

`GlmMoeDsaConfig` subclasses `Glm4MoeLiteConfig`. **DSA is gated by `is_v32 = hasattr(config,"index_topk")`**
(the base has no indexer).

### §B2 — Config: default vs real-checkpoint override

⚠️ **Read this as paired columns. The §3-style "defaults" table alone is a trap — the real checkpoint
overrides several values.** Always size from explicit dims, never from `config.head_dim`.

| Field | `GlmMoeDsaConfig()` default | **Real `zai-org/GLM-5.2` override** |
|---|---|---|
| `hidden_size` | 6144 | (same) |
| `num_hidden_layers` | 78 | (same) |
| `n_routed_experts` | 256 | (same) |
| `num_experts_per_tok` | **8** | (same) |
| `n_group` / `topk_group` | **1 / 1 (grouping OFF)** | (same) |
| `routed_scaling_factor` | **2.5** | (same) |
| `num_attention_heads` / `num_key_value_heads` | 64 / 64 | (same) |
| `q_lora_rank` | 2048 | (same) |
| `kv_lora_rank` | 512 | (same) |
| `qk_nope_head_dim` | 192 | (same) |
| `qk_rope_head_dim` | 64 | (same) |
| `v_head_dim` | 256 | (same) |
| `qk_head_dim` (derived) | 256 | (same) |
| `vocab_size` | 154880 | (same) |
| `rms_norm_eps` | 1e-5 | (same) |
| `intermediate_size` (dense FFN) | 12288 | (same) |
| `moe_intermediate_size` | 2048 | (same) |
| **`max_position_embeddings`** | **202752** | **1 048 576** ⚠️ |
| **`head_dim`** | **64** (`=qk_rope_head_dim`) | **64** — `config.json` sets 192 on disk, but `__post_init__` (`configuration_glm_moe_dsa.py:150`) overwrites `head_dim = qk_rope_head_dim`, so the live value is **always 64** ⚠️ |
| **`rope_parameters`** | `{rope_theta: 10000, rope_type: 'default'}` | `{rope_theta: 8 000 000, rope_type: 'default'}` |
| **`index_topk_freq`** | **1** | **4** ⚠️ |
| **`index_skip_topk_offset`** | **2** | **3** ⚠️ |
| scoring | sigmoid (hardcoded; see §B9) | (same) |
| `attention_bias` / `tie_word_embeddings` | False / False | (same) |
| `hidden_act` | "silu" | (same) |

- **§B1 — Source of truth.** Defaults verified via `GlmMoeDsaConfig()`; the real `config.json` overrides
  and is authoritative. Read every value from `hf_config`. **Never rely on `config.head_dim`** — size MLA
  explicitly from `qk_head_dim=256` / `qk_nope=192` / `qk_rope=64`.
- **§B3 — RoPE config.** Key is **`rope_parameters`** (a dict), NOT `rope_scaling`. Both default and real
  use `rope_type='default'` (plain RoPE) ⇒ **YaRN is never enabled, the `mscale²` branch is never taken**
  (and HF 5.12.1 == vLLM on the mscale formula anyway). The only real override here is `rope_theta`
  (8M vs 10k) and `max_position_embeddings` (1M vs 202752).
- **§B4 — Layer schedule: dense vs MoE.** `first_k_dense_replace` (default 3) **IS** a first-class HF
  field (`configuration_glm_moe_dsa.py:128`) and the primary driver: `__post_init__` derives
  `mlp_layer_types = ["dense"]*min(first_k_dense_replace,L) + ["sparse"]*rest` (`:151-154`). Read
  `first_k_dense_replace` directly; accept an explicit `mlp_layer_types` override. vLLM keys on the same field.
- **§B5 — Layer schedule: indexer full vs shared.** `indexer_types` is computed in `__post_init__`
  (`:135-146`) from `index_topk_pattern` (explicit `'F'`/`'S'`) if present, else from `index_topk_freq`
  + `index_skip_topk_offset` via `'full' if (max(i-offset+1,0) % freq)==0 else 'shared'`. **With the
  default `index_topk_freq=1`, every layer is "full"** → any test config **must** set `index_topk_freq>1`
  (or an explicit pattern) to exercise the reuse path. Every attention layer is DSA-sparse; "full"/"shared"
  only controls whether the indexer *scores* are recomputed; a "shared" layer reuses the previous layer's
  `topk_indices`. Real config (`freq=4`/`offset=3`) → "full" at layers 0,1,2 then every 4th (6,10,14,…),
  the rest "shared" (mostly shared).
- **§B6 — DSA indexer fields.** `index_topk=2048`, `index_head_dim=128`, `index_n_heads=32`,
  `index_topk_freq=4`, `index_skip_topk_offset=3`, `indexer_types` (explicit 78-entry array in the real
  config). All from `hf_config`, never hardcoded. *(vLLM's inline comment `index_n_heads = 64` at
  `deepseek_v2.py:622` is **stale** — GLM is 32. Trust `hf_config`, never vLLM comments.)*
- **§B7 — Quantization / fp8-on-disk (the servable checkpoint).** `zai-org/GLM-5.2-FP8` ships
  `torch.float8_e4m3fn` weights, **128×128 block-quantized**, scales named **`weight_scale_inv`**
  (**UE8M0 exponents**), `activation_scheme:"dynamic"`. **Rule: treat UE8M0 scales as uint8 and bitcast —
  never `astype`, never construct an on-device `float8_e8m0fnu`.** The bf16 repo is a separate ~1.5 TB /
  multi-node artifact. ⇒ **An fp8/dequant weight-load path is a Phase-R prerequisite, not optional** (built
  on random weights in Phase 5). vLLM canonical key `kStatic128BlockScale`; block-scale shape
  `[ceil(out/128), ceil(in/128)]`.
- **§B8 — Indexer-cache layer-set.** The indexer (and its key cache) exists **only on "full" layers** —
  5.12.1 nulls `self_attn.indexer` on "shared" layers (`modeling_glm_moe_dsa.py:406-407`:
  `self.indexer = None if self.skip_topk else GlmMoeDsaIndexer(...)`), so "shared" layers cache no keys
  and reuse the prior full layer's `topk_indices` (`:454-457`). Matches vLLM (`deepseek_v2.py:1023`). Real
  config → **"full" = 21 of 78 → ~5.25 GB bf16 KV @ 1M (not 19.5 GB)**. **Drive per-layer allocation off
  `indexer_types` from `hf_config`; never hardcode 78.**
- **§B9 — Dead config keys.** Present on the real checkpoint but the HF code never reads them:
  `rope_interleave=true`, `indexer_rope_interleave=true`, `pretraining_tp=1`, `scoring_func="sigmoid"`.
  Absent from a default `GlmMoeDsaConfig()` (stored as unknown kwargs only when the checkpoint sets them).
  HF hardcodes its RoPE conventions and `router_logits.sigmoid()` (`:591`) regardless. ⚠️
  `indexer_rope_interleave=true` **is load-bearing for vLLM** (→ interleaved indexer) while HF ignores it
  (→ rotate-half) — a real HF-vs-vLLM divergence on the real checkpoint (§D2, §R item 2).

### §B10 — Tiny config (the fast dev fixture)

`hidden=512`, `layers=4` (≥1 dense + ≥1 MoE + ≥1 indexer "full" + ≥1 "shared"), `heads=8`,
`index_topk=64`, `n_routed_experts=8`, `num_experts_per_tok=2`, `page_size=16`, `index_topk_freq=4`,
`index_skip_topk_offset=3`, **default rope**. Keep `kv_lora_rank=512 / qk_rope=64 / qk_nope=192 /
v_head_dim=256 / index_head_dim=128` at **real** values (they auto-pad identically, cost ~nothing).
Two seq regimes: ~32 (≤topk, dense-equiv) and ~128 (>topk=64, true sparsity).

### §B11 — Medium config (real-per-layer-dim — reaches production code paths tiny cannot)

Real per-layer dims, reduced depth/experts: `hidden=6144, num_attention_heads=64, qk_nope=192/qk_rope=64/
v_head_dim=256, kv_lora_rank=512, index_head_dim=128, index_n_heads=32, page_size=128, index_topk=2048,
index_topk_freq=4/index_skip_topk_offset=3`, with `layers=6` (3 dense + 3 sparse), `n_routed_experts=16`,
`moe_intermediate_size=512`, `vocab≈2048` → ~1.7B params / ~7 GB fp32, **CPU-eager-feasible**. Two seq
regimes around `index_topk`: ~2040 (≤topk) and ~3000 (>topk).

**Why it exists:** the tiny config reaches **none** of these production paths — `page_size=16` ⇒
`bkv_sz % 128 != 0` ⇒ the MLA kernel's shipped **fast-masking path** (`kernels/mla/v2/kernel.py:581-602`)
is never tested; MoE expert distribution is **degenerate** (`8//8=1`, `num_experts // ep_size`,
`fused_moe_gmm.py:478` — the production fused backend, §I4); `index_topk
=2048` and 64-head/model-axis divisibility never manifest. **Caveat:** medium **random** weights buy real
**shape** coverage, not real-**weight** top-k selection fidelity (random near-ties mask selection bugs — §H11a).

---

## §C — Norms (canonical eps / class / bias)

**C1 — One wrong eps = a silent math-gate failure. This table is the canonical mapping.**

| param | class | dim | eps | bias |
|---|---|---|---|---|
| `self_attn.q_a_layernorm` | RMSNorm | `q_lora_rank=2048` | **1e-6** (class default, **not** `rms_norm_eps`) | no |
| `self_attn.kv_a_layernorm` | RMSNorm | `kv_lora_rank=512` | **1e-6** (class default) | no |
| `self_attn.indexer.k_norm` | **LayerNorm** | `index_head_dim=128` | **1e-6** | **yes** |
| `input_layernorm`, `post_attention_layernorm` | RMSNorm | `hidden_size=6144` | `rms_norm_eps`=**1e-5** | no |
| `model.norm` (final) | RMSNorm | `hidden_size=6144` | `rms_norm_eps`=**1e-5** | no |

- **§C2 — HF-vs-vLLM disagreement.** vLLM uses `rms_norm_eps` (1e-5) for `q_a_layernorm`/`kv_a_layernorm`;
  **match HF's 1e-6.** `k_norm` is the **only LayerNorm** in an otherwise all-RMSNorm model (the only norm
  with a bias). **No per-head q/k norm.** The three `rms_norm_eps`=1e-5 rows are the only norms reading it.

---

## §D — RoPE conventions (the single highest porting risk)

- **§D1 — The split convention.** **MLA RoPE is INTERLEAVED** (`apply_rotary_pos_emb_interleave`, pairs
  adjacent even/odd dims; `modeling_glm_moe_dsa.py:434`, fn `:302-338`). **Indexer RoPE is rotate-half**
  (`apply_rotary_pos_emb`, `:239` / `:141-162`, `rotate_half`:133).
- **§D2 — MLA agree / indexer diverge.** MLA: HF and vLLM **AGREE** (both interleaved; vLLM
  `is_neox_style=False`). Indexer: they **DIVERGE** on the real checkpoint — HF hardcodes rotate-half and
  ignores `indexer_rope_interleave`; vLLM honors it (`is_neox_style = not indexer_rope_interleave`,
  `deepseek_v2.py:1028`) → vLLM runs the indexer **interleaved**. **v1 follows the HF oracle (rotate-half
  indexer)**; the real-weight adjudication is a Phase-R item (§R item 2).
- **§D3 — Implementation.** MLA: adapt the repo's `DeepseekScalingRotaryEmbedding` (has the correct
  interleave **but applies YaRN unconditionally** — construct with `scaling_factor=1` / `mscale_value=1`
  to collapse the YaRN math to plain RoPE; it is **not** a drop-in plain-RoPE class) **or** port
  `apply_rotary_pos_emb_interleave` standalone. Indexer: base rotate-half `RotaryEmbedding` (`rope.py`).
  **Do not reuse the base rotate-half class for MLA.**
- **§D4 — The two bit-for-bit tests (own anchor; pinned in Phase 1a).** Exact / 1e-6: MLA vs
  `apply_rotary_pos_emb_interleave` (interleaved); indexer vs `apply_rotary_pos_emb` (rotate-half).
- **§D5 — Splits.** Main attention nope-first `[192,64]`; indexer rope-first `[64,64]` (2nd chunk = non-rope
  "pass", not a second rope slice).

---

## §E — DSA / indexer deltas

- **§E1 — KEEP (GLM/V3.2 form).** q-up-proj from the shared q-LoRA residual; **`relu` on scores**; per-head
  `weights_proj × n_heads**-0.5`; head-sum; `top_k`.
- **§E2 — DROP (V4-fork-only).** The GLM indexer is built fresh from the HF oracle, so these fork pieces are
  simply not reimplemented: the **compressor/CSA** (GLM's K is instead direct `k_norm(wk(hidden_states))`
  per-token — the single biggest math change), **SWA**, **mHC sinkhorn**, **APE**. `k_norm` is **LayerNorm**,
  not RMSNorm.
- **§E3 — fp32-pinning (the production bf16 path MUST mirror this).** HF pins `indexer.weights_proj` to fp32
  (`:692`) and runs the **entire** indexer score + weight/head-sum reduction in fp32 regardless of model
  dtype: `scores = relu(q.float()@k.float())*scale` (`:246-247`), `weights = weights_proj(x).float()*
  n_heads**-0.5` (`:250`), `index_scores = weights@scores` (`:251`). The production bf16 indexer **must**
  compute weights_proj / scoring / head-sum in fp32 and downcast only **after** top-k, to hit the 1e-3 fp32
  gate. `e_score_correction_bias` is fp32-strict (`:690`).
- **§E4 — Indexer scale.** `index_head_dim**-0.5 = 128**-0.5 ≈ 0.0884` (`:195`), applied fp32 at `:246`.
- **§E5 — Indexer causality (pre-topk, distinct from the MLA mask).** Add a causal additive mask to the
  indexer's **own** fp32 `index_scores` **before** `top_k`, or the selected set can include future tokens
  (breaks A5). HF: head-0 slice of the 4D causal mask `indexer_mask = attention_mask[:,0,:,:]` (`:445`),
  added `:254-255`; no-mask path builds causality from positions via strict inequality
  `key_positions[None,None,:] > position_ids[:,:,None]` + `masked_fill(...,-inf)` (`:257-259`), then topk
  over the masked scores (`:261-262`). Separate from the additive `-inf` MLA mask built from the selected
  indices (`:467-468`). The jnp ref must replicate the **strict `>`** cutoff on fp32 scores pre-topk.
- **§E6 — Two distinct softmax scales — never conflate.** MLA scale `256**-0.5` (`:398`); indexer scale
  `128**-0.5` (`:195`). The mscale branch (under YaRN) touches **only** the MLA scale; the **indexer scale
  is fixed, no mscale ever**. Both are never under YaRN for this checkpoint (§B3).
- **§E7 — Router / MoE deltas.** Router gate in **fp32**, **sigmoid**; `e_score_correction_bias` added
  before top-k **for selection only**; gathered weights use bias-free sigmoid; `norm_topk_prob` renorm
  `/ (sum + 1e-20)`; `routed_scaling_factor=2.5` applied to the routed-expert **output** (not logits);
  shared expert added **unscaled**.
- **§E8 — PyTorch quirks — nothing to port.** Do **not** carry over DeepSeek-V3's layer-0 residual `.clone()`
  or fp16-overflow rescale; the HF GLM-DSA port already dropped them (plain residual add).

---

## §F — Weight names & loader mapping

- **§F1 — Principle.** Map the **unfused HF names** (vLLM's `fused_qkv_a_proj`/`wk_weights_proj` are internal
  post-load fusions, NOT on disk). Prefix `model.layers.{i}.`.
- **§F2 — Attention.** `self_attn.{q_a_proj, q_a_layernorm, q_b_proj, kv_a_proj_with_mqa, kv_a_layernorm,
  kv_b_proj, o_proj}.weight`; `kv_a_proj_with_mqa` concatenates `[kv_lora_rank(512) | qk_rope_head_dim(64)]`;
  split `kv_b_proj` → `k_up`/`v_up`.
- **§F3 — Indexer** (present **iff** `indexer_types[i] != "shared"` — a "shared" layer has `indexer=None`,
  **no** params on disk; `:406-407`): `self_attn.indexer.{wq_b, wk, weights_proj}.weight` +
  `self_attn.indexer.k_norm.{weight, bias}` (LayerNorm ⇒ has bias).
- **§F4 — MoE (sparse layers).** `mlp.gate.weight` + `mlp.gate.e_score_correction_bias`; **fused**
  `mlp.experts.gate_up_proj` (3-D `[E, 2·moe_inter, hidden]`, **split gate/up**) + `mlp.experts.down_proj`
  (3-D `[E, hidden, moe_inter]`); `mlp.shared_experts.{gate_proj, up_proj, down_proj}.weight` (unfused).
- **§F5 — Dense layers** (0..`first_k_dense_replace`-1): `mlp.{gate_proj, up_proj, down_proj}.weight` (unfused).
- **§F6 — Top-level + drops.** `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`
  (**untied**). Drop `model.layers.78.*` (MTP).
- **§F7 — Converter contract.** Name it `convert_hf_weights`/`t2j_weights` (a raw dtype-preserving `t2j`
  already exists at `utils.py:78` with ~18 importers — build on it, don't shadow it). Transposes every
  linear (JAX `x@kernel (in,out)` vs HF `nn.Linear (out,in)`; `embed_tokens` excepted; `lm_head`
  transposes; split fused `gate_up_proj`). A `maxabs` helper upcasts to fp32 **inside** maxabs; an
  identical-weights checksum is asserted before every parity run.
- **§F8 — Loader machinery.** `JaxAutoWeightsLoader` (`weight_utils.py:870`) via `LoadableWithIterator`,
  brought up **early** (the most expensive prior bug, S1, lives in the multi-device sharded weight-load
  path — §H6). For fp8 **real** weights (Phase R) reuse `Fp8Config` / `get_tpu_quantization_config`
  (defined `layers/jax/quantization/__init__.py:24`; call site `model_loader.py:318`) /
  `MLAEinsum.load_weights` / block dequant + the fork's `deepseek_v4_loader.py` recipe. **Do not route GLM
  around the fp8 loader** (the generic loader upcasts to bf16 and cannot read the fp8-on-disk checkpoint).

---

## §G — Code-reuse map

New file: **`tpu_inference/models/jax/glm_moe_dsa.py`**, reusing DeepSeek modules by import.

- **§G1 — Reuse as-is (import) from `deepseek_v3.py`** (re-locate by symbol): `DeepseekV3MLA` (587),
  `DeepseekV2Moe` (840), `DeepSeekV3Router` (983), `DeepseekV3MLP` (738), `SharedFusedMoe` (815) / `JaxMoE`
  (`moe.py:129`), `DeepseekV3DecoderLayer` (937), `MLAEinsum` (480). Norms `JaxRmsNorm`/`JaxLayerNorm`
  (`norm.py`); linears `JaxEinsum`/`JaxLmHead` (`linear.py`); `JaxEmbed` (`embed.py`); `UnquantizedConfig({})`.
  RoPE: base `RotaryEmbedding` (rotate-half, **indexer**); `DeepseekScalingRotaryEmbedding` (interleaved,
  **MLA**) — or a ported `apply_rotary_pos_emb_interleave`.
- **§G2 — Write GLM-specific deltas only.** (1) config-driven hparams from `hf_config`; (2) `GlmMoeDsaIndexer`
  + top-k (the genuinely new piece, §A3/§E); (3) RoPE wiring (MLA interleaved / indexer rotate-half) + GLM
  router constants; (4) DSA sparse kernel (Phase 2).
- **§G3 — Behavior-preserving param-izing.** `DeepseekV2Moe.__init__` (and the `SharedFusedMoe(...)` block,
  incl. the `expert_axis_name` global `:908`) read ~10 module globals. Add optional kwargs defaulting to the
  current globals so existing DeepSeek construction tests stay green **as a gate**.
- **§G4 — DSA hook points.** jnp-ref path: indexer + additive `-inf` mask into `GlmMoeDsaAttention.__call__`
  (a `dsa` flag toggles dense/sparse). Kernel path: hook is `DeepseekV3MLA.compute_attention` (`:663`)
  immediately before `mla_attention(...)` (`:702`), threading `topk_indices` into the kernel.
- **§G5 — Kernel-precision wrapper.** `mla_attention` (`attention_interface.py:565-579`) calls the MLA kernel
  **without** `s_dtype`/`p_same_dtype_as_v`/`two_step_flash_attention`, so they fall to bf16 defaults
  (`kernel.py:2165,2168`). Thread them through so the kernel can run fp32 (algebra check) then bf16
  (shipped). The call site is already `jax.shard_map`-wrapped (`:584`, `check_vma=False`) → pass the new
  kwargs / sparse kernel **inside** that closure, not re-wrapped. For the absorbed kernel, supply a
  **standalone no-quant `kv_b` split** producing `k_up`/`v_up` (created at weight-load; `MLAEinsum.load_weights`
  asserts `quant_config is not None` at `:525`, so write a standalone split rather than touching it).
- **§G6 — Registration.** `_MODEL_REGISTRY["GlmMoeDsaForCausalLM"]` (`model_loader.py:73-100` inline-import);
  add to `_PP_DISABLED_MODELS` (`:60-61`). The resolver `_get_model_architecture` keys only on
  `config.architectures` strings (not `auto_map`); the real config declares
  `architectures: ["GlmMoeDsaForCausalLM"]`, so the registry insert suffices.
- **§G7 — Oracles & references** (re-locate by symbol). **HF oracle** `modeling_glm_moe_dsa.py` (5.12.1) —
  `GlmMoeDsaIndexer` (relu `:247`, `weights_proj` `:250`, top-k `:261-262`), `GlmMoeDsaAttention`,
  `GlmMoeDsaTopkRouter`, `GlmMoeDsaMoE` (`routed_scaling` `:612`), `apply_rotary_pos_emb` (rotate-half,
  indexer-only), `GlmMoeDsaNaiveMoe` (the **default** eager MoE ref).
  **vLLM (read-only GPU ref)** `models/deepseek_v2.py` + `layers/sparse_attn_indexer.py` (fp8 path); its
  inline comments are stale — trust `hf_config`, not the comments. **Fork** `claude-deepseek-v4`: JAX
  indexer `deepseek_v4_attention.py:447-505` (keep relu `:484`/weights/top_k; **drop compressor** `:475`,
  replace with direct `k_norm(wk(hidden))`); Mosaic sparse kernel `kernels/sparse_attn/kernel.py` (one-hot
  gather, `-1` sentinel, `attn_sink` to drop, already wired via `c2557e26`); parity-harness blueprint
  `tests/models/jax/test_deepseek_v4.py`.
- **§G8 — Real kernels (run on TPU from the start).** MLA `kernels/mla/v2/kernel.py` (auto-pads head dims to
  128-mult; fp32 output special-cased `:1938-1939`; bf16 intermediates default via `s_dtype` `:2165` /
  `p_same_dtype_as_v` `:2168`); microtest pattern `mla_v2_test.py:49-151`. **The ported one-hot gather is
  O(N) total context, not O(`index_topk`)** — `onehot[K,N] @ kv[N,D]` reads every resident KV row
  (`sparse_attn/kernel.py:76,87-90`): a correctness-only / short-context bridge. **Phase 2 measures the O(N)
  cliff; Phase 5 rewrites it to the gen-6 DMA scalar-prefetch idiom (own gate) — recipe + cites live in
  phase-5.**

---

## §H — Testing methodology

- **§H1 — Oracle.** The in-tree `transformers GlmMoeDsaForCausalLM` eager forward (torch, CPU, random
  weights, no download), built via `_from_config(cfg, attn_implementation="eager",
  experts_implementation="eager")` — **`_from_config` is mandatory** (no public `from_config`;
  `experts_implementation` must route through it; eager forces the genuine per-expert `GlmMoeDsaNaiveMoe`
  loop — the default fused `grouped_mm` has no CPU fallback). Deterministic init **including buffers**
  (`e_score_correction_bias`, else the bias-for-selection delta passes vacuously). Model-under-test runs on
  TPU; comparison is a host-side `maxabs` (JAX → host numpy), upcasting both sides to fp32 **inside** maxabs.
  No `jax_enable_x64`.
- **§H2 — The triangulation (the core of the strategy).** Three implementations, three single-delta
  comparisons — never a fused HF-vs-bf16-kernel verdict:
  ```
  HF-eager (CPU, fp32) ──[MATH gate, 1e-3]──► jnp-ref (TPU, fp32) ──[KERNEL gate]──► Pallas kernel (TPU)
       independent oracle                       our answer key                        shipped path
  ```
- **§H3 — MATH gate** (HF-fp32 vs jnp-ref-fp32-on-TPU, rtol≈1e-3): catches every §E math/spec bug. Prove it
  with an **injected-1%-error test** (perturb a scale, assert the fp32 gate catches it). Without an fp32
  reference **on TPU** there is no math gate — the bf16 kernel noise floor (~4e-3/element, compounding over
  depth) is larger than a real ~1% bug.
- **§H4 — KERNEL gate** (jnp-ref vs kernel, on TPU): run the kernel **twice** — (a) **fp32 algebra check** at
  tight tolerance (cast `q`/`ql_nope` to fp32 — the fp32 output path is gated on `q_dtype`, `:1938`, and
  `:370` asserts output dtype == q_dtype — plus `s_dtype=fp32, p_same_dtype_as_v=False`); (b) **bf16 shipped**
  vs jnp-ref-bf16 at the **empirically-measured** bf16 floor (run the jnp-ref in bf16 to derive it). Repo
  precedent (`ragged_paged_attention_kernel_v2_test.py:129-134`, `mla_v2_test.py:405`): bf16 attention
  ~0.1–0.2, MoE/elementwise ~5e-2; full-forward shipped gate ~5e-2…2e-1.
  **Double duty:** the non-absorbed jnp-ref deliberately does **not** absorb, so a bug in the **kv_b
  absorption** shows up only here → add a **second injected-error test perturbing the kv_b absorption split**
  that must trip the fp32 kernel-algebra gate.
- **§H5 — Tiered comparison (each submodule at the layer where it lives).**

  | Check | Tolerance / method |
  |---|---|
  | RoPE — MLA interleaved, indexer rotate-half (§D4) | exact / 1e-6 |
  | Router | top-k indices **exact** (fp32, well-separated) + weights 1e-2 |
  | RMSNorm / LayerNorm, dense FFN, embed/lm_head (→ §C1) | 1e-3 (fp32 math gate) |
  | MLA attention | math 1e-3; kernel-algebra 1e-3 fp32 (+ absorption injected-error); shipped bf16 at floor |
  | MoE experts (vs eager `GlmMoeDsaNaiveMoe`; assert `_experts_implementation=="eager"`) | 1e-2 |
  | Indexer ("full" layers; absent on "shared") | boundary-aware tie-tolerant top-k set equality + `index_scores` 1e-3 fp32 (covers the §E5 pre-topk causal mask) |
  | Dense==sparse (§A5, seq ≤ topk) | **fp32 ≈1e-6 on TPU** — no HF dependency |
  | Full forward | per-layer index-set equality (tie-tolerant) + 1e-3 / bf16 floor + argmax ≥0.95 |

- **§H6 — Multi-device gate (required, not optional) — THE definition.** S1 (uninit-HBM-on-reshard) is a
  *sharding/reshard* phenomenon, **not** host-count → a single-host **v6e-8 (8-chip)** sharded mesh
  reproduces it. Via the **real sharded loader**: **N-device fp32 == 1-device fp32** (compare 1/2/4/8-device,
  value-invariant); exercise `expert_axis_name` sharding. **NaN-poison HBM** before load (catches
  NaN-surfacing variants). **NaN-poison is necessary but NOT sufficient** — the real S1 corruption was
  *coherent finite garbage* (absmax flips, ~20% gsum drift), not NaN. **When the gate trips, LOCALIZE, don't
  bisect:** (1) **per-parameter post-load checksum, 1-dev vs N-dev** — a read-only `shard_map` returning
  reorder-immune `[gsum, sqsum, absmax]` per weight; the first weight whose sqsum/absmax differs pins the
  corrupted weight + sharding axis (this isolated S1 to routed W1/W3 axis-1 leaves while W2 axis-0 was
  byte-identical); (2) **per-stage activation-divergence sweep**; (3) **dump the realized sharded layout**.
  The fix is **host-stacking weights into the sharded layout** (no device reshard); `gmm_v2(zero_initialize=
  True)` was tested and **disproven**. Run the gate against the §I5 sharding-geometry stress fixture, not
  just the tiny config (or it passes vacuously). **A single-device v6e pass is NOT TPU validation.**
- **§H7 — Determinism & ties.** TPU is seeded-reproducible but **not** bit-equal to the torch ref
  (non-associative reductions, order differs, more so multi-device). Top-k ties are a documented hazard.
  Hence: the fp32 math gate uses well-separated scores (exact equality); the indexer set compare is
  **boundary-aware tie-tolerant** (adjudicate boundary swaps by recomputed fp32 scores); `index_scores` is
  gated separately at 1e-3 so a relu/scale bug shows up even when the set survives. Pin the torch ref:
  fixed seeds, `WORLD_SIZE=1`, fp32, eager.
- **§H8 — Decode / generation / sampler gates (beyond forward parity).** Forward `maxabs` / `argmax≥0.95` is
  **not** decode or generation correctness (0.95 top-1 ⇒ ~1-in-20 positions diverge, cascading under
  autoregression). **Decode equivalence:** step-N decode logits == length-N prefill logits (fp32 1e-3 +
  **argmax-exact**) on **peaked** logits. **Greedy generation:** greedy-generate K≈32 ==
  `hf_model.generate(do_sample=False)` **token-exact**. **Sampler unit tests:** on-TPU `sample()`
  (`sampling.py`) vs a numpy ref — temp/top_k/top_p (tie-tolerant at mask boundaries), greedy=argmax,
  `is_greedy` (temp<1e-5) short-circuit, fixed-seed `jax.random.categorical` reproducibility across **both**
  RNG paths (`tpu_runner.py:1580` vs `decode_loop.py::_split_rngs`), + an injected-error test.
  **Sampled-token determinism + batch-invariance:** fixed seed → identical sequence; a greedy request's
  tokens **independent of co-batched random requests** (`do_sampling` is a batch-level static field,
  `sampling_metadata.py:115`). **EOS/stop + logprobs:** `max_tokens`/`min_tokens`/`eos_token_id` halting;
  top-k logprobs + sampled-token rank + prompt_logprobs vs HF.
- **§H9 — Sampler scope (reject + schedule).** The TPU sampler implements **only** temp/top_k/top_p
  (+logprobs); no penalties/min_p/logit_bias/bad_words/allowed_ids/per-request-seed (TPU replaces the whole
  worker, `tpu_platform.py:314-315`). Pin the ordering (temp→top_k→top_p) vs vLLM; the server must
  **reject/ignore** unsupported params (no silent no-op); they are a scheduled production item.
- **§H10 — Static vLLM-divergence gate (no GPU, no weights).** vLLM cannot be a runtime oracle here (CPU
  hard-blocks MLA + sparse attention; the indexer is fp8/CUDA-only). Instead a unit test imports vLLM's
  `deepseek_v2` indexer/RoPE construction and **asserts our choices on the documented divergence points** —
  indexer `is_neox_style`/rotate-half (`:1028`), q_a/kv_a layernorm eps (1e-6 vs vLLM 1e-5),
  mscale-never-on-indexer — **match HF and consciously differ from vLLM**. ⚠️ **The HF oracle is itself a
  documented approximation of the real indexer** — HF's `GlmMoeDsaIndexer` docstring (`:210-213`) states it
  is the "bf16 equivalent" that **skips** the real indexer's **Hadamard transform** (`rotate_activation`)
  and **fp8 scoring** (`fp8_index`), valid only in real arithmetic. vLLM (the real path) does **not** skip
  them → the fp32-rotate-half-no-Hadamard selected set can differ from the fp8-interleaved-Hadamard set near
  the boundary. This is reconciled only at Phase R (§R item 2/4), not by any pre-R gate.
- **§H11 — Methodology guardrails (the meta-bugs that burned the most V4 sessions).**
  - (a) **Random small-weights actively mislead.** `normal*0.02` produces bf16 near-ties that mimic
    structural bugs. Every **decode/selection** fixture must use **peaked/confident logits + an fp32-vs-bf16
    A/B** (discriminator: fp32 relErr ~2.6e-4 vs bf16 ~0.227) **+ cross-process determinism** (2 fresh
    engines). Random-weight medium buys real-**shape** coverage, not real-**weight** selection fidelity. ⇒
    DSA top-k selection is only truly exercised on real well-separated weights (Phase R) **unless** you
    inject synthetic well-separated indexer weights (Phase 2 fix).
  - (b) **Trace-time-baked guards silently never fire** — any slice/clamp-to-real-length must key on a
    **traced** scalar (a warmup full-bucket `L_real==T` compiles the guard away).
  - (c) **CPU losslessness ≠ v6e losslessness** — add **on-device** cast-equivalence checks for any new
    dtype path (fp8); CPU-only is insufficient.
  - (d) **Never gate on liveness/health metrics** — `nan_to_num`, HTTP 200, `completion_tokens`,
    `ends_clean` read healthy on corrupted output. Gate on values **upstream** of the defensive `nan_to_num`,
    via sustained-decode + ragged-batch + cross-process determinism.
- **§H12 — CI / test placement.** All CI runs on **TPU agents** (`.buildkite/pipeline_jax.yml`; no CPU
  runner → no skip-if-no-TPU marker). **Single-device tests** (math gate, weight-map golden, construction,
  greedy-generation, decode-equivalence, sampler unit, static-vLLM-divergence) go in
  `tests/models/jax/test_glm_moe_dsa.py`, auto-collected by `${TPU_VERSION}_test_7_2` (single chip) with no
  pipeline edit; mind the `--fail-under=68` coverage gate in `test_7_3`; broken-by-import → add to
  `--ignore=`. **Multi-device v6e-8 gate** → a dedicated step mirroring `${TPU_VERSION}_test_9` (queue
  `tpu_v6e_8_queue`, gated `if: build.env("NIGHTLY")=="1" || RUN_GLM_DSA_MULTIDEVICE=="1"`); the same step
  carries the Phase-1c/3/4 multi-device gates.
- **§H13 — The four overarching guardrails.** (1) a pure-`jnp` fp32 reference **runnable on TPU** is the
  kernel's answer key (co-locate with the kernel, else a CPU-vs-TPU fp32 reduction-order delta pollutes the
  one clean comparison); (2) staged triangulation — a tight fp32 *math* gate distinct from a loose bf16
  *kernel-noise* gate, never collapsed into one number; (3) multi-device on the real v6e is **required**,
  armed on a geometry that triggers S1; (4) **decode is validated from the start**, not after the giant
  model. Develop on the real v6e running the **real Pallas/Mosaic kernels** — CPU/single-device structurally
  cannot reproduce the S1 reshard class.

---

## §I — Sharding contract (TP / EP / DP / SP)

- **§I1 — Mesh axes (reused by import).** Six logical axes `('data','attn_dp','attn_dp_expert','expert',
  'model','dcp')` (`sharding.py`, `MESH_AXIS_NAMES`); `ShardingAxisName` specs reused unchanged. **Degrees**
  are a runtime knob from `vllm_config.parallel_config` + `additional_config['sharding']['sharding_strategy']`
  (`ShardingStrategy.from_vllm_config`) — set per-run, no code change.
- **§I2 — Inherited MLA contract.** Query `P(None, ATTN_DATA, None)`; kv_cache `P(BATCH, None, ATTN_HEAD)`.
  The kernel runs as **MQA with one kv head** (`num_key_value_heads=1` when `use_mla_kernel`) → the 576-wide
  latent cache stays **REPLICATED** across the model/TP axis (KV memory does **not** shrink with TP);
  **`attn_dp` is the KV-memory lever**. The split `k_up`/`v_up` shard on `ATTN_HEAD`.
- **§I3 — 64-head divisibility.** 64 query heads shard on the model axis → **64 must be divisible by the
  model-axis size** on the 8-chip mesh; verify before Phase 1b. GLM's 64 heads (vs DeepSeek 128) change
  divisibility/attn_dp — re-derive, don't copy DeepSeek's.
- **§I4 — EP vs TP mutual exclusion (load-bearing for the mirroring fidelity).** **Production routed-expert
  sharding is pure even EP** (256 experts ÷ v6e-8 = 32/shard; no empty/uneven shards). The current code
  (`deepseek_v3.py:1140-1144`) gates `use_ep = num_expert_parallelism>1 AND total_TP==1` (`total_TP =
  tp_size*attn_dp_size`) — **EP and TP/attn-DP are mutually exclusive** (`select_moe_backend` → `GMM_EP` /
  `GMM_TP`, `moe/utils.py`). **Both route through the SAME fused backend**: `moe_apply` (`layers/common/moe.py:73`,
  case at `:133`) → `fused_moe_func` (`:141`, from `layers/common/fused_moe_gmm.py`). They differ only in the
  reshard/reduction axis — `reduction_axis = MLP_TENSOR if parallelism=="tp" else ShardingAxisName.EXPERT`
  (`fused_moe_gmm.py:231-232`): **EP** tiles `num_experts // ep_size` (`:478`) and reshards activations via
  `ragged_gather` (`:704-709`) + `psum` over `EXPERT` (`:359`); **TP** does a real `psum` over `MLP_TENSOR`
  (`:359`) — **not** a trivial passthrough. ⚠️ The `sparse_moe.py` `local_permute`/`ragged_all_to_all` path is
  a **DIFFERENT, off-by-default backend** (`MEGABLX_GMM`, `moe.py:190-199`, gated by `USE_UNFUSED_MEGABLOCKS`) —
  **not** the production EP path; never gate on it. ⇒ **the multi-device gate must exercise BOTH modes**
  (Phase 1b): the production pure-EP reshard is not validated by a TP-mode fixture. (`expert_axis_name =
  ATTN_DATA_EXPERT`; on the dev box `dcp=1` so its product coincides with the fused path's `ShardingAxisName.EXPERT`
  at 8 — they diverge only on meshes with `dcp>1`.)
- **§I5 — Stress-fixture caveat.** The fused EP/TP backend (§I4) **silently masks** unowned-expert rows
  (`valid_rows_mask`, `fused_moe_gmm.py:282-287`; zeroed before reduction `:322`) and does **not** fault on
  uneven/empty shards — so an uneven expert count can *hide* corruption (zero-masked) rather than arm S1.
  Therefore arm the EP-mode gate on the **even production geometry** (the real `num_experts // ep_size` EXPERT
  reshard — e.g. 256÷8, or the medium config's even split); the Phase-1b gate must assert `ep_size =
  product(mesh, ShardingAxisName.EXPERT)` divides `n_routed_experts` evenly (the production invariant). A
  separate **TP-mode** fixture (experts not a multiple of the `MLP_TENSOR` axis; idle data-shard rows) stresses
  the GMM_TP reduction tiling. *(The old "empty EP shard faults on the `//` reshape" reasoning was the MEGABLX
  path — off by default, §I4 — and does not apply to the fused production path.)*
- **§I6 — New DSA tensor: indexer projections** (`wq_b`, `wk`, `weights_proj`, `k_norm`). Shard `wq_b`/`wk`
  outputs on `ATTN_HEAD` (per-head, matching the per-head `weights_proj` + head-sum); keep `k_norm`
  (LayerNorm) **replicated**; run through the Phase-1b N-dev==1-dev + NaN-poison gate.
- **§I7 — New DSA tensor: indexer-key KV cache** `[tokens, index_head_dim=128]` (single index head). Needs
  **bespoke TPU machinery — it does NOT flow through the existing hybrid path** (the native-JAX/`flax_nnx`
  branch of `get_kv_cache_spec`, `kv_cache_manager.py:471/484-565`, has **no** indexer awareness; the
  `is_cache_for_ds_v4` / `_hybrid_uniform_page_size_bytes` logic lives only in the vLLM/torchax `else:`
  branch `:566-668`, unreachable here; the hybrid path's only real tenant is non-paged mamba). **Build, don't
  inherit:** a second per-layer spec on the native path; its **own allocation branch** in
  `initialize_kv_cache` (`:795-886`, it cannot ride `create_kv_caches(use_mla=...)`); its **own block table**
  in `MultiGroupBlockTable` (which supports per-group `block_sizes`, `block_table.py:92`); its **own write
  op** (there is **no `slot_mapping`** — the MLA write index `[kv_len−q_len:kv_len]` is computed in-kernel);
  a new per-token-index `AttentionMetadata` field modeled on `mamba_state_indices`; lift the
  `len(block_ids)>1` KV-transfer `NotImplementedError` (`:1202-1206`) as a tracked sub-task. **Allocate on
  "full" layers only** (§B8, 21/78; drive off `indexer_types`). Sharding mirrors the latent cache
  `P(BATCH, None, ATTN_HEAD)`. ⚠️ **S1 watch:** indexer KV on EMPTY attn_dp token shards seeded uninit HBM
  into replicated decode state (fork `deepseek_v4_attention.py:542-547`) → host-stack into the sharded
  layout and keep empty-shard zeroing; do not device-reshard it.
- **§I8 — New DSA tensor: `topk_indices`** `[tokens, index_topk]` int32 (`-1` sentinel) in
  `AttentionMetadata`. Shard on the **token axis** `P(ATTN_DATA, None)` (`ATTN_DATA = ('data','attn_dp',
  'attn_dp_expert')`) so it rides q into the sparse kernel's `shard_map` with no extra gather. Add it to
  `AttentionMetadata`'s `register_dataclass` `data_fields` to be a sharded leaf. **Coordinate space:
  per-request-local (0..seq_len−1)** — the kernel demuxes per `seq_idx` via `cu_kv_lens` + `block_table`, so
  a selected index never crosses a request boundary (global-flat-physical indices are rejected — they need a
  cross-request guarantee the indexer cannot give).
- **§I9 — Scope.** Phases 0–2 inherit the contract on the tiny/medium config and gate degrees per-run on the
  8-chip mesh. The new DSA-tensor specs (I6–I8) are built in Phases 2–4. The continuous-batching matrix is
  Phase 4. Production TP/EP degree tuning + multi-host are Phase R (orchestration pre-validated in Phase Mh).

---

## §J — Hardware / env reality

- **§J1 — Dev box.** A single-host **`v6e-8`** (one process, 8 Trillium chips). `jax.devices()` returns all
  8 — no multi-host init hang. Single-host is enough for the S1 gate (S1 is sharding-triggered, not
  host-count-triggered); 8 chips (vs 4) give the richer attn_dp + expert geometry that *arms* the gate.
- **§J2 — Capacity reality (load-bearing).** GLM 5.2 ≈ **744B params** (256 experts dominate): ≈**1.5 TB
  bf16 / 744 GB fp8** on disk. A v6e-8 = **256 GB** (32 GB/chip in-code `utils.py:181-190`, ~31.25 GiB
  usable). **Weights, not KV, are the binding constraint** → the real model needs a **multi-host slice sized
  for weights + KV** (Phase R) and **never fits the dev box**. The dev box certifies correctness +
  shape-driven capacity math, **not** real-model residency. The servable checkpoint is `zai-org/GLM-5.2-FP8`
  (fp8 on disk, §B7) → an fp8/dequant load path is a Phase-R prerequisite.
- **§J3 — venv.** System py3.10 cannot install the pinned `jax==0.10.1` (needs ≥3.11); reuse the existing
  py3.12 venv. Pins: `jax[tpu]==0.10.1`, `libtpu==0.0.41`, `flax==0.12.4`, **`transformers==5.12.1`** (the
  validated oracle pin), `torch`, `numpy`, `pytest`, `pytest-mock`. The persistent XLA **compilation cache
  is on by default** (`compilation_manager.py`) → edit→run is near-instant after the first cold compile.
  *(Box/runtime/venv-path specifics live in `CLAUDE.md`.)*
- **§J4 — Severable v1 fallback.** If the Mosaic sparse kernel slips, v1 may ship as the **dense backbone**
  (Phase 1) green single + multi-device, with DSA validated in **jnp-ref form only** (indexer math + the §A5
  dense≡sparse equivalence + index-set parity), the sparse *kernel* dropped to v1.1. ⚠️ This is a **correct**
  artifact, not a fully **servable** one — Phases 3–5 (concurrent decode / continuous batching / fp8) all
  assume the real DSA sparse path + indexer-KV cache, so the fallback strands the production-serving goal
  until the kernel lands.
