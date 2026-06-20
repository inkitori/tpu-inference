# Phase 1a — single-device dense-MLA backbone

Dense-MLA GLM backbone on single-device TPU: the real `mla/v2` kernel **and** the pure-`jnp` fp32 jnp-ref
oracle, triangulated against the HF oracle. (DSA is dense-equivalent at seq ≤ `index_topk`, core §A5, so the
backbone is validated dense first.)

**Precondition:** Phase 0 (HF oracle + incremental decode oracle + tiny & medium configs + harness helpers).
**Core anchors:** §A2, §A5, §B2, §B3, §B4, §B10, §B11, §C1, §C2, §D (all), §E6, §E7, §E8, §F1–§F7,
§G1–§G5, §H2, §H3, §H4, §H5, §H8, §H11, §H12.

## Deliverables
- **Param-ize `DeepseekV2Moe`/`SharedFusedMoe`** (incl. `expert_axis_name`) behavior-preservingly (core §G3)
  — existing DeepSeek construction tests stay green as a gate.
- **The two RoPE bit-for-bit tests** (core §D4): MLA vs `apply_rotary_pos_emb_interleave`; indexer vs
  `apply_rotary_pos_emb`.
- **Both MLA paths on single-device TPU** (core §G5): the pure-`jnp` non-absorbed `GlmMoeDsaAttention` (fp32
  oracle) and the absorbed `mla/v2` kernel (with the standalone no-quant `k_up`/`v_up` split). Thread
  `s_dtype`/`p_same_dtype_as_v`/`two_step_flash_attention` so the kernel runs fp32 (algebra) then bf16 (shipped).
- **Triangulated per-submodule parity** (core §H2, §H5): router, RMSNorm/LayerNorm, dense FFN, MoE experts
  (vs eager `GlmMoeDsaNaiveMoe`), embed/lm_head, MLA.
- **Weight-mapping golden test** (core §F3): load the HF `state_dict` via the converter; assert every param
  maps (no `KeyError`, no leftover random-init), fused `gate_up_proj` split correct, indexer names present on
  **"full"** layers / absent on **"shared"** (gate on `indexer_types[i]`), `layers.78` dropped.
- **(FIX) Greedy `generate()`-vs-HF token-exact gate** (core §H8). On the tiny config, dense-equiv regime
  (seq ≤ `index_topk`, DSA is identity), greedy-generate K≈32 and assert **token-exact** equality to
  `hf_model.generate(do_sample=False, max_new_tokens=K)`. Isolates the decode loop + argmax handoff from DSA;
  the single highest-value forward→generation gate.
- **(FIX) Medium-config real-shape coverage.** Re-run the math + kernel-algebra gates on the **medium** config
  (short seq) so the MLA fast-masking path (`kernel.py:581-602`), the non-degenerate MoE path
  (`fused_moe_gmm.py:478`, `num_experts // ep_size`), and 64-head/model-axis divisibility are exercised — the tiny config reaches none.
- **(FIX) `max_model_len=1M` variant + near-1M RoPE bit-for-bit test** (shared with Phase 1c). Set
  `max_model_len=1048576` on the small/medium model so the MLA kernel compiles its wide `pages_per_seq≈8192`
  program (different gather loop + SMEM-OOM minibatch-split path, `attention_interface.py:139-152`) and the
  RoPE sin/cos table materializes at full width — both are otherwise first-contacted only at real scale. Add
  a RoPE bit-for-bit case at positions **near 1M** (large-angle fp32 precision is untested by small-position
  tests); core §B2 (`max_position_embeddings` real=1,048,576), §D4.
- **(FIX) bf16-floor depth-compounding slope.** Run the jnp-ref in bf16 at medium per-layer dims with depth
  dialed to an **intermediate** L (≈24–40 layers, random weights, drop experts to stay CPU-feasible) to
  characterize how the bf16 floor grows with depth — so the depth-78 floor is *predicted*, not first
  measured on the real checkpoint (core §H3/§H4).

## Acceptance gates (numeric)
- Full dense backbone forward, seq < `index_topk`: **fp32 math gate rtol≈1e-3 vs HF**; bf16 shipped within
  empirical floor (~5e-2…2e-1) + **argmax ≥0.95**.
- Weight-map golden passes (no KeyError / leftover-init; gate split correct; indexer present on "full" /
  absent on "shared"; `layers.78` dropped).
- **Injected-1%-error trips the fp32 math gate**; **injected kv_b-absorption error trips the fp32
  kernel-algebra gate** (core §H3/§H4).
- **(FIX)** greedy `generate()` token-exact vs HF; medium-config math + kernel-algebra gates green;
  near-1M-position RoPE bit-for-bit holds; depth-compounding slope characterized.

## Phase-specific risks & fixtures
- **kv_b absorption bug invisible to the jnp-ref** (only the kernel absorbs) → the dedicated fp32
  kernel-algebra gate + absorption injected-error test (core §H4).
- **Param-izing regresses DeepSeek** (esp. `expert_axis_name`) → construction regression tests (core §G3).
- **bf16 tolerance inflation hides a real ~1% bug** → fp32 math gate distinct from bf16; floor derived
  empirically; argmax is a backstop, not the verdict (core §H3).
