# Hy3-preview-4bit bring-up (torchax/vLLM, TPU v6e-8) — design

Date: 2026-06-22

## Goal / acceptance

Greedy-decode **coherent, on-topic** text from `mlx-community/Hy3-preview-4bit`
via the torchax path (`MODEL_IMPL_TYPE=vllm`), tp=8, weights streamed from the
existing gcsfuse mount. Acceptance = eyeball coherence on a few greedy prompts.

## Context

**Model** (`HYV3ForCausalLM`, `model_type=hy_v3`, Tencent Hunyuan 3.0 preview):
80 layers, hidden 4096, 64 Q / 8 KV heads, head_dim 128 (q_proj→8192), vocab
120832, RMSNorm eps 1e-5, RoPE theta 11158840 (no scaling), qk_norm (per-head
RMSNorm on q/k before RoPE), **no** attention bias, untied embeddings.
MoE: 192 experts, top-8, **1 shared expert**, moe_inter 1536, layer 0 dense
(`first_k_dense_replace=1`). Router: **sigmoid + expert bias (selection-only) +
route_norm + router_scaling_factor=2.826** (DeepSeek-V3 style). MTP head
(`num_nextn_predict_layers=1`) absent from checkpoint → droppable.

**Quant** (MLX): bits=4, group_size=64, mode=affine (`w = scale*q + bias`).
Per-module override: `model.layers.{1..79}.mlp.router.gate` → **8-bit**.
embed/lm_head dequantized to bf16 at load (existing transform).

**Checkpoint access:** gcsfuse mount already present (read-only) at
`/tmp/gcs/bucket`; VM and bucket both in us-east5-b. Use
`HF_HOME=/tmp/gcs/bucket/vllm`; model id `mlx-community/Hy3-preview-4bit`.
Local disk has only ~43 GB free → **no local download**; stream from the mount.
Checkpoint weight names already MLX-stacked (`switch_mlp.{gate,up,down}_proj`,
`mlp.router.gate`, `mlp.router.expert_bias`, `shared_mlp`) — matches what
`mlx_weight_transform.py` un-stacks today.

**What already works:** vLLM (vendored at `/home/enyouki/vllm`, v0.20.1-dev)
ships a complete, correct `HYV3ForCausalLM` (`hy_v3.py`). The torchax path
defers to vLLM's registry with **no tpu-inference allowlist gate** — no model
code to write. tpu-inference's MLX MoE scaffolding (gmm_v2 + groupbias, w13
in-kernel int4, per-expert un-stacking) exists from the Qwen3 work. TPU gating
already honors sigmoid scoring, route_norm, and the shared expert (the latter
added by vLLM's `MoERunner` outside the quant kernel).

## Work items

### 1. MoE routing correctness (the must-do; new vs. remediation plan)
On TPU the MoE gating is recomputed in `fused_moe_gmm.py`, NOT by vLLM. It
currently selects top-k on the **unbiased** sigmoid and never applies the
scaling factor → wrong experts + wrong magnitude → incoherent output.

- Plumb `e_score_correction_bias` (per-expert tensor) and `routed_scaling_factor`
  (float) from the `FusedMoE` layer through `layers/vllm/interface/moe.py` →
  `layers/common/moe.py::moe_apply` → `layers/common/fused_moe_gmm.py`
  (`fused_moe_func` signature + gating). Both attrs already exist on the vLLM
  `FusedMoE` object.
- Gating math (replicate vLLM `grouped_topk`):
  `s = sigmoid(logits); sel = s + bias; idx = topk(sel, 8);
   w = s.gather(idx); w = w / (w.sum(-1) + 1e-20); w = w * routed_scaling_factor`.
- Move `expert_bias` parameter onto the JAX device.
- Default `bias=None`, `scaling=1.0` → exact no-op for Qwen3 (softmax, no bias).
- **CPU unit test:** numerical parity of the new gating vs. vLLM's grouped_topk
  reference on random logits/bias/scaling.

### 2. 8-bit router gate (mixed precision)
- `VllmMLXConfig.from_config`: parse per-module `quantization` overrides; accept
  the `mlp.router.gate` 8-bit affine entry, reject other unsupported overrides
  loudly (fail-fast).
- Dequant the tiny `[192×4096]` gate to bf16 at load in `mlx_weight_transform.py`
  (load-time, not in-forward). Triplet-integrity guard (codes+scale+bias present).

### 3. w2 (down_proj) experts 4-bit in-kernel (mandatory for HBM)
- Mirror the w13 path: keep packed int4 in HBM; dequant inside `gmm2` via
  `gmm_v2` per-group scale + groupbias (offset fold `groupbias = bias + 8*scale`).
- Hy3 down dims (in=1536, gs=64 → 24 groups) shard cleanly at tp=8.
- **Test:** extend the existing w13 exact-match sharding regression to w2 (TPU).

### 4. Generic expert-count + shapes
- Resolve `num_experts=192` and the shared expert generically in the loader
  (drop Qwen3-specific 128 assumptions).
- Verify / relax shape & sharding asserts for Hy3 dims (hidden 4096, moe_inter
  1536, gs 64); fail loudly on a real mismatch rather than silently mis-loading.

### 5. Run + coherence validation
- Launch vLLM offline `LLM` with `HF_HOME=/tmp/gcs/bucket/vllm`,
  `MODEL_IMPL_TYPE=vllm`, `tensor_parallel_size=8`, greedy sampling, a handful of
  prompts formatted with the model's chat template.
- Acceptance: fluent, on-topic completions. Bisect any garbage in order
  routing → quant → loading.

## Memory plan
Experts 4-bit (w13 + w2 in-kernel), attention/embed/dense bf16 → ~20.8 GB/chip
of 32 GB. Fits. Attention per-step in-forward dequant left as-is (perf only, not
a correctness blocker); revisit only if it OOMs or is too slow.

## Testing strategy
- CPU unit tests: routing-parity vs grouped_topk; MLX unpack/dequant (extend
  existing).
- TPU exact-match: w2 groupbias sharding at tp>1 (extend w13 test).
- E2E: greedy coherence run is the acceptance gate.

## Out of scope
MTP/speculative head; attention int4-in-kernel fusion; prefill throughput tuning;
numerical parity vs an external reference (coherence eyeball is the bar);
non-greedy sampling quality.

## Relation to existing docs
Supersedes the framing of `docs/superpowers/plans/2026-06-22-mlx-4bit-hy3-remediation.md`,
which excluded the model definition AND missed the routing math (bias + scaling).
Reuses its quant/memory tasks (per-module override parse, load-time dequant,
generic expert-count, w2 4-bit). Its tp>1 proof becomes work item 3's test; its
attention-bf16-materialization (Task 9) is deferred as optional perf.
