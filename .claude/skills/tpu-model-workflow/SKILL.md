---
name: tpu-model-workflow
description: >-
  Workflow + testing/benchmarking discipline for adding model, quantization, or
  speculative-decoding support — or doing performance optimization — on
  tpu-inference's torchax/vLLM path (MODEL_IMPL_TYPE=vllm). Use this whenever the
  task is bringing up a new model architecture, adding a quant format, wiring
  spec-decode, or speeding up a kernel/layer/model on TPU — even if the user
  never says "unit test" or "benchmark". Defines the kernel→layer→synthetic-model
  correctness ladder, the reference oracles, synthetic-config rules, EP+TP mesh
  fidelity, and the isolated-kernel/layer benchmarking method that lets you
  iterate to near-certainty WITHOUT loading the full checkpoint in the loop.
---

# Model / quant / spec-decode support + perf work

The goal of this workflow: iterate entirely in a cheap unit-test environment that
makes the full `vllm serve` model *very likely* correct and faster — without ever
loading a full real checkpoint in the loop. Passing the ladder is near-certainty,
not proof; the real serve run stays the user's manual final check.

**Scope.** Only `MODEL_IMPL_TYPE=vllm` (torchax). Always on TPU — no CPU/sim fallback.

**Golden rule.** Iterate on synthetic small configs + isolated kernels/layers; **never load a full real checkpoint in the loop.** The real `vllm serve` run is the user's manual final check. Report honestly: tests make the full model *very likely* correct/faster — not proven.

**Test at every rung: kernel → layer → synthetic full model.** Both correctness and perf.

## Correctness
**Reference oracles** (best → last resort):
1. **vLLM-eager** — torchax runs vLLM's *own* model code, so run that same model native/eager on CPU and compare. Same definition → no cross-impl convention bugs, transitively HF-validated. Swapped ops (Pallas attention, quant kernels, padded head_dim) → tolerance; rest tight.
2. **Naive/numpy** — for those TPU-only kernels (no eager equivalent); transparently-correct ref, e.g. `tests/layers/jax/moe/test_moe.py`.
3. **Self-consistency (exact greedy match)** — *only* for quant/perf/refactor of an already-trusted model (bf16 vs int4); proves a change preserved behavior, says nothing about a new model. Ref: `tests/models/vllm/test_qwen3_moe_mlx_int4_e2e.py`.
4. **HF-eager** — broken at full/layer by a rope mismatch (#1604); avoid.

### Correctness ladder
1. **Kernel** vs oracle, per-dtype tol. (`tests/layers/jax/`)
2. **Layer/block** (attn, MoE/MLP, decoder layer) vs oracle on shared weights.
3. **Synthetic e2e via `get_model()`** (loads vLLM model → torchax `functional_call` → `shard_model_to_tpu` → `step_fun` → `compute_logits`) — never instantiate the class. New model: match **vLLM-eager** (proves assembly + weight-load + sharding). Quant/perf change: exact greedy match vs trusted bf16.
4. **Weight-load completeness** — every key consumed, every param populated.
5. **Decode + batch>1**, varied seq lens.

## Synthetic config
Keep everything affecting shape/sharding/numerics real (`hidden_size`, head counts, `head_dim`, `intermediate_size`, `vocab_size`, rope, norm eps, MoE experts/top_k, tie-embeddings); shrink **only** `num_hidden_layers` → 2–4. Real weight shapes, fixed seed. Quant tests must use the real quantized checkpoint path (e.g. MLX auto-switches `load_format` to `tpu_streaming_loader`) — dummy weights don't exercise the transform.

## Environment fidelity
- **Sharding:** build a **multi-device mesh over `jax.devices()`** with real **EP (expert) + TP (model)** specs — the conftest `mesh` fixture is single-device, which hides sharding bugs.
- **Dtype:** bf16 is the serving target. fp32 ok for early correctness; switch to bf16 before "done."
- Always enter via `get_model()`.

## Performance (same kernel → layer → model ladder)
1. **Kernel isolated** — weights as args via `device_put(NamedSharding)`, chained in `lax.scan` to amortize the per-call dispatch floor; warmup + median around `block_until_ready`; subtract the measured dispatch floor.
2. **Layer/block isolated** — same method; catches fusion/comm cost the kernel sum misses.
3. **Full-model projection** from **measured layer time**: `tpot ≈ Σ layer_us + fixed_overhead` (not `op_us × num_layers`).
- All benchmarks on the real EP+TP mesh. **Record a baseline first** (a claim with no pre-change baseline is invalid). Split **device vs host** time. Re-run correctness after — a faster wrong kernel is a regression.

## Anti-patterns
Full checkpoint in the loop · instantiating the model class instead of `get_model()` · single-device sharding/timing · only prefill / only batch=1 · benchmarking a kernel but never the layer · projecting from `op × layers` instead of measured layer time · perf claim with no baseline · claiming correctness/perf from synthetic tests alone.

## Done
Kernel + layer green · synthetic e2e via `get_model` = exact greedy match vs bf16 self-ref · weight-load complete · decode+batch>1 · bf16 · perf: kernel+layer speedup on EP+TP mesh vs recorded baseline + projection holds · TPU freed (`~/tpu-tooling/free-tpu.sh`).
