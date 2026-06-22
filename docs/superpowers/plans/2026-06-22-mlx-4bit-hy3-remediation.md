# MLX 4-bit MoE — Hy3-preview Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MLX 4-bit MoE quant scaffolding (torchax/vLLM path) correct, performant, and arch-general enough to serve `mlx-community/Hy3-preview-4bit` once its base architecture is added to torchax — fixing per-module/mixed-precision handling, w2 HBM bloat, per-step attention dequant, and arch-specific assumptions.

**Architecture:** Six remediation areas land as independent, TDD'd tasks on branch `hy3`. Config/transform/loader changes generalize the scaffolding (per-module bits, 8-bit router dequant-at-load, fail-loud guards, generic expert enumeration). MoE changes push w2 into the existing in-kernel 4-bit `gmm_v2` path. Linear changes move dequant to load time. All tasks are CPU/TPU-kernel unit-testable **now**; anything requiring the model to actually *run* (full Hy3 e2e, prefill perf gate) is explicitly deferred to the user's later base-model effort.

**Tech Stack:** Python, JAX/XLA, Pallas/Mosaic (`gmm_v2`), torchax, vLLM quant-method API, pytest. Target checkpoint: `mlx-community/Hy3-preview-4bit` (base `tencent/Hy3-preview`, `HYV3ForCausalLM`, dense-attention + MoE).

## Global Constraints

- **Target checkpoint facts (verbatim from real `config.json` + safetensors index):** `model_type: hy_v3`; 80 layers; `hidden_size: 4096`; `num_experts: 192`; `num_experts_per_tok: 8`; `num_shared_experts: 1`; `moe_intermediate_size: 1536`; `first_k_dense_replace: 1` (layer 0 is dense `mlp.{gate,up,down}_proj`). Quant block: `group_size: 64`, `bits: 4`, `mode: affine`, weight-only — **with one per-module override repeated for layers 1–79:** `model.layers.{i}.mlp.router.gate = {bits: 8, group_size: 64}`. Tensor names: experts `mlp.switch_mlp.{gate,up,down}_proj` (stacked), shared expert `mlp.shared_mlp.{gate,up,down}_proj` (4-bit), router `mlp.router.gate` (8-bit), `mlp.router.expert_bias` (bf16), `model.embed_tokens` + `lm_head` (4-bit), all norms (bf16).
- **MLX affine dequant is `w = scale * q + bias`** (additive bias, not zero-point). Signed-int4 fold for in-kernel paths: `codes_signed = (mlx_unpack(packed, bits) - 8).astype(jnp.int4)`, `groupbias = mlx_bias + 8.0 * scale`. This is exact: `(q-8)*scale + (bias+8*scale) = q*scale + bias`.
- **Do NOT change the bf16×bf16-after-unpack design** of `gmm_v2`'s affine path (`maybe_quantize_lhs = rhs_groupbias is None`). It is the correct weight-only (W4A16) pattern for memory-bound decode; the int-matmul (`quantize_lhs`) path is incompatible with `group_size=64` affine scales.
- **Base-model support for `HYV3ForCausalLM` is OUT OF SCOPE.** No task may add a model definition or attempt to run the model. Tests use the synthetic fixture `tests/utils/mlx_synthetic.py` (CPU) or the kernel-level fixtures (TPU). Tasks requiring a running model are listed under "Deferred / Blocked".
- **Commit identity:** author `inkitori`. Commit after each green step. Run CPU tests with `JAX_PLATFORMS=cpu python -m pytest <path> -q`; TPU kernel tests run on the v6e host without the platform override.

---

## Task 1: Extend synthetic fixture with 8-bit affine quant + per-module override emission

**Files:**
- Modify: `tests/utils/mlx_synthetic.py` (add `pack_u8`, `quantize_affine_8bit`; add `router_bits_8: bool = False` and `stray_quant: bool = False` options to `build_synthetic_mlx_moe`)
- Test: `tests/utils/test_mlx_synthetic.py`

**Interfaces:**
- Produces: `pack_u8(vals: np.ndarray) -> np.ndarray` (4 uint8 values per uint32, little-endian within the word). `quantize_affine_8bit(w: np.ndarray, group_size: int) -> tuple[packed_uint32, scales_bf16, biases_bf16, golden_bf16]` mirroring the existing 4-bit `_quantize_affine` but with `255`/`clip(0,255)`/`pack_u8`. When `router_bits_8=True`, `build_synthetic_mlx_moe` emits `{layer_prefix}.mlp.gate.{weight,scales,biases}` via `quantize_affine_8bit` and writes `f"model.layers.{L}.mlp.gate": {"bits": 8, "group_size": group_size}` into both `cfg["quantization"]` and `cfg["quantization_config"]`. When `stray_quant=True`, emits an extra DEQUANT-TARGET-shaped partial triplet for the Task 5 guard test.
- Consumes: existing `_quantize_affine`, `add_quant`, `pack_u4` helpers.

- [ ] **Step 1: Write the failing test** (`tests/utils/test_mlx_synthetic.py`)

```python
def test_quantize_affine_8bit_roundtrips_within_tolerance():
    import numpy as np
    from tests.utils.mlx_synthetic import quantize_affine_8bit, pack_u8
    rng = np.random.default_rng(0)
    out, in_, gs = 8, 128, 64
    w = rng.standard_normal((out, in_)).astype(np.float32)
    packed, scales, biases, golden = quantize_affine_8bit(w, gs)
    # 8-bit packs 4 values/uint32:
    assert packed.shape == (out, in_ // 4) and packed.dtype == np.uint32
    assert scales.shape == (out, in_ // gs)
    # golden is the dequant of the stored (bf16-rounded) scale/bias and must
    # track the original within 8-bit affine error:
    assert np.abs(golden.astype(np.float32) - w).max() < 0.1

def test_pack_u8_low_byte_first():
    import numpy as np
    from tests.utils.mlx_synthetic import pack_u8
    vals = np.array([0x10, 0x32, 0x54, 0x76], dtype=np.uint32)  # 4 bytes
    assert pack_u8(vals)[0] == 0x76543210
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/utils/test_mlx_synthetic.py -k "8bit or low_byte" -q`
Expected: FAIL — `cannot import name 'quantize_affine_8bit'`.

- [ ] **Step 3: Write minimal implementation** in `tests/utils/mlx_synthetic.py`

```python
def pack_u8(vals: np.ndarray) -> np.ndarray:
    vals = vals.astype(np.uint32)
    *lead, n = vals.shape
    v = vals.reshape(*lead, n // 4, 4)
    word = np.zeros((*lead, n // 4), np.uint32)
    for k in range(4):
        word |= (v[..., k] & 0xFF) << np.uint32(8 * k)
    return word

def quantize_affine_8bit(w, group_size):
    import ml_dtypes
    out, in_ = w.shape
    g = in_ // group_size
    wg = w.reshape(out, g, group_size)
    lo, hi = wg.min(-1, keepdims=True), wg.max(-1, keepdims=True)
    scale = np.where((hi - lo) == 0, 1.0, (hi - lo) / 255.0)
    bias = lo
    q = np.round((wg - bias) / scale).clip(0, 255).astype(np.uint32)
    s_bf = scale.reshape(out, g).astype(ml_dtypes.bfloat16)
    b_bf = bias.reshape(out, g).astype(ml_dtypes.bfloat16)
    golden = (q.astype(np.float32) * s_bf.astype(np.float32).reshape(out, g, 1)
              + b_bf.astype(np.float32).reshape(out, g, 1)).reshape(out, in_)
    return pack_u8(q.reshape(out, in_)), s_bf, b_bf, golden.astype(ml_dtypes.bfloat16)
```

Then thread `router_bits_8` / `stray_quant` through `build_synthetic_mlx_moe` (emit the 8-bit `mlp.gate` triplet + per-module override; under `stray_quant`, emit a partial dequant-target triplet, see Task 5).

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/utils/test_mlx_synthetic.py -q`
Expected: PASS, output pristine.

- [ ] **Step 5: Commit**

```bash
git add tests/utils/mlx_synthetic.py tests/utils/test_mlx_synthetic.py
git commit -m "test(quant): synthetic fixture emits 8-bit affine quant + per-module override"
```

---

## Task 2: Parse per-module quant overrides in `VllmMLXConfig.from_config`

**Files:**
- Modify: `tpu_inference/layers/vllm/quantization/mlx.py` (`VllmMLXConfig.__init__` ~67-75, `from_config` ~81-85)
- Test: `tests/layers/vllm/quantization/test_mlx_config.py`

**Interfaces:**
- Produces: `VllmMLXConfig.per_module_quant: dict[str, tuple[int, int]]` (full-module-path → `(bits, group_size)`); `VllmMLXConfig.get_module_quant(self, name: str) -> tuple[int, int]` (first substring match wins, else `(self.bits, self.group_size)`). Tasks 3 and 4 consume both.
- Consumes: the raw quant dict passed to `from_config`.

- [ ] **Step 1: Write the failing test**

```python
def test_from_config_parses_per_module_8bit_router_override():
    cfg = VllmMLXConfig.from_config({
        "group_size": 64, "bits": 4,
        "model.layers.1.mlp.router.gate": {"bits": 8, "group_size": 64},
    })
    assert cfg.bits == 4 and cfg.group_size == 64
    assert cfg.per_module_quant["model.layers.1.mlp.router.gate"] == (8, 64)
    assert cfg.get_module_quant("model.layers.1.mlp.router.gate.weight") == (8, 64)
    assert cfg.get_module_quant("model.layers.1.self_attn.q_proj.weight") == (4, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/layers/vllm/quantization/test_mlx_config.py -k per_module -q`
Expected: FAIL — `AttributeError: 'VllmMLXConfig' object has no attribute 'per_module_quant'`.

- [ ] **Step 3: Write minimal implementation**

In `from_config`, build `per_module_quant` from every item whose value is a `dict` containing `"bits"` (skip reserved top-level keys `group_size`, `bits`, `modules_to_not_convert`, `quant_method`, `mode`): `per_module_quant[path] = (int(v["bits"]), int(v.get("group_size", group_size)))`. Pass into `__init__`, store, and add `get_module_quant` iterating `self.per_module_quant.items()` returning the first `path in name` match else `(self.bits, self.group_size)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/layers/vllm/quantization/test_mlx_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tpu_inference/layers/vllm/quantization/mlx.py tests/layers/vllm/quantization/test_mlx_config.py
git commit -m "feat(quant): parse MLX per-module quant overrides (8-bit router)"
```

---

## Task 3: Refine mixed-precision guard — accept 8-bit router, reject what we can't handle

**Files:**
- Modify: `tpu_inference/layers/vllm/quantization/mlx.py` (validator invoked at end of `from_config`)
- Test: `tests/layers/vllm/quantization/test_mlx_config.py`

**Interfaces:**
- Consumes: `per_module_quant` (Task 2).
- Produces: `from_config` raises `ValueError` for unsupported per-module overrides; the allowed-dequant set is the module substrings `{"router.gate", "mlp.gate", "embed_tokens", "lm_head"}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_accepts_8bit_router_only():
    VllmMLXConfig.from_config({"group_size": 64, "bits": 4,
        "model.layers.1.mlp.router.gate": {"bits": 8, "group_size": 64}})  # no raise

def test_rejects_non4bit_expert_weight():
    import pytest
    with pytest.raises(ValueError, match="switch_mlp"):
        VllmMLXConfig.from_config({"group_size": 64, "bits": 4,
            "model.layers.1.mlp.switch_mlp.gate_proj": {"bits": 8, "group_size": 64}})

def test_rejects_override_on_undequanted_tensor():
    import pytest
    with pytest.raises(ValueError, match="q_proj"):
        VllmMLXConfig.from_config({"group_size": 64, "bits": 4,
            "model.layers.1.self_attn.q_proj": {"bits": 8, "group_size": 64}})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/layers/vllm/quantization/test_mlx_config.py -k "rejects or accepts_8bit" -q`
Expected: FAIL — no exception raised for the two reject cases.

- [ ] **Step 3: Write minimal implementation**

After building `per_module_quant`, for each `(path, (b, gs))`:
- If `"switch_mlp" in path or "shared_mlp" in path`: `raise ValueError(f"MLX expert weight {path!r} has a per-module override (bits={b}); the in-kernel gmm path requires uniform 4-bit experts.")`
- Else if no allowed substring in `{"router.gate", "mlp.gate", "embed_tokens", "lm_head"}` matches `path`: `raise ValueError(f"MLX per-module override on {path!r} is unsupported (only router gate / embed / lm_head are dequantized at load).")`

Mirror the `ValueError` idiom of `compressed_tensors_moe.py:53-55` and `compressed_tensors_w4a8_fp8.py:71-74`.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/layers/vllm/quantization/test_mlx_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tpu_inference/layers/vllm/quantization/mlx.py tests/layers/vllm/quantization/test_mlx_config.py
git commit -m "feat(quant): mixed-precision guard accepts 8-bit router, rejects non-4bit experts"
```

---

## Task 4: Load-time dequant of per-module (8-bit router) targets in the transform

**Files:**
- Modify: `tpu_inference/models/vllm/mlx_weight_transform.py` (`transform_mlx_weights` signature + dequant-target matching, lines 78-126)
- Modify: `tpu_inference/models/vllm/vllm_model_loader.py` (~105-108, pass `per_module_quant` to the transform)
- Test: `tests/models/vllm/test_mlx_weight_transform.py`

**Interfaces:**
- Consumes: `VllmMLXConfig.per_module_quant` (Task 2). `_dequant_to_bf16` (existing, already `bits`-generic — `mlx_unpack`/`mlx_dequantize` handle 8-bit).
- Produces: `transform_mlx_weights(..., per_module_quant: dict[str, tuple[int, int]] = {})`; emits a single plain bf16 `<module>.weight` for each per-module target (drops `.scales`/`.biases`), dequanting with that module's own `(bits, group_size)`.

- [ ] **Step 1: Write the failing test**

```python
def test_router_gate_8bit_dequantized_to_bf16():
    import numpy as np, torch
    from tests.utils.mlx_synthetic import quantize_affine_8bit
    from tpu_inference.models.vllm.mlx_weight_transform import transform_mlx_weights
    out, in_, gs = 8, 64, 64
    w = np.random.default_rng(1).standard_normal((out, in_)).astype(np.float32)
    pw, ps, pb, golden = quantize_affine_8bit(w, gs)
    name = "model.layers.1.mlp.router.gate"
    stream = [(f"{name}.weight", torch.from_numpy(pw.astype(np.uint32))),
              (f"{name}.scales", torch.from_numpy(np.asarray(ps).astype(np.float32)).to(torch.bfloat16)),
              (f"{name}.biases", torch.from_numpy(np.asarray(pb).astype(np.float32)).to(torch.bfloat16))]
    out_map = dict(transform_mlx_weights(stream, group_size=64, bits=4, num_experts=1,
                                         per_module_quant={name: (8, 64)}))
    assert set(out_map) == {f"{name}.weight"}
    w_out = out_map[f"{name}.weight"]
    assert w_out.dtype == torch.bfloat16 and type(w_out) is torch.Tensor
    np.testing.assert_allclose(w_out.float().numpy(),
                               np.asarray(golden).astype(np.float32), atol=2e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/models/vllm/test_mlx_weight_transform.py -k router_gate -q`
Expected: FAIL — `transform_mlx_weights() got an unexpected keyword argument 'per_module_quant'`.

- [ ] **Step 3: Write minimal implementation**

Add `per_module_quant: dict[str, tuple[int, int]] = {}` param. Generalize the dequant-target branch: a name is a dequant target if it matches a `_DEQUANT_PREFIXES` prefix (4-bit, embed/lm_head) **or** if `base = next(p for p in per_module_quant if name.startswith(p) and name[len(p):] in (".weight", ".scales", ".biases"))`. Buffer the triplet in `pending` keyed by base; when complete, call `_dequant_to_bf16(weight, scales, biases, group_size=gs, bits=bits)` using that target's `(bits, gs)` — `(group_size, bits)` for embed/lm_head, `per_module_quant[base]` for overrides. In `vllm_model_loader.py`, build the config via `VllmMLXConfig.from_config(quant_dict)` and pass `per_module_quant=cfg.per_module_quant` into the transform call.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/models/vllm/test_mlx_weight_transform.py -q`
Expected: PASS (existing pass-through + embed/lm_head tests still green).

- [ ] **Step 5: Commit**

```bash
git add tpu_inference/models/vllm/mlx_weight_transform.py tpu_inference/models/vllm/vllm_model_loader.py tests/models/vllm/test_mlx_weight_transform.py
git commit -m "feat(quant): load-time dequant of 8-bit MLX router gate to bf16"
```

---

## Task 5: Fail-loud triplet-integrity guard for dequant-at-load targets

**Files:**
- Modify: `tpu_inference/models/vllm/mlx_weight_transform.py` (leftover-`pending` loop, lines 122-126)
- Test: `tests/models/vllm/test_mlx_weight_transform.py`

**Design note (read before implementing):** The transform passes ALL quantized linear weights (attention q/k/v/o, dense MLP, shared MLP) through individually — they are legitimately consumed packed by `VllmMLXLinearMethod`. The transform therefore *cannot* structurally distinguish an unexpected quantized tensor from a valid pass-through one, so this guard does NOT try to. Instead it guards the cases it CAN prove wrong: a **dequant-at-load target** (embed/lm_head/per-module override) that arrives with a PARTIAL triplet — some but not all of `{weight, scales, biases}` — would otherwise leak a packed `uint32` weight downstream and silently corrupt. The broader "is every quantized tensor actually consumed by a quant Linear?" check is naturally enforced at base-model integration: vLLM's weight loader raises on an unexpected `.scales`/`.biases` param for an unquantized module. A stronger post-load packed-dtype assertion is listed under Deferred.

**Interfaces:** behavioral only — a dequant target with a partial triplet raises `ValueError` naming the module. The legitimate embed/lm_head "plain bf16, no scales/biases at all" case (weight only, zero companions) still passes through unchanged.

- [ ] **Step 1: Write the failing test**

```python
def test_partial_router_triplet_raises():
    import torch, pytest
    from tpu_inference.models.vllm.mlx_weight_transform import transform_mlx_weights
    name = "model.layers.1.mlp.router.gate"
    # weight + scales but NO biases -> partial dequant target
    stream = [(f"{name}.weight", torch.zeros(8, 1, dtype=torch.uint32)),
              (f"{name}.scales", torch.zeros(8, 1, dtype=torch.bfloat16))]
    with pytest.raises(ValueError, match="router.gate"):
        list(transform_mlx_weights(stream, group_size=64, bits=4, num_experts=1,
                                   per_module_quant={name: (8, 64)}))

def test_plain_bf16_head_passes_through():
    import torch
    from tpu_inference.models.vllm.mlx_weight_transform import transform_mlx_weights
    # lm_head shipped as a single plain weight (tie/unquantized) -> allowed
    stream = [("lm_head.weight", torch.zeros(8, 4, dtype=torch.bfloat16))]
    out = dict(transform_mlx_weights(stream, group_size=64, bits=4, num_experts=1))
    assert out["lm_head.weight"].dtype == torch.bfloat16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/models/vllm/test_mlx_weight_transform.py -k "partial_router or plain_bf16" -q`
Expected: FAIL — partial triplet currently passes through silently (no raise).

- [ ] **Step 3: Write minimal implementation**

In the leftover-`pending` loop: for each buffered `base`, if `slot.keys()` is a non-empty PROPER subset of `{"weight", "scales", "biases"}` **and** `slot.keys() != {"weight"}` (a lone `.weight` with no companions is the legitimate plain-bf16 head case), `raise ValueError(f"MLX dequant target {base!r} arrived with a partial quant triplet {sorted(slot)}; refusing to leak a packed weight downstream.")`. Otherwise (lone `.weight`) pass through unchanged as today.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/models/vllm/test_mlx_weight_transform.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tpu_inference/models/vllm/mlx_weight_transform.py tests/models/vllm/test_mlx_weight_transform.py
git commit -m "feat(quant): fail-loud on partial MLX dequant-target triplet in transform"
```

---

## Task 6: Generic MoE expert-count resolution in the loader

**Files:**
- Modify: `tpu_inference/models/vllm/vllm_model_loader.py` (~108, `num_experts=hf_config.num_experts`)
- Test: `tests/models/vllm/test_vllm_model_loader_experts.py` (new)

**Interfaces:**
- Produces: module-level `_resolve_num_experts(hf_config) -> int` checking `num_experts → n_routed_experts → num_local_experts`.

- [ ] **Step 1: Write the failing test**

```python
from tpu_inference.models.vllm.vllm_model_loader import _resolve_num_experts
class _C: pass
def test_resolve_prefers_num_experts():
    c = _C(); c.num_experts = 192
    assert _resolve_num_experts(c) == 192
def test_resolve_falls_back_to_n_routed_experts():
    c = _C(); c.n_routed_experts = 192
    assert _resolve_num_experts(c) == 192
def test_resolve_falls_back_to_num_local_experts():
    c = _C(); c.num_local_experts = 8
    assert _resolve_num_experts(c) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/models/vllm/test_vllm_model_loader_experts.py -q`
Expected: FAIL — `cannot import name '_resolve_num_experts'`.

- [ ] **Step 3: Write minimal implementation**

```python
def _resolve_num_experts(hf_config) -> int:
    for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
        v = getattr(hf_config, attr, None)
        if v is not None:
            return int(v)
    raise ValueError(
        "Could not resolve MoE expert count from hf_config "
        "(tried num_experts / n_routed_experts / num_local_experts).")
```

Replace the `num_experts=hf_config.num_experts` use at line 108 with `num_experts=_resolve_num_experts(hf_config)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/models/vllm/test_vllm_model_loader_experts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tpu_inference/models/vllm/vllm_model_loader.py tests/models/vllm/test_vllm_model_loader_experts.py
git commit -m "feat(loader): generic MoE expert-count resolution"
```

---

## Task 7: w2 (down_proj) experts 4-bit in-kernel (mirror w13)

**Key finding (decides scope):** the `gmm_v2` down-path ALREADY supports per-group `w2_groupbias` end-to-end — `gmm_wrapper` forwards it for both gmm1 and gmm2 (`fused_moe_gmm.py:213-244`), `process_moe_weights` reshapes it (`moe_weights.py:295-298`), and `shard_moe_weights` shards it via `_w2_grouped_p_spec` (`moe_weights.py:499,522`). **No kernel or sharding-plumbing change is needed** — only `mlx.py` weight-prep. (Confirmed against the shipping `compressed_tensors_moe_w4a8.py:234-244` template, which keeps both w13 and w2 int4 in-kernel under TP.)

**HBM saved:** w2 bf16 = `E·H·I·2` bytes. Hy3 per MoE layer: `192·4096·1536·2 ≈ 2.42 GB` → int4 + scale/groupbias ≈ `0.60 + 0.15 = 0.75 GB`. **≈ 1.67 GB saved per layer × 79 MoE layers ≈ 132 GB** total.

**Files:**
- Modify: `tpu_inference/layers/vllm/quantization/mlx.py` (`VllmMLXMoEMethod`: `_process` ~442-473, store-back ~489, `apply_monolithic` ~503-511, docstrings ~339-345/406-420/480-483)
- Test: `tests/layers/vllm/quantization/test_mlx_moe_method.py`

**Interfaces:**
- Consumes: loaded `w2_weight` uint32 packed `[E, H, I//pf]`, `w2_scales`/`w2_biases` bf16 `[E, H, I//gs]`.
- Produces: `FusedMoEWeights.w2_weight` = signed `jnp.int4` codes; `w2_weight_scale` f32 `[E, H, I//gs]`; `w2_groupbias` f32 `[E, H, I//gs]` = `w2_bias + 8*w2_scale`. `layer.w2_weight_scale` and `layer.w2_groupbias` registered as `Parameter`s.

- [ ] **Step 1: Write the failing test** — extend `test_process_weights_dequant_matches_golden_experts`, replacing the existing w2 assertion block (the current bf16 check) with the signed-int4 reconstruction check (reuse the existing `_reconstruct_w13_from_stored` helper; it is w2-agnostic):

```python
    w2_codes = np.asarray(jax_view(layer.w2_weight))
    w2_scale = np.asarray(jax_view(layer.w2_weight_scale))
    w2_gbias = np.asarray(jax_view(layer.w2_groupbias))
    assert w2_codes.min() < 0 and w2_codes.min() >= -8 and w2_codes.max() <= 7
    assert w2_gbias.shape == w2_scale.shape and np.abs(w2_gbias).max() > 0
    recon_w2 = _reconstruct_w13_from_stored(w2_codes, w2_scale, w2_gbias)
    np.testing.assert_allclose(recon_w2, ref_w2, atol=2e-2, rtol=2e-2)
    # teeth: wrong sign-fold and dropped bias must BOTH break the match
    assert not np.allclose(_reconstruct_w13_from_stored(w2_codes + 8, w2_scale, w2_gbias), ref_w2, atol=2e-2, rtol=2e-2)
    assert not np.allclose(_reconstruct_w13_from_stored(w2_codes, w2_scale, np.zeros_like(w2_gbias)), ref_w2, atol=2e-2, rtol=2e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/layers/vllm/quantization/test_mlx_moe_method.py -k dequant -q`
Expected: FAIL — `layer.w2_groupbias` does not exist / `w2_weight` is bf16.

- [ ] **Step 3: Write minimal implementation** — in `_process`, replace the w2 bf16 dequant (~455-456, 463-465) with the same fold w13 uses:

```python
            w2_codes = (mlx_unpack(w2q, bits) - 8).astype(jnp.int4)
            w2_scale = w2s.astype(jnp.float32)
            w2_groupbias = (w2b.astype(jnp.float32) + 8.0 * w2_scale)
```

Set `w2_weight=w2_codes`, `w2_weight_scale=w2_scale`, `w2_groupbias=w2_groupbias` on `FusedMoEWeights`. Add store-back after ~489: `layer.w2_weight_scale` and `layer.w2_groupbias` `Parameter`s. In `apply_monolithic`, pass `w2_weight=jax_view(layer.w2_weight).astype(jnp.int4)`, `w2_weight_scale=jax_view(layer.w2_weight_scale)`, `w2_groupbias=jax_view(layer.w2_groupbias)`. Update docstrings: drop the "w2 stays bf16 / cannot shard at tp=8" claim; record the real rule `(I/tp) % group_size == 0` (Hy3: 1536/8=192, 192%64=0 ✓). Keep the `FUSED_MOE` backend guard `w2_groupbias is None` (`moe_weights.py:314`) — Hy3 resolves to GMM_TP/GMM_EP.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/layers/vllm/quantization/test_mlx_moe_method.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tpu_inference/layers/vllm/quantization/mlx.py tests/layers/vllm/quantization/test_mlx_moe_method.py
git commit -m "feat(quant): MLX MoE w2 in-kernel 4-bit (int4 + per-group scale+groupbias via gmm_v2)"
```

---

## Task 8: Prove w2 groupbias sharding at tp>1 (exact-match) — TPU

**Files:**
- Modify: `tests/models/vllm/test_qwen3_moe_mlx_int4_e2e.py` (add a sharded-w2 variant)

**Caveat (must satisfy to actually exercise sharded w2):** at tp>1 the w2 contraction dim is split, so w2's per-shard block count is `(moe_inter / tp) // group_size`. To force `num_blocks > 1` per shard (so `_w2_grouped_p_spec` returns `P(None, MLP_TENSOR)` and groupbias actually shards rather than replicates), the synthetic model needs `moe_inter = 256` at `tp=2` (`256/2=128`, `128/64=2`). `build_synthetic_mlx_moe`/`build_bf16_reference_moe` accept `moe_inter` (only head dims are fixed).

**Interfaces:** consumes Task 7's stored `w2_weight`/`w2_weight_scale`/`w2_groupbias`. No production change.

- [ ] **Step 1: Write the new test** (runs on TPU; skips when chips < tp)

```python
@pytest.mark.parametrize("tensor_parallel_size", [2])
def test_synthetic_mlx_moe_w2_sharded_matches_bf16(tensor_parallel_size):
    # moe_inter=256 -> per-shard w2 num_blocks = (256/2)/64 = 2 -> groupbias SHARDS
    # body identical to test_synthetic_mlx_moe_logits_match_bf16_reference but with
    # build_synthetic_mlx_moe(..., moe_inter=256, group_size=64) and the bf16 ref built
    # at the same moe_inter; assert exact token match.
    ...
```

- [ ] **Step 2: Run to verify it passes after Task 7** (it must PASS — it is a regression gate, not a red→green for new prod code)

Run: `python -m pytest tests/models/vllm/test_qwen3_moe_mlx_int4_e2e.py -k w2_sharded -q`
Expected: PASS on TPU. **Teeth check:** locally mutate the w2 groupbias spec to `P()` (force replicate) and confirm the test FAILS, proving it exercises sharded groupbias; then revert.

- [ ] **Step 3: Commit**

```bash
git add tests/models/vllm/test_qwen3_moe_mlx_int4_e2e.py
git commit -m "test(quant): prove w2 groupbias sharding at tp>1 (sharded num_blocks>1, exact-match)"
```

---

## Task 9: Materialize attention/linear weights to bf16 at load (kill per-step dequant)

**Bug:** `VllmMLXLinearMethod.apply` (`mlx.py:256-260`) calls `mlx_dequantize` inside the jit'd forward; `process_weights_after_loading` (`mlx.py:177-247`) keeps weights uint32-packed. Weights are traced jit args (`vllm_model_wrapper.py:340,368`), so XLA re-dequantizes EVERY decode step. Fix mirrors the MoE w2 load-time dequant and AWQ's `process_weights_after_loading`: dequant ONCE at load, store one bf16 `layer.weight`, simplify `apply`.

**HBM tradeoff (document in the class docstring):** attention/linear weights become bf16 (~4× their packed size). For Hy3-preview attention ≈ 6B params → ~+9 GB across 80 layers vs 4-bit. Correctness + idiomatic (matches w2/AWQ); the int4-fuse alternative that preserves 4-bit is the optional follow-up below.

**Files:**
- Modify: `tpu_inference/layers/vllm/quantization/mlx.py` (`VllmMLXLinearMethod.process_weights_after_loading` 177-247, `apply` 249-272; `create_weights` 137-175 unchanged)
- Test: `tests/layers/vllm/quantization/test_mlx_linear_method.py`

**Interfaces:**
- Consumes: `layer.weight` uint32 `[out, in//8]`, `layer.scales`/`layer.biases` bf16 `[out, in//64]`; `mlx_dequantize(packed, scales, biases, group_size, bits) -> bf16 [out, in]`; `t2j(t, use_dlpack=False)`.
- Produces: single `layer.weight = Parameter(bf16 [out, in])`; `scales`/`biases` removed via `delattr`; `apply` reads only `layer.weight`.

- [ ] **Step 1: Write the failing test** (reuse existing `_make_linear_config`, `_load_and_process`, `_apply_np`, `_quantize_affine`)

```python
def test_process_materializes_bf16_weight_and_apply_has_no_dequant(monkeypatch):
    import numpy as np
    import tpu_inference.layers.vllm.quantization.mlx as mlxmod
    out = 96
    method, layer = _make_linear_config(out)  # existing helper builds method+layer
    rng = np.random.default_rng(0)
    w = rng.standard_normal((out, IN_FEATURES)).astype(np.float32)
    packed, scales, biases, golden = _quantize_affine(w, GROUP_SIZE, force_negative_scale=True)
    _load_and_process(method, layer, packed, scales, biases)

    assert layer.weight.dtype == torch.bfloat16
    assert tuple(layer.weight.shape) == (out, IN_FEATURES)
    assert not hasattr(layer, "scales") and not hasattr(layer, "biases")
    w_jax = np.asarray(jax_view(layer.weight)).astype(np.float32)
    np.testing.assert_allclose(w_jax, golden.astype(np.float32), atol=2e-2, rtol=2e-2)

    calls = []
    monkeypatch.setattr(mlxmod, "mlx_dequantize", lambda *a, **k: calls.append(1))
    x = rng.standard_normal((4, IN_FEATURES)).astype(np.float32)
    y = _apply_np(method, layer, x)
    assert calls == []  # apply must NOT dequant per step
    np.testing.assert_allclose(y, x @ golden.astype(np.float32).T, atol=2e-2, rtol=2e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/layers/vllm/quantization/test_mlx_linear_method.py -k materializes -q`
Expected: FAIL — `scales` still present; `mlx_dequantize` called in apply.

- [ ] **Step 3: Write minimal implementation**

In `process_weights_after_loading`: DELETE the word/group shard-alignment guard (193-220) — bf16 shards freely so it is unnecessary. Keep the `fuse_matmuls or len(output_sizes)==1` assert (226-228) and reorder logic (229-240). Inside the `@jax.jit _process`: `t2j` weight/scales/biases, reorder as today, then `w = mlx_dequantize(arr_w, arr_s, arr_b, group_size, bits)`; `jax.device_put(w, NamedSharding(mesh, wsh))`; store `layer.weight = Parameter(torch_view(...), requires_grad=False)`; `delattr(layer, "scales"); delattr(layer, "biases")`. In `apply`: drop the `mlx_dequantize` call; `weight = jax_view(layer.weight)`; keep the `jnp.einsum("bd,fd->bf", x_jax, weight)`, bias add, and `slice_sharded_tensor_for_concatenation` + concat (267-272) unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/layers/vllm/quantization/test_mlx_linear_method.py -q`
Expected: PASS. (`test_rowparallel_input_dim_sharding_dequant_consistency` calls `mlx_dequantize` directly, not via the method, so it stays green.)

- [ ] **Step 5: Commit**

```bash
git add tpu_inference/layers/vllm/quantization/mlx.py tests/layers/vllm/quantization/test_mlx_linear_method.py
git commit -m "fix(quant): MLX linear dequant once at load -> bf16 weight (kill per-step XLA dequant)"
```

---

## Deferred / Blocked (require base-model support or are optional optimizations)

- **OPTIONAL — int4-fuse for linear (HBM-preserving alternative to Task 9).** Keep `layer.weight` int4 + per-group scale and fuse dequant into the matmul via the repo's `blockwise_quantized_matmul_kernel` (`tpu_inference/kernels/quantized_matmul/blockwise_kernel.py`) through `sharded_quantized_matmul` (`tpu_inference/layers/common/linear.py:40`), mirroring the w13 MoE in-kernel pattern. Preserves 4-bit attention weights (saves the ~+9 GB from Task 9). Defer until the bf16 fix is validated and HBM pressure is shown to matter.
- **DEFERRED (blocked on base support) — prefill perf gate.** The bf16×bf16 w13 design is correct; the only un-gated risk is compute-bound PREFILL (int4→bf16 unpack is pure VPU overhead). Gate via `examples/tpu_profiling.py` (prefill `--input-len 8192 --output-len 1`; decode `--input-len 1 --output-len N --batch-size 256`) + `vllm bench serve`, comparing Stage-2 4-bit vs a bf16 baseline. Cannot run until `HYV3ForCausalLM` executes on torchax.
- **DEFERRED (blocked) — full Hy3-preview e2e numeric validation** (real checkpoint, real arch, tp=8). Needs the base model. The synthetic tp=2 exact-match gates (Task 8 for w2, existing test for w13) shard the same axes identically and are the available proof until then.
- **DEFERRED (blocked) — post-load packed-dtype assertion.** After the model is built and weights loaded, assert no parameter remains uint32-packed unless it is a registered MLX quantized param — the strongest "fail loud on unanticipated quantized tensor" check. Requires the running model; complements Task 5's transform-level triplet guard.
- **VERIFY at integration — `shared_mlp` and dense layer-0 `mlp.{gate,up,down}_proj`** must route onto quantized Linears in the future HYV3 model definition (they pass through the transform packed and rely on `VllmMLXLinearMethod`). Cannot be tested until the arch exists; vLLM's loader will raise on a mismatch.

---

## Self-Review

- **Spec coverage:** per-module/mixed-precision → Tasks 2,3,4,5; w2 HBM → Tasks 7,8; per-step attention dequant → Task 9; arch generalization (expert count) → Task 6; perf → Deferred gate; fixture support → Task 1. All five originally-proposed fixes + the scaffolding-readiness items are represented.
- **Corrected from the cluster drafts:** Task 5 was reframed from "reject any unhandled packed tensor" (which would wrongly reject the real quantized `q/k/v/o_proj` that must pass through) to a triplet-integrity guard, with the broader check documented as loader-enforced + a deferred post-load assertion.
- **Type/name consistency:** `per_module_quant: dict[str, tuple[int,int]]` and `get_module_quant` (Task 2) are consumed identically by Tasks 3,4. `w2_weight`/`w2_weight_scale`/`w2_groupbias` (Task 7) match the `FusedMoEWeights` fields the existing `gmm_wrapper` already forwards. `_resolve_num_experts` (Task 6) names match its test.
- **Ordering/deps:** 1→(2→3,4→5),6 independent, 7→8, 9 independent. Tasks 6, 7, 9 can run in parallel during execution.
