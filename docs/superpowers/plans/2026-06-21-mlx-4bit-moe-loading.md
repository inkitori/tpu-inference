# MLX 4-bit MoE Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load `mlx-community/Qwen3-30B-A3B-4bit` on tpu-inference, keeping weights 4-bit in HBM and dequantizing on the fly.

**Architecture:** Quantization is threaded as a `quant_method` strategy object (mirroring the existing FP8 path), not as scattered branches. The model arch (`qwen3_moe.py`) and the MoE layer (`JaxMoE`) are untouched. Phase 1 keeps weights packed in HBM and dequantizes to bf16 in XLA at apply-time (correct, validates loader + format). Phase 2 keeps the expert matmul int4-in-kernel via `gmm_v2`, adding MLX's affine bias term (computed as a separate cheap `dot_general` outside the kernel — "Approach A").

**Tech Stack:** JAX, Flax NNX, Pallas (TPU kernels), vLLM config plumbing, safetensors, MLX format.

## Global Constraints

- **MLX affine format:** each quantized linear = 3 tensors — `…weight` uint32 (packed 8×4-bit/word, element 0 = low nibble), `…scales` bf16 `[out, in/group_size]`, `…biases` bf16 `[out, in/group_size]`. Dequant per group along the **input** dim: `w[o,i] = scales[o, i//gs]·q[o,i] + biases[o, i//gs]`. Scales may be **negative**; bias is a real float offset (not an integer zero-point).
- **group_size = 64, bits = 4** for the base repo (read from `config.json["quantization"]`). Pack factor = `32/bits`; `in = weight.shape[-1]·pack_factor`; `num_groups = in/group_size == scales.shape[-1]`.
- **Per-module overrides:** `config["quantization"]` may carry dotted keys (e.g. `…mlp.gate → {bits:8}`, the Instruct variant). Read bit-width **per module**; never hardcode.
- **Experts are STACKED** under `switch_mlp` (leading dim = num_experts = 128), separate gate/up/down (NOT fused). No per-expert numbered keys exist.
- **Quantized:** all linears incl. `mlp.gate` (router), `embed_tokens`, `lm_head`. **bf16 (not quantized):** all norms incl. `q_norm`/`k_norm`. Dispatch on suffix: a `.scales` sibling ⇒ quantized; lone `.weight` ⇒ plain.
- **Sharding:** packed weights pack along the input dim → shard the contraction dim only in multiples of 64; clean axes are the output dim + expert-parallel.
- Mirror existing patterns. The FP8 path is the precedent for every integration point; read it before writing the int4 sibling.

---

### Task 1: MLX dequant primitives (pure functions)

Self-contained JAX functions for unpacking + affine dequant. No codebase deps. Foundation for every later task.

**Files:**
- Modify: `tpu_inference/layers/common/quantization/__init__.py` (add functions next to the existing `awq_u32_unpack_u4` / `dequantize_tensor`)
- Test: `tests/layers/common/quantization/test_mlx_dequant.py`

**Interfaces:**
- Produces:
  - `mlx_unpack(packed: jax.Array, bits: int) -> jax.Array` — `[…, n/(32//bits)]` uint32 → `[…, n]` int32 values in `[0, 2**bits)`.
  - `mlx_dequantize(packed, scales, biases, group_size: int, bits: int) -> jax.Array` — returns bf16 `[…, out, in]` (leading expert dim optional).

- [ ] **Step 1: Write the failing test**

```python
# tests/layers/common/quantization/test_mlx_dequant.py
import jax.numpy as jnp
import numpy as np
import pytest
from tpu_inference.layers.common.quantization import mlx_unpack, mlx_dequantize


def _pack_u4(vals_row):
    # vals_row: list[int] length multiple of 8 -> list[uint32], element 0 = low nibble
    words = []
    for i in range(0, len(vals_row), 8):
        w = 0
        for k in range(8):
            w |= (vals_row[i + k] & 0xF) << (4 * k)
        words.append(w)
    return words


def test_unpack_low_nibble_first():
    # word with nibbles 0..7 in order -> 0x76543210
    packed = jnp.asarray([[0x76543210]], dtype=jnp.uint32)
    out = mlx_unpack(packed, bits=4)
    np.testing.assert_array_equal(np.asarray(out), [[0, 1, 2, 3, 4, 5, 6, 7]])


def test_dequantize_affine_with_negative_scale_and_bias():
    # one output row, in=64 (one group). q = i % 16.
    q = [(i % 16) for i in range(64)]
    packed = jnp.asarray([_pack_u4(q)], dtype=jnp.uint32)           # [1, 8]
    scales = jnp.asarray([[-0.5]], dtype=jnp.bfloat16)              # [1, 1]
    biases = jnp.asarray([[3.25]], dtype=jnp.bfloat16)             # [1, 1]
    w = mlx_dequantize(packed, scales, biases, group_size=64, bits=4)
    expected = (np.asarray(q, dtype=np.float32) * -0.5) + 3.25
    np.testing.assert_allclose(np.asarray(w, dtype=np.float32), expected[None, :], atol=0.05)


def test_dequantize_two_groups_distinct_scales():
    q = [(i % 16) for i in range(128)]                              # in=128 -> 2 groups
    packed = jnp.asarray([_pack_u4(q)], dtype=jnp.uint32)           # [1, 16]
    scales = jnp.asarray([[0.5, 2.0]], dtype=jnp.bfloat16)          # [1, 2]
    biases = jnp.asarray([[0.0, 1.0]], dtype=jnp.bfloat16)         # [1, 2]
    w = mlx_dequantize(packed, scales, biases, group_size=64, bits=4)
    q_np = np.asarray(q, dtype=np.float32)
    expected = np.concatenate([q_np[:64] * 0.5 + 0.0, q_np[64:] * 2.0 + 1.0])
    np.testing.assert_allclose(np.asarray(w, dtype=np.float32), expected[None, :], atol=0.05)


def test_dequantize_stacked_experts_leading_dim():
    q = [(i % 16) for i in range(64)]
    packed = jnp.asarray([[_pack_u4(q)], [_pack_u4(q)]], dtype=jnp.uint32)  # [E=2, out=1, 8]
    scales = jnp.asarray([[[1.0]], [[-1.0]]], dtype=jnp.bfloat16)           # [2, 1, 1]
    biases = jnp.asarray([[[0.0]], [[5.0]]], dtype=jnp.bfloat16)            # [2, 1, 1]
    w = mlx_dequantize(packed, scales, biases, group_size=64, bits=4)
    assert w.shape == (2, 1, 64)
    q_np = np.asarray(q, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(w[0], dtype=np.float32), q_np[None, :], atol=0.05)
    np.testing.assert_allclose(np.asarray(w[1], dtype=np.float32), (q_np * -1.0 + 5.0)[None, :], atol=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/layers/common/quantization/test_mlx_dequant.py -v`
Expected: FAIL with `ImportError: cannot import name 'mlx_unpack'`

- [ ] **Step 3: Write minimal implementation**

```python
# tpu_inference/layers/common/quantization/__init__.py  (append)
import jax
import jax.numpy as jnp


def mlx_unpack(packed: jax.Array, bits: int) -> jax.Array:
    """uint32 packed -> int32 values, MLX order (element 0 = low bits)."""
    per_word = 32 // bits
    mask = (1 << bits) - 1
    shifts = (jnp.arange(per_word, dtype=jnp.uint32) * bits)
    vals = (packed[..., None] >> shifts) & jnp.uint32(mask)        # [..., n_words, per_word]
    vals = vals.reshape(*packed.shape[:-1], -1)                    # [..., n]  (index = word*per_word + k)
    return vals.astype(jnp.int32)


def mlx_dequantize(packed: jax.Array, scales: jax.Array, biases: jax.Array,
                   group_size: int, bits: int) -> jax.Array:
    """MLX affine dequant. packed [..., out, n_words]; scales/biases [..., out, n_groups]."""
    q = mlx_unpack(packed, bits)                                   # [..., out, in]
    scales_e = jnp.repeat(scales, group_size, axis=-1)            # [..., out, in]
    biases_e = jnp.repeat(biases, group_size, axis=-1)
    w = q.astype(jnp.float32) * scales_e.astype(jnp.float32) + biases_e.astype(jnp.float32)
    return w.astype(jnp.bfloat16)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/layers/common/quantization/test_mlx_dequant.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Cross-check against MLX reference (optional, only if `mlx` importable)**

```python
# append to the test file
def test_matches_mlx_core_if_available():
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 128)).astype(np.float32)
    wq, s, b = mx.quantize(mx.array(w), group_size=64, bits=4)
    ref = np.asarray(mx.dequantize(wq, s, b, group_size=64, bits=4))
    ours = np.asarray(mlx_dequantize(
        jnp.asarray(np.asarray(wq).view(np.uint32)),
        jnp.asarray(np.asarray(s)), jnp.asarray(np.asarray(b)),
        group_size=64, bits=4), dtype=np.float32)
    np.testing.assert_allclose(ours, ref, atol=0.05)
```

Run: `pytest tests/layers/common/quantization/test_mlx_dequant.py -v` (skips cleanly if `mlx` absent)

- [ ] **Step 6: Commit**

```bash
git add tpu_inference/layers/common/quantization/__init__.py tests/layers/common/quantization/test_mlx_dequant.py
git commit -m "feat(quant): MLX affine unpack + dequantize primitives"
```

---

### Task 2: Synthetic MLX-format MoE checkpoint builder

A test util that writes a tiny Qwen3-MoE-shaped MLX checkpoint (few layers, 8 experts, group-64) with **deliberately negative scales + nonzero bias**, plus its bf16 golden reference. Enables loader + kernel iteration without the 17 GB download.

**Files:**
- Create: `tests/utils/mlx_synthetic.py`
- Test: `tests/utils/test_mlx_synthetic.py`

**Interfaces:**
- Produces:
  - `build_synthetic_mlx_moe(dir: Path, *, layers=2, experts=8, hidden=128, moe_inter=64, group_size=64, seed=0) -> dict` — writes `config.json` + `model.safetensors` (MLX-format keys), returns `{"golden": {param_name: np.ndarray bf16-dequantized}}`.
  - `pack_u4(vals: np.ndarray) -> np.ndarray` — `[..., n]` ints → `[..., n/8]` uint32, MLX order.

- [ ] **Step 1: Write the failing test**

```python
# tests/utils/test_mlx_synthetic.py
import numpy as np
from pathlib import Path
from tests.utils.mlx_synthetic import build_synthetic_mlx_moe, pack_u4
from tpu_inference.layers.common.quantization import mlx_dequantize
import jax.numpy as jnp
import json
from safetensors.numpy import load_file


def test_pack_roundtrip():
    vals = np.arange(64).reshape(1, 64) % 16
    packed = pack_u4(vals)                       # [1, 8]
    assert packed.shape == (1, 8) and packed.dtype == np.uint32


def test_build_writes_mlx_keys_and_negative_scales(tmp_path: Path):
    info = build_synthetic_mlx_moe(tmp_path, layers=1, experts=8, hidden=128, moe_inter=64)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["quantization"]["group_size"] == 64 and cfg["quantization"]["bits"] == 4
    assert cfg["architectures"] == ["Qwen3MoeForCausalLM"]
    st = load_file(tmp_path / "model.safetensors")
    # stacked experts, leading dim = 8
    gk = "model.layers.0.mlp.switch_mlp.gate_proj"
    assert st[gk + ".weight"].dtype == np.uint32
    assert st[gk + ".weight"].shape[0] == 8
    assert (st[gk + ".scales"] < 0).any()        # adversarial: some negative scales


def test_golden_matches_dequant(tmp_path: Path):
    info = build_synthetic_mlx_moe(tmp_path, layers=1, experts=8, hidden=128, moe_inter=64)
    st = load_file(tmp_path / "model.safetensors")
    gk = "model.layers.0.mlp.switch_mlp.gate_proj"
    w = mlx_dequantize(jnp.asarray(st[gk + ".weight"]),
                       jnp.asarray(st[gk + ".scales"]),
                       jnp.asarray(st[gk + ".biases"]),
                       group_size=64, bits=4)
    np.testing.assert_allclose(np.asarray(w, dtype=np.float32),
                               info["golden"][gk].astype(np.float32), atol=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/utils/test_mlx_synthetic.py -v`
Expected: FAIL with `ModuleNotFoundError: tests.utils.mlx_synthetic`

- [ ] **Step 3: Write minimal implementation**

```python
# tests/utils/mlx_synthetic.py
import json
import numpy as np
from pathlib import Path
from safetensors.numpy import save_file


def pack_u4(vals: np.ndarray) -> np.ndarray:
    vals = vals.astype(np.uint32)
    *lead, n = vals.shape
    v = vals.reshape(*lead, n // 8, 8)
    word = np.zeros((*lead, n // 8), dtype=np.uint32)
    for k in range(8):
        word |= (v[..., k] & 0xF) << np.uint32(4 * k)
    return word


def _quantize_affine(w: np.ndarray, group_size: int, force_negative_scale: bool):
    # w: [out, in] -> packed uint32 [out, in/8], scales/biases bf16 [out, in/gs], plus bf16 golden
    out, in_ = w.shape
    g = in_ // group_size
    wg = w.reshape(out, g, group_size)
    lo = wg.min(-1, keepdims=True)
    hi = wg.max(-1, keepdims=True)
    scale = (hi - lo) / 15.0
    scale = np.where(scale == 0, 1.0, scale)
    bias = lo
    if force_negative_scale:               # adversarial: flip sign on half the groups
        flip = (np.arange(g) % 2 == 0)
        scale = scale.copy()
        scale[:, flip, :] *= -1.0
        bias = (hi if force_negative_scale else lo)  # keep q in range when scale flips
        bias = np.where(flip[None, :, None], hi, lo)
    q = np.round((wg - bias) / scale).clip(0, 15).astype(np.uint32)
    golden = (q.astype(np.float32) * scale + bias).reshape(out, in_)
    packed = pack_u4(q.reshape(out, in_))
    s = scale.reshape(out, g).astype(np.float32).astype("bfloat16") if False else scale.reshape(out, g)
    # store scales/biases as bf16 via ml_dtypes
    import ml_dtypes
    s_bf = scale.reshape(out, g).astype(ml_dtypes.bfloat16)
    b_bf = bias.reshape(out, g).astype(ml_dtypes.bfloat16)
    return packed, s_bf, b_bf, golden.astype(ml_dtypes.bfloat16)


def build_synthetic_mlx_moe(dir: Path, *, layers=2, experts=8, hidden=128,
                            moe_inter=64, group_size=64, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    tensors, golden = {}, {}

    def add_quant(name, w, negate):
        p, s, b, gold = _quantize_affine(w, group_size, negate)
        tensors[name + ".weight"] = p
        tensors[name + ".scales"] = s
        tensors[name + ".biases"] = b
        golden[name] = np.asarray(gold).astype(np.float32)

    def add_quant_stacked(name, w_stack, negate):  # w_stack: [E, out, in]
        ps, ss, bs, gs = [], [], [], []
        for e in range(w_stack.shape[0]):
            p, s, b, gold = _quantize_affine(w_stack[e], group_size, negate)
            ps.append(p); ss.append(s); bs.append(b); gs.append(np.asarray(gold).astype(np.float32))
        tensors[name + ".weight"] = np.stack(ps)
        tensors[name + ".scales"] = np.stack(ss)
        tensors[name + ".biases"] = np.stack(bs)
        golden[name] = np.stack(gs)

    import ml_dtypes
    for L in range(layers):
        pre = f"model.layers.{L}"
        # attention (dense linears)
        add_quant(f"{pre}.self_attn.q_proj", rng.standard_normal((hidden, hidden)).astype(np.float32), L == 0)
        add_quant(f"{pre}.self_attn.k_proj", rng.standard_normal((hidden, hidden)).astype(np.float32), False)
        add_quant(f"{pre}.self_attn.v_proj", rng.standard_normal((hidden, hidden)).astype(np.float32), False)
        add_quant(f"{pre}.self_attn.o_proj", rng.standard_normal((hidden, hidden)).astype(np.float32), False)
        # norms (bf16, not quantized)
        for nm in ["input_layernorm", "post_attention_layernorm"]:
            tensors[f"{pre}.{nm}.weight"] = np.ones(hidden, dtype=ml_dtypes.bfloat16)
        # router gate (quantized) + stacked experts
        add_quant(f"{pre}.mlp.gate", rng.standard_normal((experts, hidden)).astype(np.float32), False)
        add_quant_stacked(f"{pre}.mlp.switch_mlp.gate_proj",
                          rng.standard_normal((experts, moe_inter, hidden)).astype(np.float32), L == 0)
        add_quant_stacked(f"{pre}.mlp.switch_mlp.up_proj",
                          rng.standard_normal((experts, moe_inter, hidden)).astype(np.float32), False)
        add_quant_stacked(f"{pre}.mlp.switch_mlp.down_proj",
                          rng.standard_normal((experts, hidden, moe_inter)).astype(np.float32), False)

    tensors["model.norm.weight"] = np.ones(hidden, dtype=ml_dtypes.bfloat16)

    save_file(tensors, str(dir / "model.safetensors"), metadata={"format": "mlx"})
    cfg = {
        "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
        "hidden_size": hidden, "num_hidden_layers": layers, "num_experts": experts,
        "num_experts_per_tok": min(2, experts), "moe_intermediate_size": moe_inter,
        "vocab_size": 256, "tie_word_embeddings": False,
        "quantization": {"group_size": group_size, "bits": 4},
        "quantization_config": {"group_size": group_size, "bits": 4},
    }
    (dir / "config.json").write_text(json.dumps(cfg))
    return {"golden": golden}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/utils/test_mlx_synthetic.py -v`
Expected: PASS (3 tests). If `ml_dtypes` missing: `pip install ml_dtypes` (already a JAX dep).

- [ ] **Step 5: Commit**

```bash
git add tests/utils/mlx_synthetic.py tests/utils/test_mlx_synthetic.py
git commit -m "test: synthetic MLX-format MoE checkpoint builder with adversarial scales"
```

---

### Task 3: MLX quant-config detection + dispatch registration

Teach `get_tpu_quantization_config` to recognize the MLX `"quantization"` block and return an `Int4Config`. Add `int4` to the method dict and the TPU allow-list.

**Files:**
- Modify: `tpu_inference/layers/jax/quantization/__init__.py` (the `method_to_config` dict ~lines 31-34 and `get_tpu_quantization_config` ~lines 24-46)
- Create: `tpu_inference/layers/jax/quantization/int4.py` (only `Int4Config` in this task; methods in Tasks 4-5)
- Test: `tests/layers/jax/quantization/test_int4_config.py`

**Interfaces:**
- Consumes: the FP8 precedent — read `Fp8Config` (`tpu_inference/layers/jax/quantization/fp8.py:631-687`) and `get_tpu_quantization_config` before writing.
- Produces:
  - `Int4Config.from_hf_quant_config(hf_quant_config: dict) -> Int4Config` with fields `group_size:int`, `bits:int`, `overrides: dict[str,dict]` (per-module bit/group), and `bits_for(module_name: str) -> tuple[int,int]` returning `(bits, group_size)`.
  - `get_tpu_quantization_config(vllm_config)` returns `Int4Config` when the HF config carries an MLX `"quantization"` block.

- [ ] **Step 1: Read the precedent.** Read `fp8.py:631-687` (`Fp8Config`) and `__init__.py:24-46`. Note how `method_to_config` maps and how `hf_config.quantization_config` is read.

- [ ] **Step 2: Write the failing test**

```python
# tests/layers/jax/quantization/test_int4_config.py
from tpu_inference.layers.jax.quantization.int4 import Int4Config


def test_parses_base_repo_block():
    cfg = Int4Config.from_hf_quant_config({"group_size": 64, "bits": 4})
    assert cfg.group_size == 64 and cfg.bits == 4
    assert cfg.bits_for("model.layers.0.mlp.switch_mlp.gate_proj") == (4, 64)


def test_per_module_override_8bit_router():
    raw = {"group_size": 64, "bits": 4,
           "model.layers.0.mlp.gate": {"group_size": 64, "bits": 8}}
    cfg = Int4Config.from_hf_quant_config(raw)
    assert cfg.bits_for("model.layers.0.mlp.gate") == (8, 64)
    assert cfg.bits_for("model.layers.0.mlp.switch_mlp.gate_proj") == (4, 64)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/layers/jax/quantization/test_int4_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `Int4Config`**

```python
# tpu_inference/layers/jax/quantization/int4.py
from dataclasses import dataclass, field


@dataclass
class Int4Config:
    group_size: int = 64
    bits: int = 4
    overrides: dict = field(default_factory=dict)  # module-name -> {"bits", "group_size"}

    @classmethod
    def from_hf_quant_config(cls, q: dict) -> "Int4Config":
        gs = int(q.get("group_size", 64))
        bits = int(q.get("bits", 4))
        overrides = {k: v for k, v in q.items()
                     if isinstance(v, dict) and "bits" in v}
        return cls(group_size=gs, bits=bits, overrides=overrides)

    def bits_for(self, module_name: str) -> tuple[int, int]:
        for k, v in self.overrides.items():
            if module_name.endswith(k) or k in module_name:
                return int(v["bits"]), int(v.get("group_size", self.group_size))
        return self.bits, self.group_size
```

- [ ] **Step 5: Run config test to verify it passes**

Run: `pytest tests/layers/jax/quantization/test_int4_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Wire detection + dispatch** in `__init__.py`. Add `"int4": Int4Config` to `method_to_config`. In `get_tpu_quantization_config`, before the NotImplementedError, detect the MLX block:

```python
# inside get_tpu_quantization_config, after reading hf_quant_config:
#   hg_quant_config = getattr(model_config.hf_config, "quantization_config", {}) or {}
# MLX nests under "quantization" with no quant_method key:
mlx_block = getattr(model_config.hf_config, "quantization", None) or hg_quant_config.get("quantization")
if mlx_block and "group_size" in mlx_block and "bits" in mlx_block:
    from .int4 import Int4Config
    return Int4Config.from_hf_quant_config(mlx_block)
```

(Match the file's actual variable names — confirm against what Step 1 found.)

- [ ] **Step 7: Add allow-list test + verify**

```python
# append to test_int4_config.py
def test_dispatch_returns_int4config(monkeypatch):
    from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
    class HF:  # minimal stand-in for model_config.hf_config
        quantization = {"group_size": 64, "bits": 4}
        quantization_config = {"quantization": {"group_size": 64, "bits": 4}}
    class MC:
        hf_config = HF(); quantization = None
    class VC:
        model_config = MC()
    out = get_tpu_quantization_config(VC())
    assert out.__class__.__name__ == "Int4Config"
```

Run: `pytest tests/layers/jax/quantization/test_int4_config.py -v`
Expected: PASS (3 tests). Adjust the `VC/MC/HF` stand-in to match the real `get_tpu_quantization_config` signature found in Step 1.

- [ ] **Step 8: Commit**

```bash
git add tpu_inference/layers/jax/quantization/int4.py tpu_inference/layers/jax/quantization/__init__.py tests/layers/jax/quantization/test_int4_config.py
git commit -m "feat(quant): detect MLX int4 config and register Int4Config"
```

---

### Task 4: `Int4LinearMethod` (dense linears, Phase-1 XLA dequant)

The quant method for dense linears (attention qkv/o_proj, lm_head, embed). Declares packed params; `apply_jax` dequantizes via Task 1 then does the normal einsum.

**Files:**
- Modify: `tpu_inference/layers/jax/quantization/int4.py`
- Test: `tests/layers/jax/quantization/test_int4_linear.py`

**Interfaces:**
- Consumes: `mlx_dequantize` (Task 1); `Int4Config` (Task 3). Read the FP8 linear method precedent `tpu_inference/layers/jax/quantization/fp8.py` (`Fp8*LinearMethod`, `create_weights_jax` / `apply_jax`) and `tpu_inference/layers/jax/linear.py:86-93` (the `quant_method.apply_jax` dispatch seam).
- Produces: `Int4LinearMethod(config: Int4Config, bits: int, group_size: int)` with `create_weights_jax(layer, in, out, ...)` (declares `weight` uint32, `scales`/`biases` bf16) and `apply_jax(layer, x, *, einsum_str) -> jax.Array`.

- [ ] **Step 1: Write the failing test** — build a layer stub holding packed params for a known `[out,in]`, assert `apply_jax(x)` matches `x @ dequant(w).T`.

```python
# tests/layers/jax/quantization/test_int4_linear.py
import jax.numpy as jnp, numpy as np
from tests.utils.mlx_synthetic import pack_u4
from tpu_inference.layers.jax.quantization.int4 import Int4LinearMethod, Int4Config


def _quant(w, gs=64):
    out, in_ = w.shape; g = in_ // gs
    wg = w.reshape(out, g, gs); lo, hi = wg.min(-1, keepdims=True), wg.max(-1, keepdims=True)
    s = np.where(hi == lo, 1.0, (hi - lo) / 15.0); q = np.round((wg - lo) / s).clip(0, 15)
    import ml_dtypes
    return (pack_u4(q.reshape(out, in_).astype(np.uint32)),
            s.reshape(out, g).astype(ml_dtypes.bfloat16), lo.reshape(out, g).astype(ml_dtypes.bfloat16),
            (q.astype(np.float32) * s + lo).reshape(out, in_))


def test_int4_linear_apply_matches_reference():
    rng = np.random.default_rng(0); w = rng.standard_normal((128, 64)).astype(np.float32)
    packed, s, b, gold = _quant(w)
    m = Int4LinearMethod(Int4Config(), bits=4, group_size=64)
    layer = type("L", (), {})()
    layer.weight = type("P", (), {"value": jnp.asarray(packed)})()
    layer.scales = type("P", (), {"value": jnp.asarray(s)})()
    layer.biases = type("P", (), {"value": jnp.asarray(b)})()
    x = jnp.asarray(rng.standard_normal((3, 64)).astype(np.float32))
    out = m.apply_jax(layer, x, einsum_str="mn,pn->mp")  # out[p] = sum_n x[n] w[p,n]
    np.testing.assert_allclose(np.asarray(out, np.float32), np.asarray(x) @ gold.T, atol=0.2)
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/layers/jax/quantization/test_int4_linear.py -v` → FAIL (`Int4LinearMethod` undefined).

- [ ] **Step 3: Implement** `Int4LinearMethod` in `int4.py`. `apply_jax` reads `layer.weight/.scales/.biases`, calls `mlx_dequantize(...).astype(x.dtype)`, then `jnp.einsum(einsum_str, x, w)`. `create_weights_jax` declares the three params with shapes from `(in, out, bits, group_size)` and the partition specs used by the FP8 precedent (output-dim + no contraction-dim sharding finer than group_size). Mirror `Fp8*LinearMethod` structure exactly for param creation and the `QuantizeMethodBase` interface.

- [ ] **Step 4: Run to verify it passes** — `pytest tests/layers/jax/quantization/test_int4_linear.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tpu_inference/layers/jax/quantization/int4.py tests/layers/jax/quantization/test_int4_linear.py
git commit -m "feat(quant): Int4LinearMethod with XLA dequant (phase 1)"
```

---

### Task 5: `Int4FusedMoEMethod` (experts, Phase-1 XLA dequant)

The MoE quant method. Mirrors `Fp8FusedMoEMethod` lifecycle but stores packed uint32 + bf16 scales/biases per stacked expert; `apply_jax` dequantizes all experts to bf16 then routes through the existing **unquantized** GMM path. (Performance comes in Phase 2; this rung proves correctness.)

**Files:**
- Modify: `tpu_inference/layers/jax/quantization/int4.py`
- Test: `tests/layers/jax/quantization/test_int4_moe.py`

**Interfaces:**
- Consumes: `mlx_dequantize` (Task 1); `Int4Config` (Task 3). **Read first:** `Fp8FusedMoEMethod` (`fp8.py:345-628`) — `create_weights_jax` (415-471), `process_weights_after_loading` (473-574), `apply_jax` (576-628) — and `JaxMoE.__call__` (`layers/jax/moe/moe.py:173-187`), `moe_apply` (`layers/common/moe.py`).
- Produces: `Int4FusedMoEMethod(config, bits, group_size)` implementing the same four hooks. Param names: `kernel_gating_EDF{,_scales,_biases}`, `kernel_up_proj_EDF{...}`, `kernel_down_proj_EFD{...}` (packed uint32 + bf16 scales/biases, leading dim E).

- [ ] **Step 1: Write the failing test** — using `build_synthetic_mlx_moe` (Task 2), build a 1-layer/8-expert block, attach `Int4FusedMoEMethod`, run `apply_jax` for a few tokens, and compare against a numpy reference that dequantizes each expert (Task 1 golden) and runs the routed FFN by hand.

```python
# tests/layers/jax/quantization/test_int4_moe.py  (skeleton — fill expert/router refs to match JaxMoE math)
import numpy as np, jax.numpy as jnp
from pathlib import Path
from tests.utils.mlx_synthetic import build_synthetic_mlx_moe
# build block, load packed params into a JaxMoE stub, assert apply_jax ≈ numpy routed-FFN over dequant experts
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`Int4FusedMoEMethod` undefined).

- [ ] **Step 3: Implement** `Int4FusedMoEMethod`:
  - `create_weights_jax`: declare packed uint32 weights + bf16 scales/biases for gate/up/down (leading dim E), partition specs mirroring FP8's (expert + output-dim).
  - `process_weights_after_loading`: stack per-expert tensors into leading-dim-E layout (experts arrive **pre-stacked** from MLX, so mostly shape/transpose bookkeeping — keep packed, do NOT cast).
  - `apply_jax`: `w = mlx_dequantize(...)` per projection → bf16 → call the existing unquantized `moe_apply`/GMM path with bf16 experts. Run the router from the model's gate (already produces `router_logits`).

- [ ] **Step 4: Run to verify it passes** — `pytest tests/layers/jax/quantization/test_int4_moe.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tpu_inference/layers/jax/quantization/int4.py tests/layers/jax/quantization/test_int4_moe.py
git commit -m "feat(quant): Int4FusedMoEMethod with XLA dequant (phase 1)"
```

---

### Task 6: uint32-aware weight loader path

Stop the base loader from corrupting packed weights (force-cast to bf16, unconditional transpose). Carry `scales`/`biases`; preserve stacked experts; honor per-module bits.

**Files:**
- Modify: `tpu_inference/models/jax/utils/weight_utils.py` (`_load_and_shard_weight` — cast ~line 356, transpose ~lines 426-429, `DTYPE_VIEW_MAP` ~lines 55-59)
- Test: `tests/models/jax/utils/test_int4_loader.py`

**Interfaces:**
- Consumes: `Int4Config.bits_for` (Task 3). **Read first:** `_load_and_shard_weight` (320-459) and `DTYPE_VIEW_MAP` (55-59).
- Produces: when the active quant config is `Int4Config`, a load path that for `.weight` keys with a `.scales` sibling: skips the bf16 cast, skips transpose, keeps uint32; loads `.scales`/`.biases` as bf16; leaves the leading expert dim intact.

- [ ] **Step 1: Write the failing test** — point the loader at a `build_synthetic_mlx_moe` dir; assert a loaded expert weight param is uint32 with shape `[E, out, in/8]` (not bf16, not transposed) and that scales/biases are present and bf16.

- [ ] **Step 2: Run to verify it fails** — FAIL (weight comes back bf16 / transposed).

- [ ] **Step 3: Implement** the guard. Add `jnp.dtype(jnp.uint32): torch.uint32` to `DTYPE_VIEW_MAP`. In `_load_and_shard_weight`, gate the cast (line 356) and transpose (426-429) behind `not _is_mlx_packed(hf_key)`, where `_is_mlx_packed` checks the key ends in `.weight` and a `.scales` sibling exists for the same prefix (or the active config is `Int4Config` and the tensor dtype is uint32). Route `.scales`/`.biases` straight through (no cast/transpose).

- [ ] **Step 4: Run to verify it passes** — PASS.

- [ ] **Step 5: Commit**

```bash
git add tpu_inference/models/jax/utils/weight_utils.py tests/models/jax/utils/test_int4_loader.py
git commit -m "feat(loader): uint32-aware path preserving MLX packed weights + scales/biases"
```

---

### Task 7: Phase-1 end-to-end on synthetic checkpoint

Wire it together: load the synthetic MLX checkpoint through the real model-load path and assert a forward pass matches the numpy golden reference. This is the Phase-1 gate.

**Files:**
- Test: `tests/models/jax/test_qwen3_moe_mlx_int4_e2e.py`

**Interfaces:**
- Consumes: everything from Tasks 1-6.

- [ ] **Step 1: Write the failing test** — build a 2-layer synthetic checkpoint, instantiate `Qwen3MoeForCausalLM` via the normal loader with the MLX config, run a forward on a fixed token batch, and compare logits to a numpy reference that dequantizes all weights (Task 1) and runs the same math. Tolerance `atol≈0.3` (bf16).

- [ ] **Step 2: Run to verify it fails / iterate** — debug name-mapping mismatches (HF MLX keys ↔ model params) here; this task is where the loader name_map for `switch_mlp.*`, `mlp.gate`, `embed_tokens`, `lm_head` gets finalized.

- [ ] **Step 3: Make it pass.**

- [ ] **Step 4: Real-model smoke (hardware-gated).** Mark `@pytest.mark.skipif(no_tpu_or_lt_32gb)`. Load `mlx-community/Qwen3-30B-A3B-4bit`, greedy-decode a short prompt, assert it produces coherent tokens vs HF transformers reference (or at least non-degenerate logits). Document the v6e/multi-chip requirement in the test docstring.

- [ ] **Step 5: Commit**

```bash
git add tests/models/jax/test_qwen3_moe_mlx_int4_e2e.py
git commit -m "test: phase-1 MLX int4 end-to-end (synthetic + hardware-gated real model)"
```

**PHASE 1 COMPLETE** — correct numerics, weights 4-bit in HBM, dequant in XLA. Stop here for review before Phase 2.

---

### Task 8: Affine bias-correction term (pure function, Approach A)

The one new piece of math for Phase 2: the data-dependent bias term `Σ_g bias[o,g]·groupsum(x)`, computed outside the kernel.

**Files:**
- Modify: `tpu_inference/layers/common/quantization/__init__.py`
- Test: `tests/layers/common/quantization/test_mlx_bias_term.py`

**Interfaces:**
- Produces: `mlx_bias_correction(x: jax.Array, biases: jax.Array, group_size: int) -> jax.Array` — `x [T, in]`, `biases [out, n_groups]` → `[T, out]`. (MoE variant: `biases [E, out, n_groups]`, applied per routed expert.)

- [ ] **Step 1: Write the failing test**

```python
# tests/layers/common/quantization/test_mlx_bias_term.py
import jax.numpy as jnp, numpy as np
from tpu_inference.layers.common.quantization import mlx_bias_correction, mlx_dequantize
from tests.utils.mlx_synthetic import pack_u4
import ml_dtypes


def test_bias_term_equals_full_minus_scaled():
    # full affine matmul = scaled-matmul + bias-correction ; verify the decomposition
    rng = np.random.default_rng(0)
    out, in_, gs, T = 8, 128, 64, 4
    w = rng.standard_normal((out, in_)).astype(np.float32)
    g = in_ // gs; wg = w.reshape(out, g, gs)
    lo, hi = wg.min(-1, keepdims=True), wg.max(-1, keepdims=True)
    s = np.where(hi == lo, 1.0, (hi - lo) / 15.0); q = np.round((wg - lo) / s).clip(0, 15)
    packed = pack_u4(q.reshape(out, in_).astype(np.uint32))
    scales = s.reshape(out, g).astype(ml_dtypes.bfloat16); biases = lo.reshape(out, g).astype(ml_dtypes.bfloat16)
    x = rng.standard_normal((T, in_)).astype(np.float32)

    w_deq = np.asarray(mlx_dequantize(jnp.asarray(packed), jnp.asarray(scales), jnp.asarray(biases), gs, 4), np.float32)
    full = x @ w_deq.T
    qf = np.asarray(mlx_dequantize(jnp.asarray(packed), jnp.asarray(scales),
                                   jnp.zeros_like(jnp.asarray(biases)), gs, 4), np.float32)  # scale-only
    scaled = x @ qf.T
    corr = np.asarray(mlx_bias_correction(jnp.asarray(x), jnp.asarray(biases), gs), np.float32)
    np.testing.assert_allclose(scaled + corr, full, atol=0.2)
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`mlx_bias_correction` undefined).

- [ ] **Step 3: Implement**

```python
def mlx_bias_correction(x, biases, group_size):
    *lead, in_ = x.shape
    g = in_ // group_size
    xg = x.reshape(*lead, g, group_size).sum(-1)          # [..., g]
    # biases [out, g] -> [T, out]
    return jnp.einsum("tg,og->to", xg.astype(jnp.float32), biases.astype(jnp.float32))
```

- [ ] **Step 4: Run to verify it passes** — PASS.

- [ ] **Step 5: Commit**

```bash
git add tpu_inference/layers/common/quantization/__init__.py tests/layers/common/quantization/test_mlx_bias_term.py
git commit -m "feat(quant): MLX affine bias-correction term (approach A)"
```

---

### Task 9: Route int4 experts through `gmm_v2` packed (Phase-2 performant)

Switch `Int4FusedMoEMethod.apply_jax` from XLA-dequant to the packed GMM path: weights stay uint32, `gmm_v2` applies group-64 scales in-kernel, and the bias term (Task 8) is added per routed expert outside the kernel. Verify numerics match Phase 1 (incl. adversarial negative scales).

**Files:**
- Modify: `tpu_inference/layers/jax/quantization/int4.py` (`Int4FusedMoEMethod.apply_jax`, `process_weights_after_loading`)
- Modify: `tpu_inference/layers/common/fused_moe_gmm.py` (pass int4 packed `rhs` + `rhs_scale` to `gmm_v2`; mirror the FP8 GMM call)
- Test: extend `tests/layers/jax/quantization/test_int4_moe.py`

**Interfaces:**
- Consumes: `gmm_v2` (`tpu_inference/kernels/megablox/gmm_v2.py`); `mlx_bias_correction` (Task 8). **Read first:** how `Fp8FusedMoEMethod.apply_jax` builds the GMM call and the `rhs_scale` shape `[E, num_blocks, 1, N]`.

- [ ] **Step 1: Write the failing test** — same synthetic block as Task 5, but assert the **packed** apply (gmm_v2 + bias term) matches the Phase-1 XLA-dequant apply to `atol≈0.2`, including on the adversarial (negative-scale) experts. Add an env/flag to select packed vs XLA path so both run in the test.

- [ ] **Step 2: Run to verify it fails** — FAIL (packed path not implemented / numerics off by the bias term).

- [ ] **Step 3: Implement.** Reshape MLX `scales [E, out, in/64]` into the gmm_v2 `rhs_scale [E, in/64, 1, out]` layout (group dim = K-block). Pass packed uint32 `rhs` with `should_bitcast` set for int4. Compute `mlx_bias_correction` per routed expert on the grouped activations and add to the GMM output. Confirm the MoE backend resolves to GMM_TP/GMM_EP (guard: if `FUSED_MOE`, raise a clear NotImplementedError per the spec's backend-routing note).

- [ ] **Step 4: Run to verify it passes** — PASS (packed ≈ XLA-dequant, incl. adversarial).

- [ ] **Step 5: Commit**

```bash
git add tpu_inference/layers/jax/quantization/int4.py tpu_inference/layers/common/fused_moe_gmm.py tests/layers/jax/quantization/test_int4_moe.py
git commit -m "feat(quant): route int4 MoE experts through gmm_v2 packed + bias term (phase 2)"
```

---

### Task 10: `gmm_v2` MLX nibble-order + negative-scale verification

Confirm/extend the kernel so its in-kernel unpack matches MLX's nibble order and tolerates negative scales. Most of this may already hold — verify with a kernel-level test before changing code.

**Files:**
- Test: `tests/kernels/megablox/test_gmm_v2_mlx_int4.py`
- Modify (only if test fails): `tpu_inference/kernels/megablox/gmm_v2.py`

**Interfaces:**
- Consumes: `gmm_v2`; `mlx_dequantize` (Task 1) as the reference.

- [ ] **Step 1: Write the test** — feed `gmm_v2` a packed int4 `rhs` + MLX-layout `rhs_scale` (no bias; bias handled in Task 9) and compare to a dense `lhs @ mlx_dequantize(scale-only).T` reference, including a group with **negative scale**. Single group, single expert.

- [ ] **Step 2: Run** — `pytest tests/kernels/megablox/test_gmm_v2_mlx_int4.py -v` (needs TPU). If PASS, the kernel already matches MLX order → no code change. If FAIL, the nibble order differs from `pltpu.bitcast`.

- [ ] **Step 3: Fix if needed** — adjust the unpack (nibble order) inside `gmm_v2`'s `should_bitcast` path, or pre-permute the packed words at load so `bitcast` yields MLX order. Prefer the load-time permute (keeps the kernel generic).

- [ ] **Step 4: Verify** — PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/kernels/megablox/test_gmm_v2_mlx_int4.py tpu_inference/kernels/megablox/gmm_v2.py
git commit -m "test(kernel): verify gmm_v2 matches MLX nibble order + negative scales"
```

---

### Task 11: Phase-2 end-to-end + memory check (hardware-gated)

**Files:**
- Test: extend `tests/models/jax/test_qwen3_moe_mlx_int4_e2e.py`

- [ ] **Step 1:** With the packed GMM path enabled, re-run the synthetic e2e (Task 7) and assert logits match the Phase-1 golden to `atol≈0.3`.
- [ ] **Step 2 (hardware-gated):** Load `mlx-community/Qwen3-30B-A3B-4bit` on a v6e, decode a prompt, and assert peak HBM for the weights is ≈ packed-4-bit size (≈ half of bf16-dequant), confirming weights stayed packed. Compare output tokens to the Phase-1 (XLA-dequant) run.
- [ ] **Step 3: Commit**

```bash
git commit -am "test: phase-2 MLX int4 end-to-end + HBM stays-packed check"
```

**PHASE 2 COMPLETE** — weights stay 4-bit in HBM, expert matmul dequantizes in-kernel, bias term correct.

---

## Self-Review

**Spec coverage:** §2 format → Tasks 1,2,6; §3 bias term → Tasks 8,9; §4 tiers/backend guard → Tasks 3,5,9; §5 components → Tasks 3-6; §6 phases → Tasks 7 (P1), 11 (P2); §7 sharding → Tasks 4,5 (partition specs), 6 (load); §8 testing → Tasks 1,2,7,8,9,10,11; §9 risks: negative scale → Tasks 1,9,10; mixed precision → Task 3; hardware → Tasks 7,11. No gaps.

**Placeholder scan:** integration Tasks (4,5,6,9) deliberately defer boilerplate to "mirror named FP8 method at file:line" rather than fabricate unread signatures; every such task starts with a "read the precedent" step and states exact param names/shapes it must produce. Pure-logic Tasks (1,2,8) carry complete code.

**Type consistency:** `mlx_unpack`/`mlx_dequantize`/`mlx_bias_correction` signatures consistent across Tasks 1,5,8,9. `Int4Config.bits_for` used consistently in Tasks 3,6. Param names `kernel_*_EDF{,_scales,_biases}` consistent Tasks 5,9.
