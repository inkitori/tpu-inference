# MLX 4-bit MoE (torchax path) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `mlx-community/Qwen3-30B-A3B-4bit` through the torchax path (`MODEL_IMPL_TYPE=vllm`) with accurate outputs, weights kept 4-bit (packed uint32 + scales + biases) in HBM and dequantized in the forward pass.

> **PRODUCTION FINISH LINE = Task 10 + the Task 12 perf gate.** The user's real goal is serving this model on OpenRouter at *usable coding speed*, which requires the production end-state: true 4-bit experts in HBM (~2GB/chip) with **in-kernel dequant of the ROUTED experts only** (Task 10), validated by a tokens/sec + TTFT/TPOT perf gate (Task 12). **Task 8 is a CORRECTNESS milestone only** — at Task 8 the experts are bf16 in HBM (~7.6GB/chip, ~3.8× the memory) and it dequants ALL 128 experts every step (correct but slow). Do NOT stop at Task 8 and call it done.
>
> **Status:** Tasks 1–7 complete (HEAD `5119597b`); see the `.superpowers/sdd/progress.md` ledger. Tasks 8–13 remain.

**Architecture:** A new vLLM quantization method (`VllmMLXConfig` + linear + MoE methods) mirroring `awq.py`, plus a load-time weight-stream transform that bridges the MLX checkpoint naming/layout (`switch_mlp`, stacked experts, quantized embed/lm_head) to what vLLM's `Qwen3MoeForCausalLM` expects. Quant `apply` crosses into JAX (`jax_view` → `jnp` dequant+matmul → `torch_view`), reusing `mlx_unpack`/`mlx_dequantize`.

**Tech Stack:** Python, vLLM 0.22.0, JAX 0.9.2, torchax, TPU v6e-8. Venv at `/home/enyouki/tpu-inference/.venv`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-21-mlx-4bit-moe-torchax-design.md`.
- MLX format: affine 4-bit, `group_size=64`, `bits=4`, packed 8 nibbles/uint32 along the **input** dim, low-nibble-first; dequant `w = scale*q + bias` per group of 64 along input. No per-module overrides for this model.
- Run env: always `source /home/enyouki/tpu-inference/.venv/bin/activate` first.
- Tests / runs use `MODEL_IMPL_TYPE=vllm`. Add `SKIP_JAX_PRECOMPILE=1` for quick runs.
- Reuse `mlx_unpack`/`mlx_dequantize` from `tpu_inference/layers/common/quantization/__init__.py` (do NOT delete them in cleanup).
- Primary rule: **accurate, coherent outputs**. If a problem arises, consult tpu-inference PRs first.
- Commit after each task. Do not push. Branch is `hy3`.
- Commit message footer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Cleanup — remove all JAX-path MLX code and obsolete docs

**Files:**
- Delete: `tpu_inference/layers/jax/quantization/int4.py`
- Modify: `tpu_inference/layers/jax/quantization/__init__.py` (remove int4 registration + MLX detection)
- Modify: `tpu_inference/models/jax/utils/weight_utils.py` (remove `_is_mlx_packed`, `_get_active_int4_config`, `_normalize_mlx_switch_mlp_keys`, the `jnp.uint32` view-map entry, and their call sites)
- Modify: `tpu_inference/models/common/model_loader.py` (remove the `enable_weights_track = False` MLX toggle only)
- Delete: `tests/models/jax/test_qwen3_moe_mlx_int4_e2e.py` and any committed `tests/.../test_int4_*.py` (`test_int4_linear.py`, `test_int4_moe.py`, `test_int4_config.py`, `test_int4_loader.py`)
- Delete: `HY3_SPEC.md`, `.sdd/` (whole dir), and the JAX-era `docs/superpowers/specs/2026-06-21-mlx-4bit-moe-loading-design.md` + `docs/superpowers/plans/2026-06-21-mlx-4bit-moe-loading.md` if present
- Keep (do NOT touch): `tpu_inference/layers/common/quantization/__init__.py` (`mlx_unpack`/`mlx_dequantize`), `tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py`, `tests/utils/mlx_synthetic.py`

**Interfaces:**
- Produces: a repo with zero `Int4`/MLX-quant references on the JAX path; `mlx_unpack`/`mlx_dequantize` still importable.

- [ ] **Step 1: Locate every JAX MLX reference**

Run: `cd /home/enyouki/tpu-inference && git grep -n -E "Int4|int4|_is_mlx_packed|switch_mlp|_get_active_int4|enable_weights_track|mlx_4bit" -- tpu_inference tests | grep -v "layers/common/quantization"`
Expected: a finite list of hits in the files above. Note each.

- [ ] **Step 2: Delete the standalone files**

```bash
cd /home/enyouki/tpu-inference
git rm -f tpu_inference/layers/jax/quantization/int4.py
git rm -f tests/models/jax/test_qwen3_moe_mlx_int4_e2e.py 2>/dev/null || rm -f tests/models/jax/test_qwen3_moe_mlx_int4_e2e.py
# delete committed int4 unit tests (adjust paths to wherever grep found them)
git ls-files 'tests/**/test_int4_*.py' | xargs -r git rm -f
rm -rf HY3_SPEC.md .sdd
git rm -f docs/superpowers/specs/2026-06-21-mlx-4bit-moe-loading-design.md docs/superpowers/plans/2026-06-21-mlx-4bit-moe-loading.md 2>/dev/null || true
```

- [ ] **Step 3: Revert the in-file MLX hooks**

In `tpu_inference/layers/jax/quantization/__init__.py`: remove the `"int4": Int4Config` registry entry, the `Int4Config` import, and the MLX-detection short-circuit (restore the function to its pre-MLX form — compare against `git log -p` for commit `ddbb783c`).

In `tpu_inference/models/jax/utils/weight_utils.py`: delete `_is_mlx_packed`, `_get_active_int4_config`, `_normalize_mlx_switch_mlp_keys`, remove the `jnp.uint32 -> torch.uint32` entry from `DTYPE_VIEW_MAP`, and remove every call site of those helpers (restore the surrounding loops to their pre-MLX form).

In `tpu_inference/models/common/model_loader.py`: remove only the `enable_weights_track = False` lines added for the JAX MLX path (leave everything else).

- [ ] **Step 4: Verify import + no dangling references**

Run:
```bash
cd /home/enyouki/tpu-inference && source .venv/bin/activate
python -c "import tpu_inference; from tpu_inference.layers.common.quantization import mlx_unpack, mlx_dequantize; print('ok')"
git grep -n -E "Int4|_is_mlx_packed|switch_mlp|_get_active_int4|enable_weights_track" -- tpu_inference | grep -v "layers/common/quantization" || echo "clean"
```
Expected: prints `ok` then `clean` (no remaining references outside `layers/common/quantization`).

- [ ] **Step 5: Run the JAX quantization unit tests that remain to confirm nothing else broke**

Run: `cd /home/enyouki/tpu-inference && source .venv/bin/activate && python -m pytest -q tests/layers/jax/quantization 2>/dev/null || echo "no such dir — skip"`
Expected: pass or "no such dir".

- [ ] **Step 6: Commit**

```bash
cd /home/enyouki/tpu-inference
git add -A
git commit -m "chore(quant): remove JAX-path MLX int4 implementation and obsolete docs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Harness sanity — confirm the vLLM/torchax path still serves a model

**Files:** none (verification milestone).

**Interfaces:**
- Produces: confidence that cleanup didn't break the vllm path; a known-good baseline command.

- [ ] **Step 1: Run a tiny model through the vllm path**

Run:
```bash
cd /home/enyouki/tpu-inference && source .venv/bin/activate
MODEL_IMPL_TYPE=vllm SKIP_JAX_PRECOMPILE=1 \
  python examples/offline_inference.py --model Qwen/Qwen3-0.6B --max-model-len 1024 --max-tokens 32 --tensor-parallel-size 1
```
Expected: prints `Prompt: ... / Generated text: ...` with coherent text. If it fails, STOP and consult tpu-inference PRs / fix before proceeding.

- [ ] **Step 2: No commit** (verification only). Record the working command in the task notes.

---

### Task 3: MLX-detection helper + register `VllmMLXConfig` skeleton

**Files:**
- Modify: `tpu_inference/layers/common/quant_methods.py` (add `MLX = "mlx"`)
- Create: `tpu_inference/layers/vllm/quantization/mlx.py` (config skeleton)
- Modify: `tpu_inference/layers/vllm/quantization/__init__.py` (import + `method_to_config` entry + MLX detection)
- Modify: `tpu_inference/platforms/tpu_platform.py` (add `"mlx"` to `supported_quantization`)
- Test: `tests/layers/vllm/quantization/test_mlx_config.py`

**Interfaces:**
- Produces: `VllmMLXConfig(group_size:int, bits:int, modules_to_not_convert:list|None)` with `get_name()=="mlx"`, `from_config(cls, config: dict)`, `get_quant_method(self, layer, prefix)`. Helper `is_mlx_quantized(hf_config) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/layers/vllm/quantization/test_mlx_config.py
from tpu_inference.layers.vllm.quantization.mlx import VllmMLXConfig, is_mlx_quantized

class _HF:  # minimal stand-in for an HF config carrying an MLX quant block
    def __init__(self):
        self.quantization_config = {"group_size": 64, "bits": 4}

def test_is_mlx_quantized_true_for_groupsize_bits_without_quant_method():
    assert is_mlx_quantized(_HF()) is True

def test_is_mlx_quantized_false_when_quant_method_present():
    hf = _HF(); hf.quantization_config = {"group_size": 64, "bits": 4, "quant_method": "awq"}
    assert is_mlx_quantized(hf) is False

def test_from_config_parses_group_size_and_bits():
    cfg = VllmMLXConfig.from_config({"group_size": 64, "bits": 4})
    assert cfg.group_size == 64 and cfg.bits == 4
    assert VllmMLXConfig.get_name() == "mlx"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/enyouki/tpu-inference && source .venv/bin/activate && python -m pytest tests/layers/vllm/quantization/test_mlx_config.py -v`
Expected: FAIL with `ModuleNotFoundError: ...mlx`.

- [ ] **Step 3: Add the method constant**

In `tpu_inference/layers/common/quant_methods.py`, after the existing constants (`UNQUANTIZED`, `MXFP4`, `AWQ`, `COMPRESSED_TENSORS`, `FP8`), add:
```python
MLX = "mlx"
```

- [ ] **Step 4: Create the config skeleton**

```python
# tpu_inference/layers/vllm/quantization/mlx.py
from typing import Any, Optional, Union

import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)

from tpu_inference.layers.common.quant_methods import MLX
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig


def is_mlx_quantized(hf_config) -> bool:
    """MLX checkpoints carry a quant block with group_size+bits and NO quant_method."""
    for attr in ("quantization_config", "quantization"):
        q = getattr(hf_config, attr, None)
        if isinstance(q, dict) and "group_size" in q and "bits" in q \
                and "quant_method" not in q:
            return True
    return False


@register_quantization_config(MLX)
class VllmMLXConfig(QuantizationConfig, VllmQuantConfig):

    def __init__(self, group_size: int, bits: int,
                 modules_to_not_convert: Optional[list[str]] = None):
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.pack_factor = 32 // bits  # 8 for 4-bit
        self.modules_to_not_convert = modules_to_not_convert or []

    @classmethod
    def get_name(cls) -> str:
        return MLX

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "VllmMLXConfig":
        return cls(group_size=config["group_size"], bits=config["bits"],
                   modules_to_not_convert=config.get("modules_to_not_convert"))

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    def get_quant_method(self, layer, prefix: str
                         ) -> Optional[Union[QuantizeMethodBase]]:
        # Methods added in Tasks 5 and 7; return None for now.
        return None
```

- [ ] **Step 5: Register in the dispatch + add MLX detection**

In `tpu_inference/layers/vllm/quantization/__init__.py`: add `from tpu_inference.layers.vllm.quantization.mlx import VllmMLXConfig, is_mlx_quantized` and add `quant_methods.MLX: VllmMLXConfig` to `method_to_config`. Immediately after `model_config = copy.deepcopy(vllm_config.model_config)` and before the dict lookup, insert:
```python
    if model_config.quantization is None and is_mlx_quantized(model_config.hf_config):
        model_config.quantization = quant_methods.MLX
```

- [ ] **Step 6: Add to platform allow-list**

In `tpu_inference/platforms/tpu_platform.py`, add `"mlx"` to the `supported_quantization` list.

- [ ] **Step 7: Run the test**

Run: `cd /home/enyouki/tpu-inference && source .venv/bin/activate && python -m pytest tests/layers/vllm/quantization/test_mlx_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat(quant): register VllmMLXConfig + MLX format detection (torchax path)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Pin down the MLX dequant math against the synthetic oracle

**Files:**
- Test: `tests/layers/common/quantization/test_mlx_dequant.py`

**Interfaces:**
- Consumes: `mlx_dequantize(packed, scales, biases, group_size, bits) -> bf16 jax.Array` (input `packed [..., out, in//8]`, `scales/biases [..., out, in//64]`); `tests/utils/mlx_synthetic._quantize_affine(w, group_size, force_negative_scale) -> (packed, scales_bf16, biases_bf16, golden_bf16)`.
- Produces: a guard proving the reused primitive matches the checkpoint math, including negative scales.

- [ ] **Step 1: Write the failing test**

```python
# tests/layers/common/quantization/test_mlx_dequant.py
import jax.numpy as jnp
import numpy as np
import pytest

from tests.utils.mlx_synthetic import _quantize_affine
from tpu_inference.layers.common.quantization import mlx_dequantize


@pytest.mark.parametrize("force_neg", [False, True])
def test_mlx_dequantize_matches_affine_golden(force_neg):
    rng = np.random.default_rng(0)
    out_features, in_features, group_size = 16, 128, 64
    w = rng.standard_normal((out_features, in_features)).astype(np.float32)
    packed, scales, biases, golden = _quantize_affine(w, group_size, force_neg)

    got = np.asarray(mlx_dequantize(
        jnp.asarray(packed), jnp.asarray(scales), jnp.asarray(biases),
        group_size=group_size, bits=4)).astype(np.float32)

    # Exact reconstruction of the *quantized* (golden) weights, modulo bf16 rounding.
    np.testing.assert_allclose(got, golden.astype(np.float32), atol=1e-2, rtol=1e-2)
```

- [ ] **Step 2: Run and verify it fails or passes**

Run: `cd /home/enyouki/tpu-inference && source .venv/bin/activate && python -m pytest tests/layers/common/quantization/test_mlx_dequant.py -v`
Expected: PASS if the primitive is already correct. If it FAILS, fix `mlx_unpack`/`mlx_dequantize` (check nibble order: element 0 = low nibble, flat input index = `word*8 + k`; affine `q*scale + bias`; group repeat along the last axis) until it passes. Do not change the test's math.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test(quant): pin MLX affine dequant against synthetic oracle

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `VllmMLXLinearMethod` (keep-4bit, dequant-in-apply)

**Files:**
- Modify: `tpu_inference/layers/vllm/quantization/mlx.py` (add `VllmMLXLinearMethod`; wire into `get_quant_method`)
- Test: `tests/layers/vllm/quantization/test_mlx_linear.py`

**Reference (mirror closely):** `tpu_inference/layers/vllm/quantization/awq.py:91-232` (`VllmAWQLinearMethod`), especially `_apply_fused` (181-204) and `_apply_split` (206-232). Key differences for MLX: weight is `uint32` packed along the **input** dim with shape `[out, in//8]` (AWQ packs along output); use `scales`+`biases` (affine) not `scales`+`qzeros`; dequant via `mlx_dequantize`, not `(q - z) * s`.

**Interfaces:**
- Consumes: `VllmQuantConfig.get_linear_config(layer) -> VllmQuantLinearConfig` (fields: `output_sizes: list[int]`, `n_shards: int`, `weight_sharding`, `bias_sharding`, `fuse_matmuls`, `num_proj`).
- Produces: `VllmMLXLinearMethod(quant_config, linear_config)` with `create_weights`, `process_weights_after_loading`, `apply(layer, x, bias=None) -> torch.Tensor`. Stored params on `layer`: `weight` (uint32 `[out, in//8]`), `scales` (bf16 `[out, in//64]`), `biases` (bf16 `[out, in//64]`).

- [ ] **Step 1: Write the failing test (numerical correctness of a quantized linear)**

```python
# tests/layers/vllm/quantization/test_mlx_linear.py
import jax.numpy as jnp
import numpy as np

from tests.utils.mlx_synthetic import _quantize_affine
from tpu_inference.layers.common.quantization import mlx_dequantize


def test_mlx_linear_dequant_then_matmul_matches_golden():
    """The apply() math: y = x @ dequant(weight).T must match x @ golden.T."""
    rng = np.random.default_rng(1)
    out_f, in_f, gs = 32, 128, 64
    w = rng.standard_normal((out_f, in_f)).astype(np.float32)
    packed, scales, biases, golden = _quantize_affine(w, gs, force_negative_scale=True)
    x = rng.standard_normal((4, in_f)).astype(np.float32)

    weight = mlx_dequantize(jnp.asarray(packed), jnp.asarray(scales),
                            jnp.asarray(biases), group_size=gs, bits=4)  # [out, in]
    y = np.asarray(jnp.einsum("bd,fd->bf", jnp.asarray(x), weight)).astype(np.float32)
    y_ref = x @ golden.astype(np.float32).T
    np.testing.assert_allclose(y, y_ref, atol=2e-2, rtol=2e-2)
```

This locks the exact einsum/layout the `apply` body must use (`"bd,fd->bf"`, contracting the input dim, weight `[out, in]`).

- [ ] **Step 2: Run to verify it passes (it tests the math contract)**

Run: `cd /home/enyouki/tpu-inference && source .venv/bin/activate && python -m pytest tests/layers/vllm/quantization/test_mlx_linear.py -v`
Expected: PASS. (This guards the layout the implementation must follow.)

- [ ] **Step 3: Implement `create_weights`**

Add to `mlx.py`. Mirror how vLLM's AWQ linear registers typed parameters, but with MLX shapes. Use `from vllm.model_executor.parameter import PackedvLLMParameter, GroupQuantScaleParameter, ModelWeightParameter`. The weight packs along the input dim, so `packed_dim=1`/`input_dim=1`, `output_dim=0`; fusion/sharding act on the (unpacked) output dim 0.

```python
class VllmMLXLinearMethod(QuantizeMethodBase):

    def __init__(self, quant_config: "VllmMLXConfig", linear_config):
        self.quant_config = quant_config
        self.linear_config = linear_config

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        gs = self.quant_config.group_size
        pf = self.quant_config.pack_factor
        out = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = PackedvLLMParameter(
            data=torch.empty(out, input_size_per_partition // pf, dtype=torch.uint32),
            output_dim=0, input_dim=1, packed_dim=1, packed_factor=pf,
            weight_loader=weight_loader)
        scales = GroupQuantScaleParameter(
            data=torch.empty(out, input_size_per_partition // gs, dtype=params_dtype),
            output_dim=0, input_dim=1, weight_loader=weight_loader)
        biases = GroupQuantScaleParameter(
            data=torch.empty(out, input_size_per_partition // gs, dtype=params_dtype),
            output_dim=0, input_dim=1, weight_loader=weight_loader)
        layer.register_parameter("weight", weight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("biases", biases)
```

> NOTE for implementer: cross-check `PackedvLLMParameter`/`GroupQuantScaleParameter` constructor kwarg names against the installed vLLM (`.venv/.../vllm/model_executor/parameter.py`) and AWQ's `create_weights`. Adjust kwarg names if they differ; the synthetic e2e test (Task 7) is the final arbiter.

- [ ] **Step 4: Implement `process_weights_after_loading` (shard, stay packed)**

Mirror AWQ's `process_weights_after_loading` (`awq.py:98`) but WITHOUT unpacking/dequantizing — only `t2j` each of `weight`/`scales`/`biases`, shard along the output dim using `self.linear_config.weight_sharding`, and re-store as torch params via `torch_view`. Use `shard_linear_weights(LinearWeights(weight=..., weight_scale=..., zero_point=..., bias=None), mesh=self.linear_config.mesh, weight_p_spec=self.linear_config.weight_sharding, bias_p_spec=self.linear_config.bias_sharding, transposed=False)` analog — but since MLX keeps three packed tensors, the simplest correct approach is to shard each tensor along axis 0 (output) with a `NamedSharding`. Follow the helper used by AWQ; keep the packed `uint32` weight unmodified.

```python
    def process_weights_after_loading(self, layer):
        import jax
        from jax.sharding import NamedSharding, PartitionSpec as P
        from torchax.interop import jax_view, torch_view
        mesh = self.linear_config.mesh
        wsh = self.linear_config.weight_sharding  # P(out_axis, None)
        def _shard(t, spec):
            arr = jax_view(t)
            return torch_view(jax.device_put(arr, NamedSharding(mesh, spec)))
        layer.weight = torch.nn.Parameter(_shard(layer.weight, wsh), requires_grad=False)
        layer.scales = torch.nn.Parameter(_shard(layer.scales, wsh), requires_grad=False)
        layer.biases = torch.nn.Parameter(_shard(layer.biases, wsh), requires_grad=False)
```

> NOTE: if `weight_sharding` is a 2-tuple `P(axis, None)` it applies directly to the `[out, in//pf]` packed weight and `[out, in//gs]` scales/biases (axis 0 = output). Verify the partition axis name against AWQ; adjust if AWQ shards differently.

- [ ] **Step 5: Implement `apply` (dequant-in-XLA, handle fused output split)**

```python
    def apply(self, layer, x, bias=None):
        import jax
        import jax.numpy as jnp
        from torchax.interop import jax_view, torch_view
        from tpu_inference.layers.common.quantization import mlx_dequantize
        from tpu_inference.utils import slice_sharded_tensor_for_concatenation  # path per AWQ import
        with jax.named_scope(layer._get_name()):
            x_jax = jax_view(x)
            weight = mlx_dequantize(
                jax_view(layer.weight), jax_view(layer.scales), jax_view(layer.biases),
                group_size=self.quant_config.group_size, bits=self.quant_config.bits)
            outs = jnp.einsum("bd,fd->bf", x_jax, weight)
            if bias is not None and not layer.skip_bias_add:
                outs = outs + jax_view(bias)
            outs = slice_sharded_tensor_for_concatenation(
                outs, self.linear_config.output_sizes, self.linear_config.n_shards)
            return torch_view(jnp.concatenate(outs, axis=-1))
```

> NOTE: copy the exact import path of `slice_sharded_tensor_for_concatenation` from the top of `awq.py`. If the layer is not fused (`num_proj == 1`), `slice_.../concatenate` is a no-op pass-through; keep it for parity with AWQ, or guard on `len(output_sizes) == 1`.

- [ ] **Step 6: Wire dispatch in `get_quant_method`**

Replace the linear branch in `VllmMLXConfig.get_quant_method`:
```python
        from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
        match layer:
            case LinearBase():
                linear_config = self.get_linear_config(layer)
                if is_layer_skipped(prefix, self.modules_to_not_convert):
                    from tpu_inference.layers.vllm.quantization.unquantized import VllmUnquantizedLinearMethod
                    return VllmUnquantizedLinearMethod(linear_config)
                return VllmMLXLinearMethod(self, linear_config)
            case _:
                return None
```

- [ ] **Step 7: Run the math test again + import check**

Run: `cd /home/enyouki/tpu-inference && source .venv/bin/activate && python -m pytest tests/layers/vllm/quantization/test_mlx_linear.py -v && python -c "from tpu_inference.layers.vllm.quantization.mlx import VllmMLXLinearMethod; print('ok')"`
Expected: PASS + `ok`.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat(quant): VllmMLXLinearMethod with dequant-in-XLA apply

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Weight-stream transform (rename/un-stack experts, dequant embed/lm_head)

**Files:**
- Create: `tpu_inference/models/vllm/mlx_weight_transform.py`
- Modify: `tpu_inference/models/vllm/vllm_model_loader.py` (override `get_all_weights` to apply the transform when MLX is active)
- Test: `tests/models/vllm/test_mlx_weight_transform.py`

**Reference:** `IncrementalModelLoader(DefaultModelLoader)` at `vllm_model_loader.py:73`; inherited `get_all_weights(self, model_config, model) -> Generator[tuple[str, Tensor]]` (vLLM `default_loader.py:288`).

**Interfaces:**
- Produces: `transform_mlx_weights(weights: Iterable[tuple[str, Tensor]], *, group_size:int, bits:int, num_experts:int) -> Iterator[tuple[str, Tensor]]`. Behavior:
  - `...mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` (stacked `[E, out, in*]`) → emit per-expert `...mlp.experts.{e}.{gate,up,down}_proj.{weight,scales,biases}` (slice axis 0); weight stays `uint32` packed.
  - `...embed_tokens.{weight,scales,biases}` and `lm_head.{weight,scales,biases}` → emit a single dequantized `....weight` (bf16) via `mlx_dequantize`; drop scales/biases.
  - everything else (attention q/k/v/o, `mlp.gate`, norms) → pass through unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/vllm/test_mlx_weight_transform.py
import numpy as np
import torch

from tests.utils.mlx_synthetic import _quantize_affine, pack_u4
from tpu_inference.models.vllm.mlx_weight_transform import transform_mlx_weights


def _packed_t(out, in_, gs):
    w = np.random.default_rng(0).standard_normal((out, in_)).astype(np.float32)
    packed, scales, biases, golden = _quantize_affine(w, gs, False)
    return (torch.from_numpy(packed.view(np.uint32).astype(np.uint32)),
            torch.from_numpy(scales), torch.from_numpy(biases), golden)

def test_experts_unstacked_and_renamed_kept_packed():
    E, out, in_, gs = 4, 16, 64, 64
    w = np.random.default_rng(0).standard_normal((E, out, in_)).astype(np.float32)
    # stack per-expert affine packs
    packs = [ _quantize_affine(w[e], gs, False) for e in range(E) ]
    stk_w = torch.from_numpy(np.stack([p[0] for p in packs]).astype(np.uint32))
    stk_s = torch.from_numpy(np.stack([p[1] for p in packs]))
    stk_b = torch.from_numpy(np.stack([p[2] for p in packs]))
    stream = [
        ("model.layers.0.mlp.switch_mlp.gate_proj.weight", stk_w),
        ("model.layers.0.mlp.switch_mlp.gate_proj.scales", stk_s),
        ("model.layers.0.mlp.switch_mlp.gate_proj.biases", stk_b),
    ]
    out_map = dict(transform_mlx_weights(stream, group_size=gs, bits=4, num_experts=E))
    for e in range(E):
        k = f"model.layers.0.mlp.experts.{e}.gate_proj.weight"
        assert k in out_map and out_map[k].dtype == torch.uint32
        assert f"model.layers.0.mlp.experts.{e}.gate_proj.scales" in out_map
    assert "switch_mlp" not in " ".join(out_map)

def test_embed_and_lm_head_dequantized_to_bf16_weight():
    out, in_, gs = 32, 64, 64
    pw, ps, pb, golden = _packed_t(out, in_, gs)
    stream = [
        ("model.embed_tokens.weight", pw),
        ("model.embed_tokens.scales", ps),
        ("model.embed_tokens.biases", pb),
    ]
    out_map = dict(transform_mlx_weights(stream, group_size=gs, bits=4, num_experts=1))
    assert set(out_map) == {"model.embed_tokens.weight"}
    w = out_map["model.embed_tokens.weight"]
    assert w.dtype == torch.bfloat16 and tuple(w.shape) == (out, in_)

def test_attention_and_norms_pass_through():
    t = torch.zeros(4, 4, dtype=torch.uint32)
    stream = [("model.layers.0.self_attn.q_proj.weight", t),
              ("model.layers.0.input_layernorm.weight", torch.zeros(4))]
    out_map = dict(transform_mlx_weights(stream, group_size=64, bits=4, num_experts=1))
    assert "model.layers.0.self_attn.q_proj.weight" in out_map
    assert "model.layers.0.input_layernorm.weight" in out_map
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/enyouki/tpu-inference && source .venv/bin/activate && python -m pytest tests/models/vllm/test_mlx_weight_transform.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the transform**

```python
# tpu_inference/models/vllm/mlx_weight_transform.py
import re
from typing import Iterable, Iterator

import jax.numpy as jnp
import torch
from torchax.interop import jax_view, torch_view

from tpu_inference.layers.common.quantization import mlx_dequantize

_SWITCH = re.compile(r"^(.*)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$")
_DEQUANT_PREFIXES = ("model.embed_tokens", "lm_head")


def _dequant_to_bf16(weight: torch.Tensor, scales: torch.Tensor,
                     biases: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    w = mlx_dequantize(jax_view(weight.to(torch.uint32) if weight.dtype != torch.uint32 else weight),
                       jax_view(scales), jax_view(biases),
                       group_size=group_size, bits=bits)
    return torch_view(w.astype(jnp.bfloat16))


def transform_mlx_weights(weights: Iterable[tuple[str, torch.Tensor]], *,
                          group_size: int, bits: int, num_experts: int
                          ) -> Iterator[tuple[str, torch.Tensor]]:
    # Buffer embed/lm_head triplets to dequant when all three parts have arrived.
    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        m = _SWITCH.match(name)
        if m is not None:
            prefix, proj, suffix = m.group(1), m.group(2), m.group(3)
            for e in range(num_experts):
                yield (f"{prefix}.mlp.experts.{e}.{proj}.{suffix}", tensor[e].contiguous())
            continue
        base = next((p for p in _DEQUANT_PREFIXES
                     if name.startswith(p) and name[len(p):] in (".weight", ".scales", ".biases")), None)
        if base is not None:
            slot = pending.setdefault(base, {})
            slot[name[len(base) + 1:]] = tensor
            if {"weight", "scales", "biases"} <= slot.keys():
                yield (f"{base}.weight",
                       _dequant_to_bf16(slot["weight"], slot["scales"], slot["biases"],
                                        group_size, bits))
                del pending[base]
            continue
        yield (name, tensor)
    # Any embed/lm_head that was already plain bf16 (no scales/biases) passes through:
    for base, slot in pending.items():
        for suffix, tensor in slot.items():
            yield (f"{base}.{suffix}", tensor)
```

- [ ] **Step 4: Override `get_all_weights` in the loader**

In `tpu_inference/models/vllm/vllm_model_loader.py`, add to `IncrementalModelLoader`:
```python
    def get_all_weights(self, model_config, model):
        weights = super().get_all_weights(model_config, model)
        hf_config = model_config.hf_config
        from tpu_inference.layers.vllm.quantization.mlx import is_mlx_quantized
        if is_mlx_quantized(hf_config):
            from tpu_inference.models.vllm.mlx_weight_transform import transform_mlx_weights
            q = getattr(hf_config, "quantization_config", None) or getattr(hf_config, "quantization")
            return transform_mlx_weights(
                weights, group_size=q["group_size"], bits=q["bits"],
                num_experts=hf_config.num_experts)
        return weights
```

- [ ] **Step 5: Run the test**

Run: `cd /home/enyouki/tpu-inference && source .venv/bin/activate && python -m pytest tests/models/vllm/test_mlx_weight_transform.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(loader): MLX weight-stream transform (unstack experts, dequant embed/lm_head)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: `VllmMLXMoEMethod` — Stage 1 baseline (dequant-at-load → bf16, model serves correctly)

**Files:**
- Modify: `tpu_inference/layers/vllm/quantization/mlx.py` (add `VllmMLXMoEMethod`; wire MoE branch in `get_quant_method`)
- Test: `tests/models/vllm/test_qwen3_moe_mlx_int4_e2e.py`

**Reference:** `VllmAWQMoEMethod` (`awq.py:235-501`) for `create_weights` param registration + `__init__` (backend selection); `VllmUnquantizedFusedMoEMethod.apply_monolithic` (`unquantized.py:370-391`) for the apply body to mirror. `process_moe_weights`/`shard_moe_weights` expect `[E, out, in]` and swapaxes internally.

**Interfaces:**
- Consumes: `get_moe_config(layer)`; `FusedMoEWeights(w13_weight, w13_weight_scale, w13_bias, w2_weight, w2_weight_scale, w2_bias)`; `vllm_moe_apply(layer, weights, quant_method_instance, x, router_logits)`; `process_moe_weights(weights, moe_backend, w13_reorder_size=None, w13_interleave=False)`; `shard_moe_weights(weights, moe_backend, mesh)`.
- Produces: `VllmMLXMoEMethod(quant_config, layer, mesh)` with `is_monolithic=True`, packed stacked params (`w13_qweight` uint32 `[E, 2I, H//8]`, `w13_scales`/`w13_biases` bf16 `[E, 2I, H//64]`, `w2_qweight` `[E, H, I//8]`, `w2_scales`/`w2_biases` `[E, H, I//64]`), and for milestone A stores dequantized bf16 `w13_weight [E,2I,H]`/`w2_weight [E,H,I]` after loading.

- [ ] **Step 1: Write the failing e2e numerical test (MLX vs bf16 reference, same vllm path)**

```python
# tests/models/vllm/test_qwen3_moe_mlx_int4_e2e.py
import os
import tempfile

import numpy as np
import pytest

os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")

import jax  # noqa: E402
from tests.utils.mlx_synthetic import build_synthetic_mlx_moe, build_bf16_reference_moe  # noqa: E402


@pytest.mark.skipif(not jax.devices(), reason="requires TPU")
def test_synthetic_mlx_moe_logits_match_bf16_reference():
    from vllm import LLM, SamplingParams
    with tempfile.TemporaryDirectory() as mlx_dir, tempfile.TemporaryDirectory() as ref_dir:
        meta = build_synthetic_mlx_moe(mlx_dir, layers=2, experts=8, hidden=128, moe_inter=64)
        build_bf16_reference_moe(ref_dir, meta["golden"], layers=2, experts=8, hidden=128, moe_inter=64)

        sp = SamplingParams(max_tokens=8, temperature=0.0, logprobs=5)
        prompt_ids = [1, 5, 9, 13, 2, 7]

        mlx = LLM(model=mlx_dir, tensor_parallel_size=1, max_model_len=64,
                  enforce_eager=True, dtype="bfloat16")
        out_mlx = mlx.generate({"prompt_token_ids": prompt_ids}, sp)
        del mlx

        ref = LLM(model=ref_dir, tensor_parallel_size=1, max_model_len=64,
                  enforce_eager=True, dtype="bfloat16")
        out_ref = ref.generate({"prompt_token_ids": prompt_ids}, sp)
        del ref

        assert out_mlx[0].outputs[0].token_ids == out_ref[0].outputs[0].token_ids
```

> NOTE for implementer: confirm `build_synthetic_mlx_moe`/`build_bf16_reference_moe` write a `config.json` vLLM accepts as `Qwen3MoeForCausalLM`. If `mlx_synthetic.py` needs a tokenizer or extra config keys for `LLM(...)` to load, extend it minimally (it already writes a Qwen3-MoE config). If greedy token-id equality is too brittle at bf16, relax to comparing top-1 logits of the first decoded step within `atol=0.5`.

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/enyouki/tpu-inference && source .venv/bin/activate && MODEL_IMPL_TYPE=vllm python -m pytest tests/models/vllm/test_qwen3_moe_mlx_int4_e2e.py -v -s`
Expected: FAIL (MoE method returns None → model build or load error).

- [ ] **Step 3: Implement `__init__` + `create_weights` (packed stacked params)**

Mirror `VllmAWQMoEMethod.__init__` (`awq.py:237`) for backend selection. Register packed params (plain `Parameter(torch.empty(...))` + `set_weight_attrs`, exactly as AWQ MoE does — NOT typed Parameter classes):

```python
class VllmMLXMoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config, layer, mesh, ep_axis_name="model"):
        from tpu_inference.layers.common.moe import select_moe_backend_from_fused_moe_config
        super().__init__(layer.moe_config)
        self.quant_config = quant_config
        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)
        self.extra_backend_kwargs = {}

    @property
    def is_monolithic(self):
        return True

    def get_fused_moe_quant_config(self, layer):
        return None

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **extra):
        from vllm.model_executor.utils import set_weight_attrs
        gs, pf = self.quant_config.group_size, self.quant_config.pack_factor
        E, H, I = num_experts, hidden_size, intermediate_size_per_partition
        def reg(name, shape, dtype):
            p = torch.nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, extra)
        reg("w13_qweight", (E, 2 * I, H // pf), torch.uint32)
        reg("w2_qweight",  (E, H, I // pf),     torch.uint32)
        reg("w13_scales",  (E, 2 * I, H // gs), params_dtype)
        reg("w2_scales",   (E, H, I // gs),     params_dtype)
        reg("w13_biases",  (E, 2 * I, H // gs), params_dtype)
        reg("w2_biases",   (E, H, I // gs),     params_dtype)
```

> NOTE: AWQ registers experts as `[E, hidden, 2I//pf]` (it packs along output). MLX packs along the **input** dim, so `w13` is `[E, 2I, H//pf]` (output-major, input packed) and `w2` is `[E, H, I//pf]`. Confirm vLLM's `FusedMoE.weight_loader` writes per-expert slices into these shapes given the transformed per-expert names from Task 6; the e2e test verifies routing.

- [ ] **Step 4: Implement `process_weights_after_loading` (milestone A: dequant → bf16, kernel layout, shard)**

```python
    def process_weights_after_loading(self, layer):
        import jax
        import jax.numpy as jnp
        from torchax.interop import jax_view, torch_view
        from tpu_inference.layers.common.quantization import mlx_dequantize
        from tpu_inference.layers.common.process_weights.moe_weights import (
            FusedMoEWeights, process_moe_weights, shard_moe_weights)
        gs, bits = self.quant_config.group_size, self.quant_config.bits

        @jax.jit
        def _dequant(w13q, w13s, w13b, w2q, w2s, w2b):
            w13 = mlx_dequantize(w13q, w13s, w13b, group_size=gs, bits=bits)  # [E,2I,H]
            w2 = mlx_dequantize(w2q, w2s, w2b, group_size=gs, bits=bits)      # [E,H,I]
            return w13, w2

        w13, w2 = _dequant(
            jax_view(layer.w13_qweight), jax_view(layer.w13_scales), jax_view(layer.w13_biases),
            jax_view(layer.w2_qweight), jax_view(layer.w2_scales), jax_view(layer.w2_biases))
        weights = FusedMoEWeights(
            w13_weight=w13, w13_weight_scale=None, w13_bias=None,
            w2_weight=w2, w2_weight_scale=None, w2_bias=None)
        weights = process_moe_weights(weights, self.moe_backend)
        weights = shard_moe_weights(weights, self.moe_backend, self.mesh)
        for name in ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales",
                     "w13_biases", "w2_biases"):
            delattr(layer, name)
        layer.w13_weight = torch.nn.Parameter(torch_view(weights.w13_weight), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch_view(weights.w2_weight), requires_grad=False)
```

- [ ] **Step 5: Implement `apply_monolithic` (mirror unquantized)**

```python
    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        from torchax.interop import jax_view
        from tpu_inference.layers.common.process_weights.moe_weights import FusedMoEWeights
        from tpu_inference.layers.vllm.interface.moe import vllm_moe_apply
        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight), w13_weight_scale=None, w13_bias=None,
            w2_weight=jax_view(layer.w2_weight), w2_weight_scale=None, w2_bias=None)
        return vllm_moe_apply(layer=layer, weights=weights, quant_method_instance=self,
                              x=x, router_logits=router_logits)
```

- [ ] **Step 6: Wire MoE dispatch in `get_quant_method`**

Add to the `match layer` block in `VllmMLXConfig.get_quant_method`:
```python
            case FusedMoE():
                layer.moe_config = self.get_moe_config(layer)
                return VllmMLXMoEMethod(self, layer, self.mesh)
```
(Add `from vllm.model_executor.layers.fused_moe import FusedMoE` and `from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase` imports.)

- [ ] **Step 7: Run the e2e test**

Run: `cd /home/enyouki/tpu-inference && source .venv/bin/activate && MODEL_IMPL_TYPE=vllm python -m pytest tests/models/vllm/test_qwen3_moe_mlx_int4_e2e.py -v -s`
Expected: PASS (MLX logits/tokens match the bf16 reference). Debug routing/shape errors against AWQ until green. **This is the primary correctness gate.**

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat(quant): VllmMLXMoEMethod stage-1 baseline (dequant-at-load), synthetic e2e passes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Production wiring + real 30B bring-up (correctness milestone)

**Files:**
- Modify: `tpu_inference/layers/vllm/quantization/__init__.py` (or wherever vLLM resolves `load_format`/loader) — auto-select the streaming loader when MLX is detected.
- Modify (likely): `tpu_inference/layers/vllm/quantization/mlx.py` (`VllmMLXLinearMethod.process_weights_after_loading` sharding for RowParallel input dim).
- Optionally add a gated smoke test to `tests/models/vllm/test_qwen3_moe_mlx_int4_e2e.py`.

**Interfaces:**
- Consumes: the whole Stage-1 pipeline. Model id `mlx-community/Qwen3-30B-A3B-4bit`.

This locks in **"the model serves correctly"** before any kernel surgery. Experts are bf16 in HBM at this stage (~7.6GB/chip, fits) — this is a CORRECTNESS milestone, NOT the production memory/perf end-state (that is Task 10). Acceptance here is coherent output + a recorded Stage-1 perf baseline; the rigorous correctness proof is Task 13.

- [ ] **Step 1: PREREQUISITE — download the real MLX weights (~17GB)**

The old plan's "already cached" assumption was **WRONG**: only `config.json` + the safetensors index (~148K) are present, none of the ~17GB of shards. Download first:
```bash
cd /home/enyouki/tpu-inference && source .venv/bin/activate
huggingface-cli download mlx-community/Qwen3-30B-A3B-4bit
```
Confirm 4 shards land under `~/.cache/huggingface/hub/models--mlx-community--Qwen3-30B-A3B-4bit/`.

- [ ] **Step 2: PRODUCTION LOAD WIRING — auto-select the streaming loader for MLX**

Today the synthetic e2e (Task 7) only loads because it hardcodes `load_format="tpu_streaming_loader"`. Under a plain `LLM(model=...)` (`load_format="auto"`), vLLM picks `DefaultModelLoader` — the Task-6 `transform_mlx_weights` (override on `IncrementalModelLoader`) **never runs**, so load crashes on the MLX-only `lm_head.biases` (and `switch_mlp` names never get un-stacked). For production `LLM(model=...)` to work, auto-select the streaming loader (`IncrementalModelLoader`) when `is_mlx_quantized(hf_config)`.

Investigate where vLLM resolves `load_format` into a loader on the torchax path (likely near the quant-detection injection in `tpu_inference/layers/vllm/quantization/__init__.py`, or model-config / `vllm_model_wrapper.py` setup where the loader is chosen). Wire it so that when MLX is detected, `load_format` is forced to the streaming loader (the one whose `get_all_weights` override applies `transform_mlx_weights`). Verify a plain `LLM(model="mlx-community/Qwen3-30B-A3B-4bit", ...)` reaches `transform_mlx_weights` (add a temporary log/breakpoint or unit-assert the resolved loader class is `IncrementalModelLoader`).

- [ ] **Step 3: tp=8 SHARDING FIX — RowParallel input-dim sharding of packed tensors**

`VllmMLXLinearMethod.process_weights_after_loading` shards all 3 packed tensors (`weight`/`scales`/`biases`) with `self.linear_config.weight_sharding`. But RowParallelLinear (attention `o_proj`) uses `P(None, ATTN_HEAD)` — it shards the **INPUT** dim. For MLX, the input dim is packed (`in//8`, 8 nibbles/uint32 word) and grouped (`in//64`, one scale/bias per group of 64). Sharding that input dim at uint32-word / 64-group granularity via `weight_sharding` splits a uint32 word or a quant group across chips → **wrong dequant at TP>1**. This is the Stage-1 bug (linears stay packed), so it bites here.

Fix it so the RowParallel input-dim split lands on group/word boundaries (e.g. shard the unpacked logical input dim consistently across `weight` `[out, in//8]`, `scales`/`biases` `[out, in//64]` so each shard owns whole words AND whole groups; `I/tp` must be divisible by both 8 and 64). Resolve and verify (single-layer numeric check or the synthetic e2e at tp=2 if feasible) BEFORE the real run.

- [ ] **Step 4: Run real 30B at tp=8; confirm coherent + capture Stage-1 perf baseline**

Run (the Task-2 known-good offline_inference command, scaled to the 30B):
```bash
cd /home/enyouki/tpu-inference && source .venv/bin/activate
MODEL_IMPL_TYPE=vllm SKIP_JAX_PRECOMPILE=1 \
  python examples/offline_inference.py \
    --model mlx-community/Qwen3-30B-A3B-4bit \
    --max-model-len 2048 --max-tokens 64 --tensor-parallel-size 8
```
Expected: loads via the auto-selected streaming loader (Step 2) and prints **coherent, non-degenerate** completions. If garbage: suspect (a) nibble order / affine sign in dequant, (b) qkv/gate_up fusion of scales/biases, (c) expert routing/un-stack order, (d) the tp=8 input-dim sharding of Step 3. Bisect with the synthetic test + a single-layer comparison.

Acceptance is **NOT a human eyeball alone**:
- Minimum bar to pass Task 8: coherent output **AND** a recorded Stage-1 throughput baseline (tokens/sec) — run one Task-12-style `vllm bench throughput` pass and write the number into `.superpowers/sdd/progress.md` (Task 12 compares Stage-2 against it).
- The rigorous independent correctness check is **Task 13** (run at least its PREFERRED dequant-transcription check before declaring real acceptance).

- [ ] **Step 5 (optional): Add a gated smoke test**

Add to `tests/models/vllm/test_qwen3_moe_mlx_int4_e2e.py` a `@pytest.mark.skipif`-gated test (env flag `RUN_REAL_MLX=1`) that runs the 30B via plain `LLM(model=...)` (no hardcoded load_format) with greedy decoding on one prompt and asserts non-empty, ASCII-coherent output — this also guards the Step-2 auto-loader wiring.

- [ ] **Step 6: Commit (if a test/wiring change was made)**

```bash
git add -A && git commit -m "feat(loader): auto-select MLX streaming loader + tp=8 sharding fix; real 30B bring-up

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Stage 2 — extend `gmm_v2` with a per-group bias (`rhs_groupbias`)

**Scope: ~13 edit sites in `gmm_v2.py`** — this is a structural clone of the per-group scale path. The kernel unit test (Step 2) is the arbiter.

**Files:**
- Modify: `tpu_inference/kernels/megablox/gmm_v2.py` (add the per-group additive-bias path, mirroring the existing per-group scale path)
- Test: `tests/kernels/megablox/test_gmm_v2_groupbias.py`

**Why:** `gmm_v2` already bitcast-unpacks uint32 int4 in-kernel and applies a per-group `rhs_scale [G, num_blocks, 1, N]` inside the k-loop. MLX is affine (`w = scale·q + bias`); the existing `rhs_bias [G,1,N]` (`:471-474`) is per-output-channel, NOT per-group, so it cannot represent the affine term. We add a parallel **per-group** bias whose contribution to output `[t,n]` is `Σ_g bias[g,n]·groupsum_x[t,g]`, accumulated in the k-loop before the fused activation (Approach B — the only approach compatible with w13's in-kernel `fuse_act`).

**CRITICAL — there are TWO k-loop application sites, not one.** The new `rhs_groupbias` must be applied at BOTH:
- `gmm_v2.py:386-391` — the **unquantized-lhs** path (the old plan named only the next one and missed this).
- `gmm_v2.py:457-462` — the **quantized-lhs** path.
Missing 386-391 silently produces wrong results on whichever path the MoE actually takes.

**Exact edit-site list (~13, from the remediation map):**
- public `gmm_v2` signature param `rhs_groupbias=None` — `:1134-1135`
- `InputConfigs.has_scale` → add `has_groupbias` — `:175`
- `validate_inputs` — `:927`, `:944-947`
- `make_gmm_configs` signature+body + its call — `:1017`, `:1035-1051`, `:1185-1189`
- `get_cost_estimate` — `:988-990`
- scale `index_map` clone for groupbias — `:237-245`
- `generate_block_specs` block_spec + `WeightsRef` wiring — `:291-301`
- HBM `BlockSpec` (+ float32 cast) — `:1204-1207`
- `WeightsRef` construction — `:1255`; `pallas_call` `in_specs` — `:1264-1268`
- Ref accessors: `RhsRef.get_scale` `:91-93`, `WeightsRef` field+getter `:106-114`, **`FusedWeightsRef.get_scale` gate/up concat split `:134-137`** (groupbias needs the SAME split)
- **k-loop application — BOTH `:386-391` (unquantized-lhs) AND `:457-462` (quantized-lhs)**

New param `rhs_groupbias` is shaped/index-mapped exactly like `rhs_scale` (`[G, num_blocks, 1, N]`, float32), and is **zero-overhead when `None`** (fully backward-compatible — existing callers unaffected).

**Interfaces:**
- Produces: `gmm_v2(..., rhs_groupbias: jax.Array | None = None, ...)` where `rhs_groupbias` has the same shape/index-map as `rhs_scale` (`[G, num_blocks, 1, N]`, float32). When provided, the kernel adds `groupbias[b_id, :, n] * jnp.sum(block_lhs, axis=1, keepdims=True)` to the block accumulator each k-block, before `apply_act_fn`.

- [ ] **Step 1: Spike — map the existing scale path wiring**

Read `gmm_v2.py` and write down (in the task notes) every place `rhs_scale` is threaded — use the ~13-site list above as the checklist. The groupbias path is a structural clone of each. Pay particular attention to the TWO application points: the quantized-lhs inner loop `:457-462` AND the unquantized-lhs inner loop `:386-391` (BOTH must apply groupbias), and the `FusedWeightsRef` gate/up concat split `:134-137` (groupbias needs the same split as scale).

- [ ] **Step 2: Write the failing kernel unit test (affine grouped-matmul reference)**

```python
# tests/kernels/megablox/test_gmm_v2_groupbias.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2


@pytest.mark.skipif(not jax.devices(), reason="requires TPU")
def test_gmm_v2_per_group_bias_matches_affine_reference():
    G, M, K, N, gs = 2, 32, 128, 64, 64
    num_blocks = K // gs
    rng = np.random.default_rng(0)
    lhs = rng.standard_normal((M, K)).astype(np.float32)
    q = rng.integers(0, 16, size=(G, K, N)).astype(np.int32)          # 4-bit codes
    scale = rng.standard_normal((G, num_blocks, 1, N)).astype(np.float32)
    gbias = rng.standard_normal((G, num_blocks, 1, N)).astype(np.float32)
    group_sizes = jnp.array([M // 2, M - M // 2], dtype=jnp.int32)

    # Reference: dequantize affine then grouped matmul.
    def deq(g):
        s = np.repeat(scale[g, :, 0, :], gs, axis=0)   # [K, N]
        b = np.repeat(gbias[g, :, 0, :], gs, axis=0)   # [K, N]
        return q[g].astype(np.float32) * s + b         # [K, N]
    ref = np.zeros((M, N), np.float32)
    row = 0
    for g in range(G):
        n = int(group_sizes[g]); w = deq(g)
        ref[row:row + n] = lhs[row:row + n] @ w
        row += n

    out = gmm_v2(jnp.asarray(lhs), jnp.asarray(q), group_sizes,
                 rhs_scale=jnp.asarray(scale), rhs_groupbias=jnp.asarray(gbias))
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-1, rtol=1e-1)
```

> NOTE: match the actual `gmm_v2` calling convention for `q`/scale dtype and packing from Step 1 — the test above passes unpacked int codes for clarity; if `gmm_v2` requires uint32-packed rhs, pack `q` with the repo's helper and set the quant dtype as the scale path expects. Adjust kwarg names to the real signature.

- [ ] **Step 3: Run to verify it fails**

Run: `cd /home/enyouki/tpu-inference && source .venv/bin/activate && python -m pytest tests/kernels/megablox/test_gmm_v2_groupbias.py -v`
Expected: FAIL (`gmm_v2() got an unexpected keyword argument 'rhs_groupbias'`).

- [ ] **Step 4: Implement the per-group bias path**

Mirror the scale path point by point, hitting all ~13 sites in the edit-site list above:
1. Add `rhs_groupbias=None` to the `gmm_v2` signature (`:1134-1135`) and thread it like `rhs_scale`.
2. Add `InputConfigs.has_groupbias` (`:175`); wire `validate_inputs` (`:927`, `:944-947`), `make_gmm_configs` (`:1017`, `:1035-1051`, `:1185-1189`), `get_cost_estimate` (`:988-990`).
3. Clone the scale `index_map` (`:237-245`) and `generate_block_specs` block_spec + `WeightsRef` wiring (`:291-301`) for `rhs_groupbias`.
4. Wire its HBM `BlockSpec` (+ float32 cast, `:1204-1207`), `WeightsRef` construction (`:1255`), and `pallas_call` `in_specs` (`:1264-1268`).
5. Add a ref accessor (e.g. `get_groupbias`) on the ref classes (`RhsRef` `:91-93`, `WeightsRef` field+getter `:106-114`), **including the `FusedWeightsRef` gate/up concat split `:134-137`** (same split as scale).
6. In the inner k-loop, immediately after the scale multiply and before `apply_act_fn`, add the groupbias accumulation:
```python
if cfg.has_groupbias:
    gb = rhs_groupbias_ref[...]            # slice for this block/n-range, like scale
    block_acc += gb * jnp.sum(block_lhs, axis=1, keepdims=True)
```
**Add this at BOTH k-loop sites: `:386-391` (unquantized-lhs) AND `:457-462` (quantized-lhs).** Missing either silently produces wrong results on whichever path the MoE takes. Reuse the existing ragged-`valid_k` masking already applied to `block_lhs`.

> NOTE: this is intricate Pallas code; the kernel unit test in Step 2 is the arbiter. Keep `rhs_groupbias=None` fully backward-compatible (zero overhead when absent) so all existing `gmm_v2` callers are unaffected — run a couple of existing megablox tests to confirm.

- [ ] **Step 5: Run the test + an existing gmm_v2 regression test**

Run:
```bash
cd /home/enyouki/tpu-inference && source .venv/bin/activate
python -m pytest tests/kernels/megablox/test_gmm_v2_groupbias.py -v
python -m pytest tests/kernels/megablox -k "gmm" -q   # existing callers unaffected
```
Expected: new test PASS; existing gmm tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(kernel): gmm_v2 per-group additive bias (rhs_groupbias) for affine quant

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: Stage 2 — wire `VllmMLXMoEMethod` to the in-kernel packed path

**Scope: ~30 edit sites across 4 files — this is NOT "one wiring edit".** The old plan claimed "one wiring edit in `fused_moe_gmm.py`" and offered "(a) repurpose `w13_bias`" — **BOTH are deleted as wrong.** `w13_bias`/`w2_bias` are a per-channel MLP bias `[E,1,out]` applied ONCE post-matmul; the MLX affine bias is per-quant-block `[E,num_blocks,1,N]` applied INSIDE the k-loop. They are NOT interchangeable. Stage-2 requires a **NEW field `w13_groupbias`/`w2_groupbias`**, threaded everywhere `w13_weight_scale`/`w2_weight_scale` is threaded.

**Files:**
- Modify: `tpu_inference/layers/common/process_weights/moe_weights.py` (10a — new groupbias field + process/shard, ~10 sites)
- Modify: `tpu_inference/layers/common/fused_moe_gmm.py` (10b — thread groupbias through the gmm chain, ~17 sites)
- Modify: `tpu_inference/layers/common/moe.py` (10c — dispatch reads groupbias, 1–2 sites)
- Modify: `tpu_inference/layers/vllm/quantization/mlx.py` (10c — `VllmMLXMoEMethod.process_weights_after_loading` producer + `apply_monolithic`)
- Test: `tests/models/vllm/test_qwen3_moe_mlx_int4_e2e.py` (unchanged — must still pass)

**Interfaces:**
- Consumes: `gmm_v2(..., rhs_groupbias=...)` from Task 9; `process_moe_weights`/`shard_moe_weights`; `FusedMoEWeights`.
- Produces: experts kept as uint32-packed int4 + per-group scale + per-group bias in HBM (true 4-bit, ~2GB/chip); dequant happens in `gmm_v2` for only the routed experts.

- [ ] **Step 1: PRE-CHECKS — backend + tp=8 group-divisibility**

(a) Confirm the active MoE backend routes through `gmm_v2`: inspect `self.moe_backend` (or temporary print/assert in `apply_monolithic` during the Stage-1 e2e) — it must be `GMM_EP`/`GMM_TP`, NOT `DENSE_MAT`/`MEGABLX_GMM` (those `raise NotImplementedError` in `process_moe_weights`, `moe_weights.py:396-403`). Record the backend.

(b) **tp=8 group-divisibility pre-check.** Qwen3-30B has `I=768`. At tp=8, `I/tp = 96`. For the packed-w2 path the input dim is sharded; `96 // 64 = 1.5` is **non-integer**, so w2 scales/biases (one per group of 64 along the input) **cannot shard at group granularity** — this will fail loudly. Determine before threading whether w2 is sharded on the grouped input dim under the active backend (GMM_TP vs GMM_EP differ here). If it is, this blocks true-4bit w2 at tp=8; surface it (do not let it silently corrupt). Record the finding and decision (e.g. EP keeps full experts per chip → no input-dim group split; or a different tp; or w2 stays bf16). Resolve before 10a.

- [ ] **Step 2 (10a): `moe_weights.py` — new `w13_groupbias`/`w2_groupbias` field (~10 sites)**

Add the new field and thread it through process/shard, mirroring `w13_weight_scale`/`w2_weight_scale` exactly (copy the site list):
1. Dataclass `FusedMoEWeights` fields `w13_groupbias`/`w2_groupbias` — `:40-48`
2. Extract in `process_moe_weights` — `~:234-236`
3. `FUSED_MOE` transform block — `~:329-340`
4. `GMM_TP` transform (`process_w13_for_gmm`, `concat_dim=3`) — `:363-371`
5. `GMM_EP` transform — `:384-392`
6. Return `FusedMoEWeights(...)` — `:405-412`
7. `shard_moe_weights` EP shardings — `:423-430`
8. `shard_moe_weights` TP shardings — `:440-464`
9. Layout block `FUSED_MOE` — `:468-475`
10. Layout block `GMM_TP`/`EP` — `:477-484`

Lay groupbias into the same `[E, num_blocks, 1, N]` form `process_moe_weights` produces for scale. Keep the per-channel `w13_bias`/`w2_bias` (the MLP bias) untouched.

- [ ] **Step 3 (10b): `fused_moe_gmm.py` — thread groupbias through the gmm chain (~17 sites)**

groupbias is a **NEW sharded array** → it needs its own `shard_map` `in_specs` entry AND its own positional arg in BOTH `tensor_parallel_gmm` AND `expert_parallel_gmm` (it CANNOT reuse scale's — these are not free):
13. `fused_moe_func` signature — `:426-433`
14. → `expert_parallel_gmm` call — `:560-576`
15. → `tensor_parallel_gmm` call — `:578-594`
16. `expert_parallel_gmm`: signature + scale_spec defs + `shard_map` `in_specs` + positional args — `:322-339`, `:347-350`, `:362-374`, `:377-389`
17. `tensor_parallel_gmm`: same — `:248-265`, `:272-280`, `:292-304`, `:307-319`
18. `moe_gmm_local` signature — `:129-147`
19. → `gmm_wrapper` call ×3 (w1; w2 SC path; w2 non-SC path) — `:156-165`, `:186-192`, `:209-215`
20. `gmm_wrapper` signature — `:93-100`
21. → `gmm_v2` call (forward `rhs_groupbias=...`) — `:101-111`

- [ ] **Step 4 (10c): `moe.py` dispatch + `mlx.py` producer/apply**

(a) `tpu_inference/layers/common/moe.py`: `fused_moe_func(...)` dispatch reads `w13`/`w2` scale + groupbias — `:137-140` (and the `FUSED_MOE` branch `~:119-122` if in scope).

(b) `mlx.py` `process_weights_after_loading` (producer): keep packed uint32 weight, build `FusedMoEWeights` carrying scale AND the new groupbias, `process_moe_weights` → `shard_moe_weights`, store packed weight + scale + groupbias (do NOT dequant, do NOT delete the packed params):
```python
    def process_weights_after_loading(self, layer):
        from torchax.interop import jax_view, torch_view
        from tpu_inference.layers.common.process_weights.moe_weights import (
            FusedMoEWeights, process_moe_weights, shard_moe_weights)
        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_qweight), w13_weight_scale=jax_view(layer.w13_scales),
            w13_groupbias=jax_view(layer.w13_biases), w13_bias=None,
            w2_weight=jax_view(layer.w2_qweight), w2_weight_scale=jax_view(layer.w2_scales),
            w2_groupbias=jax_view(layer.w2_biases), w2_bias=None)
        weights = process_moe_weights(weights, self.moe_backend)   # now handles groupbias
        weights = shard_moe_weights(weights, self.moe_backend, self.mesh)
        # store back onto the layer: packed weight + scale + groupbias (keep packed)
        ...
```

(c) `mlx.py` `apply_monolithic`: pass packed weight + scale + groupbias straight through (no XLA dequant):
```python
    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        from torchax.interop import jax_view
        from tpu_inference.layers.common.process_weights.moe_weights import FusedMoEWeights
        from tpu_inference.layers.vllm.interface.moe import vllm_moe_apply
        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight), w13_weight_scale=jax_view(layer.w13_weight_scale),
            w13_groupbias=jax_view(layer.w13_groupbias), w13_bias=None,
            w2_weight=jax_view(layer.w2_weight), w2_weight_scale=jax_view(layer.w2_weight_scale),
            w2_groupbias=jax_view(layer.w2_groupbias), w2_bias=None)
        return vllm_moe_apply(layer=layer, weights=weights, quant_method_instance=self,
                              x=x, router_logits=router_logits)
```

- [ ] **Step 5: Re-run the synthetic e2e numerical test (must still pass)**

Run: `cd /home/enyouki/tpu-inference && source .venv/bin/activate && MODEL_IMPL_TYPE=vllm python -m pytest tests/models/vllm/test_qwen3_moe_mlx_int4_e2e.py -v -s`
Expected: PASS — identical outputs to Stage 1. If it regresses, the scale/groupbias layout or the gmm-chain threading is off; bisect against the Task 9 kernel test.

- [ ] **Step 6: Re-run the real 30B + check footprint**

Run the Task 8 offline-inference command again. Expected: coherent output AND lower HBM (experts now ~2GB/chip packed vs ~7.6GB bf16). Confirm via the loader's memory log or `jax` device memory stats. (This is the HBM drop the Task-12 perf gate also records.)

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(quant): MLX MoE in-kernel dequant (packed int4 + per-group scale+groupbias in gmm_v2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: Method-level CPU unit tests for `VllmMLXLinearMethod` + `VllmMLXMoEMethod`

**Files:**
- Test: `tests/layers/vllm/quantization/test_mlx_linear_method.py`
- Test: `tests/layers/vllm/quantization/test_mlx_moe_method.py`

**Why:** Today the *config* class is tested but the *method* classes are NOT — `grep -rn "VllmMLXLinearMethod\|VllmMLXMoEMethod" tests/` = ZERO hits. The qkv/gate_up fusion split (`apply` slice/concat) and the MoE expert un-stacking/registration logic have no isolated, fast, CPU-runnable test (the einsum-only `test_mlx_linear.py` and the TPU-only e2e don't cover them). These tests run in seconds and catch fusion/un-stacking bugs the math-only tests can't.

**Interfaces:**
- Consumes: `VllmMLXLinearMethod(quant_config, linear_config)` and `VllmMLXMoEMethod(quant_config, layer, mesh)` with fake `nn.Module` layers + fake linear/moe configs — NO TPU, NO `LLM(...)`. `tests/utils/mlx_synthetic._quantize_affine`.

- [ ] **Step 1: `VllmMLXLinearMethod` unit test (incl. fused gate_up)**

Build a stub `nn.Module` layer + a fake `linear_config` (`output_sizes`, `n_shards`, `mesh`, `weight_sharding`, `bias_sharding`, `num_proj`). Then:
- `create_weights(...)` → assert param shapes/dtypes: `weight` uint32 `[out, in//8]`; `scales`/`biases` bf16 `[out, in//64]`.
- Load synthetic `_quantize_affine` packs into the params; call `process_weights_after_loading`.
- `apply(layer, x)` and compare to `x @ golden.T` within bf16 tol — **INCLUDING a fused gate_up case (`output_sizes=[I, I]`, `num_proj=2`)** to cover the slice/concat split path (the single most untested branch).

- [ ] **Step 2: `VllmMLXMoEMethod` unit test (expert un-stacking)**

Build a fake `FusedMoE` layer (with `moe_config`) + single-device mesh:
- `create_weights(...)` → assert stacked shapes: `w13_qweight [E, 2I, H//8]`, `w13_scales/biases [E, 2I, H//64]`, `w2_qweight [E, H, I//8]`, `w2_scales/biases [E, H, I//64]`.
- Load per-expert slices via the weight_loader, run the dequant path (Stage-1 `process_weights_after_loading`), assert against golden experts — covers expert un-stacking/registration.
- If any single step is unavoidably TPU-bound (e.g. a sharding call), gate **only that assertion** with a device check; keep all shape/dtype/registration checks CPU-runnable.

- [ ] **Step 3: Run + commit**

```bash
cd /home/enyouki/tpu-inference && source .venv/bin/activate
python -m pytest tests/layers/vllm/quantization/test_mlx_linear_method.py tests/layers/vllm/quantization/test_mlx_moe_method.py -v
git add -A && git commit -m "test(quant): CPU method-level tests for VllmMLXLinear/MoE (fusion split + expert un-stacking)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 12: Performance gate (tokens/sec, TTFT, TPOT, KV-cache headroom)

**Files:** none (benchmark milestone). Record numbers in `.superpowers/sdd/progress.md`.

**Why:** The plan had ZERO perf acceptance criteria. The user's goal — usable OpenRouter coding speed — is unverifiable without a tokens/sec + latency gate. Without a Stage-1 baseline number (captured in Task 8), there is nothing to regression-check Stage-2 against.

**Tooling (verified in venv):** `vllm bench {throughput,latency,serve}` CLI (vLLM 0.22.0); repo serving harness `scripts/vllm/benchmarking/benchmark_serving.py` (emits `request_throughput`, `output_throughput`, `total_token_throughput`, full TTFT/TPOT/ITL/E2EL percentiles).

**Realistic shapes for 30B-A3B (3B active) on v6e-8 tp=8:** concurrency 16–64, prompt 1–2k tokens, output 256–1024, `--max-model-len 8192`.

- [ ] **Step 1: Offline throughput (prefill- and decode-heavy)**

```bash
cd /home/enyouki/tpu-inference && source .venv/bin/activate
# prefill-heavy
vllm bench throughput --model mlx-community/Qwen3-30B-A3B-4bit \
  --tensor-parallel-size 8 --input-len 1024 --output-len 256 --num-prompts 200
# decode-heavy (interactive-coding shape)
vllm bench throughput --model mlx-community/Qwen3-30B-A3B-4bit \
  --tensor-parallel-size 8 --input-len 256 --output-len 1024 --num-prompts 200
```
Gate on **output tokens/sec**.

- [ ] **Step 2: Online serving (coding-realistic) — TTFT + TPOT**

Launch `vllm serve mlx-community/Qwen3-30B-A3B-4bit --tensor-parallel-size 8 --max-model-len 8192`, then run `scripts/vllm/benchmarking/benchmark_serving.py` at a target concurrency (16–64). Gate on **median TTFT** and **median TPOT** (TPOT = the interactive-coding latency metric).

- [ ] **Step 3: Record KV-cache headroom**

From the vLLM startup logs, record **free HBM / KV blocks** as a budget line (capacity for concurrent coding sessions).

- [ ] **Step 4: Gate Stage-2 vs Stage-1**

Stage-2 (Task 10) must be **≥ Stage-1 (Task 8) baseline tokens/sec** (regression guard) **AND** show the HBM drop (~7.6→~2GB/chip experts — the whole point of Stage-2). Record both numbers; if Stage-2 regresses throughput, the in-kernel path has a perf bug. No commit (numbers logged in the progress ledger).

---

### Task 13: Independent correctness validation (don't trust the self-oracle)

**Files:**
- Test/script: `tests/layers/common/quantization/test_mlx_dequant_independent.py` (PREFERRED check)
- Optional script for the STRONGER / TASK-ACCURACY checks (not committed unless run).

**Why:** The only correctness gate so far is the synthetic e2e (Task 7), whose oracle (`mlx_synthetic._quantize_affine`) and `mlx_dequantize` are BOTH implementer-authored and share affine math — a shared bug passes Tasks 4 AND 7 while producing wrong real outputs. `mlx`/`mlx_lm` won't run here (Apple-only). So we need an *independent* check. Gate Task 8's real acceptance on at least the PREFERRED check below.

- [ ] **Step 1 (PREFERRED): independent dequant transcription vs `mlx_dequantize` on real shards**

Requires the real MLX weights (Task 8 Step 1 download). Independently transcribe MLX's *published* `mx.dequantize` algorithm (group_size=64, 4-bit, affine `w = scale*q + bias`, low-nibble-first, 8 nibbles/uint32 along input) as a **standalone numpy function** — written from the published spec, **NOT copied from `mlx_dequantize`**. Load a few real `.weight/.scales/.biases` shards from the checkpoint and assert the independent numpy dequant agrees with `mlx_dequantize` (tight tol). Document explicitly that the transcription is genuinely independent (else it can't catch a shared bug).

- [ ] **Step 2 (STRONGER, optional): bf16 behavioral reference**

Download bf16 `Qwen/Qwen3-30B-A3B` (~60GB), load via `transformers` (`Qwen3MoeForCausalLM`), generate reference logits/text on ~5 fixed coding prompts. Compare vs our vLLM 4-bit path with **quant-tolerant metrics** — top-1 / top-5 token agreement, KL or cosine on logits — **NOT exact match** (quant drift is expected). This is the real "is our serving correct" behavioral check.

- [ ] **Step 3 (TASK-ACCURACY, optional): small eval slice**

`pip install lm-eval`; run a small MMLU/GSM8K slice against the served endpoint, 4-bit vs bf16 upcast. Slowest but model-agnostic and implementer-independent.

- [ ] **Step 4: Commit the PREFERRED test**

```bash
cd /home/enyouki/tpu-inference && source .venv/bin/activate
python -m pytest tests/layers/common/quantization/test_mlx_dequant_independent.py -v
git add -A && git commit -m "test(quant): independent MLX dequant cross-check vs published mx.dequantize on real shards

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Recommended execution order for remaining work

The `Task N` headings are kept in spec order (1–13) so the brief-extractor parses them, but the remaining work is best executed in this dependency order:

**11** (CPU method tests — fast, do first; catches fusion/un-stacking bugs cheaply) → **8** (production bring-up — needs weights download + `load_format` auto-select + tp=8 sharding fix; captures the Stage-1 perf baseline) → **13** (independent validation — gate Task 8's real acceptance on at least its PREFERRED check) → **9** (gmm_v2 per-group bias kernel + kernel test) → **10** (Stage-2 wire, ~30 sites) → **12** (perf gate — Stage-1 baseline was captured at Task 8, Stage-2 compared at Task 10).

---

## Self-Review

**Task map (staged):**
- Stage 0 — Cleanup + scaffolding: Tasks 1 (cleanup), 2 (harness sanity), 3 (register + detect), 4 (dequant math gate).
- **Stage 1 — Correctness milestone** (NOT the production end-state): Tasks 5 (linear), 6 (weight transform), 7 (MoE dequant-at-load → bf16), 8 (production load wiring + real 30B serves coherently + Stage-1 perf baseline). At the end of Task 8 the model is *correct* but experts are bf16 in HBM (~7.6GB/chip, ~3.8× memory, dequants all experts/step) — do NOT stop here.
- **Stage 2 — Production end-state** (true 4-bit, ~2GB/chip, in-kernel dequant of ROUTED experts only): Tasks 9 (`gmm_v2` per-group bias + kernel test, ~13 sites), 10 (wire packed int4 + scale + NEW `*_groupbias` field; ~30 sites across 4 files; re-validate; true 4-bit footprint).
- **Cross-cutting (NEW):** Task 11 (CPU method-level unit tests), Task 12 (performance gate — tokens/sec, TTFT, TPOT, KV headroom; Stage-2 ≥ Stage-1), Task 13 (independent correctness validation — don't trust the self-oracle).
- **Production finish line = Task 10 + the Task 12 perf gate.** See "Recommended execution order" above for the dependency-ordered execution sequence (11 → 8 → 13 → 9 → 10 → 12).

**Spec coverage:**
- Registration (5 edits) + detection/config injection → Task 3. If vLLM crashes building `VllmConfig` on the bare quant block, add an early HF-config patch injecting `quant_method="mlx"`. ✓
- Linear method (keep-4bit, dequant-in-apply) → Task 5. ✓
- MoE Stage 1 (dequant-at-load) → Task 7; MoE Stage 2 (in-kernel, true 4-bit) → Tasks 9+10. ✓
- Weight-stream transform (rename/un-stack, dequant embed/lm_head) → Task 6. ✓
- `gmm_v2` per-group `rhs_groupbias` (the only kernel gap for affine; TWO k-loop sites) → Task 9. ✓
- Production load wiring (auto-select streaming loader under `load_format="auto"`) + tp=8 input-dim sharding fix → Task 8. ✓
- Full cleanup → Task 1. ✓
- Method-level CPU unit tests (fusion split + expert un-stacking) → Task 11. ✓
- Performance gate (tokens/sec, TTFT, TPOT, KV headroom; Stage-2 ≥ Stage-1) → Task 12. ✓
- Independent correctness validation (published `mx.dequantize` transcription; optional bf16 reference / lm-eval) → Task 13. ✓
- Validation: sanity (Task 2), numerical synthetic e2e (Task 7, re-run at 10), real 30B (Task 8, re-run at 10), independent (Task 13). ✓
- Reuse `mlx_unpack`/`mlx_dequantize` → Tasks 4,5,6,7. ✓

**Placeholder scan:** Code shown for every implementation step except the Task 9 Pallas internals, which are honestly scoped as spike→mirror-the-scale-path→kernel-unit-test (the scale path is the exact template; exact Pallas wiring can't be transcribed without the kernel in front of the implementer). The Task 9 kernel test, the Task 7/10 synthetic e2e, the Task 11 method tests, the Task 12 perf gate, and the Task 13 independent check are the objective gates. All `NOTE` blocks are verification instructions, not missing logic. Tasks 9 (~13 sites) and 10 (~30 sites across 4 files) carry explicit edit-site lists from the spec-review remediation map.

**Type consistency:** `VllmMLXConfig` (group_size, bits, pack_factor, modules_to_not_convert) consistent across Tasks 3/5/7/10/11. `mlx_dequantize(packed, scales, biases, group_size=, bits=)` identical in Tasks 4/5/6/7/13. `FusedMoEWeights` is extended in Task 10 with a **NEW `w13_groupbias`/`w2_groupbias` field** carrying the per-quant-block affine bias `[E,num_blocks,1,N]` into `gmm_v2(rhs_groupbias=...)` — the per-channel `w13_bias`/`w2_bias` `[E,1,out]` MLP bias is **left untouched** (they are NOT interchangeable; the old "repurpose w13_bias" option was invalid and is deleted). `transform_mlx_weights(weights, *, group_size, bits, num_experts)` defined in Task 6, called identically in the loader override (auto-selected for MLX in Task 8).

**Known residual risks:** uint32 survival through safetensors→torch→`jax_view` (Task 6 `.to(torch.uint32)`); exact vLLM `PackedvLLMParameter` kwargs (Task 5 NOTE); `load_format="auto"` must auto-select the streaming loader or production load crashes on `lm_head.biases` (Task 8 Step 2); tp=8 RowParallel input-dim sharding of packed/grouped tensors must land on word+group boundaries (Task 8 Step 3); tp=8 w2 group-divisibility (`I/tp=96`, `96//64` non-integer) may block true-4bit w2 — surfaced in Task 10 Step 1b; active `MoEBackend` must be `gmm_v2`-backed (Task 10 Step 1a); `gmm_v2` groupbias must stay zero-overhead when absent (Task 9 NOTE); the self-oracle shares math with `mlx_dequantize` so it cannot prove real correctness — Task 13's independent check gates Task 8 acceptance. Stage 1 (Tasks 1–8) is independent of Stage 2 — if the kernel work stalls, the model still serves correctly from Task 8 (but NOT at production memory/speed).
