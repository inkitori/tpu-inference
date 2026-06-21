# DeepSeek-V4-Flash Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the first real DeepSeek-V4-Flash forward to run on TPU v6e-8 (torchax path) and produce coherent output for prompts ≤128 tokens, by fixing the FP4 MoE GMM block size and implementing the dense sliding-window-attention forward (`forward`/`forward_mqa`/`_o_proj`/`process_weights_after_loading`) routed through the `mla_swa` Pallas kernel for all 43 layers.

**Architecture:** The model reuses vLLM's AMD `DeepseekV4ForCausalLM`. Our TPU attention class `VllmDeepseekV4MLAAttention` **subclasses `vllm.models.deepseek_v4.attention.DeepseekV4Attention`** (the base ABC) — see `deepseek_v4_attention.py:47`. It is **symbol-patched in as a replacement for** `DeepseekV4ROCMAiterMLAAttention` on `amd.model` via `patch_deepseek_v4_mla_cls()` (`deepseek_v4_attention.py:124-135`); i.e. `DeepseekV4ROCMAiterMLAAttention` is the *patch target it displaces*, NOT its parent. The base `DeepseekV4Attention.__init__` registers the layer into `compilation_config.static_forward_context[prefix]` and sets `self.prefix` (it does NOT set `self.layer_name`). Experts stay FP4 (`float4_e2m1fn`), linears stay block-FP8; the MoE matmul runs through `kernels/megablox/gmm_v2.py` (public entrypoint `gmm_v2`), which must take the dequant-in-VMEM branch on v6e (mxu_column_size=256) by requantizing experts at a block size < 256. Attention runs the unified ragged `mla_sliding_window_ragged_paged_attention` kernel which quantizes KV and writes the cache internally; the decode/prefill split is communicated only via the `distribution` i32[3] metadata array, never a Python branch.

**Tech Stack:** JAX/jaxlib 0.10.1, torch 2.10.0+cpu, torchax 0.0.13, ml_dtypes 0.5.4, Pallas/Mosaic on TPU v6e, vLLM editable install at `/home/enyouki/vllm` (AMD/ROCm DSV4 variant forced), pytest.

## Global Constraints

- **Quantization stays on.** Experts in FP4 (`float4_e2m1fn`), linears in block FP8 (e4m3, ue8m0 scales). This is the only config that fits HBM (187/230 GiB at load). Never requantize experts to FP8.
- **torchax path only** (`MODEL_IMPL_TYPE=vllm`), not the pure-JAX model path.
- **Text-only.** Multimodal disabled. **MTP (multi-token prediction) stays disabled** throughout.
- **Never download full weights.** Mounted read-only at `/home/enyouki/dsv4-weights`.
- **Test with synthetic small-config weights** on a small DSV4 config (real quant formats), NOT the full model — for all routine/parity testing.
- **Test on the real multi-chip mesh from the start** (TP=8 + expert-parallel + DP-attention), the same mesh used for full-model serving.
- **Reserve the full (~187 GiB) model load ONLY for the milestone coherence smoke test** (final task). Never for iterative development.
- **No attention sinks in Phase 1** (`mla_swa` has no sink arg; sinks are Phase 2). Coherence holds only for ≤128-token sequences (one sliding window).
- Don't reinstall vLLM (pinned editable install at `/home/enyouki/vllm`). Don't modify `gmm_v2.py`.
- **`NEW_MODEL_DESIGN=1` is MANDATORY for every test in this plan.** Without it, `ShardingAxisName` resolves to the 2-axis `ShardingAxisName2D` (`("data","model")`, `sharding.py:34`) instead of the 6-axis `ShardingAxisNameBase` (`sharding.py:106-116`, gated on `envs.NEW_MODEL_DESIGN`), the 6-axis production mesh cannot be built, and MLA + DP-attention raise `ValueError` (`tpu_platform.py:276-283`, `sharding.py:307-310`). Set it (with `MODEL_IMPL_TYPE=vllm`) in the environment **before** the Python process starts — these are read at import time, so `os.environ[...]=...` inside a test module is too late. Either export them in the shell (`NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest ...`) or set them in a `tests/dsv4/conftest.py` that does `os.environ.setdefault(...)` at the very top (before any tpu_inference import). This plan assumes a `tests/dsv4/conftest.py` created in Task 2 does this (see Task 2).

---

## Conventions used throughout this plan

- All test files live under `/home/enyouki/tpu-inference/tests/`. New Phase-1 harness/tests live under `tests/dsv4/`.
- All tests run on the production mesh built by `get_spmd_mesh(num_devices=8, enable_attn_dp=True)` (6-axis `MESH_AXIS_NAMES = ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp")`, shape `(1, 2, 1, 1, 4, 1)`; the **head/group-parallel `model` axis is size 4**, not 8). This requires `NEW_MODEL_DESIGN=1` (see Global Constraints) — only then does `ShardingAxisName` resolve to the 6-axis `ShardingAxisNameBase`. Pair it with the constants `ShardingAxisName.ATTN_DATA`, `.ATTN_HEAD`, `.EXPERT`, `.BATCH`, `.VOCAB` (all exist on `ShardingAxisNameBase`; `ShardingAxisName` is a lazy proxy `sharding.py:133` that picks `ShardingAxisNameBase` under `NEW_MODEL_DESIGN=1`). The exact logits/weight PartitionSpecs must be read from the live committed array (`print(arr.sharding)`) before asserting — do not assume.
- Parity uses `import numpy as np; np.testing.assert_allclose(...)` (there is NO custom `assert_allclose` helper in this repo).
- **Tolerance ladder (spec §6.4):** bf16 exact-math ops `rtol=atol=1e-5`; RoPE near bit-exact (`rtol=atol=1e-5`, RoPE-norm preservation `atol=1e-4`); FP8/FP4 GEMM and attention kernels `rtol=atol=0.1` (matches `mla_test.py:394` and `mla_swa_test.py:405`); full-forward logits = per-token top-1 agreement + loose logit atol.
- Run a single test with (the env vars are mandatory — see Global Constraints):
  `cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest <path>::<Class>::<test> -x -s`.
  Every `python -m pytest ...` command in this plan is shown with that env prefix; when a Task's test does not touch tpu_inference sharding/mesh/model code (e.g. the pure-torch Tasks 5/6/7), the prefix is harmless but kept for uniformity.
- Assert shard-equivalence determinism at the top of every multi-chip test:
  ```python
  assert jax.config.jax_threefry_partitionable  # default True; bit-identical sharded RNG
  ```

---

### Task 1: Synthetic DSV4 mini-config factory

**Files:**
- Create `tests/dsv4/__init__.py` (empty)
- Create `tests/dsv4/mini_config.py`
- Test `tests/dsv4/test_mini_config.py`

**Interfaces:**
- Produces `make_dsv4_mini_config() -> dict` returning a HuggingFace-style config dict for a tiny DeepSeek-V4 model that includes ≥1 dense, ≥1 CSA (ratio-4), ≥1 HCA (ratio-128) layer, real quant formats, and **mesh-divisible dims for the production DP-attention mesh**.
- Produces `MINI_NUM_LAYERS: int`, `MINI_NUM_HEADS: int`, `MINI_NUM_EXPERTS: int` module constants used by later tasks.

Rationale (verified): real model facts (spec §4) are `hidden=4096, num_attention_heads=64, head_dim=512 (nope=448+rope=64), kv_heads=1, q_lora_rank=1024, o_lora_rank=1024, o_groups=8, sliding_window=128, rms_norm_eps=1e-6, 256 routed + 1 shared expert, top-6, moe_inter=2048, rope_theta=10000, compress_rope_theta=160000, vocab=129280`. `compress_ratios` per layer: dense→1, CSA→4, HCA→128 (base clamps to `max(1, ratio)`). The mini config keeps `head_dim=512` (nope=448+rope=64) because `mla_swa.quantize_kv_inputs` asserts `actual_head_dim == 512` (**`mla_swa.py:877`**, inside `quantize_kv_inputs`); shrink everything else.

> **CORRECTED mesh-divisibility (was wrong in an earlier draft):** the head/group parallel axis on the production DP-attention mesh is the **`model` axis, whose size is 4**, NOT 8. `get_spmd_mesh(num_devices=8, enable_attn_dp=True)` computes `attn_dp_size=2`, `model_size = 8//2 = 4`, giving mesh shape `(1, 2, 1, 1, 4, 1)` (`utils.py:35-37`). Heads shard over `ATTN_HEAD = ('model','expert','dcp')` (`sharding.py:47`) and the base attention does `n_local_heads = n_heads // tp_size`, `n_local_groups = n_groups // tp_size` (`attention.py:166,173`). So the real divisibility requirement is by the **model axis (4)**, not 8. Experts shard over the `expert`/`model` axes. To shard cleanly AND keep `n_local_groups >= 1` we pick `num_attention_heads=8` (8 % 4 == 0 → 2 heads/shard), `o_groups=4` (4 % 4 == 0 → `n_local_groups=1`/shard), and `n_routed_experts=8` (8 % 4 == 0 and divisible by the EP axis). Do NOT assume `o_groups=8` or `n_local_groups=8` survive sharding — at the production mesh `n_local_groups = o_groups // 4`. The `_o_proj`/`process_weights_after_loading` BMM layout (Tasks 8/10) must use the **local** `self.n_local_groups`, never the config `o_groups`.

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_mini_config.py
import numpy as np

from tests.dsv4.mini_config import (MINI_NUM_EXPERTS, MINI_NUM_HEADS,
                                     MINI_NUM_LAYERS, make_dsv4_mini_config)


def test_mini_config_has_all_three_regimes():
    cfg = make_dsv4_mini_config()
    ratios = cfg["compress_ratios"]
    assert len(ratios) == cfg["num_hidden_layers"] == MINI_NUM_LAYERS
    # base clamps to max(1, ratio): dense->1, CSA->4, HCA->128
    assert 1 in ratios, "need >=1 dense layer (ratio 1)"
    assert 4 in ratios, "need >=1 CSA layer (ratio 4)"
    assert 128 in ratios, "need >=1 HCA layer (ratio 128)"


def test_mini_config_is_mesh_divisible():
    # Production DP-attention mesh: model axis = 4 (NOT 8). Heads/groups shard
    # over the 4-way `model` axis; experts over expert/model. So dims must be
    # divisible by 4, and o_groups must be a multiple of 4 to keep n_local_groups>=1.
    cfg = make_dsv4_mini_config()
    assert cfg["num_attention_heads"] % 4 == 0
    assert cfg["n_routed_experts"] % 4 == 0
    assert cfg["o_groups"] % 4 == 0
    # o_groups must also divide num_attention_heads (heads_per_group integer).
    assert cfg["num_attention_heads"] % cfg["o_groups"] == 0
    assert MINI_NUM_HEADS == cfg["num_attention_heads"]
    assert MINI_NUM_EXPERTS == cfg["n_routed_experts"]


def test_mini_config_preserves_real_attention_dims():
    cfg = make_dsv4_mini_config()
    # mla_swa.quantize_kv_inputs asserts actual_head_dim == 512.
    assert cfg["head_dim"] == 512
    assert cfg["qk_rope_head_dim"] == 64
    assert cfg["head_dim"] - cfg["qk_rope_head_dim"] == 448  # nope dim
    assert cfg["sliding_window"] == 128
    assert cfg["rms_norm_eps"] == 1e-6
    assert cfg["num_experts_per_tok"] == 6
    assert cfg["expert_dtype"] == "fp4"
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_mini_config.py -x -s
```
Expected: `ModuleNotFoundError: No module named 'tests.dsv4.mini_config'` (or `ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
# tests/dsv4/__init__.py
```

```python
# tests/dsv4/mini_config.py
"""Synthetic DeepSeek-V4-Flash *mini* config for routine/parity testing.

Tiny dims so the model instantiates in seconds, but keeps:
  * all three attention regimes (dense / CSA ratio-4 / HCA ratio-128),
  * the real quant formats (FP4 e2m1 experts, FP8 e4m3 block linears),
  * mesh-divisible dims for the DP-attention production mesh, whose head/group
    parallel `model` axis is size 4 (NOT 8): num_attention_heads % 4 == 0,
    n_routed_experts % 4 == 0, o_groups % 4 == 0 and o_groups | num_attention_heads,
  * head_dim == 512 (nope 448 + rope 64) — mla_swa.quantize_kv_inputs asserts this.
"""
from __future__ import annotations

# 4 layers: dense, CSA(ratio 4), HCA(ratio 128), dense.
MINI_COMPRESS_RATIOS = [1, 4, 128, 1]
MINI_NUM_LAYERS = len(MINI_COMPRESS_RATIOS)
MINI_NUM_HEADS = 8          # % 4 (model axis) == 0 -> 2 heads/shard
MINI_NUM_EXPERTS = 8        # % 4 == 0 and divisible by the EP axis
MINI_HIDDEN = 256
MINI_O_GROUPS = 4           # % 4 == 0 -> n_local_groups = 1/shard on the 4-way model axis


def make_dsv4_mini_config() -> dict:
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "hidden_size": MINI_HIDDEN,
        "intermediate_size": 512,
        "num_hidden_layers": MINI_NUM_LAYERS,
        "num_attention_heads": MINI_NUM_HEADS,
        "num_key_value_heads": 1,
        # Attention latent / MLA dims (head_dim must stay 512 for mla_swa).
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 128,
        "o_lora_rank": 128,
        "kv_lora_rank": 512,
        "o_groups": MINI_O_GROUPS,
        "sliding_window": 128,
        # Per-layer attention regime selector (base clamps to max(1, ratio)).
        "compress_ratios": list(MINI_COMPRESS_RATIOS),
        # MoE.
        "n_routed_experts": MINI_NUM_EXPERTS,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 256,
        "first_k_dense_replace": 0,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        # Norm / RoPE.
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "compress_rope_theta": 160000,
        "max_position_embeddings": 4096,
        "rope_scaling": {
            "type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 256,
        },
        # Quant: FP4 experts, FP8 block linears.
        "expert_dtype": "fp4",
        "moe_quant_algo": "MXFP4",
        "vocab_size": 1280,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
```

- [ ] **Step 4: Run test to verify it passes**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_mini_config.py -x -s
```
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tests/dsv4/__init__.py tests/dsv4/mini_config.py tests/dsv4/test_mini_config.py && git commit -m "DSV4 Phase1: synthetic mini-config factory (3 regimes, mesh-divisible, real quant)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Multi-chip mesh fixture + sharding-assertion helper + conftest env gate

**Files:**
- Create `tests/dsv4/conftest.py` (sets `NEW_MODEL_DESIGN`/`MODEL_IMPL_TYPE` at import time, before any tpu_inference import)
- Create `tests/dsv4/mesh_fixtures.py`
- Test `tests/dsv4/test_mesh_fixtures.py`

**Interfaces:**
- Consumes nothing from earlier tasks.
- Produces `tests/dsv4/conftest.py` — a defensive `os.environ.setdefault("NEW_MODEL_DESIGN","1")` + `setdefault("MODEL_IMPL_TYPE","vllm")` at the **top of the file** (pytest imports `conftest.py` before collecting test modules, and these env vars are read at `tpu_inference` import time). This is a belt-and-suspenders backstop; the shell prefix in every run command is the primary mechanism. **It cannot help if `tpu_inference`/`jax` were already imported with the wrong env by an earlier non-dsv4 conftest** — so always also pass the shell prefix.
- Produces `dsv4_mesh()` pytest fixture → yields a `jax.sharding.Mesh` from `get_spmd_mesh(num_devices=8, enable_attn_dp=True)`.
- Produces `assert_sharded_like(arr: jax.Array, mesh: jax.sharding.Mesh, spec: jax.sharding.PartitionSpec) -> None` which asserts on the **committed** `arr.sharding` (NOT `eval_shape`).
- Produces `assert_threefry_partitionable() -> None`.

Rationale (verified): `get_spmd_mesh(num_devices=1, enable_attn_dp=False)` is at `tests/layers/common/utils.py:26` (both args default; `num_devices=1, enable_attn_dp=False`). With `enable_attn_dp=True` and 8 devices it builds the 6-axis production mesh **only when `NEW_MODEL_DESIGN=1`** (the axis names come from `MESH_AXIS_NAMES`, `sharding.py:32-33`); the produced shape is `(1, 2, 1, 1, 4, 1)` (`attn_dp_size=2`, `model_size=8//2=4`; `utils.py:35-37`). Plain `jax.eval_shape` returns `.sharding = None` (spec §6.3 H), so we assert on a committed array's `.sharding`.

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_mesh_fixtures.py
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tests.dsv4.mesh_fixtures import (assert_sharded_like,
                                      assert_threefry_partitionable,
                                      dsv4_mesh)


def test_mesh_is_8_chip_attn_dp(dsv4_mesh):
    # 6-axis production mesh, total 8 devices.
    assert dsv4_mesh.devices.size == 8
    assert set(dsv4_mesh.axis_names) == {
        "data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp"
    }


def test_assert_sharded_like_passes_for_committed_array(dsv4_mesh):
    assert_threefry_partitionable()
    # 8 rows so it shards across the 4-way "model" axis cleanly.
    x = jnp.arange(8 * 4, dtype=jnp.float32).reshape(8, 4)
    sh = NamedSharding(dsv4_mesh, P("model", None))
    x = jax.device_put(x, sh)
    assert_sharded_like(x, dsv4_mesh, P("model", None))  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_mesh_fixtures.py -x -s
```
Expected: `ModuleNotFoundError: No module named 'tests.dsv4.mesh_fixtures'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tests/dsv4/conftest.py
"""Env gate for all DSV4 tests. MUST run before any tpu_inference/jax import.

NEW_MODEL_DESIGN=1 selects the 6-axis ShardingAxisNameBase + production mesh;
without it the mesh fixture and any MLA/DP-attention model build raise ValueError.
These are read at import time, so set them here (setdefault, so a shell-provided
value wins) AND prefer the shell prefix `NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm`.
"""
import os

os.environ.setdefault("NEW_MODEL_DESIGN", "1")
os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")


def pytest_configure(config):
    # `slow` is NOT registered by the repo's tests/conftest.py (only
    # disable_jax_cache, bvt). Register it here so Task 15's @pytest.mark.slow
    # and `-m slow` selection work without an "unknown marker" warning.
    config.addinivalue_line(
        "markers", "slow: loads the real ~187 GiB model; coherence milestone only.")
```

```python
# tests/dsv4/mesh_fixtures.py
"""Production-mesh fixture (TP=8 + EP + DP-attention) and sharding oracle for DSV4 tests."""
import jax
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.layers.common.utils import get_spmd_mesh


def assert_threefry_partitionable() -> None:
    """8-way-sharded threefry RNG is bit-identical to single-device only when on."""
    assert jax.config.jax_threefry_partitionable, (
        "jax_threefry_partitionable must be True for shard-equivalence")


@pytest.fixture(scope="module")
def dsv4_mesh():
    """Production sharding: TP=8 split as 2-way attn-DP x 4-way model, + EP axes."""
    if len(jax.devices()) < 8:
        pytest.skip("DSV4 production mesh requires 8 TPU devices")
    mesh = get_spmd_mesh(num_devices=8, enable_attn_dp=True)
    yield mesh


def assert_sharded_like(arr: jax.Array, mesh: Mesh, spec: P) -> None:
    """Assert on the COMMITTED sharding of a real array (eval_shape gives None)."""
    assert isinstance(arr.sharding, NamedSharding), (
        f"expected NamedSharding, got {type(arr.sharding)}")
    expected = NamedSharding(mesh, spec)
    assert arr.sharding == expected, (
        f"sharding mismatch: got {arr.sharding!r}, want {expected!r}")
```

- [ ] **Step 4: Run test to verify it passes**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_mesh_fixtures.py -x -s
```
Expected: `2 passed` (or `2 skipped` if fewer than 8 devices — run on the v6e-8 host so it must pass).

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tests/dsv4/conftest.py tests/dsv4/mesh_fixtures.py tests/dsv4/test_mesh_fixtures.py && git commit -m "DSV4 Phase1: 8-chip production mesh fixture + committed-sharding oracle + env conftest

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: AOT compile gate utility

**Files:**
- Create `tests/dsv4/aot_gate.py`
- Test `tests/dsv4/test_aot_gate.py`

**Interfaces:**
- Consumes nothing from earlier tasks.
- Produces `aot_compile(fn, *args, mesh=None, static_argnums=(), static_argnames=()) -> jax.stages.Compiled`. It jits `fn`, calls `.lower(*args)` then `.compile()` so Mosaic passes run and surface compile errors (e.g. the FP4 GMM MosaicError) with no full run. Returns the compiled executable. `.lower()` alone is insufficient (spec §6.3 A) — `.compile()` must be called.
- Produces `make_aval(shape, dtype, mesh=None, spec=None) -> jax.ShapeDtypeStruct` to build dummy avals.

Rationale (verified): `compilation_manager.py:151` does `lowered = fn.lower(*args, **call_kwargs)` and `:165` `compiled = lowered.compile()` on an **already-jitted** `fn` inside `with jax.set_mesh(mesh):` (`:163`). Our `aot_compile` instead jits the raw `fn` first (so callers pass a plain function) — equivalent. The `jax.ShapeDtypeStruct(shape=..., dtype=..., sharding=...)` aval idiom is at `tpu_runner.py:2665` (NOT in `compilation_manager.py`, which uses concrete `jnp.ones` dummies via `_create_dummy_tensor` at `:96`). `jax.set_mesh` is a real, heavily-used API in this jax 0.10.1.

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_aot_gate.py
import jax
import jax.numpy as jnp

from tests.dsv4.aot_gate import aot_compile, make_aval


def test_aot_compile_returns_compiled_executable():
    def f(x, y):
        return x @ y

    a = make_aval((16, 16), jnp.bfloat16)
    b = make_aval((16, 16), jnp.bfloat16)
    compiled = aot_compile(f, a, b)
    assert isinstance(compiled, jax.stages.Compiled)


def test_aot_compile_surfaces_compile_error():
    # A shape-incompatible matmul must fail at lower/compile time, not silently.
    def bad(x, y):
        return x @ y  # (16,16) @ (8,8) -> contraction mismatch

    a = make_aval((16, 16), jnp.float32)
    b = make_aval((8, 8), jnp.float32)
    import pytest
    with pytest.raises(Exception):
        aot_compile(bad, a, b)
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_aot_gate.py -x -s
```
Expected: `ModuleNotFoundError: No module named 'tests.dsv4.aot_gate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tests/dsv4/aot_gate.py
"""AOT compile gate: jit(fn).lower(*avals).compile() forces Mosaic passes.

Use as a pre-flight before any full run: it surfaces untraceable/Mosaic compile
errors (e.g. the FP4 GMM MosaicError) in seconds with no weights/data.
"""
import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def make_aval(shape, dtype, mesh=None, spec=None) -> jax.ShapeDtypeStruct:
    if mesh is not None and spec is not None:
        return jax.ShapeDtypeStruct(shape=shape, dtype=dtype,
                                    sharding=NamedSharding(mesh, spec))
    return jax.ShapeDtypeStruct(shape=shape, dtype=dtype)


def aot_compile(fn, *args, mesh=None, static_argnums=(),
                static_argnames=()) -> jax.stages.Compiled:
    """Lower AND compile fn against the given avals/arrays. .compile() is what
    runs the Mosaic backend; .lower() alone only serializes (spec 6.3 A)."""
    jitted = jax.jit(fn, static_argnums=static_argnums,
                     static_argnames=static_argnames)
    if mesh is not None:
        with jax.set_mesh(mesh):
            return jitted.lower(*args).compile()
    return jitted.lower(*args).compile()
```

- [ ] **Step 4: Run test to verify it passes**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_aot_gate.py -x -s
```
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tests/dsv4/aot_gate.py tests/dsv4/test_aot_gate.py && git commit -m "DSV4 Phase1: AOT compile gate (lower+compile) pre-flight utility

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: FP4 GMM block-size fix

**Files:**
- Modify `tpu_inference/layers/vllm/quantization/mxfp4.py:57` (the `REQUANTIZED_BLOCK_SIZE` constant)
- Test `tests/dsv4/test_fp4_gmm_blocksize.py`

**Interfaces:**
- Consumes `aot_compile`, `make_aval` (Task 3); `dsv4_mesh` (Task 2).
- Produces no new public symbols. The change is: `REQUANTIZED_BLOCK_SIZE = 512` → `REQUANTIZED_BLOCK_SIZE = 32`.

Rationale (verified): `REQUANTIZED_BLOCK_SIZE` is defined once (`mxfp4.py:57`) and used once (`mxfp4.py:171`, the 3rd positional arg to `quantize_moe_weights(..., jnp.float4_e2m1fn, REQUANTIZED_BLOCK_SIZE, ...)`). `gmm_v2.should_dequantize_before_matmul` (`gmm_v2.py:197-215`) returns `quant_block_size < pltpu.get_tpu_info().mxu_column_size`; on v6e `mxu_column_size == 256`, so 512 → False → quantized-matmul/f8-cast path (`gmm_v2.py:495-506`) which Mosaic cannot compile for native f4; 32 → True → dequant-in-VMEM bf16 path (`gmm_v2.py:395-409`). NVFP4 precedent uses block 16 (`nvfp4.py:444`). `gmm_v2` is TPU-only even under `interpret=True` (`pltpu.get_tpu_info()` called in the Python wrapper at trace time — the `should_dequantize_before_matmul` property at `gmm_v2.py:214` and unconditionally at `gmm_v2.py:1269,1313`; verified to raise `ValueError` on a CPU-only host), so ALL gmm validation happens on the real v6e mesh and the FP4-fix compile proof is Task 13, not a standalone gmm call. The DSV4 dispatch to `VllmMxfp4MoEMethod` is at `deepseek_v4_fp8.py:83` (the `expert_dtype == "fp4"` branch, line 76, after the NVFP4 guard).

> **NOTE for the worker:** the public entrypoint in `kernels/megablox/gmm_v2.py` is the `@jax.jit`-wrapped function **`gmm_v2`** (def at `gmm_v2.py:1206`), NOT `gmm` (no symbol named `gmm` exists). Its signature is `gmm_v2(lhs, rhs, group_sizes, rhs_scale=None, rhs_bias=None, group_offset=None, *, tile_info=calculate_tiling, vmem_limit_bytes=None, precision=..., preferred_element_type=None, acc_dtype=None, maybe_quantize_lhs=True, zero_initialize=True, fuse_act=None)`. A hand-assembled mxfp4 `gmm_v2` micro-test is **low-leverage and error-prone** (you must build `rhs` as packed `float4_e2m1fn`, the `rhs_scale` ue8m0 block-scale tensor at block 32, and the `group_sizes`). **Therefore this task's test is scoped to (a) the constant value and (b) the dispatch assertions below — neither calls `gmm_v2`.** The actual proof that the block-size fix makes `gmm_v2` compile on v6e is delivered by **Task 13's AOT gate on the full forward** (the load-bearing FP4-fix validation). Do not attempt a standalone `gmm_v2` micro-test. Note also: `gmm_v2` calls `pltpu.get_tpu_info()` in its Python wrapper at trace time (`gmm_v2.py:1269` etc.), so it cannot run on a CPU-only host even under `interpret=True` — all gmm validation is TPU-only.

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_fp4_gmm_blocksize.py
import pytest

import tpu_inference.layers.vllm.quantization.mxfp4 as mxfp4


def test_requant_block_size_below_mxu_column_size():
    # v6e mxu_column_size == 256; block size must be < 256 to take the
    # dequant-in-VMEM branch in gmm_v2.should_dequantize_before_matmul.
    assert mxfp4.REQUANTIZED_BLOCK_SIZE < 256
    # Match the native MXFP4 group (and the NVFP4 precedent's small block).
    assert mxfp4.REQUANTIZED_BLOCK_SIZE == 32


def test_should_dequantize_true_for_new_block_size():
    # Mirror gmm_v2's decision on v6e: quant_block_size < mxu_column_size.
    import jax
    if jax.devices()[0].platform != "tpu":
        pytest.skip("gmm_v2 / get_tpu_info is TPU-only")
    from jax.experimental.pallas import tpu as pltpu
    mxu = pltpu.get_tpu_info().mxu_column_size
    assert mxu == 256, f"expected v6e mxu_column_size 256, got {mxu}"
    assert mxfp4.REQUANTIZED_BLOCK_SIZE < mxu


def test_dsv4_dispatches_to_mxfp4_moe_method():
    # Confirm the FP4 expert path routes to VllmMxfp4MoEMethod. Verified:
    # `if self.expert_dtype == "fp4":` (deepseek_v4_fp8.py:76) and
    # `return VllmMxfp4MoEMethod(...)` (deepseek_v4_fp8.py:83) both live in the
    # body of the regular instance method VllmDeepseekV4Fp8Config.get_quant_method.
    import inspect

    from tpu_inference.layers.vllm.quantization.deepseek_v4_fp8 import \
        VllmDeepseekV4Fp8Config
    body = inspect.getsource(VllmDeepseekV4Fp8Config.get_quant_method)
    assert "VllmMxfp4MoEMethod" in body
    assert 'self.expert_dtype == "fp4"' in body
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_fp4_gmm_blocksize.py -x -s
```
Expected: `test_requant_block_size_below_mxu_column_size` FAILS with `assert 512 < 256` / `assert 512 == 32`.

- [ ] **Step 3: Write minimal implementation**

Edit `tpu_inference/layers/vllm/quantization/mxfp4.py` line 57:
```python
# was: REQUANTIZED_BLOCK_SIZE = 512
REQUANTIZED_BLOCK_SIZE = 32  # < v6e mxu_column_size (256) => dequant-in-VMEM bf16 path in gmm_v2; experts stay FP4 in HBM (see spec 3)
```

- [ ] **Step 4: Run test to verify it passes**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_fp4_gmm_blocksize.py -x -s
```
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tpu_inference/layers/vllm/quantization/mxfp4.py tests/dsv4/test_fp4_gmm_blocksize.py && git commit -m "DSV4 Phase1: set mxfp4 requant block size 32 (<256) for dequant-in-VMEM GMM path

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Pure-torch reference building blocks (RMSNorm + GPT-J RoPE + FP4/ue8m0/FP8 dequant)

**Files:**
- Create `tests/dsv4/torch_ref.py`
- Test `tests/dsv4/test_torch_ref.py`

**Interfaces:**
- Consumes nothing from earlier tasks.
- Produces, all pure-torch / CPU-runnable:
  - `rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor` (returns fp32)
  - `rmsnorm_with_weight(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor`
  - `build_cos_sin_cache(rope_dim: int, max_pos: int, theta: float, mscale: float = 1.0) -> torch.Tensor` → `[max_pos, rope_dim]` = `cat((cos, sin), -1)`
  - `rotate_gptj(x: torch.Tensor) -> torch.Tensor`
  - `apply_rope_gptj_last_k(x, positions, cos_sin_cache, rope_dim) -> torch.Tensor` (forward GPT-J interleaved RoPE on last `rope_dim` dims)
  - `apply_inverse_rope_gptj_last_k(x, positions, cos_sin_cache, rope_dim) -> torch.Tensor` (inverse: `sin -> -sin`)
  - `break_fp4_e2m1(packed_u8: torch.Tensor, out_dtype) -> torch.Tensor` (re-export wrapper around vLLM `break_fp4_bytes`)
  - `upcast_e8m0_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor` (re-export wrapper)

Rationale (verified): GPT-J `rotate_gptj` even/odd interleave is at vLLM `rotary_embedding/common.py:24-29`; cos_sin_cache layout `cat((cos, sin), -1)` at `deepseek_scaling_rope.py:250-261`; GPT-J application uses `repeat_interleave(2)` (not `cat((cos,cos))`) at `deepseek_scaling_rope.py:284-297`; inverse RoPE = `sin -> -sin` (`test_fused_inv_rope_fp8_quant.py:143`); `rmsnorm_no_weight` returns fp32 (`test_fused_deepseek_v4_qnorm_rope_kv_insert.py:109-119`); `break_fp4_bytes` at `nvfp4_emulation_utils.py:328` with table `[0,.5,1,1.5,2,3,4,6]` (low nibble first); `_upcast_e8m0_to_fp32` at `fp8_utils.py:1049`.

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_torch_ref.py
import torch

from tests.dsv4 import torch_ref


def test_rmsnorm_no_weight_matches_manual():
    x = torch.randn(4, 16)
    out = torch_ref.rmsnorm_no_weight(x, eps=1e-6)
    var = x.float().pow(2).mean(-1, keepdim=True)
    expected = x.float() * torch.rsqrt(var + 1e-6)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_rotate_gptj_is_even_odd_interleave():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # pairs (1,2),(3,4)
    # rotate_gptj: stack((-odd, even)) interleaved -> (-2,1,-4,3)
    out = torch_ref.rotate_gptj(x)
    torch.testing.assert_close(out, torch.tensor([[-2.0, 1.0, -4.0, 3.0]]))


def test_rope_then_inverse_is_identity_on_rope_dims():
    rope_dim, max_pos = 64, 256
    cache = torch_ref.build_cos_sin_cache(rope_dim, max_pos, theta=10000.0)
    x = torch.randn(5, 2, 512)  # head_dim 512, last 64 are rope dims
    pos = torch.arange(5)
    rotated = torch_ref.apply_rope_gptj_last_k(x, pos, cache, rope_dim)
    back = torch_ref.apply_inverse_rope_gptj_last_k(rotated, pos, cache, rope_dim)
    torch.testing.assert_close(back.float(), x.float(), rtol=1e-4, atol=1e-4)


def test_break_fp4_e2m1_lookup_table():
    # byte 0x21 -> low nibble 0x1 (=0.5), high nibble 0x2 (=1.0)
    packed = torch.tensor([[0x21]], dtype=torch.uint8)
    out = torch_ref.break_fp4_e2m1(packed, torch.float32)
    torch.testing.assert_close(out, torch.tensor([[0.5, 1.0]]))


def test_upcast_e8m0_is_power_of_two():
    # e8m0 byte 127 (bias) -> exponent 0 -> 1.0; byte 128 -> 2.0
    s = torch.tensor([127, 128], dtype=torch.uint8)
    out = torch_ref.upcast_e8m0_to_fp32(s)
    torch.testing.assert_close(out, torch.tensor([1.0, 2.0]))
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_torch_ref.py -x -s
```
Expected: `ImportError` / `AttributeError: module 'tests.dsv4.torch_ref' has no attribute 'rmsnorm_no_weight'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tests/dsv4/torch_ref.py
"""Pure-torch, CPU-runnable reference building blocks for DSV4 Phase-1 parity.

Reuses vLLM dequant utils; the vLLM AMD model cannot run on CPU, so these
reproduce the exact reference numerics (spec 6.3 C).
"""
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import \
    _upcast_e8m0_to_fp32
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import \
    break_fp4_bytes


def rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(var + eps)


def rmsnorm_with_weight(x: torch.Tensor, w: torch.Tensor,
                        eps: float) -> torch.Tensor:
    return (rmsnorm_no_weight(x, eps) * w.float()).to(x.dtype)


def build_cos_sin_cache(rope_dim: int, max_pos: int, theta: float,
                        mscale: float = 1.0) -> torch.Tensor:
    """[max_pos, rope_dim] == cat((cos, sin), -1) (deepseek_scaling_rope.py)."""
    half = rope_dim // 2
    inv_freq = 1.0 / (theta**(torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos() * mscale
    sin = freqs.sin() * mscale
    return torch.cat((cos, sin), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _gptj_cos_sin(positions, cos_sin_cache, rope_dim):
    cs = cos_sin_cache[positions.long()].to(torch.float32)  # [..., rope_dim]
    half = rope_dim // 2
    cos = cs[..., :half].repeat_interleave(2, dim=-1)  # GPT-J: interleave, not cat
    sin = cs[..., half:].repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope_gptj_last_k(x, positions, cos_sin_cache,
                           rope_dim) -> torch.Tensor:
    cos, sin = _gptj_cos_sin(positions, cos_sin_cache, rope_dim)
    # broadcast over the head axis: x is [T, H, D]; cos/sin are [T, rope_dim]
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    out = x.clone().float()
    rot = out[..., -rope_dim:]
    out[..., -rope_dim:] = rot * cos + rotate_gptj(rot) * sin
    return out.to(x.dtype)


def apply_inverse_rope_gptj_last_k(x, positions, cos_sin_cache,
                                   rope_dim) -> torch.Tensor:
    cos, sin = _gptj_cos_sin(positions, cos_sin_cache, rope_dim)
    cos = cos.unsqueeze(-2)
    sin = -sin.unsqueeze(-2)  # inverse RoPE: negate sin
    out = x.clone().float()
    rot = out[..., -rope_dim:]
    out[..., -rope_dim:] = rot * cos + rotate_gptj(rot) * sin
    return out.to(x.dtype)


def break_fp4_e2m1(packed_u8: torch.Tensor, out_dtype) -> torch.Tensor:
    return break_fp4_bytes(packed_u8, out_dtype)


def upcast_e8m0_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor:
    return _upcast_e8m0_to_fp32(scale_u8)
```

- [ ] **Step 4: Run test to verify it passes**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_torch_ref.py -x -s
```
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tests/dsv4/torch_ref.py tests/dsv4/test_torch_ref.py && git commit -m "DSV4 Phase1: pure-torch reference blocks (RMSNorm, GPT-J RoPE +/- inverse, FP4/e8m0 dequant)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Golden capture/replay helper (.npz persist)

**Files:**
- Create `tests/dsv4/golden.py`
- Test `tests/dsv4/test_golden.py`

**Interfaces:**
- Consumes nothing.
- Produces:
  - `save_golden(path: str, **arrays) -> None` — persists named numpy arrays to a `.npz`.
  - `load_golden(path: str) -> dict[str, np.ndarray]` — loads them back.
  - `GOLDEN_DIR: str` — absolute dir `tests/dsv4/goldens/` where goldens live (created on demand).
  - `assert_close_to_golden(name: str, actual, golden: dict, rtol: float, atol: float) -> None` — wraps `np.testing.assert_allclose(golden[name], np.asarray(actual), ...)`.

Rationale (verified): no golden persist/replay exists in the repo (spec §6.5 says BUILD). TPU tests replay against saved `.npz` so the reference never re-runs and the full model never loads.

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_golden.py
import os

import numpy as np

from tests.dsv4 import golden


def test_save_and_load_roundtrip(tmp_path):
    p = os.path.join(str(tmp_path), "g.npz")
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    golden.save_golden(p, attn_out=a)
    loaded = golden.load_golden(p)
    np.testing.assert_array_equal(loaded["attn_out"], a)


def test_assert_close_to_golden(tmp_path):
    p = os.path.join(str(tmp_path), "g.npz")
    a = np.ones((4,), dtype=np.float32)
    golden.save_golden(p, x=a)
    g = golden.load_golden(p)
    golden.assert_close_to_golden("x", a + 1e-6, g, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_golden.py -x -s
```
Expected: `ModuleNotFoundError: No module named 'tests.dsv4.golden'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tests/dsv4/golden.py
"""Golden capture/replay: persist reference outputs to .npz; TPU tests replay."""
import os

import numpy as np

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "goldens")


def save_golden(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})


def load_golden(path: str) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def assert_close_to_golden(name: str, actual, golden: dict, rtol: float,
                           atol: float) -> None:
    np.testing.assert_allclose(golden[name], np.asarray(actual), rtol=rtol,
                               atol=atol)
```

- [ ] **Step 4: Run test to verify it passes**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_golden.py -x -s
```
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tests/dsv4/golden.py tests/dsv4/test_golden.py && git commit -m "DSV4 Phase1: golden capture/replay (.npz) helper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: MoE router parity (sqrtsoftplus + noaux_tc top-6) + FP4 expert dequant reference

**Files:**
- Create `tests/dsv4/moe_ref.py`
- Test `tests/dsv4/test_moe_router_parity.py`

**Interfaces:**
- Consumes Task 5 (`break_fp4_e2m1`, `upcast_e8m0_to_fp32`).
- Produces, pure-torch / CPU-runnable:
  - `topk_softplus_sqrt(gating: torch.Tensor, e_score_correction_bias: torch.Tensor | None, topk: int, renormalize: bool, routed_scaling_factor: float) -> tuple[torch.Tensor, torch.Tensor]` returning `(topk_weights, topk_indices)`.
  - `dequant_fp4_expert(packed_u8: torch.Tensor, e8m0_scale_u8: torch.Tensor, block_size: int, out_dtype) -> torch.Tensor` — block-scaled FP4→bf16/fp32 dequant matching the GMM dequant-in-VMEM math.

Rationale (verified): the router reference `_topk_softplus_sqrt_torch` is at `vllm/.../fused_moe/router/fused_topk_bias_router.py:59` — `scores = sqrt(softplus(gating.float()))`; bias added ONLY for top-k selection (`scores_for_choice = scores + bias`); weights gathered from the UNBIASED scores; renormalize `weights / sum.clamp(min=1e-20)`; then `* routed_scaling_factor` (lines 71-102). FP4 dequant uses `break_fp4_bytes` table `[0,.5,1,1.5,2,3,4,6]` (`nvfp4_emulation_utils.py:328`) × `_upcast_e8m0_to_fp32` scales (`fp8_utils.py:1049`), block-scaled — the same e2m1 decode the GMM dequant-in-VMEM branch performs (spec §3). This closes the §7.4-step-2 MoE/router parity target.

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_moe_router_parity.py
import torch
import torch.nn.functional as F

from tests.dsv4 import moe_ref


def test_router_selects_topk_and_scales():
    torch.manual_seed(0)
    n_tok, n_exp, topk = 4, 16, 6
    gating = torch.randn(n_tok, n_exp)
    bias = torch.randn(n_exp)
    w, idx = moe_ref.topk_softplus_sqrt(
        gating, bias, topk=topk, renormalize=True, routed_scaling_factor=2.5)
    assert idx.shape == (n_tok, topk)
    assert w.shape == (n_tok, topk)
    # exactly topk distinct experts per token
    for r in range(n_tok):
        assert len(set(idx[r].tolist())) == topk
    # selection uses biased scores; weights come from UNBIASED scores
    scores = torch.sqrt(F.softplus(gating.float()))
    _, expected_idx = torch.topk(scores + bias.float(), k=topk, dim=-1)
    torch.testing.assert_close(idx, expected_idx)
    # renormalized then scaled by routed_scaling_factor
    raw = scores.gather(1, expected_idx)
    expected_w = raw / raw.sum(-1, keepdim=True).clamp(min=1e-20) * 2.5
    torch.testing.assert_close(w, expected_w, rtol=1e-5, atol=1e-5)


def test_fp4_expert_dequant_block_scaled():
    # one block (block_size=2 here for test): bytes -> e2m1 values * e8m0 scale
    packed = torch.tensor([[0x21]], dtype=torch.uint8)   # -> [0.5, 1.0]
    scale = torch.tensor([[128]], dtype=torch.uint8)     # e8m0 128 -> 2.0
    out = moe_ref.dequant_fp4_expert(packed, scale, block_size=2,
                                     out_dtype=torch.float32)
    torch.testing.assert_close(out, torch.tensor([[1.0, 2.0]]))
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_moe_router_parity.py -x -s
```
Expected: `ModuleNotFoundError: No module named 'tests.dsv4.moe_ref'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tests/dsv4/moe_ref.py
"""Pure-torch MoE router (sqrtsoftplus + noaux_tc) + FP4 expert dequant reference."""
import torch
import torch.nn.functional as F

from tests.dsv4.torch_ref import break_fp4_e2m1, upcast_e8m0_to_fp32


def topk_softplus_sqrt(gating, e_score_correction_bias, topk, renormalize,
                       routed_scaling_factor):
    scores = torch.sqrt(F.softplus(gating.float()))
    scores_for_choice = scores
    if e_score_correction_bias is not None:
        scores_for_choice = scores + e_score_correction_bias.float()
    _, indices = torch.topk(scores_for_choice, k=topk, dim=-1)
    weights = scores.gather(1, indices)  # weights from UNBIASED scores
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    weights = weights * routed_scaling_factor
    return weights, indices


def dequant_fp4_expert(packed_u8, e8m0_scale_u8, block_size, out_dtype):
    # packed_u8: [rows, cols] uint8 (2 e2m1 per byte). Decode to [rows, cols*2].
    vals = break_fp4_e2m1(packed_u8, torch.float32)  # [rows, cols*2]
    scales = upcast_e8m0_to_fp32(e8m0_scale_u8)      # [rows, num_blocks]
    rows, n = vals.shape
    num_blocks = n // block_size
    vals = vals.reshape(rows, num_blocks, block_size)
    vals = vals * scales.reshape(rows, num_blocks, 1)
    return vals.reshape(rows, n).to(out_dtype)
```

- [ ] **Step 4: Run test to verify it passes**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_moe_router_parity.py -x -s
```
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tests/dsv4/moe_ref.py tests/dsv4/test_moe_router_parity.py && git commit -m "DSV4 Phase1: MoE router parity (sqrtsoftplus+noaux_tc) + FP4 expert dequant reference

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: `_o_proj` (inverse GPT-J RoPE + grouped wo_a BMM + wo_b)

**Files:**
- Modify `tpu_inference/layers/vllm/custom_ops/deepseek_v4_attention.py:90-92` (the `_o_proj` stub) and its imports.
- Test `tests/dsv4/test_o_proj.py`

**Interfaces:**
- Consumes Task 5 (`apply_inverse_rope_gptj_last_k`, `build_cos_sin_cache`) for the parity reference; Task 2 (`dsv4_mesh`).
- Produces `VllmDeepseekV4MLAAttention._o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor` returning `[N, hidden_size]`.
- **Cross-task ordering:** `_o_proj` reads `self.wo_a_bf16`, which is created by `process_weights_after_loading` in **Task 10**. That is fine: this task's TEST only exercises the standalone reference math (`_ref_o_proj`, never the real `_o_proj`), and the real `_o_proj` is not *called* until Task 12 (which runs after Task 10 builds `self.wo_a_bf16`). Do not try to call the real `_o_proj` before Task 10.

Rationale (verified):
- The reference `_o_proj` is on the ROCm subclass at `/home/enyouki/vllm/vllm/models/deepseek_v4/amd/rocm.py:590-601`; it calls `rocm_inv_rope_einsum` (`/home/enyouki/vllm/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:1004-1027`) then `self.wo_b(z.flatten(1))`.
- `_o_proj` = inverse GPT-J RoPE on the last `rope_head_dim=64` dims per head (computed in fp32, downcast to bf16) → reshape to `[N, n_local_groups, -1]` → grouped contraction `einsum("tgd,grd->tgr", o_grouped, wo_a_bf16)` (where `wo_a.is_bmm=True`, `wo_a.bmm_batch_size = self.n_local_groups`) → flatten → `wo_b` (`RowParallelLinear`). The attributes `self.rotary_emb`, `self.rope_head_dim`, `self.n_local_groups`, `self.o_lora_rank`, `self.wo_a`, `self.wo_b`, `self.n_local_heads`, `self.head_dim` are all set by the base `DeepseekV4Attention.__init__` (`attention.py:148-316`).
- **Use `self.n_local_groups`, NOT the config `o_groups`.** The base sets `n_local_groups = n_groups // tp_size` (`attention.py:173`); on the production mesh (model axis = 4) this is `o_groups // 4`. Never hard-code 8.

> **CRITICAL — `wo_a` is FP8-block-quantized, NOT ready-to-use bf16 (verified):** `self.wo_a` is a `ColumnParallelLinear(..., quant_config=quant_config)` (`attention.py:213-220`). After loading on TPU, `self.wo_a.weight` is **2D `float8_e4m3fn`** of shape `(n_local_groups*o_lora_rank, heads_per_group*head_dim)` with a separate `self.wo_a.weight_scale_inv` (ue8m0 block scales). The 2D→3D grouped reshape that GPU does in `deepgemm_post_process_fp8_weight_block` is **CUDA-only** (`deep_gemm.py:46-52` gates on `current_platform.is_cuda()`), so on TPU it never runs. Therefore **`self.wo_a.weight.float()` is WRONG** — it upcasts raw fp8 codes without applying scales and keeps the 2D layout. The correct bf16 grouped weight must be produced once in `process_weights_after_loading` (Task 10) by replicating `_get_cached_wo_a_bf16` (`rocm_aiter_mla_sparse.py:967-1001`): `view(g, r, d).to(fp32)` × `_expand_2d_block_scales(weight_scale_inv ...)` (ue8m0-decoded via the `<<23` upcast, `fp8_utils.py:1049`) → `.to(bf16)`, stored as `self.wo_a_bf16`. `_o_proj` reads `self.wo_a_bf16`. The TEST below asserts parity of the **standalone math** (inverse-RoPE + grouped einsum + linear) against the Task-5 pure-torch reference using a plain bf16 weight — it does not exercise the fp8 dequant (that is pinned in Task 10's pwal test + Task 12 end-to-end).

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_o_proj.py
"""Parity for _o_proj's math: inverse GPT-J RoPE + grouped einsum + wo_b linear.

We do not instantiate the full attention module here (that needs a built model);
instead we test the standalone math the implementation must reproduce, so the
implementation's einsum/inverse-rope are pinned to the pure-torch reference.
This guards the contraction shapes and the inverse-rope sign convention.
"""
import torch

from tests.dsv4 import torch_ref


def _ref_o_proj(o, positions, cos_sin_cache, rope_dim, n_groups, wo_a_w,
                wo_b_w):
    # o: [N, n_heads, head_dim]
    o = torch_ref.apply_inverse_rope_gptj_last_k(o, positions, cos_sin_cache,
                                                 rope_dim).float()
    n = o.shape[0]
    o_g = o.reshape(n, n_groups, -1)               # [N, G, heads_per_g*head_dim]
    z = torch.einsum("tgd,grd->tgr", o_g, wo_a_w)  # [N, G, o_lora_rank]
    return torch.matmul(z.reshape(n, -1), wo_b_w.t())  # [N, hidden]


def test_o_proj_math_shapes_and_inverse_rope():
    torch.manual_seed(0)
    N, n_heads, head_dim = 5, 8, 512
    rope_dim, n_groups, o_lora, hidden = 64, 8, 128, 256
    heads_per_g = n_heads // n_groups
    o = torch.randn(N, n_heads, head_dim)
    positions = torch.arange(N)
    cache = torch_ref.build_cos_sin_cache(rope_dim, 256, theta=10000.0)
    wo_a_w = torch.randn(n_groups, o_lora, heads_per_g * head_dim)  # [g, r, d]
    wo_b_w = torch.randn(hidden, n_groups * o_lora)
    out = _ref_o_proj(o, positions, cache, rope_dim, n_groups, wo_a_w, wo_b_w)
    assert out.shape == (N, hidden)
    # inverse-rope must change the rope dims (sign convention sanity)
    rot = torch_ref.apply_inverse_rope_gptj_last_k(o, positions, cache, rope_dim)
    assert not torch.allclose(rot[..., -rope_dim:], o[..., -rope_dim:])
```

- [ ] **Step 2: Run test to verify it fails**

This test exercises the reference math (Task 5) only, so it should PASS immediately — its purpose is to pin the contraction convention the implementation must match. To make Step 2 a real red→green, first add a deliberately-broken assertion, run, then fix. Concretely:

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_o_proj.py -x -s
```
If it already passes, that confirms the reference convention; proceed to Step 3 to implement `_o_proj` to match. (The implementation's red→green is verified by Task 12's end-to-end parity, which calls the real module `_o_proj` against this same reference.)

- [ ] **Step 3: Write minimal implementation**

In `deepseek_v4_attention.py`, replace the `_o_proj` stub (lines 90-92). Add at top of file (after existing imports):
```python
from torchax.interop import jax_view, torch_view
import jax.numpy as jnp
```

Replace the stub:
```python
    def _o_proj(self, o: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        # o: [N, n_local_heads, head_dim]. Step 7 of the attention dataflow.
        # 1) inverse GPT-J RoPE on the last rope_head_dim dims, computed in fp32.
        o = self._inverse_rope_gptj(o, positions).to(o.dtype)
        # 2) group view -> grouped BMM via einsum("tgd,grd->tgr") over n_local_groups.
        #    wo_a_bf16 is the dequantized+reshaped weight built in
        #    process_weights_after_loading (Task 10): the raw self.wo_a.weight is
        #    2D float8_e4m3fn, NOT this layout — do NOT read it directly here.
        n = o.shape[0]
        o_g = o.reshape(n, self.n_local_groups, -1)   # [N, G, heads_per_g*head_dim]
        wo_a_w = self.wo_a_bf16  # bf16 [n_local_groups, o_lora_rank, heads_per_g*head_dim]
        z = torch.einsum("tgd,grd->tgr", o_g.float(),
                         wo_a_w.float()).to(o.dtype)  # [N, G, o_lora_rank]
        # 3) wo_b RowParallelLinear over the flattened [N, G*o_lora_rank].
        return self.wo_b(z.reshape(n, -1))

    def _inverse_rope_gptj(self, x: torch.Tensor,
                           positions: torch.Tensor) -> torch.Tensor:
        # GPT-J interleaved inverse RoPE on the last rope_head_dim dims (fp32).
        rope_dim = self.rope_head_dim
        cache = self.rotary_emb.cos_sin_cache  # [max_pos, rope_dim] = cat(cos, sin)
        cs = cache[positions.long()].float()
        half = rope_dim // 2
        cos = cs[..., :half].repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = -cs[..., half:].repeat_interleave(2, dim=-1).unsqueeze(-2)
        out = x.clone().float()
        rot = out[..., -rope_dim:]
        x1 = rot[..., ::2]
        x2 = rot[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        out[..., -rope_dim:] = rot * cos + rotated * sin
        return out
```

> `self.wo_a_bf16` is produced once in Task 10's `process_weights_after_loading` (dequant fp8→bf16 with ue8m0 block scales + 2D→3D grouped reshape). Verify by printing `self.wo_a.weight.shape`, `self.wo_a.weight.dtype` (expect 2D `float8_e4m3fn`), and `hasattr(self.wo_a, "weight_scale_inv")` (expect True) in Task 10 before relying on the dequant. If `weight_scale_inv` is absent (a non-quantized synthetic build), fall back to `self.wo_a.weight.view(g, r, d).to(bf16)` (the `else` branch of `_get_cached_wo_a_bf16`).

- [ ] **Step 4: Run test to verify it passes**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_o_proj.py -x -s
```
Expected: `1 passed` (reference-convention pin). Real-module parity is verified end-to-end in Task 12.

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tpu_inference/layers/vllm/custom_ops/deepseek_v4_attention.py tests/dsv4/test_o_proj.py && git commit -m "DSV4 Phase1: implement _o_proj (inverse GPT-J RoPE + grouped wo_a einsum + wo_b)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: `forward_mqa` (mla_swa kernel call + ragged metadata)

**Files:**
- Modify `tpu_inference/layers/vllm/custom_ops/deepseek_v4_attention.py:86-88` (the `forward_mqa` stub) + imports.
- Test `tests/dsv4/test_forward_mqa_metadata.py` (metadata builders, run on TPU)

**Interfaces:**
- Consumes Task 2 (`dsv4_mesh`); the kernel `mla_sliding_window_ragged_paged_attention` from `tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py:932`.
- Produces:
  - `VllmDeepseekV4MLAAttention.forward_mqa(self, q, kv, positions, output) -> None` — writes attention output into `output` in place.
  - module-level helper `build_swa_ragged_metadata(attn_metadata) -> tuple` returning `(kv_lens, page_indices, cu_q_lens, distribution)` derived from the vLLM `AttentionMetadata` fields.

Rationale (verified): the kernel signature (`mla_swa.py:932-960`) is:
```
mla_sliding_window_ragged_paged_attention(
    q [max_tok, num_q_heads, head_dim] bf16,
    new_kv [max_tok, head_dim] bf16,          # raw bf16; kernel quantizes internally
    cache_kv [pages, phys//4, 4, 640] uint8,  # donated; DSv4 FP8
    kv_lens i32[max_seqs],
    page_indices i32[max_seqs*pages_per_seq],
    cu_q_lens i32[max_seqs+1],
    distribution i32[3],
    *, sm_scale=1.0, sliding_window:int, logical_page_size:int,
       mask_value=DEFAULT, chunk_prefill_size=None,
       num_kv_pages_per_block, num_queries_per_block,
       vmem_limit_bytes=DEFAULT, unnormalized_output=False)
  -> (out [max_tok, num_q_heads, head_dim], updated_cache_kv, l, m)
```
`distribution=(i,j,k)`: `seqs[0:i]` decode-only, `seqs[i:j]` chunked-prefill, `seqs[j:k]` mixed, `k`=total seqs (docstring `mla_swa.py:971-973`). The kernel def is at `mla_swa.py:932`; `sliding_window` and `logical_page_size` are **required** kwargs (no default), and `num_kv_pages_per_block`/`num_queries_per_block` default to `None` but raise `ValueError` if left `None` (`mla_swa.py:995-998`) — so always pass them (test uses `num_queries_per_block=8, num_kv_pages_per_block=2`). Returns the 4-tuple `(output, updated_kv, l, m)` (`mla_swa.py:1273`). `cu_q_lens = concat([[0], cumulative_sum(new_kv_lens)])` (mla_swa_test.py:438-440). The metadata field names on `AttentionMetadata` are `seq_lens` (kv lengths, `attention_metadata.py:41`), `block_tables` (page indices, `:39`), `query_start_loc` (cu_q_lens, `:43`), `request_distribution` (distribution, `:45`). There is NO Python decode/prefill branch (the split is inside the kernel via `distribution`).

> **layer_name vs prefix (VERIFIED — important):** for THIS class the lookup key is **`self.prefix`**, and the implementation below uses it consistently. Unlike the R1 `VllmMLAAttention` (which has a *separate inner* `MLAAttention` keyed by `self.layer_name`), `VllmDeepseekV4MLAAttention` subclasses `DeepseekV4Attention`, whose base `__init__` registers the **outer module itself** into `compilation_config.static_forward_context[prefix]` and sets `self.prefix = prefix` (no `self.layer_name`) (`/home/enyouki/vllm/vllm/models/deepseek_v4/attention.py` ~148, ~295). The runner builds `layer_name_to_kvcache_index` from `static_forward_context` keys (`runner/kv_cache_manager.py:604,824,904` via `get_layers_from_vllm_config`), so the dict is keyed by the **same `prefix`** the base registered under. Therefore `layer_name_to_kvcache_index[self.prefix]` and the attention-context lookup by `self.prefix` are correct. Do NOT use `self.layer_name` here — `DeepseekV4Attention` never sets it. (If, when running Task 12, the lookup KeyErrors, print `list(ctx.layer_name_to_kvcache_index.keys())` and `self.prefix` to confirm the exact registered string before adjusting.)
>
> **Runtime-glue uncertainty (spec §7.5):** the exact wiring of `output`/`positions`/`kv` into the kernel, and where the KV cache jax array lives in the torchax wrapper context, must be read from the live `attention_impl` (`attention.py:426-505`) and `mla_attention.py:164-203` (R1 template) BEFORE writing the call. NOTE the base `forward` does NOT call `forward_mqa` directly — it calls `attention_impl` (`attention.py:351`) which calls `forward_mqa` (`attention.py:505`); our Phase-1 `forward` override (Task 10) bypasses `attention_impl` and calls `forward_mqa` directly to force the dense path. Use `jax_view`/`torch_view` at the torch<->jax boundary; the R1 boundary template is at `tpu_inference/layers/vllm/backends/flash_attn_mla.py:112-216` (NOTE: `backends/`, not `custom_ops/`). This task's TEST validates only the **pure metadata builders** against the kernel-test construction pattern; the kernel-call wiring's correctness is proven end-to-end in Task 12 + the AOT gate (Task 13).

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_forward_mqa_metadata.py
"""Validate the ragged-metadata builders match the mla_swa kernel-test pattern."""
import jax.numpy as jnp
import numpy as np

from tpu_inference.layers.vllm.custom_ops.deepseek_v4_attention import \
    build_swa_ragged_metadata


class _MD:
    """Minimal stand-in for AttentionMetadata (only the 4 fields read)."""
    def __init__(self, seq_lens, block_tables, query_start_loc,
                 request_distribution):
        self.seq_lens = seq_lens
        self.block_tables = block_tables
        self.query_start_loc = query_start_loc
        self.request_distribution = request_distribution


def test_build_metadata_passes_through_canonical_fields():
    # Mirror mla_swa_test step-2: 2 decode + 1 prefill(len 3); total seqs = 3.
    seq_lens = jnp.array([5, 5, 7], dtype=jnp.int32)
    block_tables = jnp.arange(3 * 4, dtype=jnp.int32)           # flat page table
    cu_q = jnp.array([0, 1, 2, 5], dtype=jnp.int32)             # decode,decode,prefill(3)
    distribution = jnp.array([2, 2, 3], dtype=jnp.int32)        # [decode, prefill_bnd, total]
    md = _MD(seq_lens, block_tables, cu_q, distribution)

    kv_lens, page_indices, cu_q_lens, dist = build_swa_ragged_metadata(md)
    np.testing.assert_array_equal(np.asarray(kv_lens), np.asarray(seq_lens))
    np.testing.assert_array_equal(np.asarray(page_indices),
                                  np.asarray(block_tables))
    np.testing.assert_array_equal(np.asarray(cu_q_lens), np.asarray(cu_q))
    np.testing.assert_array_equal(np.asarray(dist), np.asarray(distribution))
    assert dist.shape == (3,)
    assert cu_q_lens.shape[0] == seq_lens.shape[0] + 1
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_forward_mqa_metadata.py -x -s
```
Expected: `ImportError: cannot import name 'build_swa_ragged_metadata'`.

- [ ] **Step 3: Write minimal implementation**

In `deepseek_v4_attention.py`, add the module-level builder (after `patch_deepseek_v4_mla_cls` or near the top imports — place it above the class) and replace the `forward_mqa` stub. Add imports:
```python
from tpu_inference.kernels.experimental.deepseek_v4.mla_swa import \
    mla_sliding_window_ragged_paged_attention
from vllm.model_executor.layers.attention.attention import \
    get_attention_context
```

Module-level builder:
```python
def build_swa_ragged_metadata(attn_metadata):
    """Map vLLM AttentionMetadata -> mla_swa kernel ragged args.

    attention_metadata.py fields:
      seq_lens             -> kv_lens     i32[max_seqs]
      block_tables         -> page_indices i32[max_seqs*pages_per_seq] (flat)
      query_start_loc      -> cu_q_lens    i32[max_seqs+1]
      request_distribution -> distribution i32[3] = [decode, prefill_bnd, total]
    """
    return (attn_metadata.seq_lens, attn_metadata.block_tables,
            attn_metadata.query_start_loc, attn_metadata.request_distribution)
```

`forward_mqa` (replace lines 86-88):
```python
    def forward_mqa(self, q: torch.Tensor, kv: torch.Tensor,
                    positions: torch.Tensor, output: torch.Tensor) -> None:
        # Single ragged kernel: it quantizes new_kv, writes the cache, and
        # applies BOTH causal and sliding-window masks internally. No Python
        # decode/prefill branch (split is in `distribution`). No sink (Phase 1).
        attn_metadata, _, _, _ = get_attention_context(self.prefix)
        if attn_metadata is None:
            # Warmup dummy: no metadata -> produce zeros.
            output.zero_()
            return
        kv_lens, page_indices, cu_q_lens, distribution = \
            build_swa_ragged_metadata(attn_metadata)

        from tpu_inference.models.vllm.vllm_model_wrapper_context import \
            get_vllm_model_wrapper_context
        ctx = get_vllm_model_wrapper_context()
        kv_cache_index = ctx.layer_name_to_kvcache_index[self.prefix]
        cache_kv = ctx.kv_caches[kv_cache_index]

        q_j = jax_view(q.to(torch.bfloat16))     # [N, n_q_heads, head_dim]
        kv_j = jax_view(kv.to(torch.bfloat16))   # [N, head_dim] raw bf16
        out_j, updated_cache, _l, _m = (
            mla_sliding_window_ragged_paged_attention(
                q_j, kv_j, jax_view(cache_kv),
                jax_view(kv_lens), jax_view(page_indices),
                jax_view(cu_q_lens), jax_view(distribution),
                sm_scale=self.scale,
                sliding_window=self.window_size,
                # WARNING: logical_page_size is the LOGICAL page size (block_size,
                # = vllm_config.cache_config.block_size), NOT shape[1]*shape[2].
                # The kernel test passes the bare page_size (mla_swa_test.py:394).
                # The expression below is a PLACEHOLDER derivation -- confirm the
                # correct value from the KV-cache spec (block_size / 576 B
                # alignment / MLAAttentionSpec) before first run; this is a
                # top-candidate first-forward bug (spec 7.5).
                logical_page_size=self._mla_swa_logical_page_size(cache_kv),
                num_kv_pages_per_block=2,
                num_queries_per_block=8,
            ))
        ctx.kv_caches[kv_cache_index] = torch_view(updated_cache)
        output.copy_(torch_view(out_j))

    def _mla_swa_logical_page_size(self, cache_kv) -> int:
        # The logical page size = the vLLM KV-cache block_size (the kernel test
        # passes the bare page_size, mla_swa_test.py:394). Read it from the cache
        # config, NOT from the physical cache_kv tensor shape. The base
        # get_kv_cache_spec already uses vllm_config.cache_config.block_size
        # (deepseek_v4_attention.py:96) -- store that block_size on self in
        # __init__ (e.g. self._kv_block_size = vllm_config.cache_config.block_size,
        # right after the super().__init__ call) and return it here. Confirm
        # against the MLAAttentionSpec (576 B alignment) wiring before first run.
        return self._kv_block_size
```

> Read the real `attention_impl` (`attention.py:426-505`) and confirm: (a) `q` reaches `forward_mqa` as `[N, n_local_heads, head_dim]` already (it does: `q = self.wq_b(qr).view(-1, n_local_heads, head_dim)`), (b) the metadata field names exactly match (Task 9 test pins them), (c) `logical_page_size` derivation matches the KV-cache spec alignment (576 B / `MLAAttentionSpec`). On this class the lookup attribute is `self.prefix` (the base never sets `self.layer_name`); `get_attention_context` on TPU returns the single `AttentionMetadata` regardless of key (the dict-keyed branch isn't taken on the torchax path), but `layer_name_to_kvcache_index[self.prefix]` IS key-sensitive — confirm the registered key equals `self.prefix` by printing the dict keys in Task 12 if it KeyErrors.

- [ ] **Step 4: Run test to verify it passes**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_forward_mqa_metadata.py -x -s
```
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tpu_inference/layers/vllm/custom_ops/deepseek_v4_attention.py tests/dsv4/test_forward_mqa_metadata.py && git commit -m "DSV4 Phase1: implement forward_mqa (mla_swa kernel + ragged metadata builder)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: `process_weights_after_loading` (wo_a BMM layout) + remove pass-through `forward`

**Files:**
- Modify `tpu_inference/layers/vllm/custom_ops/deepseek_v4_attention.py:106-107` (`process_weights_after_loading`) and `:112-121` (replace the pass-through `forward` with the real orchestration).
- Test `tests/dsv4/test_forward_orchestration.py`

**Interfaces:**
- Consumes Task 8 (`_o_proj`), Task 9 (`forward_mqa`).
- Produces:
  - `VllmDeepseekV4MLAAttention.process_weights_after_loading(self, act_order: bool = False) -> None` — ensures `self.wo_a.weight` is in the `[n_local_groups, o_lora_rank, heads_per_g*head_dim]` BMM layout `_o_proj` consumes (reshape/store if needed); leaves FP8 linears and FP4 experts untouched.
  - `VllmDeepseekV4MLAAttention.forward(self, positions, hidden_states, llama_4_scaling=None) -> torch.Tensor` — real orchestration of steps 0-4, 6-7 (skip step 5), routing all 43 layers through `mla_swa` via `forward_mqa` then `_o_proj`.

Rationale (verified): the base `DeepseekV4Attention.forward` (`attention.py:318-364`) drives steps 1-4 via `attn_gemm_parallel_execute` → `qr_kv.split([q_lora_rank, head_dim])` (line 339) → `fused_q_kv_rmsnorm` (lines 340-346) → `attention_impl` (line 351), which internally does `wq_b(qr).view(-1, n_local_heads, head_dim)` + the fused qnorm/rope/quant/insert + `forward_mqa` (line 505), and step 7 (`_o_proj`, line 364). `attention_impl` (`attention.py:426-505`) runs indexer/compressor **conditionally** (`if self.indexer is not None` / `elif self.compressor is not None`) — for a layer where those submodules are None it takes the SWA-only `else` branch. **However**, on TPU the base `forward`/`attention_impl` chain pulls in CUDA multi-stream machinery (`execute_in_parallel`, `ln_events`, `@eager_break_during_capture`) and the GPU fused qnorm/rope/kv-insert CUDA op, none of which run on TPU. So we do NOT inherit the base `forward`; we write a **thin Phase-1 `forward` override** that replicates steps 1-4 in pure torch (so torchax traces them), calls our `forward_mqa` directly (skipping `attention_impl` and step 5 entirely so EVERY layer is forced dense through `mla_swa`, per spec §7.2), then our `_o_proj`. We call `self.fused_wqa_wkv(hidden_states)` directly (not `attn_gemm_parallel_execute`) to avoid the aux-GEMM/stream machinery. The import `from vllm.models.deepseek_v4.common.ops import fused_q_kv_rmsnorm` is valid (re-exported by the `common/ops` package `__init__`, used by `attention.py:25-28` itself); it takes `(qr, kv, q_norm.weight.data, kv_norm.weight.data, eps)` and returns `(qr, kv)`. `self.wq_b` is a `ColumnParallelLinear(q_lora_rank, n_heads*head_dim, return_bias=False)` (returns a single tensor). All referenced attributes (`fused_wqa_wkv`, `wq_b`, `q_norm`, `kv_norm`, `eps`, `q_lora_rank`, `head_dim`, `n_local_heads`, `padded_heads`, `rotary_emb`, `rope_head_dim`) are set by `DeepseekV4Attention.__init__`.

> **Note on `process_weights_after_loading`:** there is NO base method by that name on `DeepseekV4Attention` — our subclass already defines it as a stub (`deepseek_v4_attention.py:106-107`). So this is a class-specific method (the loader calls it by duck-typed name), not an override of a base contract; its signature `(self, act_order: bool = False)` is our own choice and need not match any base.

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_forward_orchestration.py
"""Static guards: forward must not be the pass-through stub; pwal must reshape wo_a."""
import inspect

from tpu_inference.layers.vllm.custom_ops.deepseek_v4_attention import \
    VllmDeepseekV4MLAAttention


def test_forward_is_not_passthrough_stub():
    src = inspect.getsource(VllmDeepseekV4MLAAttention.forward)
    assert "just a pass-through" not in src, "forward is still the stub"
    assert "return hidden_states" not in src.split("def forward")[1][:400], \
        "forward must not return hidden_states unchanged"
    # must call the two TPU ops
    assert "forward_mqa" in src
    assert "_o_proj" in src


def test_pwal_no_longer_bare_pass():
    src = inspect.getsource(
        VllmDeepseekV4MLAAttention.process_weights_after_loading)
    body = src.split(":", 1)[1].strip()
    assert body != "pass", "process_weights_after_loading is still a no-op"
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_forward_orchestration.py -x -s
```
Expected: `test_forward_is_not_passthrough_stub` FAILS (`forward is still the stub`).

- [ ] **Step 3: Write minimal implementation**

Replace `process_weights_after_loading` (lines 106-107). It must **dequantize** the FP8-block `wo_a` to bf16 AND reshape 2D→3D (the GPU `deepgemm_post_process_fp8_weight_block` reshape is CUDA-only and never runs on TPU), storing the result as `self.wo_a_bf16` for `_o_proj`. This replicates `_get_cached_wo_a_bf16` (`/home/enyouki/vllm/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:967-1001`):
```python
    def process_weights_after_loading(self, act_order: bool = False) -> None:
        # FP4 experts and FP8-block linears keep their quant; here we ONLY build
        # the bf16 grouped wo_a weight that _o_proj's einsum("tgd,grd->tgr")
        # consumes: [n_local_groups, o_lora_rank, heads_per_group*head_dim].
        # self.wo_a.weight is 2D float8_e4m3fn (n_local_groups*o_lora_rank,
        # heads_per_g*head_dim) with ue8m0 block scales in weight_scale_inv;
        # the 2D->3D reshape GPU does at load is CUDA-only, so do it here.
        g = self.wo_a.bmm_batch_size          # == self.n_local_groups
        r = self.o_lora_rank
        d = self.wo_a.weight.shape[1]         # heads_per_group * head_dim
        if hasattr(self.wo_a, "weight_scale_inv"):
            w = self.wo_a.weight.view(g, r, d).to(torch.float32)
            scale = self._expand_wo_a_block_scales(
                self.wo_a.weight_scale_inv.view(
                    g, -1, self.wo_a.weight_scale_inv.shape[-1]), r, d)
            self.wo_a_bf16 = (w * scale).to(torch.bfloat16)
        else:
            # Non-quantized (synthetic) build: just view+cast.
            self.wo_a_bf16 = self.wo_a.weight.view(g, r, d).to(torch.bfloat16)

    def _expand_wo_a_block_scales(self, scale, r, d):
        # Mirror rocm_aiter_mla_sparse._expand_2d_block_scales / _decode_e8m0_scales:
        # ue8m0 -> fp32 (exponent<<23), then repeat_interleave each 128-block up
        # to (g, r, d). scale: [g, r//block, d//block] in ue8m0 (uint8) layout.
        import torch as _t
        s = scale
        if s.dtype in (_t.uint8,) or "e8m0" in str(s.dtype):
            # ue8m0 byte -> fp32 power-of-two (matches _upcast_e8m0_to_fp32).
            s = (s.view(_t.uint8).to(_t.int32) << 23).view(_t.float32)
        else:
            s = s.to(_t.float32)
        block_r = r // s.shape[1]
        block_d = d // s.shape[2]
        s = s.repeat_interleave(block_r, dim=1).repeat_interleave(block_d, dim=2)
        return s[:, :r, :d]
```

> **Pin the scale layout in Step 2 of this task BEFORE trusting `_expand_wo_a_block_scales`.** Print `self.wo_a.weight.dtype` (expect `float8_e4m3fn`), `self.wo_a.weight.shape` (2D), `hasattr(self.wo_a, "weight_scale_inv")`, and `self.wo_a.weight_scale_inv.dtype/.shape` in a one-off debug run on the synthetic mini-model (Task 12). If the block dims don't divide evenly, read `_expand_2d_block_scales` (`rocm_aiter_mla_sparse.py:863-874`) and `_decode_e8m0_scales` (`:853-860`) and match them exactly — they are the source of truth for this checkpoint's ue8m0 block-scale format. If the synthetic loader produces non-quantized bf16 `wo_a` (no `weight_scale_inv`), the `else` branch is taken and the dequant path is exercised only under real weights (Task 15) — note that gap.

Replace the pass-through `forward` (lines 112-121) with the real dense orchestration:
```python
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Steps 0-4,6-7 of the dataflow; step 5 (compressor/indexer) skipped:
        # every layer is forced dense through mla_swa (coherent <=128 tokens).
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype, device=hidden_states.device)

        # Steps 1-2: fused WQA/WKV GEMM -> split -> fused q/kv RMSNorm.
        # Call fused_wqa_wkv DIRECTLY (a disable_tp ReplicatedLinear, returns
        # (qr_kv, None)), NOT attn_gemm_parallel_execute -- the latter drives
        # CUDA multi-stream aux GEMMs (execute_in_parallel + ln_events) for the
        # indexer/compressor, which Phase 1 skips. (attention.py:194-201,409-412)
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        # Split sizes are [q_lora_rank, head_dim] = [1024, 512] (attention.py:339);
        # there is NO kv_lora_rank in DSV4 -- the KV latent dim IS head_dim (512).
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        # fused_q_kv_rmsnorm: weighted RMSNorm of the qr (1024) and kv (512)
        # latents. CHECK at Task-12 trace time whether the imported impl is a
        # CUDA custom op (untraceable on TPU). If torchax cannot trace it,
        # replace with two pure-torch weighted RMSNorms (reuse
        # tests/dsv4/torch_ref.rmsnorm_with_weight numerics):
        #   qr = rmsnorm_with_weight(qr, self.q_norm.weight.data, self.eps)
        #   kv = rmsnorm_with_weight(kv, self.kv_norm.weight.data, self.eps)
        # which is the exact same math and is the parity reference anyway.
        from vllm.models.deepseek_v4.common.ops import fused_q_kv_rmsnorm
        qr, kv = fused_q_kv_rmsnorm(qr, kv, self.q_norm.weight.data,
                                    self.kv_norm.weight.data, self.eps)

        # Step 3: wq_b -> [N, n_local_heads, head_dim].
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)

        # Step 4: per-head weight-free RMSNorm + GPT-J RoPE (last rope dims) +
        # KV RoPE + fp8 quant + paged insert is done INSIDE the mla_swa kernel
        # for KV; q-side norm+rope is applied by the base fused op. For Phase 1
        # we apply the q-side fused qnorm/rope here via the base helper, then
        # hand raw bf16 kv to the kernel (it quantizes + inserts).
        q, kv = self._fused_qnorm_rope_kv_insert_q_only(q, kv, positions)

        # Step 6: attention. The mla_swa kernel output is [N, n_local_heads,
        # head_dim]; our get_padded_num_q_heads returns num_heads unchanged so
        # padded_heads == n_local_heads and o_padded matches the kernel output
        # shape exactly (the slice below is then a no-op). If a future change
        # makes padded_heads > n_local_heads, pass o_padded[:, :n_local_heads, :]
        # to forward_mqa instead (output.copy_ requires matching shapes).
        self.forward_mqa(q, kv, positions, o_padded)
        o = o_padded[:, :self.n_local_heads, :]

        # Step 7: inverse-rope o_proj.
        return self._o_proj(o, positions)
```

> **Step-4 helper (`_fused_qnorm_rope_kv_insert_q_only`) — read the base first.** The base `_fused_qnorm_rope_kv_insert` (`attention.py:507-594`) does the q-side per-head weight-free RMSNorm + GPT-J RoPE AND the kv-side rope+fp8-quant+paged-insert in one fused CUDA op (unrunnable on TPU). On TPU the kv-side fp8-quant + paged insert is done by `mla_swa` (it quantizes + writes the cache). So implement `_fused_qnorm_rope_kv_insert_q_only` to apply ONLY: (a) the q-side per-head **weight-free** RMSNorm over `head_dim` (returns fp32 then cast; matches `rmsnorm_no_weight`, `test_fused_deepseek_v4_qnorm_rope_kv_insert.py:109-119`), (b) forward GPT-J interleaved RoPE on the last `rope_head_dim` dims of q AND of kv, using `self.rotary_emb.cos_sin_cache` + `positions`, returning `(q, kv)` with kv rope-applied (last 64 dims) but NOT quantized (kernel does that). Reuse the exact GPT-J interleave from `_inverse_rope_gptj` (Task 8) but with FORWARD sign (`+sin`, not `-sin`). Pin its math to `tests/dsv4/torch_ref.apply_rope_gptj_last_k` + `rmsnorm_no_weight` (Task 5) — add a standalone unit test in this task's test file. (The base `_fused_qnorm_rope_kv_insert` keeps the whole RMSNorm→RoPE pipeline in fp32 and rounds once at the store; match that — do the norm+rope in fp32, cast at the end.)

Add the q-only fused helper:
```python
    def _fused_qnorm_rope_kv_insert_q_only(self, q, kv, positions):
        # q: [N, n_local_heads, head_dim]; kv: [N, head_dim].
        rope_dim = self.rope_head_dim
        # weight-free per-head RMSNorm over head_dim.
        qf = q.float()
        q = (qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + self.eps)) \
            .to(q.dtype)
        cache = self.rotary_emb.cos_sin_cache
        cs = cache[positions.long()].float()
        half = rope_dim // 2
        cos = cs[..., :half].repeat_interleave(2, dim=-1)
        sin = cs[..., half:].repeat_interleave(2, dim=-1)

        def _rope(t, cos_, sin_):
            out = t.clone().float()
            rot = out[..., -rope_dim:]
            x1, x2 = rot[..., ::2], rot[..., 1::2]
            rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
            out[..., -rope_dim:] = rot * cos_ + rotated * sin_
            return out.to(t.dtype)

        q = _rope(q, cos.unsqueeze(-2), sin.unsqueeze(-2))   # broadcast over heads
        kv = _rope(kv, cos, sin)                              # [N, head_dim]
        return q, kv
```

- [ ] **Step 4: Run test to verify it passes**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_forward_orchestration.py -x -s
```
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tpu_inference/layers/vllm/custom_ops/deepseek_v4_attention.py tests/dsv4/test_forward_orchestration.py && git commit -m "DSV4 Phase1: real forward orchestration (steps 0-4,6-7) + wo_a BMM layout in pwal

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: Dense-MLA attention parity (forward_mqa vs interpret + dense reference)

**Files:**
- Test `tests/dsv4/test_mla_swa_parity.py`

**Interfaces:**
- Consumes Task 2 (`dsv4_mesh`, `assert_threefry_partitionable`), the kernel `mla_sliding_window_ragged_paged_attention`, the mla_swa-test construction pattern.
- Produces no new symbols (a parity test).

Rationale (verified): the kernel-test reference (`mla_swa_test.py:248-270`, function `ref_implementation`) builds the dense reference per-seq: `attn = einsum("qnh,kh->nqk", q, k, preferred_element_type=fp32)*sm_scale`; causal mask `q_span < kv_span` with `q_span = kv_len - q_len + iota(axis 1)`, `kv_span = iota(axis 2)`; SWA mask `q_span - sliding_window >= kv_span` (OR'd, line 260); `attn = where(mask, mask_value, attn)`; softmax; `out = einsum("nqk,kl->qnl", probs, v)`. For ≤128 tokens within one window the SWA mask never fires → equals full causal attention. Metadata: `cu_q_lens = concat([[0], jnp.cumulative_sum(new_kv_lens)])` (`mla_swa_test.py:438-440`), `distribution = [decode, prefill_bnd, total]`, `page_indices` = the test's `swc_page_indices`. Tolerance `rtol=atol=0.1` (`mla_swa_test.py:405`). **The source kernel test uses `num_heads=128, head_dim=512, page_size=12, sliding_window=32` (`setUp`, mla_swa_test.py:292-297) — do NOT confuse with `mla_test.py`'s `64/512/16/none`.** Our parity test ports that test's `setUp`/cache-build but with mini-config-compatible dims (`num_heads=8`, `head_dim=512`, `sliding_window=128`, seqs ≤128 tok). This reuses the kernel's own construction so it is a true math check of our metadata builder + call conventions, not a re-derivation.

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_mla_swa_parity.py
"""Dense-MLA parity: the mla_swa kernel vs a pure-jax dense reference, using the
same ragged metadata our forward_mqa builds. <=window sequences => full causal."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.experimental.deepseek_v4 import mla_swa
# NOTE: there is NO `tpu_inference.kernels.utils` module. `cdiv`/`get_dtype_packing`
# are defined locally in mla_swa_test.py (lines 25, 34) AND exported by mla_swa.py
# (lines 30, 43). Import from mla_swa, or copy the local defs from the kernel test:
from tpu_inference.kernels.experimental.deepseek_v4.mla_swa import (cdiv,
                                                                    get_dtype_packing)


def _dense_ref(q, new_kv, kv_lens, cu_q_lens, sm_scale, sliding_window):
    # Per-seq dense MLA with causal + SWA mask (mla_swa_test.py:248-270).
    outs = []
    num_seqs = kv_lens.shape[0]
    for i in range(num_seqs):
        qs, qe = int(cu_q_lens[i]), int(cu_q_lens[i + 1])
        q_len = qe - qs
        if q_len == 0:
            continue
        kv_len = int(kv_lens[i])
        k = new_kv[qs:qe]                      # [kv segment] (<=window so all-new)
        q_i = q[qs:qe]                         # [q_len, heads, head_dim]
        attn = jnp.einsum("qnh,kh->nqk", q_i, k,
                          preferred_element_type=jnp.float32) * sm_scale
        q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32,
                                                             attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span
        mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        attn = jnp.where(mask, -1e30, attn)
        probs = jax.nn.softmax(attn, axis=-1).astype(k.dtype)
        out_i = jnp.einsum("nqk,kl->qnl", probs, k[:, :512]).astype(q_i.dtype)
        outs.append(out_i)
    return jnp.concatenate(outs, axis=0)


@pytest.mark.skip(reason="full kernel-vs-ref parity requires the kernel-test's "
                  "cache build; this is the structural template. Fill in the "
                  "cache build from mla_swa_test.setUp before enabling.")
def test_mla_swa_dense_parity():
    # See mla_swa_test.py:282-417 for the exact cache/page-index build to reuse.
    pass
```

> The full parity test must lift the cache construction (`swc_cache` uint8 640-layout) and `run_and_compare_outputs` flow verbatim from `tests/kernels/deepseek_v4/mla_swa_test.py:282-417`. Rather than re-deriving it, the worker should subclass or copy that test's `setUp` + `run_and_compare_outputs`, set `sliding_window=128` and `num_heads=8` (mini-config), seqs ≤128 tokens, and assert the kernel output matches `_dense_ref` within `rtol=atol=0.1`. Keep the existing `mla_swa_test.py` as the proven oracle.

- [ ] **Step 2: Run test to verify it fails (then implement, then unskip)**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_mla_swa_parity.py -x -s
```
Expected: `1 skipped`. Remove the `@pytest.mark.skip`, port the cache build from `mla_swa_test.py:282-417`, and the test must FAIL only if the metadata/call conventions are wrong, then PASS once correct.

- [ ] **Step 3: Write minimal implementation**

Port the cache/page-index build and `run_and_compare_outputs` from `tests/kernels/deepseek_v4/mla_swa_test.py:282-417` into `test_mla_swa_dense_parity` (instance config: `batch_size=4, num_heads=8, head_dim=512, sliding_window=128, page_size=16`, all `new_kv_lens <= 128`). Call:
```python
out, _cache, _l, _m = mla_swa.mla_sliding_window_ragged_paged_attention(
    q, new_kv, swc_cache, kv_lens, page_indices, cu_q_lens, distribution,
    sm_scale=1.0, sliding_window=128,
    num_queries_per_block=8, num_kv_pages_per_block=2,
    logical_page_size=page_size)
np.testing.assert_allclose(np.asarray(_dense_ref(...)), np.asarray(out),
                           rtol=0.1, atol=0.1)
```

- [ ] **Step 4: Run test to verify it passes**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_mla_swa_parity.py -x -s
```
Expected: `1 passed` (on the v6e-8 host).

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tests/dsv4/test_mla_swa_parity.py && git commit -m "DSV4 Phase1: dense-MLA parity (mla_swa kernel vs pure-jax dense ref, <=window)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 12: Synthetic mini-model build + load + end-to-end first-divergence

**Files:**
- Create `tests/dsv4/build_mini_model.py`
- Test `tests/dsv4/test_mini_model_forward.py`

**Interfaces:**
- Consumes Task 1 (`make_dsv4_mini_config`), Task 2 (`dsv4_mesh`, `assert_threefry_partitionable`), Tasks 4/7/8/9/10 (the fixed quant + the full attention forward incl. `process_weights_after_loading` which builds `self.wo_a_bf16`). Reuses `JaxDummyModelLoader` via `get_model_loader(load_config)` with `load_format="jax_dummy"` (registered at `weight_utils.py:1079`, class def `:1080`), the torchax model wrapper `_maybe_patch_for_deepseek_v4` (`vllm_model_wrapper.py:94`). NOTE: the repo's `models/common/model_loader.py:269-278` already rewrites `load_format="dummy"` → `"jax_dummy"`, so a synthetic build typically goes through `JaxDummyModelLoader`.
- Produces:
  - `build_mini_model(mesh, cfg=None) -> tuple[model, vllm_config]` — constructs the DSV4 mini-model under the torchax path with synthetic FP4/FP8 dummy weights, no full-model load.
  - `run_mini_forward(model, vllm_config, input_ids, positions) -> torch.Tensor` — one un-jitted (eager) forward returning logits.

Rationale (verified): `JaxDummyModelLoader` is the JAX-path loader; the spec flags it must be verified/adapted for the vllm/torchax loader (§6.5, §8). The torchax model is built via the vLLM model wrapper with `MODEL_IMPL_TYPE=vllm` and `_maybe_patch_for_deepseek_v4` (forces AMD variant + patches our attention class). `debug_accuracy_for_each_op` (spec §6.3 B) runs eager (un-jitted) and drops into pdb at the first diverging op; its hardcoded `atol=1e-3` is too tight for FP4/FP8 — patch the threshold or filter ops.

> **Loader uncertainty (spec §8 — FLAGGED):** `JaxDummyModelLoader` targets the pure-JAX `JaxModule`, NOT the torchax/vllm nn.Module model. If `loader.load_weights(model, model_config)` fails on the torchax model (it iterates `model.named_parameters()` expecting jax params), the fallback is to fill each `nn.Parameter` of the built torchax model with synthetic values directly in the right quant dtype (FP4-packed experts via the inverse of `break_fp4_bytes`; FP8 e4m3 linears via random bf16 cast to `float8_e4m3fn` + ue8m0 scales), then call each module's `process_weights_after_loading`. Build a small `fill_synthetic_weights(model)` in `build_mini_model.py` for this path. **This is a real Phase-0 risk; resolve it empirically here before Task 13/14.**

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_mini_model_forward.py
"""Synthetic mini-model: build under torchax, load dummy FP4/FP8 weights, run a
<=128-token eager forward on the production mesh. Catches untraceable ops and
the FP4 GMM blocker end-to-end (no full-model load)."""
import jax
import pytest
import torch

# NEW_MODEL_DESIGN=1 and MODEL_IMPL_TYPE=vllm MUST already be set in the env
# (tests/dsv4/conftest.py sets them before collection; also pass the shell prefix).
# Do NOT rely on os.environ.setdefault here -- it runs after tpu_inference is
# imported and is therefore too late to affect ShardingAxisName/mesh selection.
from tests.dsv4.build_mini_model import build_mini_model, run_mini_forward
from tests.dsv4.mesh_fixtures import (assert_threefry_partitionable,  # noqa
                                      dsv4_mesh)


def test_mini_model_eager_forward_runs(dsv4_mesh):
    assert_threefry_partitionable()
    with jax.set_mesh(dsv4_mesh):
        model, vllm_config = build_mini_model(dsv4_mesh)
        n = 64  # <=128 (one window)
        input_ids = torch.arange(n, dtype=torch.int32) % 1280
        positions = torch.arange(n, dtype=torch.int32)
        logits = run_mini_forward(model, vllm_config, input_ids, positions)
    assert logits.shape[0] == n
    assert torch.isfinite(logits.float()).all(), "non-finite logits (NaN/Inf)"
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_mini_model_forward.py -x -s
```
Expected: `ModuleNotFoundError: No module named 'tests.dsv4.build_mini_model'`.

- [ ] **Step 3: Write minimal implementation**

Build `tests/dsv4/build_mini_model.py`. The exact vLLM config-construction + model-build glue must be read from an existing torchax model test (search `tests/` for `MODEL_IMPL_TYPE`/`vllm_model_wrapper`/`get_model` usage and mirror it). Skeleton (fill the marked spots from the real glue):
```python
# tests/dsv4/build_mini_model.py
"""Construct the DSV4 mini-model on the torchax path with synthetic weights."""
import jax
import torch

from tests.dsv4.mini_config import make_dsv4_mini_config


def build_mini_model(mesh, cfg=None):
    cfg = cfg or make_dsv4_mini_config()
    # 1) Build a VllmConfig whose hf_config carries the mini cfg dict, with
    #    quant_config = get_tpu_quantization_config(vllm_config) (DSV4_FP8 ->
    #    FP4 experts), load_format="jax_dummy", parallel/tensor/EP set for the
    #    8-chip mesh.  <-- mirror tests/models/jax/test_qwen3_moe.py:42-94 +
    #    the DSV4 wrapper path (vllm_model_wrapper.py:_maybe_patch_for_deepseek_v4).
    vllm_config = _make_vllm_config_from_mini(cfg, mesh)  # implement from real glue
    # 2) Build the torchax model under the DSV4 patch context.
    with _maybe_patch_for_deepseek_v4(vllm_config):       # vllm_model_wrapper.py:94
        model = _build_torchax_model(vllm_config, mesh)   # implement from real glue
    # 3) Synthetic weights: try JaxDummyModelLoader; fall back to direct fill.
    _load_synthetic_weights(model, vllm_config, mesh)
    return model, vllm_config


def run_mini_forward(model, vllm_config, input_ids, positions):
    # Eager (un-jitted) forward; build minimal AttentionMetadata for <=128 tok
    # single-sequence prefill: seq_lens=[n], block_tables=arange(pages),
    # query_start_loc=[0,n]. The request_distribution=(i,j,k) tuple must follow
    # the mla_swa kernel convention (seqs[0:i] decode, seqs[i:j] chunked-prefill,
    # seqs[j:k] mixed, k=total) -- do NOT guess; read exactly how the runner
    # populates request_distribution/query_start_loc for a single full-prefill
    # request (runner/persistent_batch_manager.py) and mirror it. For one full
    # prefill seq it is most likely (0, 1, 1) (1 chunked-prefill seq, 0 decode),
    # but confirm against the runner + mla_swa_test.py:438-460 before relying on it.
    # Optionally enable torchax first-divergence:
    #   import torchax
    #   torchax.default_env().config.debug_accuracy_for_each_op = True  # patch atol for FP4/FP8
    return _eager_logits(model, input_ids, positions)     # implement from real glue
```

> Implement `_make_vllm_config_from_mini`, `_build_torchax_model`, `_load_synthetic_weights`, `_eager_logits`, and the `AttentionMetadata` build by reading the real torchax model-build path and `runner/persistent_batch_manager.py` (how `request_distribution`/`query_start_loc` are populated) and the R1 attention test if one exists. Keep `n<=128`. For first-divergence, set `debug_accuracy_for_each_op=True` and bump its `atol` (spec §6.3 B: hardcoded 1e-3 is too tight) — run un-jitted.

- [ ] **Step 4: Run test to verify it passes**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_mini_model_forward.py -x -s
```
Expected: `1 passed` (finite logits for a 64-token forward on synthetic weights). If `debug_accuracy_for_each_op` drops into pdb, that pinpoints the first diverging op — fix it, then re-run.

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tests/dsv4/build_mini_model.py tests/dsv4/test_mini_model_forward.py && git commit -m "DSV4 Phase1: synthetic mini-model build/load + eager forward (finite logits, no full load)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 13: AOT compile gate on the full mini-forward + FP4 GMM compile proof

**Files:**
- Test `tests/dsv4/test_forward_aot_gate.py`

**Interfaces:**
- Consumes Task 3 (`aot_compile`, `make_aval`), Task 12 (`build_mini_model`), Task 2 (`dsv4_mesh`).
- Produces no new symbols.

Rationale (verified): the AOT gate (`.lower().compile()`) runs Mosaic passes and surfaces the FP4 GMM MosaicError with no full run (spec §6.3 A, §7.1 step 2). After the Task-4 block-size fix, the full forward (incl. the MoE GMM) must compile on v6e. `gmm_v2` is TPU-only at trace time, so this test runs on the v6e-8 host.

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_forward_aot_gate.py
"""AOT gate: the full mini-forward (incl. FP4 MoE GMM) must compile on v6e.
Proves the Task-4 block-size fix routes through the dequant-in-VMEM path."""
import jax
import pytest
import torch

from tests.dsv4.aot_gate import aot_compile
from tests.dsv4.build_mini_model import build_mini_model
from tests.dsv4.mesh_fixtures import dsv4_mesh  # noqa


def test_full_mini_forward_compiles_on_v6e(dsv4_mesh):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("Mosaic compile is TPU-only")
    with jax.set_mesh(dsv4_mesh):
        model, vllm_config = build_mini_model(dsv4_mesh)
        # Wrap the jitted step the model exposes; AOT-compile it against dummy
        # avals for a 64-token single-seq prefill. The model's jitted forward
        # entrypoint + its arg avals come from the same glue as Task 12.
        compiled = _aot_compile_mini_forward(model, vllm_config, dsv4_mesh,
                                             num_tokens=64)
    assert compiled is not None


def _aot_compile_mini_forward(model, vllm_config, mesh, num_tokens):
    # Build ShapeDtypeStruct avals for (input_ids, positions, attn_metadata,
    # kv_caches) matching the model's jitted forward signature, then:
    #   return aot_compile(jitted_forward, *avals, mesh=mesh)
    raise NotImplementedError(
        "fill avals from the Task-11 jitted forward signature")
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_forward_aot_gate.py -x -s
```
Expected: FAILS with `NotImplementedError` (avals not yet wired). Before the Task-4 fix it would instead surface a MosaicError on the `tpu.unpack_subelements` f4 op (spec §3) — the fix removes that.

- [ ] **Step 3: Write minimal implementation**

Implement `_aot_compile_mini_forward`: read the model's jitted forward entrypoint (the `jit_step_func`-style callable used by the runner) and build `jax.ShapeDtypeStruct` avals for its inputs (input_ids `[64]` i32, positions `[64]` i32, the `AttentionMetadata` leaves, and the kv_caches list) using `make_aval(..., mesh=mesh, spec=...)` with `ShardingAxisName.ATTN_DATA` for token-axis arrays and `ShardingAxisName.BATCH` for the kv_cache. Then `return aot_compile(jitted_forward, *avals, mesh=mesh)`.

> If wiring full-forward avals is heavy, the acceptable fallback is to AOT-compile the **MoE layer's own forward** (the `VllmMxfp4MoEMethod`/fused-MoE call built by the mini-model, which internally calls `gmm_v2` on the requantized FP4 experts) in isolation against `make_aval`-built activation/expert avals — NOT a hand-assembled raw `gmm_v2(...)` call (that path is forbidden in Task 4: assembling packed `float4_e2m1fn` rhs + ue8m0 `rhs_scale` + `group_sizes` by hand is error-prone). Reach the GMM through the built MoE module so the requant block size (Task 4) is the one actually in the weights. Either form proves the load-bearing claim: the FP4 fix makes the MoE GMM compile on v6e via the dequant-in-VMEM path.

- [ ] **Step 4: Run test to verify it passes**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_forward_aot_gate.py -x -s
```
Expected: `1 passed` (compiles on v6e after the Task-4 fix).

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tests/dsv4/test_forward_aot_gate.py && git commit -m "DSV4 Phase1: AOT gate proves full mini-forward (FP4 MoE GMM) compiles on v6e

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 14: Phase-1 invariants (causality/SWA, RoPE norm, MoE top-6, output sharding)

**Files:**
- Test `tests/dsv4/test_phase1_invariants.py`

**Interfaces:**
- Consumes Task 12 (`build_mini_model`, `run_mini_forward`), Task 2 (`dsv4_mesh`, `assert_sharded_like`), Task 5 (`apply_rope_gptj_last_k`, `build_cos_sin_cache`).
- Produces no new symbols.

Rationale (verified, spec §6.3 G + §7.4 step 4): reference-free property tests — causality/SWA (future tokens can't change earlier outputs; ≤128 tok ⇒ window covers whole sequence), RoPE per-pair L2 norm preservation (`atol=1e-4`), MoE top-6 (exactly 6 experts selected per token via `num_experts_per_tok`), output sharding spec on the committed logits.

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_phase1_invariants.py
import jax
import numpy as np
import torch

from tests.dsv4 import torch_ref
from tests.dsv4.build_mini_model import build_mini_model, run_mini_forward
from tests.dsv4.mesh_fixtures import (assert_sharded_like,  # noqa
                                      assert_threefry_partitionable, dsv4_mesh)
from jax.sharding import PartitionSpec as P
from tpu_inference.layers.common.sharding import ShardingAxisName


def test_rope_preserves_pair_l2_norm():
    rope_dim = 64
    cache = torch_ref.build_cos_sin_cache(rope_dim, 256, theta=10000.0)
    x = torch.randn(7, 2, 512)
    pos = torch.arange(7)
    rot = torch_ref.apply_rope_gptj_last_k(x, pos, cache, rope_dim).float()
    # per even/odd pair magnitude is preserved by rotation
    a = x[..., -rope_dim:].float()
    pa = (a[..., ::2]**2 + a[..., 1::2]**2)
    pb = (rot[..., -rope_dim:][..., ::2]**2 + rot[..., -rope_dim:][..., 1::2]**2)
    np.testing.assert_allclose(pa.numpy(), pb.numpy(), rtol=1e-4, atol=1e-4)


def test_causality_within_window(dsv4_mesh):
    assert_threefry_partitionable()
    with jax.set_mesh(dsv4_mesh):
        model, cfg = build_mini_model(dsv4_mesh)
        n = 32
        ids = torch.arange(n, dtype=torch.int32) % 1280
        pos = torch.arange(n, dtype=torch.int32)
        base = run_mini_forward(model, cfg, ids, pos).float()
        # Perturb the LAST token's id; earlier logits must be unchanged.
        ids2 = ids.clone(); ids2[-1] = (ids2[-1] + 1) % 1280
        pert = run_mini_forward(model, cfg, ids2, pos).float()
    np.testing.assert_allclose(base[:-1].numpy(), pert[:-1].numpy(),
                               rtol=1e-3, atol=1e-3)


def test_logits_sharding_spec(dsv4_mesh):
    with jax.set_mesh(dsv4_mesh):
        model, cfg = build_mini_model(dsv4_mesh)
        n = 16
        ids = torch.arange(n, dtype=torch.int32) % 1280
        pos = torch.arange(n, dtype=torch.int32)
        from torchax.interop import jax_view
        logits = jax_view(run_mini_forward(model, cfg, ids, pos))
    # logits token axis is replicated/data-sharded; vocab axis on model axes.
    assert_sharded_like(logits, dsv4_mesh,
                        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.VOCAB))
```

> The exact output PartitionSpec for logits must be read from the model's lm_head sharding before finalizing `test_logits_sharding_spec` — assert the real committed spec. The causality and RoPE-norm tests are reference-free and stand as written.

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_phase1_invariants.py -x -s
```
Expected: import/build failures if Task 12 glue is incomplete; otherwise the RoPE-norm test passes and the model-level invariants exercise the real forward.

- [ ] **Step 3: Write minimal implementation**

No production code; finalize the logits PartitionSpec in `test_logits_sharding_spec` from the real lm_head sharding (print `logits.sharding` once, then assert it). If `test_causality_within_window` fails, that is a real attention-masking bug — debug via `superpowers:systematic-debugging` and the Task-10 dense parity before relaxing the test.

- [ ] **Step 4: Run test to verify it passes**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_phase1_invariants.py -x -s
```
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tests/dsv4/test_phase1_invariants.py && git commit -m "DSV4 Phase1: invariants (RoPE norm, causality<=window, logits sharding)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 15: Coherence smoke (MILESTONE GATE — real weights, ≤128 tokens)

**Files:**
- Test `tests/dsv4/test_coherence_smoke.py`

**Interfaces:**
- Consumes the full implemented forward (Tasks 4/7/8/9). This is the ONLY task that loads the real ~187 GiB weights from `/home/enyouki/dsv4-weights`.
- Produces no new symbols.

Rationale (verified, spec §7.4 step 5 + §7.3): for `pos < 128` the SWA mask never fires → `mla_swa` attends the full causal prefix → mathematically equals full attention; the SWA cache holds real uncompressed per-token KV for every layer, so routing all 43 layers through `mla_swa` reads complete KV. The one fidelity gap vs reference is the missing sink (small softmax drift — coherent, not parity; closed Phase 2). Output for a ≤128-token prompt must read sensibly. Mark with the slow/real-weights marker; keep prompt strictly ≤128 tokens.

- [ ] **Step 1: Write the failing test**

```python
# tests/dsv4/test_coherence_smoke.py
"""MILESTONE GATE: real-weight coherence smoke. ONLY task that loads the full
~187 GiB model. Prompt strictly <=128 tokens (one sliding window) — dense-only
is coherent only within the window (spec 7.3)."""
import os

import pytest

WEIGHTS = "/home/enyouki/dsv4-weights"


@pytest.mark.skipif(not os.path.isdir(WEIGHTS),
                    reason="real DSV4 weights not mounted")
@pytest.mark.slow
def test_coherence_le_128_tokens():
    # MUST be launched with the full env block already exported (see below):
    #   NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm MOE_REQUANTIZE_WEIGHT_DTYPE=fp4 \
    #   MOE_REQUANTIZE_BLOCK_SIZE=512 SKIP_JAX_PRECOMPILE=1 HF_HUB_OFFLINE=1 \
    #   TOKENIZERS_PARALLELISM=false python -m pytest ... -m slow
    # (setting os.environ here is too late -- these are read at import time.)
    # Build the real serving config (TP=8 + EP + DP-attention) pointing at
    # WEIGHTS, run vLLM offline_inference on a short prompt, decode <=128 total
    # tokens, and assert the output reads sensibly.
    from tests.dsv4.coherence_run import run_real_prompt  # built in Step 3
    prompt = "The capital of France is"
    text = run_real_prompt(prompt, max_total_tokens=128)
    assert isinstance(text, str) and len(text.strip()) > 0
    # Coherence heuristic: output is ASCII-ish words, not repeated gibberish.
    words = text.split()
    assert len(words) >= 3
    assert len(set(words)) >= 2, f"degenerate/repeated output: {text!r}"
    print("COHERENCE OUTPUT:", text)
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /home/enyouki/tpu-inference && NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm python -m pytest tests/dsv4/test_coherence_smoke.py -x -s
```
Expected: `ModuleNotFoundError: No module named 'tests.dsv4.coherence_run'` (or `skipped` if weights unmounted).

- [ ] **Step 3: Write minimal implementation**

Build `tests/dsv4/coherence_run.py` with `run_real_prompt(prompt, max_total_tokens)`. Mirror `examples/offline_inference.py` (verified path; it is a thin `LLM(**args)` + `llm.generate(...)` wrapper). The required env block (verified against the working launcher; export these BEFORE the process starts):
```
NEW_MODEL_DESIGN=1            # selects 6-axis sharding + production mesh (mandatory)
MODEL_IMPL_TYPE=vllm          # torchax path (mandatory)
MOE_REQUANTIZE_WEIGHT_DTYPE=fp4
MOE_REQUANTIZE_BLOCK_SIZE=512 # note: this is the env knob; our Task-4 fix is the in-code REQUANTIZED_BLOCK_SIZE=32, which is the one that actually flips the GMM path
SKIP_JAX_PRECOMPILE=1
HF_HUB_OFFLINE=1
TOKENIZERS_PARALLELISM=false
```
The vLLM serving wiring (no `--data-parallel-size`; DP comes from `enable_dp_attention`):
```
--tensor-parallel-size 8 --enable-expert-parallel \
--additional-config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true, "expert_parallelism": 8, "tensor_parallelism": 1}}, "replicate_attn_weights": "True", "sparse_matmul": "True"}'
```
Point `model=` at `/home/enyouki/dsv4-weights`, keep `max_tokens` small so total ≤128, generate, return the decoded string. This is the one place real weights load (~12 min I/O).

- [ ] **Step 4: Run test to verify it passes** (full env block — this loads real weights)

```
cd /home/enyouki/tpu-inference && \
  NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm MOE_REQUANTIZE_WEIGHT_DTYPE=fp4 \
  MOE_REQUANTIZE_BLOCK_SIZE=512 SKIP_JAX_PRECOMPILE=1 HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  python -m pytest tests/dsv4/test_coherence_smoke.py -x -s -m slow
```
Expected: `1 passed`, and the printed `COHERENCE OUTPUT:` reads as sensible English continuing the prompt. **This is the Phase-1 exit gate** (paired with the Task-12 AOT GMM-compile proof).

- [ ] **Step 5: Commit**

```
cd /home/enyouki/tpu-inference && git add tests/dsv4/coherence_run.py tests/dsv4/test_coherence_smoke.py && git commit -m "DSV4 Phase1: coherence smoke gate (real weights, <=128 tokens) — Phase 1 exit

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase-1 exit gate (spec §5)

Phase 1 is complete when BOTH hold:
1. **AOT compile gate passes for the MoE GMM** (Task 13) — the FP4 fix compiles on v6e via the dequant-in-VMEM path.
2. **Coherent output on a real-weight prompt of ≤128 tokens** (Task 15).

All component/invariant tests (Tasks 1-14) run on the synthetic mini-config on the 8-chip production mesh; only Task 15 loads real weights.
