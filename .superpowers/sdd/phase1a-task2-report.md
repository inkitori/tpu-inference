# Phase 1a Task 2 — RoPE bit-for-bit parity report

## Status: DONE

## JAX RoPE function names and signatures

All three functions live in `tpu_inference/models/jax/glm_moe_dsa.py`.
Import them as:

```python
from tpu_inference.models.jax.glm_moe_dsa import (
    build_rope_cos_sin_np,
    apply_rope_interleaved_jax,
    apply_rope_rotate_half_jax,
)
```

### `build_rope_cos_sin_np(positions, rope_theta, head_dim) -> (cos, sin)`

Host-side numpy fp32 cos/sin table builder. Called OUTSIDE any JAX device mesh
(V4 lesson). Returns numpy float32 arrays of shape `[T, head_dim]`.

- `positions`: 1-D integer/float array, shape `[T]`
- `rope_theta`: float (e.g. `10000.0` for default, `8_000_000.0` for prod)
- `head_dim`: int — `config.qk_rope_head_dim = 64` for GLM 5.2 MLA;
  `config.index_head_dim = 128` for the indexer path (Phase 2)

Construction matches `GlmMoeDsaRotaryEmbedding.forward()` exactly:
```python
k = arange(0, head_dim, 2, int64→float32)
inv_freq = 1.0 / (rope_theta ** (k / head_dim))    # [d/2]
freqs = positions[:, None] * inv_freq[None, :]       # [T, d/2]
emb = cat(freqs, freqs, axis=-1)                     # [T, d]
cos, sin = cos(emb), sin(emb)                        # [T, d]
```

### `apply_rope_interleaved_jax(q, k, cos, sin) -> (q_embed, k_embed)`

MLA interleaved RoPE. Matches HF `apply_rotary_pos_emb_interleave`.

- `q`, `k`: `[B, n_heads, T, head_dim]` jnp float32
- `cos`, `sin`: `[T, head_dim]` or `[B, T, head_dim]` (broadcast handles both)

Convention:
```python
cos_half = cos[..., :d//2]              # slice to first half
q1, q2 = q[..., 0::2], q[..., 1::2]   # even/odd split
q_embed = cat([q1*cos - q2*sin, q2*cos + q1*sin], -1)
```

**Consumed by:** Task 3 (MLA attention module).

### `apply_rope_rotate_half_jax(q, k, cos, sin) -> (q_embed, k_embed)`

Indexer rotate-half RoPE. Matches HF `apply_rotary_pos_emb`.

- Same arg shapes as `apply_rope_interleaved_jax`.

Convention:
```python
# full-width cos/sin — no slicing
rotate_half(x) = cat([-x[...,d//2:], x[...,:d//2]], -1)
q_embed = q * cos + rotate_half(q) * sin
```

**Consumed by:** Task 5 (indexer RoPE, Phase 2 prep).

---

## Achieved maxabs deltas

All measured on v6e-8 box with `/home/enyouki/.venv/bin/python`:

| Test | Fixture | maxabs q | maxabs k |
|------|---------|----------|----------|
| Test 1: MLA interleaved, small pos | `[B=2,H=8,T=16,d=64]`, pos 0..15, θ=10000 | **0.00e+00** | **0.00e+00** |
| Test 2: Indexer rotate-half, small | `[B=2,H=8,T=16,d=64]`, pos 0..15, θ=10000 | **0.00e+00** | **0.00e+00** |
| Test 3: MLA interleaved, near-1M   | `[B=1,H=8,T=8,d=64]`, pos {1048575,…}, θ=8M | **0.00e+00** | **0.00e+00** |

cos/sin table maxabs (sub-assertion, all cases): **0.00e+00**

The delta is exactly 0 (bit-identical) because both sides use the same numpy
fp32 `build_rope_cos_sin_np` table, and the apply is identical arithmetic.

---

## How cos/sin was built to match HF

HF `GlmMoeDsaRotaryEmbedding.forward()` path (for `rope_type="default"`,
`attention_scaling=1.0`):

1. `config.head_dim = config.qk_rope_head_dim = 64` (set in `__post_init__`)
2. `inv_freq = 1.0 / (theta ** (arange(0,64,2,int64→fp32) / 64))`  → 32 elements
3. `freqs = inv_freq_expanded @ position_ids_expanded, transposed` → `[B,T,32]`
4. `emb = cat(freqs, freqs, -1)` → `[B,T,64]`
5. `cos/sin = emb.cos/sin() * 1.0`  (attention_scaling=1 for default rope)

The JAX `build_rope_cos_sin_np` replicates steps 1-5 on the HOST with numpy
fp32, OUTSIDE any device mesh. Because `positions * inv_freq` is IEEE fp32
multiply (correctly rounded), and numpy/torch/jnp all agree on `cos`/`sin` at
these magnitudes (empirically verified by orchestrator to ~6e-8), the table
is bit-identical to HF. The only possible divergence source — `theta**(-2k/d)`
(pow) — is eliminated by computing `inv_freq` ONCE in numpy and reusing it
on both sides.

---

## Verbatim pytest result

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1
rootdir: /home/enyouki/tpu-inference

tests/models/jax/test_glm_moe_dsa.py::test_rope_mla_interleaved_parity_small PASSED
tests/models/jax/test_glm_moe_dsa.py::test_rope_indexer_rotate_half_parity_small PASSED
tests/models/jax/test_glm_moe_dsa.py::test_rope_mla_interleaved_parity_near_1m PASSED

======================= 3 passed in 17.80s ========================
```

Full suite (31 tests, no regressions):
```
======================= 31 passed, 4 warnings in 44.33s ========================
```

---

## Files changed

- **Created:** `tpu_inference/models/jax/glm_moe_dsa.py`
  — `build_rope_cos_sin_np`, `apply_rope_interleaved_jax`,
    `apply_rope_rotate_half_jax`, `_rotate_half_jax`
- **Modified:** `tests/models/jax/test_glm_moe_dsa.py`
  — Added 3 RoPE parity tests + 3 helper functions (`_hf_cos_sin`,
    `_hf_oracle_rope_interleave`, `_hf_oracle_rope_rotate_half`)

---

## Concerns

None. The implementation is bit-identical (maxabs=0) to the HF oracle on both
conventions at both small and near-1M positions. The 1e-6 tolerance is met
with headroom to spare.

---

## Phase 1a Task 2 — Strengthening: Real Module Table Gate (follow-up)

### How the real rotary module was instantiated

`_real_module_cos_sin(positions, rope_theta, head_dim)` in
`tests/models/jax/test_glm_moe_dsa.py`:

1. Build `tiny_glm_moe_dsa_config()` (already sets `head_dim=qk_rope_head_dim=64`).
2. Patch `cfg.rope_parameters["rope_theta"] = float(rope_theta)` — the mixin
   populates `rope_parameters` in `__post_init__`, so updating the dict key
   is safe and the module picks it up immediately.
3. Instantiate `GlmMoeDsaRotaryEmbedding(cfg)` — triggers
   `compute_default_rope_parameters` which reads `rope_parameters["rope_theta"]`.
4. Call `rotary.forward(x, pos_ids)` where `x = torch.ones(1, T, 1, float32)`
   (dummy; only dtype/device matter) and `pos_ids = tensor(positions[None,:], int64)`.
5. Drop batch dim from returned `cos[0], sin[0]` → numpy `[T, head_dim]`.

### Achieved table maxabs vs REAL module

| Test | Fixture | table maxabs cos | table maxabs sin |
|------|---------|-----------------|-----------------|
| Test 1 (small, θ=10000) | pos 0..15, d=64 | **5.96e-08** | **5.96e-08** |
| Test 2 (small, θ=10000) | pos 0..15, d=64 | **5.96e-08** | **5.96e-08** |
| Test 3 (near-1M, θ=8M)  | pos {1048575,…}, d=64 | **5.96e-08** | **5.96e-08** |

All well within 1e-6 (1-bit headroom — IEEE fp32 cos/sin roundtrip error floor).

### Real construction bug found and fixed in `build_rope_cos_sin_np`

**YES — a real bug was found and fixed.**

**Root cause:** The original `build_rope_cos_sin_np` computed:
```python
k = np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32)
inv_freq = (1.0 / (rope_theta ** (k / head_dim))).astype(np.float32)
```
Here `theta ** float32_array` invokes numpy's scalar-pow with a float32 exponent,
which uses a different fp32 pow kernel than torch's `scalar ** float32_tensor`.
For `theta=8_000_000`, this produces a 1-ULP difference in `inv_freq[2]`
(`0x3ebd9850` numpy vs `0x3ebd9851` torch). At position 1048575, multiplying
by inv_freq[2] amplifies that 1-ULP error to **~3e-2** in angle — causing
`maxabs(real_module_cos, build_cos) ≈ 3.1e-2`, which would have made the model
produce wrong RoPE values vs the checkpoint.

The existing tests missed this because they compared `build_rope_cos_sin_np` only
against `_hf_cos_sin` (another numpy helper using the same flawed path). Both
had the same bug; they agreed with each other while both disagreed with the real
module.

**Fix** (`tpu_inference/models/jax/glm_moe_dsa.py`): compute `inv_freq` via
torch, matching `GlmMoeDsaRotaryEmbedding.compute_default_rope_parameters` exactly:
```python
import torch as _torch
inv_freq = (1.0 / (
    rope_theta ** (
        _torch.arange(0, head_dim, 2, dtype=_torch.int64).to(dtype=_torch.float32)
        / head_dim
    )
)).numpy()
```
This eliminates the numpy scalar-pow divergence. Post-fix maxabs = 5.96e-8 for all
three tests (numerical noise floor from `cos`/`sin` computation, not inv_freq).

### Side effect on apply assertions (Tests 1 & 2)

With the fixed `build_rope_cos_sin_np`, the apply assertions that previously fed
`_hf_cos_sin` table into the HF oracle and `jax_cos` into the JAX apply started
failing (1.3e-6) because the two tables now differ slightly. Fix: both sides now
use `real_cos[None]` / `real_sin[None]` (from the real module) as input to the
HF oracle apply, while the JAX apply uses `jax_cos`/`jax_sin` (which matches
the real module to 5.96e-8). This strictly tests the *apply function* parity
with a shared, correct table.

For Test 3 (near-1M): the secondary `_hf_cos_sin` cross-check was removed since
that helper is known to have the 1-ULP bug at `rope_theta=8_000_000`.

### Verbatim pytest result (post-fix)

```
tests/models/jax/test_glm_moe_dsa.py::test_rope_mla_interleaved_parity_small PASSED
tests/models/jax/test_glm_moe_dsa.py::test_rope_indexer_rotate_half_parity_small PASSED
tests/models/jax/test_glm_moe_dsa.py::test_rope_mla_interleaved_parity_near_1m PASSED

3 passed in 32.51s
```

Full suite (no regressions):
```
31 passed, 4 warnings in 56.70s
```
