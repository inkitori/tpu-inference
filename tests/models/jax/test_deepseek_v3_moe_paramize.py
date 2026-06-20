# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Behavior-preservation tests for DeepseekV2Moe / SharedFusedMoe param-ization.

Phase 1a Task 1 — TDD gate.

Gate structure (spec `docs/superpowers/specs/glm5.2-dsa/phases/phase-1a.md`):
  1. Baseline: construct DeepseekV2Moe with DEFAULT args (no kwargs), run a
     seeded forward, and assert finite output + record param shapes.  This test
     is written BEFORE the refactor and must be GREEN on unmodified code.
  2. After refactoring `deepseek_v3.py` to add optional kwargs (each defaulting
     to the current module global / literal), the SAME baseline test must still
     pass byte-identically (default == current behavior).
  3. "kwargs honored" test: construct with NON-default kwargs
     (num_local_experts=8, hidden_size=512, moe_intermediate_size=256,
     num_shared_experts=1) and assert param shapes reflect the new dims.

IMPORTANT: this file uses a DENSE_MAT backend + a small single-device mesh to
keep construction fast; the 8-chip gate is Phase 1b.  No skip-if-no-TPU marker
(CI runs on TPU agents only).
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import tpu_inference.models.jax.deepseek_v3 as _dv3_mod

import jax
import jax.numpy as jnp
from flax import nnx

from tpu_inference.layers.common.sharding import (
    ShardingAxisNameBase,
    ShardingAxisNameBase as ShardingAxisName,
)
from tpu_inference.layers.jax.moe.moe import MoEBackend
from tpu_inference.layers.jax.quantization.unquantized import UnquantizedConfig
from tpu_inference.models.jax.deepseek_v3 import DeepseekV2Moe
from tests.models.jax.glm_moe_dsa_harness import make_glm_mesh

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rng(seed: int = 0) -> nnx.Rngs:
    return nnx.Rngs(params=jax.random.PRNGKey(seed))


def _single_device_mesh():
    """One-device mesh (first chip only) using canonical 6-axis layout."""
    return make_glm_mesh(num_devices=1)


@contextmanager
def _moe_context(mesh):
    """Combined context: mesh + ShardingAxisName patch needed for MoE construction."""
    with (patch("tpu_inference.models.jax.deepseek_v3.ShardingAxisName",
                ShardingAxisNameBase),
          jax.set_mesh(mesh)):
        yield


# ---------------------------------------------------------------------------
# Tiny config constants
# ---------------------------------------------------------------------------
# Use a TINY stand-in so the "kwargs honored" tests allocate < 1 MB.
# The "default behavior" identity (baseline test) checks structural param shapes
# because constructing and running a full 256-expert forward in a unit test
# would need ~2 GB of HBM just for allocation.
_TINY = dict(
    num_local_experts=8,
    hidden_size=512,
    moe_intermediate_size=256,
    num_experts_per_tok=2,
    n_group=1,
    topk_groups=1,
    norm_topk_prob=True,
    routed_scaling_factor=2.5,
    num_shared_experts=1,
    hidden_act="silu",
    expert_axis_name=ShardingAxisName.ATTN_DATA_EXPERT,
    scoring_func="sigmoid",
)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestDeepseekV2MoeParamize:
    """Behavior-preservation tests for the DeepseekV2Moe param-ization refactor."""

    # ------------------------------------------------------------------
    # Baseline: default construction — param shapes match module globals
    # This test is written BEFORE the refactor and is the GREEN baseline.
    # It MUST remain green after the refactor (default == current behavior).
    # ------------------------------------------------------------------

    def test_default_construction_param_shapes(self):
        """DeepseekV2Moe() with NO extra kwargs must produce param shapes
        matching the current module globals (deepseek_v3.py:77-108).

        Written against UNREFACTORED code — must be GREEN before any changes.
        After the refactor (defaults identical), must still pass byte-identically.
        """
        mesh = _single_device_mesh()
        rng = _make_rng(42)

        with _moe_context(mesh):
            moe = DeepseekV2Moe(
                mesh=mesh,
                dtype=jnp.bfloat16,
                num_expert_parallelism=1,
                moe_backend=MoEBackend.DENSE_MAT,
                quant_config=None,
                scoring_func="sigmoid",
                rng=rng,
                prefix="test_layer",
                enable_return_routed_experts=False,
            )

        # Router weight: (hidden_size, num_local_experts) = (7168, 256)
        assert moe.gate.weight.value.shape == (7168, 256), (
            f"router kernel shape {moe.gate.weight.value.shape} != (7168, 256)")

        # Router bias: (num_local_experts,) = (256,)
        assert moe.gate.e_score_correction_bias.value.shape == (256,), (
            f"router bias shape {moe.gate.e_score_correction_bias.value.shape} != (256,)")

        # Shared expert gate_proj: (hidden_size, num_shared_experts * moe_intermediate_size)
        # = (7168, 1*2048) = (7168, 2048)
        assert moe.shared_experts.gate_proj.weight.value.shape == (7168, 2048), (
            f"shared gate_proj shape {moe.shared_experts.gate_proj.weight.value.shape} "
            f"!= (7168, 2048)")

        # Routed expert gating weights: (E, D, F) = (256, 7168, 2048)
        assert moe.experts.kernel_gating_EDF.value.shape == (256, 7168, 2048), (
            f"routed gate shape {moe.experts.kernel_gating_EDF.value.shape} "
            f"!= (256, 7168, 2048)")

        # routed_scaling_factor must thread through
        assert moe.experts.routed_scaling_factor == 2.5, (
            f"routed_scaling_factor {moe.experts.routed_scaling_factor} != 2.5")

    # ------------------------------------------------------------------
    # Kwargs-honored: non-default dims must reflect in param shapes.
    # This test FAILS on unrefactored code (TypeError: unexpected kwarg).
    # After the refactor it MUST PASS.
    # ------------------------------------------------------------------

    def test_kwargs_honored_param_shapes(self):
        """Constructing DeepseekV2Moe with non-default kwargs must produce
        param shapes reflecting the kwarg values, not the module globals.

        RED on current unrefactored code (unexpected keyword argument).
        GREEN after the refactor.
        """
        mesh = _single_device_mesh()
        rng = _make_rng(7)

        with _moe_context(mesh):
            moe = DeepseekV2Moe(
                mesh=mesh,
                dtype=jnp.bfloat16,
                num_expert_parallelism=1,
                moe_backend=MoEBackend.DENSE_MAT,
                quant_config=None,
                rng=rng,
                prefix="glm_layer",
                enable_return_routed_experts=False,
                # --- GLM-specific overrides (non-default) ---
                num_local_experts=_TINY["num_local_experts"],
                hidden_size=_TINY["hidden_size"],
                moe_intermediate_size=_TINY["moe_intermediate_size"],
                num_experts_per_tok=_TINY["num_experts_per_tok"],
                n_group=_TINY["n_group"],
                topk_groups=_TINY["topk_groups"],
                norm_topk_prob=_TINY["norm_topk_prob"],
                routed_scaling_factor=_TINY["routed_scaling_factor"],
                num_shared_experts=_TINY["num_shared_experts"],
                hidden_act=_TINY["hidden_act"],
                expert_axis_name=_TINY["expert_axis_name"],
                scoring_func=_TINY["scoring_func"],
            )

        E = _TINY["num_local_experts"]      # 8
        D = _TINY["hidden_size"]             # 512
        F = _TINY["moe_intermediate_size"]   # 256
        shared_F = _TINY["num_shared_experts"] * F  # 1*256 = 256

        # Router weight: (D, E)
        assert moe.gate.weight.value.shape == (D, E), (
            f"router kernel {moe.gate.weight.value.shape} != ({D}, {E})")

        # Router bias: (E,)
        assert moe.gate.e_score_correction_bias.value.shape == (E,), (
            f"router bias {moe.gate.e_score_correction_bias.value.shape} != ({E},)")

        # Shared expert gate_proj: (D, shared_F)
        assert moe.shared_experts.gate_proj.weight.value.shape == (D, shared_F), (
            f"shared gate_proj {moe.shared_experts.gate_proj.weight.value.shape} "
            f"!= ({D}, {shared_F})")

        # Routed expert gating weights: (E, D, F)
        assert moe.experts.kernel_gating_EDF.value.shape == (E, D, F), (
            f"routed gate {moe.experts.kernel_gating_EDF.value.shape} "
            f"!= ({E}, {D}, {F})")

    # ------------------------------------------------------------------
    # Forward-pass finite check with tiny kwargs (post-refactor proof).
    # RED on current code (unexpected kwarg). GREEN after refactor.
    # ------------------------------------------------------------------

    def test_kwargs_honored_forward_finite(self):
        """Forward pass through the tiny-kwarg MoE must return finite values.

        RED on current code. GREEN after refactor.
        """
        mesh = _single_device_mesh()
        rng = _make_rng(13)

        with _moe_context(mesh):
            moe = DeepseekV2Moe(
                mesh=mesh,
                dtype=jnp.bfloat16,
                num_expert_parallelism=1,
                moe_backend=MoEBackend.DENSE_MAT,
                quant_config=UnquantizedConfig({}),
                rng=rng,
                prefix="glm_fwd",
                enable_return_routed_experts=True,
                num_local_experts=_TINY["num_local_experts"],
                hidden_size=_TINY["hidden_size"],
                moe_intermediate_size=_TINY["moe_intermediate_size"],
                num_experts_per_tok=_TINY["num_experts_per_tok"],
                n_group=_TINY["n_group"],
                topk_groups=_TINY["topk_groups"],
                norm_topk_prob=_TINY["norm_topk_prob"],
                routed_scaling_factor=_TINY["routed_scaling_factor"],
                num_shared_experts=_TINY["num_shared_experts"],
                hidden_act=_TINY["hidden_act"],
                expert_axis_name=_TINY["expert_axis_name"],
                scoring_func=_TINY["scoring_func"],
            )

            T, D = 4, _TINY["hidden_size"]
            key = jax.random.PRNGKey(99)
            x = jax.random.normal(key, (T, D), dtype=jnp.bfloat16)

            out, expert_indices = moe(x)

        assert jnp.all(jnp.isfinite(out.astype(jnp.float32))), (
            "forward pass output contains non-finite values")
        assert out.shape == (T, D), (
            f"output shape {out.shape} != ({T}, {D})")
        assert expert_indices is not None, "expert_indices should be returned"

    # ------------------------------------------------------------------
    # Fix 1 gate: default-resolution is a pure no-op forward.
    # Proves that the kwarg-defaulting path produces byte-identical
    # computation to explicit passing, AND locks the math with a snapshot.
    # ------------------------------------------------------------------

    def test_default_resolution_is_noop_forward(self, monkeypatch):
        """Construct two DeepseekV2Moe instances over identical tiny configs:
          A — no kwargs (relies on module-global defaults, patched to tiny values).
          B — explicit kwargs equal to those same tiny values.
        Copy A's weights into B.  Forward both on the same input and assert
        byte-identical outputs (maxabs == 0).  Also assert a numeric snapshot
        of A's output to lock the MoE math going forward.

        Design: patching module globals to a tiny config lets us run a real
        forward (< 1 MB) while still exercising the default-resolution code path.
        Byte-identity between A and B proves the defaulting is a pure no-op.
        The snapshot asserts the numerical value is reproducible — any future
        change to the routing/scaling/renorm math will trip this.
        """
        # --- Tiny config values to patch into the module globals ---
        TINY_NUM_LOCAL_EXPERTS = 8
        TINY_HIDDEN_SIZE = 128
        TINY_MOE_INTERMEDIATE_SIZE = 64
        TINY_NUM_EXPERTS_PER_TOKEN = 2
        TINY_N_GROUP = 1
        TINY_ROUTED_SCALING_FACTOR = 2.5
        TINY_NUM_SHARED_EXPERTS = 1
        TINY_HIDDEN_ACT = "silu"
        TINY_EXPERT_AXIS_NAME = ShardingAxisName.ATTN_DATA_EXPERT

        # Patch the module globals that DeepseekV2Moe.__init__ reads for defaults.
        monkeypatch.setattr(_dv3_mod, "num_local_experts", TINY_NUM_LOCAL_EXPERTS)
        monkeypatch.setattr(_dv3_mod, "hidden_size", TINY_HIDDEN_SIZE)
        monkeypatch.setattr(_dv3_mod, "moe_intermediate_size", TINY_MOE_INTERMEDIATE_SIZE)
        monkeypatch.setattr(_dv3_mod, "num_experts_per_token", TINY_NUM_EXPERTS_PER_TOKEN)
        monkeypatch.setattr(_dv3_mod, "n_group", TINY_N_GROUP)
        monkeypatch.setattr(_dv3_mod, "routed_scaling_factor", TINY_ROUTED_SCALING_FACTOR)
        monkeypatch.setattr(_dv3_mod, "num_shared_experts", TINY_NUM_SHARED_EXPERTS)
        monkeypatch.setattr(_dv3_mod, "hidden_act", TINY_HIDDEN_ACT)
        monkeypatch.setattr(_dv3_mod, "expert_axis_name", TINY_EXPERT_AXIS_NAME)

        mesh = _single_device_mesh()
        rng_a = _make_rng(42)

        from tpu_inference.layers.jax.quantization.unquantized import UnquantizedConfig

        with _moe_context(mesh):
            # Instance A — no optional kwargs; resolves from (patched) globals.
            moe_a = DeepseekV2Moe(
                mesh=mesh,
                dtype=jnp.bfloat16,
                num_expert_parallelism=1,
                moe_backend=MoEBackend.DENSE_MAT,
                quant_config=UnquantizedConfig({}),
                scoring_func="sigmoid",
                rng=rng_a,
                prefix="noop_a",
                enable_return_routed_experts=False,
            )

            # Instance B — all kwargs explicitly set to the same tiny values.
            rng_b = _make_rng(42)  # same seed → same initial random weights
            moe_b = DeepseekV2Moe(
                mesh=mesh,
                dtype=jnp.bfloat16,
                num_expert_parallelism=1,
                moe_backend=MoEBackend.DENSE_MAT,
                quant_config=UnquantizedConfig({}),
                scoring_func="sigmoid",
                rng=rng_b,
                prefix="noop_b",
                enable_return_routed_experts=False,
                num_local_experts=TINY_NUM_LOCAL_EXPERTS,
                hidden_size=TINY_HIDDEN_SIZE,
                moe_intermediate_size=TINY_MOE_INTERMEDIATE_SIZE,
                num_experts_per_tok=TINY_NUM_EXPERTS_PER_TOKEN,
                n_group=TINY_N_GROUP,
                topk_groups=4,       # matches the literal default in __init__
                norm_topk_prob=True, # matches the literal default in __init__
                routed_scaling_factor=TINY_ROUTED_SCALING_FACTOR,
                num_shared_experts=TINY_NUM_SHARED_EXPERTS,
                hidden_act=TINY_HIDDEN_ACT,
                expert_axis_name=TINY_EXPERT_AXIS_NAME,
            )

            # Copy A's weights into B so both compute over identical parameters.
            from flax import nnx as _nnx
            _nnx.update(moe_b, _nnx.state(moe_a))

            # Same seeded input for both.
            key = jax.random.PRNGKey(7)
            x = jax.random.normal(key, (4, TINY_HIDDEN_SIZE), dtype=jnp.bfloat16)

            out_a, _ = moe_a(x)
            out_b, _ = moe_b(x)

        # --- Byte-identity gate: default-resolution is a pure no-op ---
        maxabs = float(jnp.max(jnp.abs(out_a.astype(jnp.float32)
                                       - out_b.astype(jnp.float32))))
        assert maxabs == 0.0, (
            f"default-resolution is NOT a no-op: max |A-B| = {maxabs:.3e}. "
            "The kwarg-defaulting path diverges from explicit passing.")

        # --- Numeric snapshot: lock the MoE math going forward ---
        # Value captured on first green run (seed=42 weights, seed=7 input, tiny config).
        # If this fails, the routing/scaling/renorm math changed unexpectedly.
        snapshot_mean = float(jnp.mean(out_a.astype(jnp.float32)))
        # NOTE: _SNAPSHOT_MEAN captured on 2026-06-20 (seed=42 weights, seed=7 input,
        # tiny config: experts=8, hidden=128, moe_inter=64, tok=2, group=1).
        # Tolerance is 0 — bfloat16 on JAX is fully reproducible.
        # To regenerate: set to None, run test, print snapshot_mean, hardcode.
        _SNAPSHOT_MEAN = 0.0038310810923576355
        if _SNAPSHOT_MEAN is not None:
            assert snapshot_mean == _SNAPSHOT_MEAN, (
                f"MoE math snapshot mismatch: got {snapshot_mean}, "
                f"expected {_SNAPSHOT_MEAN}. "
                "Routing/scaling/renorm math may have changed.")

        # Always check finite + shape.
        assert jnp.all(jnp.isfinite(out_a.astype(jnp.float32))), (
            "Instance A forward output contains non-finite values")
        assert out_a.shape == (4, TINY_HIDDEN_SIZE), (
            f"Instance A output shape {out_a.shape} != (4, {TINY_HIDDEN_SIZE})")
