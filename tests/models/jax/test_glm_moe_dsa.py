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
"""Phase 0 gates for GLM 5.2 (GlmMoeDsa) bring-up.

Covers the spec §6 Phase 0 deliverable / §14 acceptance: TPU JAX env, the
HF-eager oracle on the tiny config, and the harness helpers (converter,
maxabs, identical-weights checksum, mesh fixtures). Single-device tests are
auto-collected on the single-chip Buildkite queue; the 8-chip gate skips off
the v6e-8.
"""
import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tests.models.jax.glm_moe_dsa_harness import (
    TINY_SEQ_DENSE, TINY_SEQ_SPARSE, assert_identical_weights, build_hf_oracle,
    make_glm_mesh, maxabs, tiny_glm_moe_dsa_config, t2j_weights,
    weight_checksum)


# --- mesh fixtures (spec §6 Phase 0: 1-device + multi-device, no single- -----
# --- device assert inherited from conftest.py's `mesh`) ----------------------
@pytest.fixture
def mesh_1d():
    m = make_glm_mesh(1)
    with jax.set_mesh(m):
        yield m


@pytest.fixture
def mesh_nd():
    n = len(jax.local_devices())
    if n < 2:
        pytest.skip(f"multi-device mesh fixture needs >=2 chips, saw {n}")
    m = make_glm_mesh(n)
    with jax.set_mesh(m):
        yield m


# --- maxabs ------------------------------------------------------------------
def test_maxabs_basic():
    r = maxabs(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.5]))
    assert isinstance(r, float)
    assert r == pytest.approx(0.5)


def test_maxabs_upcasts_bf16():
    import ml_dtypes
    a = np.array([2.0, -3.0], dtype=ml_dtypes.bfloat16)
    b = np.array([2.0, -3.5], dtype=ml_dtypes.bfloat16)
    # 2.0, 3.0, 3.5 are bf16-exact; the diff must be computed in fp32.
    assert maxabs(a, b) == pytest.approx(0.5)


def test_maxabs_cross_framework_torch_vs_jax():
    import torch
    # The triangulation compares torch HF output against jax model output.
    assert maxabs(torch.tensor([1.0, 2.0]),
                  jnp.array([1.0, 2.25])) == pytest.approx(0.25)


# --- t2j_weights converter ---------------------------------------------------
def test_t2j_weights_transposes_linear():
    import torch
    name = "model.layers.0.self_attn.q_b_proj.weight"
    w = torch.arange(8.0).reshape(2, 4)  # HF nn.Linear: (out=2, in=4)
    out = t2j_weights({name: w})
    assert out[name].shape == (4, 2)  # JAX einsum wants (in, out)
    np.testing.assert_array_equal(np.asarray(out[name]),
                                  np.asarray(w).T)


def test_t2j_weights_embed_not_transposed():
    import torch
    name = "model.embed_tokens.weight"
    out = t2j_weights({name: torch.arange(24.0).reshape(6, 4)})  # (vocab, hidden)
    assert out[name].shape == (6, 4)  # lookup table, not a matmul


def test_t2j_weights_lm_head_transposed():
    import torch
    name = "lm_head.weight"
    out = t2j_weights({name: torch.arange(24.0).reshape(6, 4)})  # (vocab, hidden)
    assert out[name].shape == (4, 6)


def test_t2j_weights_norm_1d_unchanged():
    import torch
    name = "model.norm.weight"
    out = t2j_weights({name: torch.arange(4.0)})
    assert out[name].shape == (4, )


def test_t2j_weights_splits_fused_gate_up_proj():
    import torch
    E, I, H = 2, 4, 3
    fused = torch.arange(float(E * 2 * I * H)).reshape(E, 2 * I, H)
    base = "model.layers.3.mlp.experts."
    out = t2j_weights({base + "gate_up_proj": fused})
    assert base + "gate_up_proj" not in out
    g, u = out[base + "gate_proj"], out[base + "up_proj"]
    assert g.shape == (E, I, H) and u.shape == (E, I, H)
    np.testing.assert_array_equal(np.asarray(g), np.asarray(fused[:, :I, :]))
    np.testing.assert_array_equal(np.asarray(u), np.asarray(fused[:, I:, :]))


# --- weight checksum / identical-weights gate --------------------------------
def test_weight_checksum_is_reorder_immune():
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([3.0, 1.0, 2.0])  # permutation
    assert weight_checksum(a) == weight_checksum(b)


def test_assert_identical_weights_passes_for_identical():
    w = {"a": jnp.array([1.0, 2.0]), "b": jnp.ones((2, 2))}
    assert_identical_weights(w, {k: v for k, v in w.items()})


def test_assert_identical_weights_fails_on_perturbation():
    with pytest.raises(AssertionError):
        assert_identical_weights({"x": jnp.array([1.0, 2.0])},
                                 {"x": jnp.array([1.0, 2.001])})


def test_assert_identical_weights_fails_on_key_mismatch():
    with pytest.raises(AssertionError):
        assert_identical_weights({"a": jnp.zeros(1)}, {"b": jnp.zeros(1)})


# --- tiny config -------------------------------------------------------------
def test_tiny_config_layer_schedules():
    cfg = tiny_glm_moe_dsa_config()
    # 3 dense + 1 MoE (first_k_dense_replace=3); indexer full*3 + shared*1.
    assert list(cfg.mlp_layer_types) == ["dense", "dense", "dense", "sparse"]
    assert list(cfg.indexer_types) == ["full", "full", "full", "shared"]
    assert cfg.first_k_dense_replace == 3


def test_tiny_config_keeps_real_attention_dims():
    cfg = tiny_glm_moe_dsa_config()
    assert cfg.kv_lora_rank == 512
    assert cfg.qk_rope_head_dim == 64
    assert cfg.qk_nope_head_dim == 192
    assert cfg.v_head_dim == 256
    assert cfg.index_head_dim == 128
    assert cfg.index_topk == 64
    assert cfg.qk_head_dim == 256  # derived: nope(192)+rope(64)


def test_tiny_config_accepts_overrides():
    assert tiny_glm_moe_dsa_config(num_hidden_layers=2).num_hidden_layers == 2


# --- HF-eager oracle ---------------------------------------------------------
def test_oracle_resolves_eager_experts():
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import \
        GlmMoeDsaNaiveMoe
    model = build_hf_oracle()
    assert model.config._experts_implementation_internal == "eager"
    # layer 3 is the lone sparse/MoE layer in the tiny config
    assert isinstance(model.model.layers[3].mlp.experts, GlmMoeDsaNaiveMoe)


def test_oracle_randomizes_selection_bias():
    # e_score_correction_bias is zero-initialized by HF; left unrandomized the
    # bias-for-selection parity delta (Phase 1) passes vacuously.
    model = build_hf_oracle(randomize_buffers=True)
    biases = [
        b for n, b in model.named_buffers()
        if n.endswith("e_score_correction_bias")
    ]
    assert len(biases) >= 1
    assert all(float(b.norm()) > 0.0 for b in biases)


def test_oracle_default_bias_is_zero_without_randomization():
    model = build_hf_oracle(randomize_buffers=False)
    biases = [
        b for n, b in model.named_buffers()
        if n.endswith("e_score_correction_bias")
    ]
    assert len(biases) >= 1
    assert all(float(b.norm()) == 0.0 for b in biases)


def test_oracle_is_deterministic_for_fixed_seed():
    import torch
    m1, m2 = build_hf_oracle(seed=0), build_hf_oracle(seed=0)
    for (n1, p1), (_, p2) in zip(m1.named_parameters(),
                                 m2.named_parameters()):
        assert torch.equal(p1, p2), f"param {n1} differs across builds"
    for (_, b1), (_, b2) in zip(m1.named_buffers(), m2.named_buffers()):
        assert torch.equal(b1, b2)


@pytest.mark.parametrize("seq", [TINY_SEQ_DENSE, TINY_SEQ_SPARSE])
def test_oracle_forward_is_finite(seq):
    import torch
    model = build_hf_oracle()
    cfg = model.config
    torch.manual_seed(123)
    ids = torch.randint(0, cfg.vocab_size, (1, seq))
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
    assert tuple(logits.shape) == (1, seq, cfg.vocab_size)
    assert torch.isfinite(logits).all()


# --- env / mesh round-trip gates (spec §14 Phase 0 acceptance) ---------------
def test_transformers_oracle_pin():
    import transformers
    assert transformers.__version__ == "5.12.1"


def test_jax_tpu_host_roundtrip():
    x = jnp.arange(4, dtype=jnp.float32) + 1.0
    host = np.asarray(x)
    assert host.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_eight_chips_visible():
    n = len(jax.devices())
    if n < 8:
        pytest.skip(f"v6e-8 gate: only {n} chip(s) visible in this process")
    assert n == 8


def test_mesh_1d_roundtrip(mesh_1d):
    assert mesh_1d.size == 1
    ref = np.arange(4.0).reshape(1, 4)
    x = jax.device_put(ref, NamedSharding(mesh_1d, P("model")))
    y = np.asarray(x * 2.0)
    np.testing.assert_array_equal(y, ref * 2.0)


def test_mesh_nd_sharded_roundtrip(mesh_nd):
    n = mesh_nd.size
    assert n == len(jax.local_devices())
    ref = np.arange(float(n * 4)).reshape(n, 4)
    # sharded across the `model` axis, compute, then gather to host
    x = jax.device_put(ref, NamedSharding(mesh_nd, P("model")))
    y = np.asarray(x * 2.0)
    np.testing.assert_array_equal(y, ref * 2.0)
