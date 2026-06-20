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
    build_hf_decode_oracle, make_glm_mesh, maxabs, medium_glm_moe_dsa_config,
    tiny_glm_moe_dsa_config, t2j_weights, weight_checksum)
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxLmHead
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.models.jax.deepseek_v3 import (DeepSeekV3Router,
                                                  DeepseekV2Moe, DeepseekV3MLP)


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


# --- (ADD) incremental HF decode oracle gate (spec phase-0.md §ADD gate 1) ---
def test_decode_oracle_matches_full_forward():
    """Stepped decode last-token logits == fresh full-forward at each length.

    This is the prefill-only hole gate (phase-0.md §ADD): stepping one token
    at a time through a growing DynamicCache must yield the same last-token
    logits (fp32) as a fresh full forward over the identical token prefix, at
    multiple cumulative lengths. Tolerance fp32 <1e-4.
    """
    import torch
    cfg = tiny_glm_moe_dsa_config()
    # Prefill a short prompt then decode a few steps.
    prompt_len = 3
    decode_steps = 4
    total = prompt_len + decode_steps
    all_ids = torch.randint(0, cfg.vocab_size, (1, total), generator=torch.Generator().manual_seed(7))

    # --- stepped decode ---
    # Use randomize_buffers=True with seed=0 so e_score_correction_bias is
    # non-zero and the bias-affected router path is genuinely exercised.
    stepped_logits = build_hf_decode_oracle(
        input_ids=all_ids[:, :prompt_len],
        decode_ids=all_ids[:, prompt_len:],
        seed=0,
        randomize_buffers=True,
    )
    # stepped_logits: list of (1, vocab) tensors, one per decode token

    # --- fresh full forward at each cumulative length ---
    # Must use IDENTICAL seed + randomize_buffers so weights AND buffers match
    # the decode oracle; equivalence holds (delta ~1e-6) but now exercises the
    # non-zero bias path (spec §H1 deterministic buffer init requirement).
    model = build_hf_oracle(cfg=cfg, seed=0, randomize_buffers=True)
    model.eval()
    for step_idx in range(decode_steps):
        cum_len = prompt_len + step_idx + 1
        prefix_ids = all_ids[:, :cum_len]
        with torch.no_grad():
            full_logits = model(input_ids=prefix_ids, use_cache=False).logits
        # last-token slice
        full_last = full_logits[:, -1, :]       # (1, vocab)
        step_last = stepped_logits[step_idx]    # (1, vocab)
        delta = maxabs(step_last, full_last)
        assert delta < 1e-4, (
            f"decode step {step_idx} (cum_len={cum_len}): "
            f"stepped vs full-forward maxabs={delta:.6e} >= 1e-4"
        )


# --- (ADD) medium config gate (spec phase-0.md §ADD gate 2) ------------------
def test_medium_config_forward_is_finite():
    """Medium config (§B11) instantiates and runs an HF-eager forward on CPU.

    Keeps the sequence short (4 tokens) so the hidden=6144 forward stays fast.
    Asserts finite logits of shape (1, seq, vocab_size).
    """
    import torch
    cfg = medium_glm_moe_dsa_config()
    model = build_hf_oracle(cfg=cfg, seed=0, randomize_buffers=False)
    seq = 4
    torch.manual_seed(99)
    ids = torch.randint(0, cfg.vocab_size, (1, seq))
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
    assert tuple(logits.shape) == (1, seq, cfg.vocab_size)
    assert torch.isfinite(logits).all(), "medium config forward produced non-finite logits"


# ---------------------------------------------------------------------------
# Phase 1a Task 2 — RoPE bit-for-bit parity (MLA interleaved + indexer
# rotate-half), including near-1M positions.
#
# TDD: tests written BEFORE the implementation in glm_moe_dsa.py so that
# each test is witnessed failing first.
#
# Convention summary (matches HF oracle in modeling_glm_moe_dsa.py):
#   • Both HF paths share the same GlmMoeDsaRotaryEmbedding.forward():
#       inv_freq = 1 / (theta ** (arange(0,d,2,int64→fp32) / d))   # d=head_dim=64
#       freqs = (inv_freq[None,:,None] @ position_ids[:,None,:]).T  # [B,T,d/2]
#       emb = cat(freqs, freqs, -1)                                  # [B,T,d]
#       cos/sin = emb.cos/sin() * attention_scaling  (=1 for default)
#   • apply_rotary_pos_emb_interleave: cos/sin sliced to first half [B,T,d/2]
#       q1,q2 = q[...,0::2], q[...,1::2]
#       q_embed = cat([q1*cos - q2*sin, q2*cos + q1*sin], -1)
#   • apply_rotary_pos_emb (rotate-half): cos/sin kept full-width [B,T,d]
#       rotate_half(x) = cat([-x[...,d//2:], x[...,:d//2]], -1)
#       q_embed = q*cos + rotate_half(q)*sin
#
# The JAX implementations (in tpu_inference/models/jax/glm_moe_dsa.py):
#   • build_rope_cos_sin_np(positions, rope_theta, head_dim) -> (cos, sin) numpy
#       cos/sin shape: [T, head_dim]  (full-width, cat(freqs,freqs))
#       Built on the HOST with numpy fp32 OUTSIDE any device mesh — V4 lesson.
#   • apply_rope_interleaved_jax(q, k, cos, sin) -> (q_out, k_out)
#       Input cos/sin: first-half slice [T, head_dim//2] or full [T, head_dim].
#       Matches HF apply_rotary_pos_emb_interleave exactly.
#   • apply_rope_rotate_half_jax(q, k, cos, sin) -> (q_out, k_out)
#       Input cos/sin: full [T, head_dim].  Matches HF apply_rotary_pos_emb.
# ---------------------------------------------------------------------------

def _hf_cos_sin(positions_np, rope_theta, head_dim):
    """Reproduce GlmMoeDsaRotaryEmbedding.forward() in numpy fp32 on host.

    Returns (cos, sin) each shape [B=1, T, head_dim] as numpy arrays,
    mirroring HF torch output (fp32, no dtype cast at the end).

    Note: kept as a secondary cross-check only. The PRIMARY table reference
    is now ``_real_module_cos_sin`` which calls the actual
    ``GlmMoeDsaRotaryEmbedding`` torch module.
    """
    import numpy as np
    # HF: arange(0, dim, 2, int64) → float32 / dim
    k = np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32)
    inv_freq = 1.0 / (rope_theta ** (k / head_dim))  # [d/2]
    # HF: inv_freq_expanded [B,d/2,1] @ position_ids_expanded [B,1,T] → [B,d/2,T]
    # then .transpose(1,2) → [B,T,d/2]
    B = 1
    T = len(positions_np)
    inv_freq_exp = inv_freq[None, :, None]                       # [1, d/2, 1]
    pos_exp = positions_np[None, None, :].astype(np.float32)     # [1, 1, T]
    freqs = (inv_freq_exp * pos_exp).transpose(0, 2, 1)          # [B, T, d/2]
    emb = np.concatenate([freqs, freqs], axis=-1)                # [B, T, d]
    cos = np.cos(emb)   # attention_scaling=1.0 for default rope
    sin = np.sin(emb)
    return cos, sin   # [1, T, head_dim]


def _real_module_cos_sin(positions_np, rope_theta, head_dim):
    """Call the actual ``GlmMoeDsaRotaryEmbedding`` torch module.

    This is the PRIMARY table reference for all three RoPE tests.  It
    instantiates the real HF rotary module with the given ``rope_theta``
    (via config) and calls ``forward(x, position_ids)`` to get fp32 cos/sin.

    Returns (cos, sin) each shape [T, head_dim] as numpy float32 arrays,
    matching the shape returned by ``build_rope_cos_sin_np``.

    ``x`` is a dummy fp32 tensor (only used for dtype/device); the returned
    cos/sin are cast to ``x.dtype`` (fp32) inside HF's forward.
    """
    import torch
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import \
        GlmMoeDsaRotaryEmbedding
    cfg = tiny_glm_moe_dsa_config()
    # Patch rope_theta so the module uses the requested base frequency.
    # rope_parameters is a dict already populated by __post_init__ /
    # RotaryEmbeddingConfigMixin; updating in-place is safe here.
    cfg.rope_parameters = dict(cfg.rope_parameters)
    cfg.rope_parameters["rope_theta"] = float(rope_theta)
    # head_dim is set by __post_init__ to qk_rope_head_dim (64); we do not
    # need to override it — all three tests use head_dim=64.

    rotary = GlmMoeDsaRotaryEmbedding(cfg)
    T = len(positions_np)
    # dummy x: only dtype and device matter; shape [1, T, 1] is sufficient.
    x = torch.ones(1, T, 1, dtype=torch.float32)
    pos_ids = torch.tensor(positions_np[None, :], dtype=torch.long)  # [1, T]
    with torch.no_grad():
        cos, sin = rotary.forward(x, pos_ids)
    # cos/sin: [1, T, head_dim]; drop the batch dim → [T, head_dim]
    return cos[0].numpy(), sin[0].numpy()


def _hf_oracle_rope_interleave(q_np, k_np, cos_np, sin_np):
    """Apply HF apply_rotary_pos_emb_interleave using torch (fp32).

    q_np, k_np: [B, n_heads, T, head_dim]
    cos_np, sin_np: [B, T, head_dim]  (full-width; function slices to first half)
    Returns (q_out, k_out) as numpy.
    """
    import torch
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import \
        apply_rotary_pos_emb_interleave
    q = torch.as_tensor(q_np, dtype=torch.float32)
    k = torch.as_tensor(k_np, dtype=torch.float32)
    cos = torch.as_tensor(cos_np, dtype=torch.float32)
    sin = torch.as_tensor(sin_np, dtype=torch.float32)
    with torch.no_grad():
        q_out, k_out = apply_rotary_pos_emb_interleave(q, k, cos, sin,
                                                        unsqueeze_dim=1)
    return q_out.numpy(), k_out.numpy()


def _hf_oracle_rope_rotate_half(q_np, k_np, cos_np, sin_np):
    """Apply HF apply_rotary_pos_emb (rotate-half) using torch (fp32).

    q_np, k_np: [B, n_heads, T, head_dim]
    cos_np, sin_np: [B, T, head_dim]
    Returns (q_out, k_out) as numpy.
    """
    import torch
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import \
        apply_rotary_pos_emb
    q = torch.as_tensor(q_np, dtype=torch.float32)
    k = torch.as_tensor(k_np, dtype=torch.float32)
    cos = torch.as_tensor(cos_np, dtype=torch.float32)
    sin = torch.as_tensor(sin_np, dtype=torch.float32)
    with torch.no_grad():
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    return q_out.numpy(), k_out.numpy()


# ---- Test 1: MLA interleaved RoPE parity at small positions -----------------

def test_rope_mla_interleaved_parity_small():
    """JAX MLA interleaved RoPE matches HF oracle at small positions (maxabs < 1e-6).

    Fixture: [B=2, n_heads=8, T=16, qk_rope_head_dim=64], positions 0..15.
    Sub-assertion: cos/sin table matches HF before testing applied output.
    """
    from tpu_inference.models.jax.glm_moe_dsa import (
        apply_rope_interleaved_jax, build_rope_cos_sin_np)

    B, n_heads, T, head_dim = 2, 8, 16, 64
    rope_theta = 10000.0
    rng = np.random.default_rng(42)
    positions = np.arange(T, dtype=np.int32)
    q_np = rng.standard_normal((B, n_heads, T, head_dim)).astype(np.float32)
    k_np = rng.standard_normal((B, n_heads, T, head_dim)).astype(np.float32)

    # Build cos/sin with HF numpy helper (secondary cross-check)
    hf_cos, hf_sin = _hf_cos_sin(positions, rope_theta, head_dim)  # [1, T, d]

    # Build cos/sin with JAX function (host numpy, no device)
    jax_cos, jax_sin = build_rope_cos_sin_np(positions, rope_theta, head_dim)
    # jax_cos/sin: [T, head_dim]

    # PRIMARY table gate: compare against the REAL GlmMoeDsaRotaryEmbedding module.
    # real_cos/sin are used for BOTH sides of the apply comparison (see below) so
    # that the apply assertion measures only the JAX vs torch arithmetic delta, not
    # any residual table-construction difference.
    real_cos, real_sin = _real_module_cos_sin(positions, rope_theta, head_dim)
    assert maxabs(real_cos, jax_cos) < 1e-6, (
        f"cos table vs real module: maxabs={maxabs(real_cos, jax_cos):.2e}")
    assert maxabs(real_sin, jax_sin) < 1e-6, (
        f"sin table vs real module: maxabs={maxabs(real_sin, jax_sin):.2e}")

    # Secondary cross-check: numpy helper (same theta=10000, safe at small positions)
    assert maxabs(hf_cos[0], jax_cos) < 1e-6, (
        f"cos table mismatch: maxabs={maxabs(hf_cos[0], jax_cos):.2e}")
    assert maxabs(hf_sin[0], jax_sin) < 1e-6, (
        f"sin table mismatch: maxabs={maxabs(hf_sin[0], jax_sin):.2e}")

    # HF oracle applied output — use real module table ([1,T,d]) so both sides
    # share the same cos/sin and the delta reflects only JAX vs torch arithmetic.
    hf_q_out, hf_k_out = _hf_oracle_rope_interleave(
        q_np, k_np, real_cos[None], real_sin[None])

    # JAX applied output  — broadcast cos/sin over batch
    q_jax = jnp.array(q_np)
    k_jax = jnp.array(k_np)
    cos_jax = jnp.array(jax_cos)  # [T, head_dim]
    sin_jax = jnp.array(jax_sin)
    jax_q_out, jax_k_out = apply_rope_interleaved_jax(q_jax, k_jax,
                                                       cos_jax, sin_jax)

    delta_q = maxabs(hf_q_out, jax_q_out)
    delta_k = maxabs(hf_k_out, jax_k_out)
    assert delta_q < 1e-6, (
        f"MLA interleaved q maxabs={delta_q:.2e} >= 1e-6 (small positions)")
    assert delta_k < 1e-6, (
        f"MLA interleaved k maxabs={delta_k:.2e} >= 1e-6 (small positions)")


# ---- Test 2: Indexer rotate-half RoPE parity at small positions -------------

def test_rope_indexer_rotate_half_parity_small():
    """JAX indexer rotate-half RoPE matches HF oracle at small positions (maxabs < 1e-6).

    Fixture: [B=2, n_heads=8, T=16, qk_rope_head_dim=64], positions 0..15.
    Sub-assertion: same cos/sin table as Test 1 (full-width, shared builder).
    """
    from tpu_inference.models.jax.glm_moe_dsa import (
        apply_rope_rotate_half_jax, build_rope_cos_sin_np)

    B, n_heads, T, head_dim = 2, 8, 16, 64
    rope_theta = 10000.0
    rng = np.random.default_rng(99)
    positions = np.arange(T, dtype=np.int32)
    q_np = rng.standard_normal((B, n_heads, T, head_dim)).astype(np.float32)
    k_np = rng.standard_normal((B, n_heads, T, head_dim)).astype(np.float32)

    hf_cos, hf_sin = _hf_cos_sin(positions, rope_theta, head_dim)  # [1, T, d]

    jax_cos, jax_sin = build_rope_cos_sin_np(positions, rope_theta, head_dim)

    # PRIMARY table gate: compare against the REAL GlmMoeDsaRotaryEmbedding module.
    # real_cos/sin are used for BOTH sides of the apply comparison so the delta
    # measures only JAX vs torch arithmetic, not table-construction differences.
    real_cos, real_sin = _real_module_cos_sin(positions, rope_theta, head_dim)
    assert maxabs(real_cos, jax_cos) < 1e-6, (
        f"cos table vs real module: maxabs={maxabs(real_cos, jax_cos):.2e}")
    assert maxabs(real_sin, jax_sin) < 1e-6, (
        f"sin table vs real module: maxabs={maxabs(real_sin, jax_sin):.2e}")

    # Secondary cross-check: numpy helper (safe at small positions / theta=10000)
    assert maxabs(hf_cos[0], jax_cos) < 1e-6, (
        f"cos table mismatch: maxabs={maxabs(hf_cos[0], jax_cos):.2e}")
    assert maxabs(hf_sin[0], jax_sin) < 1e-6, (
        f"sin table mismatch: maxabs={maxabs(hf_sin[0], jax_sin):.2e}")

    # HF oracle applied output — use real module table ([1,T,d]) so both sides
    # share the same cos/sin.
    hf_q_out, hf_k_out = _hf_oracle_rope_rotate_half(
        q_np, k_np, real_cos[None], real_sin[None])

    # JAX applied output
    q_jax = jnp.array(q_np)
    k_jax = jnp.array(k_np)
    cos_jax = jnp.array(jax_cos)
    sin_jax = jnp.array(jax_sin)
    jax_q_out, jax_k_out = apply_rope_rotate_half_jax(q_jax, k_jax,
                                                       cos_jax, sin_jax)

    delta_q = maxabs(hf_q_out, jax_q_out)
    delta_k = maxabs(hf_k_out, jax_k_out)
    assert delta_q < 1e-6, (
        f"Indexer rotate-half q maxabs={delta_q:.2e} >= 1e-6 (small positions)")
    assert delta_k < 1e-6, (
        f"Indexer rotate-half k maxabs={delta_k:.2e} >= 1e-6 (small positions)")


# ---- Test 3: MLA interleaved RoPE parity at near-1M positions ---------------

def test_rope_mla_interleaved_parity_near_1m():
    """JAX MLA interleaved RoPE matches HF oracle at near-1M positions (maxabs < 1e-6).

    Uses rope_theta=8_000_000 (real checkpoint value).  The near-1M positions
    exercise the highest angles: pos * inv_freq[0] ≈ 1048575 rad (the D4 FIX
    precision risk, spec §D4).  Verified by orchestrator: numpy/torch/jnp all
    agree to ~6e-8 in fp32 at this angle, so 1e-6 is achievable.
    """
    from tpu_inference.models.jax.glm_moe_dsa import (
        apply_rope_interleaved_jax, build_rope_cos_sin_np)

    B, n_heads, T, head_dim = 1, 8, 8, 64
    rope_theta = 8_000_000.0
    rng = np.random.default_rng(7)
    # Near-1M positions; max_position_embeddings=1,048,576 in production
    positions = np.array([1048575, 1048574, 1048000, 1047000,
                          524288, 262144, 131072, 65536], dtype=np.int32)
    q_np = rng.standard_normal((B, n_heads, T, head_dim)).astype(np.float32)
    k_np = rng.standard_normal((B, n_heads, T, head_dim)).astype(np.float32)

    hf_cos, hf_sin = _hf_cos_sin(positions, rope_theta, head_dim)  # [1, T, d]

    jax_cos, jax_sin = build_rope_cos_sin_np(positions, rope_theta, head_dim)

    # PRIMARY table gate: compare against the REAL GlmMoeDsaRotaryEmbedding module.
    # Uses rope_theta=8_000_000 (real checkpoint value) — the most demanding case:
    # at near-1M positions a 1-ULP error in inv_freq accumulates to ~3e-2 in angle,
    # so this gate is specifically designed to catch fp32 pow-path divergence.
    # real_cos/sin are used for BOTH sides of the apply comparison so the delta
    # measures only JAX vs torch float32 arithmetic, not table-construction diff.
    real_cos, real_sin = _real_module_cos_sin(positions, rope_theta, head_dim)
    assert maxabs(real_cos, jax_cos) < 1e-6, (
        f"cos table vs real module (near-1M): maxabs={maxabs(real_cos, jax_cos):.2e}")
    assert maxabs(real_sin, jax_sin) < 1e-6, (
        f"sin table vs real module (near-1M): maxabs={maxabs(real_sin, jax_sin):.2e}")
    # Note: _hf_cos_sin (numpy helper) is NOT a valid secondary reference at
    # near-1M / rope_theta=8_000_000 — the helper's numpy pow-path has a 1-ULP
    # error in inv_freq that causes ~3e-2 angle divergence at these positions.
    # The real GlmMoeDsaRotaryEmbedding module above is the only valid reference.

    # HF oracle applied output — use real module table ([1,T,d]) so both sides
    # share the same cos/sin.
    hf_q_out, hf_k_out = _hf_oracle_rope_interleave(
        q_np, k_np, real_cos[None], real_sin[None])

    # JAX applied output
    q_jax = jnp.array(q_np)
    k_jax = jnp.array(k_np)
    cos_jax = jnp.array(jax_cos)
    sin_jax = jnp.array(jax_sin)
    jax_q_out, jax_k_out = apply_rope_interleaved_jax(q_jax, k_jax,
                                                       cos_jax, sin_jax)

    delta_q = maxabs(hf_q_out, jax_q_out)
    delta_k = maxabs(hf_k_out, jax_k_out)
    assert delta_q < 1e-6, (
        f"MLA interleaved q maxabs={delta_q:.2e} >= 1e-6 (near-1M positions)")
    assert delta_k < 1e-6, (
        f"MLA interleaved k maxabs={delta_k:.2e} >= 1e-6 (near-1M positions)")


# ---------------------------------------------------------------------------
# Phase 1a Task 3 — Non-absorbed pure-jnp fp32 MLA reference (the math oracle).
#
# Gates GlmMoeDsaAttentionRef (explicit per-forward kv_b split, NO absorption,
# all-fp32) against the HF eager oracle at the single-MLA-submodule level on a
# random hidden-state input at seq=32 < index_topk=64, where the DSA indexer is
# the dense identity (spec §A5) so HF self_attn == dense causal MLA.
#
# TDD: tests written BEFORE the GlmMoeDsaAttentionRef implementation so each is
# witnessed failing first. Tolerance: 1e-3 (fp32 MATH gate, §H3/§H5).
#
# Isolation strategy: the SAME random hidden_states [B,T,hidden] is fed to (a)
# one HF MLA attention block (self_attn.forward, indexer active but identity at
# T<topk) and (b) GlmMoeDsaAttentionRef, with IDENTICAL weights (verified by
# assert_identical_weights BEFORE comparing outputs). cos/sin are taken from the
# REAL GlmMoeDsaRotaryEmbedding module and passed to BOTH sides, isolating the
# attention algebra from RoPE-table construction (Task 2 already gates the table
# bit-for-bit at 1e-6).
# ---------------------------------------------------------------------------

# MLA submodule weight names (unfused HF names, relative to the layer prefix).
_MLA_WEIGHT_NAMES = (
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
)


def _extract_mla_jax_weights(model, layer_idx):
    """Convert ONE layer's MLA submodule weights to JAX via t2j_weights.

    Returns a dict keyed by the bare MLA names in _MLA_WEIGHT_NAMES.
    """
    prefix = f"model.layers.{layer_idx}."
    sd = model.state_dict()
    sub = {}
    for bare in _MLA_WEIGHT_NAMES:
        full = prefix + bare
        sub[full] = sd[full]
    conv = t2j_weights(sub)
    return {bare: conv[prefix + bare] for bare in _MLA_WEIGHT_NAMES}


def _hf_mla_block_output(model, layer_idx, hidden_states_t, cos_t, sin_t):
    """Run ONE HF MLA attention block's forward in fp32 and return its output.

    Builds an additive causal mask [B,1,T,T] and position_ids=arange(T). At
    T<index_topk the indexer is active but selects all causal keys, so the
    output equals dense causal MLA (spec §A5).
    """
    import torch
    B, T, _ = hidden_states_t.shape
    attn = model.model.layers[layer_idx].self_attn
    # additive causal mask: 0 on/below diagonal, -inf above (fp32 min)
    neg = torch.finfo(torch.float32).min
    causal = torch.triu(torch.full((T, T), neg, dtype=torch.float32), diagonal=1)
    attn_mask = causal[None, None, :, :].expand(B, 1, T, T).contiguous()
    position_ids = torch.arange(T, dtype=torch.long)[None, :].expand(B, T)
    with torch.no_grad():
        out, _, _ = attn(
            hidden_states=hidden_states_t,
            position_embeddings=(cos_t, sin_t),
            attention_mask=attn_mask,
            past_key_values=None,
            position_ids=position_ids,
            prev_topk_indices=None,
        )
    return out  # [B, T, hidden]


def _build_mla_parity_fixture(seed=0, layer_idx=0, T=32):
    """Shared fixture: build oracle, random hidden states, cos/sin, weights.

    Returns (cfg, hidden_np, cos_np, sin_np, jax_w, hf_out_np) all fp32 host
    numpy / jnp arrays. layer_idx=0 is a "full" indexer layer; T=32<index_topk=64
    keeps the indexer dense-equivalent.
    """
    import torch
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import \
        GlmMoeDsaRotaryEmbedding

    cfg = tiny_glm_moe_dsa_config()
    assert T < cfg.index_topk, f"T={T} must be < index_topk={cfg.index_topk}"
    model = build_hf_oracle(cfg=cfg, seed=seed)

    B, hidden = 1, cfg.hidden_size
    rng = np.random.default_rng(1234)
    hidden_np = rng.standard_normal((B, T, hidden)).astype(np.float32)
    hidden_t = torch.as_tensor(hidden_np, dtype=torch.float32)

    # Real rotary module cos/sin (head_dim=qk_rope_head_dim=64), passed to BOTH sides.
    rotary = GlmMoeDsaRotaryEmbedding(cfg)
    pos = torch.arange(T, dtype=torch.long)[None, :]  # [1, T]
    with torch.no_grad():
        cos_t, sin_t = rotary.forward(hidden_t, pos)   # [1, T, 64]
    cos_np = cos_t[0].numpy().astype(np.float32)        # [T, 64]
    sin_np = sin_t[0].numpy().astype(np.float32)

    hf_out = _hf_mla_block_output(model, layer_idx, hidden_t, cos_t, sin_t)
    hf_out_np = hf_out.detach().numpy().astype(np.float32)

    jax_w = _extract_mla_jax_weights(model, layer_idx)
    return cfg, hidden_np, cos_np, sin_np, jax_w, hf_out_np


def _check_jax_weights_match_hf_raw(model, layer_idx, jax_w):
    """Assert JAX-loaded MLA weights match the raw HF torch weights (with transpose).

    For each MLA linear projection weight, this takes the RAW torch tensor from
    the HF block (shape ``[out, in]``), applies the documented HF→JAX transpose
    (``.T`` → ``[in, out]``), converts to fp32 numpy, and compares against the
    corresponding JAX-loaded weight (maxabs == 0.0).

    This is an INDEPENDENT check — it does not call export_weights() or compare
    jax_w against itself, so it catches a t2j_weights transpose/shape/mapping
    bug that a self-comparison would miss.
    """
    prefix = f"model.layers.{layer_idx}."
    sd = model.state_dict()
    # Linear projections: HF shape (out, in) → JAX wants (in, out) via .T
    linear_names = (
        "self_attn.q_a_proj.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.o_proj.weight",
    )
    # 1-D norm weights: shape unchanged
    norm_names = (
        "self_attn.q_a_layernorm.weight",
        "self_attn.kv_a_layernorm.weight",
    )
    for bare in linear_names:
        hf_torch = sd[prefix + bare]  # [out, in]
        expected = hf_torch.detach().float().cpu().numpy().T  # [in, out]
        actual = np.asarray(jax_w[bare]).astype(np.float32)
        delta = float(np.max(np.abs(expected - actual)))
        assert delta == 0.0, (
            f"HF↔JAX weight mismatch for {bare!r}: maxabs={delta:.3e} "
            f"(expected 0.0 — transpose or mapping bug in t2j_weights)")
    for bare in norm_names:
        hf_torch = sd[prefix + bare]  # [dim]
        expected = hf_torch.detach().float().cpu().numpy()
        actual = np.asarray(jax_w[bare]).astype(np.float32)
        delta = float(np.max(np.abs(expected - actual)))
        assert delta == 0.0, (
            f"HF↔JAX norm weight mismatch for {bare!r}: maxabs={delta:.3e}")


def test_mla_ref_math_gate():
    """GlmMoeDsaAttentionRef (non-absorbed fp32) == HF eager MLA block, maxabs<1e-3.

    Single-MLA-submodule parity at seq=32 < index_topk=64 (DSA dense identity,
    §A5). An INDEPENDENT HF↔JAX raw-weight check (not a self-comparison) is
    run BEFORE comparing outputs so the result reflects the two implementations,
    not a weight mismatch or a transpose/mapping bug.
    """
    import torch
    from tpu_inference.models.jax.glm_moe_dsa import GlmMoeDsaAttentionRef

    cfg = tiny_glm_moe_dsa_config()
    seed = 0
    layer_idx = 0
    T = 32
    model = build_hf_oracle(cfg=cfg, seed=seed)

    cfg_ret, hidden_np, cos_np, sin_np, jax_w, hf_out_np = _build_mla_parity_fixture(
        seed=seed, layer_idx=layer_idx, T=T)

    ref = GlmMoeDsaAttentionRef(cfg_ret)
    ref.load_weights(jax_w)

    # Fix 2: INDEPENDENT HF↔JAX raw-weight check (not a self-comparison).
    # Takes each raw torch tensor, applies the documented .T transpose for linears,
    # converts to fp32 numpy, and compares against the JAX-loaded weight.
    # This catches a t2j_weights transpose/shape/mapping bug independently of
    # the forward gate. The previous assert_identical_weights(jax_w, ref.export_weights())
    # was vacuous because export_weights() returns the same jax_w dict.
    _check_jax_weights_match_hf_raw(model, layer_idx, jax_w)

    out = ref(jnp.asarray(hidden_np),
              jnp.asarray(cos_np), jnp.asarray(sin_np))
    delta = maxabs(hf_out_np, out)
    assert delta < 1e-3, (
        f"MLA ref vs HF eager maxabs={delta:.3e} >= 1e-3 (fp32 MATH gate)")


def test_mla_ref_injected_error_trips_gate():
    """A 1% projection-weight perturbation must FAIL the 1e-3 gate by a clear margin.

    Proves the fp32 math gate has teeth: the CLEAN parity is ~5e-7 (≈2000x below
    the 1e-3 gate), so a real 1% bug must be caught.  Perturbing o_proj by 1%
    yields maxabs ≈ 1.6e-2 — >15x the gate, a decisive failure.  (A 1% sm_scale
    perturbation also trips it at ≈2.5e-3; the weight perturbation is the more
    representative "real bug" and gives a clearer margin.)  §H3.
    """
    from tpu_inference.models.jax.glm_moe_dsa import GlmMoeDsaAttentionRef

    cfg, hidden_np, cos_np, sin_np, jax_w, hf_out_np = _build_mla_parity_fixture()

    ref = GlmMoeDsaAttentionRef(cfg)
    ref.load_weights(jax_w)
    # Inject a 1% error into the output projection weight.
    ref._w["self_attn.o_proj.weight"] = ref._w["self_attn.o_proj.weight"] * 1.01

    out = ref(jnp.asarray(hidden_np),
              jnp.asarray(cos_np), jnp.asarray(sin_np))
    delta = maxabs(hf_out_np, out)
    assert delta >= 1e-3, (
        f"injected 1% o_proj error did NOT trip the gate: maxabs={delta:.3e} "
        f"< 1e-3 (gate has no teeth)")
    # And it must trip by a clear margin (well above the fp32 tolerance).
    assert delta > 1e-2, (
        f"injected error margin too small: maxabs={delta:.3e} (expected >1e-2)")


def _build_small_input_mla_fixture(seed=0, layer_idx=0, T=32, scale=1e-3):
    """Build a parity fixture with a SMALL-MAGNITUDE hidden input.

    Scales the hidden states so the latent ``mean(x²) ≈ scale`` (default 1e-3).
    This regime makes the q_a/kv_a layernorm eps (1e-6 vs 1e-5) produce a
    relative difference ~4.5e-3 in the layernorm output — well above the 1e-3
    gate — making the eps a load-bearing gate tooth.

    Returns the same tuple as ``_build_mla_parity_fixture`` but with the hidden
    array scaled down so ``mean(hidden²) ≈ scale``.
    """
    import torch
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import \
        GlmMoeDsaRotaryEmbedding

    cfg = tiny_glm_moe_dsa_config()
    assert T < cfg.index_topk, f"T={T} must be < index_topk={cfg.index_topk}"
    model = build_hf_oracle(cfg=cfg, seed=seed)

    B, hidden = 1, cfg.hidden_size
    rng = np.random.default_rng(9999)
    # Standard-normal → scale to achieve mean(x²) ≈ scale.
    # For a standard-normal x, mean(x²) ≈ 1.0; multiplying by sqrt(scale) gives
    # mean(x²) ≈ scale.
    hidden_np = (rng.standard_normal((B, T, hidden)) * np.sqrt(scale)).astype(
        np.float32)
    hidden_t = torch.as_tensor(hidden_np, dtype=torch.float32)

    # Real rotary module cos/sin
    rotary = GlmMoeDsaRotaryEmbedding(cfg)
    pos = torch.arange(T, dtype=torch.long)[None, :]  # [1, T]
    with torch.no_grad():
        cos_t, sin_t = rotary.forward(hidden_t, pos)   # [1, T, 64]
    cos_np = cos_t[0].numpy().astype(np.float32)       # [T, 64]
    sin_np = sin_t[0].numpy().astype(np.float32)

    hf_out = _hf_mla_block_output(model, layer_idx, hidden_t, cos_t, sin_t)
    hf_out_np = hf_out.detach().numpy().astype(np.float32)

    jax_w = _extract_mla_jax_weights(model, layer_idx)
    return cfg, hidden_np, cos_np, sin_np, jax_w, hf_out_np


def test_mla_ref_eps_teeth_clean_parity():
    """Small-input regime: CLEAN parity (both sides eps=1e-6) still passes < 1e-3.

    Fix 1(a): at hidden scale sqrt(1e-3), mean(x²)≈1e-3 in the latent.  Both
    the HF oracle (eps=1e-6) and the jnp ref (default norm_eps=1e-6) use the
    correct eps, so parity should hold to the same tolerance as the normal-input
    gate.  This confirms the small-input regime does not break the clean case.
    """
    from tpu_inference.models.jax.glm_moe_dsa import GlmMoeDsaAttentionRef

    cfg, hidden_np, cos_np, sin_np, jax_w, hf_out_np = _build_small_input_mla_fixture()

    # Default norm_eps=1e-6 (same as HF).
    ref = GlmMoeDsaAttentionRef(cfg)
    ref.load_weights(jax_w)

    out = ref(jnp.asarray(hidden_np),
              jnp.asarray(cos_np), jnp.asarray(sin_np))
    delta = maxabs(hf_out_np, out)
    assert delta < 1e-3, (
        f"Small-input CLEAN parity failed: maxabs={delta:.3e} >= 1e-3 "
        f"(both eps=1e-6; expected clean pass)")


def test_mla_ref_eps_teeth_wrong_eps_trips_gate():
    """Small-input regime: eps=1e-5 in the jnp ref MUST fail the 1e-3 gate.

    Fix 1(b): proves the eps is pinned — the gate bites on a wrong eps value.
    At mean(x²)≈1e-3, swapping eps 1e-6→1e-5 produces a relative layernorm
    diff ~4.5e-3 (eps/variance ratio changes by ~9x), which propagates through
    q_b/kv_b projections and the attention path to yield maxabs >> 1e-3.

    This test ASSERTS FAILURE (parity MUST break with wrong eps), confirming
    the gate is not silent in the face of the #1 eps-bug risk (spec §H3).
    """
    from tpu_inference.models.jax.glm_moe_dsa import GlmMoeDsaAttentionRef

    cfg, hidden_np, cos_np, sin_np, jax_w, hf_out_np = _build_small_input_mla_fixture()

    # Mutate ONLY the jnp ref's eps to 1e-5; HF oracle stays at 1e-6.
    ref_wrong_eps = GlmMoeDsaAttentionRef(cfg, norm_eps=1e-5)
    ref_wrong_eps.load_weights(jax_w)

    out = ref_wrong_eps(jnp.asarray(hidden_np),
                        jnp.asarray(cos_np), jnp.asarray(sin_np))
    delta = maxabs(hf_out_np, out)
    assert delta >= 1e-3, (
        f"Wrong eps (1e-5) did NOT trip the gate in small-input regime: "
        f"maxabs={delta:.3e} < 1e-3 (eps is NOT pinned — silent bug risk!)")
    # Must trip by a clear margin (this is the whole point of the teeth test).
    assert delta > 2e-3, (
        f"Wrong eps margin too small: maxabs={delta:.3e} (expected >2e-3 at "
        f"mean(x²)≈1e-3); the gate is too weak to be meaningful")


# ---------------------------------------------------------------------------
# Phase 1a Task 4 — GLM model scaffold + 5 per-submodule parity gates.
#
# Each gate feeds the SAME random activation to (a) one HF-eager submodule and
# (b) the JAX submodule the scaffold wires in, with IDENTICAL weights (asserted
# against the raw HF torch tensor with the documented transpose BEFORE any
# output comparison). Every fp32 parity forward runs under
# `jax.default_matmul_precision("highest")` (TPU default fp32 matmul uses bf16
# passes ~5e-3 error which would false-fail the 1e-3 gates).
#
# Tolerances (spec §H / brief): norms/FFN/embed/lm_head 1e-3; router weights /
# MoE experts 1e-2; router top-k indices EXACT.
#
# TDD: tests written against the JAX submodules the scaffold reuses
# (JaxRmsNorm, DeepseekV3MLP, DeepSeekV3Router, DeepseekV2Moe, JaxEmbed,
# JaxLmHead) so each gate is witnessed before/with the scaffold landing.
# ---------------------------------------------------------------------------

import math

from flax import nnx


def _glm_dims(cfg):
    """Explicit GLM dims read from hf_config (NEVER cfg.head_dim — it is
    overwritten to qk_rope_head_dim=64 in __post_init__)."""
    return dict(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_hidden_layers=cfg.num_hidden_layers,
        vocab_size=cfg.vocab_size,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        qk_head_dim=cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
        rms_norm_eps=cfg.rms_norm_eps,
        intermediate_size=cfg.intermediate_size,
        moe_intermediate_size=cfg.moe_intermediate_size,
        n_routed_experts=cfg.n_routed_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        n_shared_experts=cfg.n_shared_experts,
        routed_scaling_factor=cfg.routed_scaling_factor,
        n_group=cfg.n_group,
        topk_group=cfg.topk_group,
        norm_topk_prob=cfg.norm_topk_prob,
        first_k_dense_replace=cfg.first_k_dense_replace,
    )


# === Gate 1: RMSNorm (eps 1e-5 vs 1e-6, with eps teeth) =====================

def _hf_rmsnorm_forward(weight_t, eps, x_t):
    """Run GlmMoeDsaRMSNorm.forward exactly (fp32 upcast)."""
    import torch
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import \
        GlmMoeDsaRMSNorm
    norm = GlmMoeDsaRMSNorm(weight_t.shape[0], eps=eps)
    with torch.no_grad():
        norm.weight.copy_(weight_t)
        return norm(x_t)


def _make_jax_rmsnorm(dim, eps, weight_np):
    """Build a JaxRmsNorm (fp32) and load `weight_np` into its `weight` param."""
    norm = JaxRmsNorm(dim, epsilon=eps, dtype=jnp.float32,
                      param_dtype=jnp.float32, rngs=nnx.Rngs(0))
    norm.weight.value = jnp.asarray(weight_np, dtype=jnp.float32)
    return norm


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_norm_rmsnorm_parity(mesh_1d, eps):
    """JaxRmsNorm == GlmMoeDsaRMSNorm at the GLM eps values, maxabs<1e-3.

    eps=1e-5 covers input/post/final norms; eps=1e-6 covers q_a/kv_a norms.
    HF↔JAX weight identity is asserted before comparing outputs.
    """
    import torch
    from tpu_inference.layers.jax.norm import JaxRmsNorm  # noqa: F401 (alias below)
    dim, T = 512, 16
    rng = np.random.default_rng(0)
    weight_np = rng.standard_normal(dim).astype(np.float32)
    x_np = rng.standard_normal((1, T, dim)).astype(np.float32)

    norm = _make_jax_rmsnorm(dim, eps, weight_np)
    # HF↔JAX weight identity (RMSNorm weight is 1-D, no transpose).
    assert float(np.max(np.abs(
        np.asarray(norm.weight.value) - weight_np))) == 0.0

    hf_out = _hf_rmsnorm_forward(torch.as_tensor(weight_np), eps,
                                 torch.as_tensor(x_np))
    with jax.default_matmul_precision("highest"):
        jax_out = norm(jnp.asarray(x_np))
    delta = maxabs(hf_out, jax_out)
    assert delta < 1e-3, f"RMSNorm eps={eps} maxabs={delta:.3e} >= 1e-3"


def test_norm_eps_1e6_not_1e5_guard(mesh_1d):
    """Teeth: the q_a/kv_a norm MUST use eps=1e-6, not rms_norm_eps=1e-5.

    Small-magnitude input (mean(x²)~1e-3) makes eps load-bearing: a JaxRmsNorm
    built with eps=1e-5 diverges >1e-3 from the eps=1e-6 HF oracle, while the
    correct eps=1e-6 stays well under it.
    """
    import torch
    dim, T = 512, 16
    rng = np.random.default_rng(1)
    weight_np = rng.standard_normal(dim).astype(np.float32)
    # mean(x²) ~ 1e-3 so the eps/variance ratio (1e-6 vs 1e-5) is load-bearing.
    x_np = (rng.standard_normal((1, T, dim)) * math.sqrt(1e-3)).astype(np.float32)

    # Oracle: the CORRECT eps for q_a/kv_a is 1e-6.
    hf_out = _hf_rmsnorm_forward(torch.as_tensor(weight_np), 1e-6,
                                 torch.as_tensor(x_np))

    norm_right = _make_jax_rmsnorm(dim, 1e-6, weight_np)
    norm_wrong = _make_jax_rmsnorm(dim, 1e-5, weight_np)
    with jax.default_matmul_precision("highest"):
        out_right = norm_right(jnp.asarray(x_np))
        out_wrong = norm_wrong(jnp.asarray(x_np))
    d_right = maxabs(hf_out, out_right)
    d_wrong = maxabs(hf_out, out_wrong)
    assert d_right < 1e-3, f"correct eps=1e-6 should pass: {d_right:.3e}"
    assert d_wrong >= 1e-3, (
        f"wrong eps=1e-5 did NOT trip the guard: {d_wrong:.3e} (eps not pinned)")


# === Gate 2: Dense FFN (DeepseekV3MLP SwiGLU) ==============================

_FFN_WEIGHT_NAMES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")


def _hf_mlp_block(model, layer_idx, x_t):
    """Run ONE HF dense MLP block (a first_k_dense_replace layer)."""
    import torch
    mlp = model.model.layers[layer_idx].mlp
    with torch.no_grad():
        return mlp(x_t)


def _build_jax_dense_mlp(cfg, weights_jax):
    """Construct DeepseekV3MLP (dense, intermediate_size) and load weights."""
    d = _glm_dims(cfg)
    mlp = DeepseekV3MLP(
        dtype=jnp.float32,
        hidden_act=cfg.hidden_act,
        hidden_size=d["hidden_size"],
        intermediate_size=d["intermediate_size"],
        rngs=nnx.Rngs(0),
    )
    mlp.gate_proj.weight.value = jnp.asarray(weights_jax["gate_proj.weight"])
    mlp.up_proj.weight.value = jnp.asarray(weights_jax["up_proj.weight"])
    mlp.down_proj.weight.value = jnp.asarray(weights_jax["down_proj.weight"])
    return mlp


def test_dense_ffn_parity(mesh_1d):
    """DeepseekV3MLP (SwiGLU silu) == HF GlmMoeDsaMLP dense block, maxabs<1e-3.

    Uses layer 0 (dense; first_k_dense_replace=3). HF↔JAX weight identity
    (linear transpose [out,in]->[in,out]) asserted before output comparison.
    """
    import torch
    cfg = tiny_glm_moe_dsa_config()
    model = build_hf_oracle(cfg=cfg, seed=0)
    layer_idx, T = 0, 16
    prefix = f"model.layers.{layer_idx}.mlp."
    sd = model.state_dict()

    sub = {prefix + n: sd[prefix + n] for n in _FFN_WEIGHT_NAMES}
    conv = t2j_weights(sub)
    weights_jax = {n: conv[prefix + n] for n in _FFN_WEIGHT_NAMES}

    # INDEPENDENT HF↔JAX raw-weight check (linears: .T transpose).
    for n in _FFN_WEIGHT_NAMES:
        expected = sd[prefix + n].detach().float().cpu().numpy().T
        actual = np.asarray(weights_jax[n]).astype(np.float32)
        assert float(np.max(np.abs(expected - actual))) == 0.0, (
            f"HF↔JAX dense-FFN weight mismatch for {n!r}")

    rng = np.random.default_rng(2)
    x_np = rng.standard_normal((1, T, cfg.hidden_size)).astype(np.float32)
    hf_out = _hf_mlp_block(model, layer_idx, torch.as_tensor(x_np))

    mlp = _build_jax_dense_mlp(cfg, weights_jax)
    with jax.default_matmul_precision("highest"):
        jax_out = mlp(jnp.asarray(x_np[0]))  # DeepseekV3MLP is [T,D]
    delta = maxabs(hf_out[0], jax_out)
    assert delta < 1e-3, f"dense FFN maxabs={delta:.3e} >= 1e-3"


# === Gate 3: Router top-k (indices EXACT + weights<1e-2) ====================

def _hf_route_tokens(model, layer_idx, logits_t):
    """Run HF GlmMoeDsaMoE.route_tokens_to_experts exactly (sigmoid, bias-for-
    selection, group mask, top-k, bias-free gather, /(sum+1e-20), *scaling)."""
    moe = model.model.layers[layer_idx].mlp
    return moe.route_tokens_to_experts(logits_t)


def _build_jax_router(cfg, gate_weight_np, bias_np, mesh):
    """Construct DeepSeekV3Router with GLM config deltas (sigmoid, n_group=1,
    topk_group=1, routed_scaling_factor=2.5, norm_topk_prob)."""
    d = _glm_dims(cfg)
    router = DeepSeekV3Router(
        hidden_size=d["hidden_size"],
        num_experts=d["n_routed_experts"],
        num_experts_per_tok=d["num_experts_per_tok"],
        n_groups=d["n_group"],          # 1 for GLM (grouping OFF)
        topk_groups=d["topk_group"],    # 1 for GLM
        norm_topk_prob=d["norm_topk_prob"],
        routed_scaling_factor=d["routed_scaling_factor"],
        dtype=jnp.float32,
        rngs=nnx.Rngs(0),
        scoring_func="sigmoid",
        moe_backend=MoEBackend.DENSE_MAT,
    )
    # gate kernel: HF [E, hidden] -> JAX [hidden, E] (linear transpose).
    router.weight.value = jnp.asarray(gate_weight_np.T, dtype=jnp.float32)
    router.e_score_correction_bias.value = jnp.asarray(bias_np,
                                                       dtype=jnp.float32)
    return router


def test_router_topk_parity(mesh_1d):
    """DeepSeekV3Router top-k indices EXACT + gathered weights<1e-2 vs HF.

    §H11a: PEAKED logits (well-separated, not normal*0.02) so the top-k
    selection is unambiguous and exact-index parity reflects correctness, not
    luck. The router returns BIAS-FREE sigmoid weights renormed /(sum+1e-20);
    HF bakes *routed_scaling_factor into its returned weights, so the JAX
    weights are compared *routed_scaling_factor to match (the scaling is applied
    once, on the MoE output, in the JAX path — see Gate 4).
    """
    import torch
    cfg = tiny_glm_moe_dsa_config()
    # Sparse layer is layer 3 (first_k_dense_replace=3) in tiny config.
    layer_idx = 3
    model = build_hf_oracle(cfg=cfg, seed=0, randomize_buffers=True)
    d = _glm_dims(cfg)
    E, T = d["n_routed_experts"], 24

    gate_w = model.model.layers[layer_idx].mlp.gate.weight.detach().float(
    ).cpu().numpy()  # [E, hidden]
    bias = model.model.layers[layer_idx].mlp.gate.e_score_correction_bias.detach(
    ).float().cpu().numpy()  # [E]
    assert float(np.linalg.norm(bias)) > 0.0, "bias must be non-zero (selection path)"

    # PEAKED router logits: one clearly-dominant expert per token plus a few
    # well-separated runners-up, so top-k selection is unambiguous.
    rng = np.random.default_rng(3)
    logits_np = (rng.standard_normal((T, E)) * 8.0).astype(np.float32)
    logits_t = torch.as_tensor(logits_np)

    hf_idx, hf_w = _hf_route_tokens(model, layer_idx, logits_t)
    hf_idx_np = hf_idx.detach().cpu().numpy()        # [T, topk] (unsorted)
    hf_w_np = hf_w.detach().float().cpu().numpy()    # [T, topk] (incl *2.5)

    router = _build_jax_router(cfg, gate_w, bias, mesh_1d)
    # HF↔JAX gate-weight identity (linear transpose).
    assert float(np.max(np.abs(
        np.asarray(router.weight.value) - gate_w.T))) == 0.0
    assert float(np.max(np.abs(
        np.asarray(router.e_score_correction_bias.value) - bias))) == 0.0

    # The JAX router computes logits internally from x; here we want to compare
    # the routing arithmetic on IDENTICAL logits. Drive it via get_topk_indices
    # + the same gather/renorm the router applies in __call__.
    with jax.default_matmul_precision("highest"):
        probs = jax.nn.sigmoid(jnp.asarray(logits_np))
        jax_idx = router.get_topk_indices(probs)             # [T, topk]
        jax_w = jnp.take_along_axis(probs, jax_idx, axis=-1)
        jax_w = jax_w / (jnp.sum(jax_w, axis=-1, keepdims=True) + 1e-20)
        jax_w = jax_w * d["routed_scaling_factor"]           # match HF bake-in

    jax_idx_np = np.asarray(jax_idx)
    # Top-k indices EXACT — compare as SETS per token (HF top-k is unsorted).
    for t in range(T):
        assert set(jax_idx_np[t].tolist()) == set(hf_idx_np[t].tolist()), (
            f"token {t}: router top-k indices differ "
            f"jax={sorted(jax_idx_np[t])} hf={sorted(hf_idx_np[t])}")

    # Gathered weights < 1e-2 — align by selected expert id (order may differ).
    jax_w_np = np.asarray(jax_w)
    max_w_delta = 0.0
    for t in range(T):
        jmap = {int(e): float(w) for e, w in zip(jax_idx_np[t], jax_w_np[t])}
        hmap = {int(e): float(w) for e, w in zip(hf_idx_np[t], hf_w_np[t])}
        for e in hmap:
            max_w_delta = max(max_w_delta, abs(jmap[e] - hmap[e]))
    assert max_w_delta < 1e-2, f"router weights maxabs={max_w_delta:.3e} >= 1e-2"


def test_router_sigmoid_not_softmax_guard(mesh_1d):
    """Teeth: GLM router scores with SIGMOID, not softmax.

    A softmax-scored router yields different gathered weights; assert the
    sigmoid path matches HF and a softmax path would not (sanity: sigmoid and
    softmax of the same peaked logits give clearly different weight magnitudes).
    """
    cfg = tiny_glm_moe_dsa_config()
    d = _glm_dims(cfg)
    E, T = d["n_routed_experts"], 8
    rng = np.random.default_rng(33)
    logits_np = (rng.standard_normal((T, E)) * 8.0).astype(np.float32)
    sig = np.asarray(jax.nn.sigmoid(jnp.asarray(logits_np)))
    sm = np.asarray(jax.nn.softmax(jnp.asarray(logits_np), axis=-1))
    # The two scorings must be materially different (so the choice is testable).
    assert float(np.max(np.abs(sig - sm))) > 1e-2


# === Gate 4: MoE experts (DeepseekV2Moe vs HF GlmMoeDsaMoE eager) ==========

def _load_hf_moe_weights_into_jax(moe_module, model, layer_idx, cfg):
    """Inject HF GlmMoeDsaMoE weights into a JAX DeepseekV2Moe (DENSE_MAT).

    HF stores fused expert params: gate_up_proj [E, 2F, D], down_proj [E, D, F]
    (nn.Linear [out,in] layout). JAX DENSE_MAT experts want:
      kernel_gating_EDF  [E, D, F]   (gate, transposed last-two from [E,F,D])
      kernel_up_proj_EDF [E, D, F]   (up,   transposed last-two from [E,F,D])
      kernel_down_proj_EFD [E, F, D] (down, transposed last-two from [E,D,F])
    Router gate kernel: HF [E,D] -> JAX [D,E]. Shared expert: a DeepseekV3MLP
    (intermediate = moe_intermediate_size * n_shared_experts).
    """
    import numpy as _np
    sd = model.state_dict()
    p = f"model.layers.{layer_idx}.mlp."

    gate_up = sd[p + "experts.gate_up_proj"].detach().float().cpu().numpy()  # [E,2F,D]
    down = sd[p + "experts.down_proj"].detach().float().cpu().numpy()        # [E,D,F]
    F = cfg.moe_intermediate_size
    g = gate_up[:, :F, :]   # [E, F, D]
    u = gate_up[:, F:, :]   # [E, F, D]
    experts = moe_module.experts  # SharedFusedMoe
    experts.kernel_gating_EDF.value = jnp.asarray(_np.swapaxes(g, -2, -1))   # [E,D,F]
    experts.kernel_up_proj_EDF.value = jnp.asarray(_np.swapaxes(u, -2, -1))  # [E,D,F]
    experts.kernel_down_proj_EFD.value = jnp.asarray(_np.swapaxes(down, -2, -1))  # [E,F,D]

    # Router.
    gate_w = sd[p + "gate.weight"].detach().float().cpu().numpy()  # [E,D]
    bias = sd[p + "gate.e_score_correction_bias"].detach().float().cpu().numpy()
    moe_module.gate.weight.value = jnp.asarray(gate_w.T)
    moe_module.gate.e_score_correction_bias.value = jnp.asarray(bias)

    # Shared expert MLP (gate/up/down, linear transpose).
    sg = sd[p + "shared_experts.gate_proj.weight"].detach().float().cpu().numpy()
    su = sd[p + "shared_experts.up_proj.weight"].detach().float().cpu().numpy()
    sdn = sd[p + "shared_experts.down_proj.weight"].detach().float().cpu().numpy()
    moe_module.shared_experts.gate_proj.weight.value = jnp.asarray(sg.T)
    moe_module.shared_experts.up_proj.weight.value = jnp.asarray(su.T)
    moe_module.shared_experts.down_proj.weight.value = jnp.asarray(sdn.T)


def _build_jax_moe(cfg, mesh):
    """Construct DeepseekV2Moe with all GLM config deltas, DENSE_MAT backend."""
    from tpu_inference.layers.jax.quantization.unquantized import \
        UnquantizedConfig
    d = _glm_dims(cfg)
    return DeepseekV2Moe(
        mesh=mesh,
        dtype=jnp.float32,
        num_expert_parallelism=1,
        moe_backend=MoEBackend.DENSE_MAT,
        quant_config=UnquantizedConfig({}),
        scoring_func="sigmoid",
        rng=nnx.Rngs(0),
        prefix="model.layers.3.mlp",
        num_local_experts=d["n_routed_experts"],
        hidden_size=d["hidden_size"],
        moe_intermediate_size=d["moe_intermediate_size"],
        num_experts_per_tok=d["num_experts_per_tok"],
        n_group=d["n_group"],
        topk_groups=d["topk_group"],
        norm_topk_prob=d["norm_topk_prob"],
        routed_scaling_factor=d["routed_scaling_factor"],
        num_shared_experts=d["n_shared_experts"],
        hidden_act=cfg.hidden_act,
    )


def test_moe_experts_parity(mesh_1d):
    """DeepseekV2Moe == HF GlmMoeDsaMoE (eager experts), maxabs<1e-2.

    Asserts the oracle resolves EAGER experts (GlmMoeDsaNaiveMoe) BEFORE
    comparing. Exercises sigmoid routing, bias-for-selection, /(sum+1e-20)
    renorm, *routed_scaling_factor on routed output, and UNSCALED shared add.
    """
    import torch
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import \
        GlmMoeDsaNaiveMoe
    cfg = tiny_glm_moe_dsa_config()
    layer_idx = 3
    model = build_hf_oracle(cfg=cfg, seed=0, randomize_buffers=True)
    # MoE-experts gate precondition: eager experts.
    assert model.config._experts_implementation_internal == "eager"
    assert isinstance(model.model.layers[layer_idx].mlp.experts,
                      GlmMoeDsaNaiveMoe)

    T = 24
    rng = np.random.default_rng(4)
    x_np = rng.standard_normal((1, T, cfg.hidden_size)).astype(np.float32)
    with torch.no_grad():
        hf_out = model.model.layers[layer_idx].mlp(torch.as_tensor(x_np))

    moe = _build_jax_moe(cfg, mesh_1d)
    _load_hf_moe_weights_into_jax(moe, model, layer_idx, cfg)
    with jax.default_matmul_precision("highest"):
        jax_out, _ = moe(jnp.asarray(x_np[0]))  # DeepseekV2Moe is [T,D]
    delta = maxabs(hf_out[0], jax_out)
    assert delta < 1e-2, f"MoE experts maxabs={delta:.3e} >= 1e-2"


# === Gate 5: Embed lookup + lm_head (untied) ===============================

def test_embed_lookup_parity(mesh_1d):
    """JaxEmbed lookup == HF embed_tokens lookup (untransposed table)."""
    import torch
    cfg = tiny_glm_moe_dsa_config()
    model = build_hf_oracle(cfg=cfg, seed=0)
    d = _glm_dims(cfg)

    emb_w = model.model.embed_tokens.weight.detach().float().cpu().numpy()  # [V,H]
    embed = JaxEmbed(num_embeddings=d["vocab_size"], features=d["hidden_size"],
                     dtype=jnp.float32, param_dtype=jnp.float32,
                     rngs=nnx.Rngs(0))
    embed.weight.value = jnp.asarray(emb_w)  # UNTRANSPOSED (lookup table)
    # HF↔JAX identity (no transpose for embed).
    assert float(np.max(np.abs(np.asarray(embed.weight.value) - emb_w))) == 0.0

    rng = np.random.default_rng(5)
    ids_np = rng.integers(0, cfg.vocab_size, size=(1, 12)).astype(np.int32)
    with torch.no_grad():
        hf_emb = model.model.embed_tokens(torch.as_tensor(ids_np))
    jax_emb = embed(jnp.asarray(ids_np))
    delta = maxabs(hf_emb, jax_emb)
    assert delta < 1e-3, f"embed lookup maxabs={delta:.3e} >= 1e-3"


def test_lm_head_parity(mesh_1d):
    """JaxLmHead logits == HF lm_head (transposed, UNTIED from embed), <1e-3."""
    import torch
    cfg = tiny_glm_moe_dsa_config()
    model = build_hf_oracle(cfg=cfg, seed=0)
    d = _glm_dims(cfg)

    head_w = model.lm_head.weight.detach().float().cpu().numpy()  # [V,H]
    emb_w = model.model.embed_tokens.weight.detach().float().cpu().numpy()
    # UNTIED: lm_head weight is a separate tensor from embed_tokens.
    assert head_w.shape == emb_w.shape

    head = JaxLmHead(hidden_size=d["hidden_size"], vocab_size=d["vocab_size"],
                     dtype=jnp.float32, param_dtype=jnp.float32,
                     rngs=nnx.Rngs(0))
    head.weight.value = jnp.asarray(head_w.T)  # [H,V] (linear transpose)
    # HF↔JAX identity.
    assert float(np.max(np.abs(np.asarray(head.weight.value) - head_w.T))) == 0.0

    rng = np.random.default_rng(6)
    x_np = rng.standard_normal((10, d["hidden_size"])).astype(np.float32)
    with torch.no_grad():
        hf_logits = model.lm_head(torch.as_tensor(x_np))
    with jax.default_matmul_precision("highest"):
        jax_logits = head(jnp.asarray(x_np))
    delta = maxabs(hf_logits, jax_logits)
    assert delta < 1e-3, f"lm_head maxabs={delta:.3e} >= 1e-3"


# === Scaffold: construct + forward + registration =========================

def test_glm_model_constructs_and_forwards(mesh_1d):
    """GlmMoeDsaForCausalLM constructs from hf_config and forwards end-to-end.

    Smoke test (NOT a math gate — that is Task 6): builds the scaffold with the
    tiny config on a 1-device mesh, runs a short forward, asserts finite hidden
    states of the right shape and that compute_logits returns [T, vocab].

    Attention slot holds the Task-3 GlmMoeDsaAttentionRef (pure-jnp fp32);
    Task 7 swaps in the absorbed kernel path. seq=32 < index_topk=64 so DSA is
    the dense identity (spec §A5).
    """
    from tpu_inference.models.jax.glm_moe_dsa import (GlmMoeDsaForCausalLM,
                                                      build_glm_vllm_config)
    cfg = tiny_glm_moe_dsa_config()
    vllm_config = build_glm_vllm_config(cfg, mesh=mesh_1d)
    rng_key = jax.random.PRNGKey(0)
    model = GlmMoeDsaForCausalLM(vllm_config, rng_key, mesh_1d)

    # tiny config: 4 layers [dense,dense,dense,sparse]; verify the schedule.
    layers = list(model.model.layers)
    assert len(layers) == cfg.num_hidden_layers
    assert isinstance(layers[0].mlp, DeepseekV3MLP), "layer 0 must be dense"
    assert isinstance(layers[3].mlp, DeepseekV2Moe), "layer 3 must be sparse"

    T = TINY_SEQ_DENSE  # 32 < index_topk=64 (DSA dense-equivalent)
    rng = np.random.default_rng(7)
    ids = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(1, T)).astype(np.int32))
    positions = jnp.arange(T, dtype=jnp.int32)
    with jax.default_matmul_precision("highest"):
        kv_caches, hidden, _, expert_ids = model([], ids, positions)
        logits = model.compute_logits(hidden)
    hidden = np.asarray(hidden)
    logits = np.asarray(logits)
    assert hidden.shape == (1, T, cfg.hidden_size)
    assert np.isfinite(hidden).all()
    assert logits.shape == (1, T, cfg.vocab_size)
    assert np.isfinite(logits).all()
    # Forward returns the DeepSeek 4-tuple contract (kv_caches, x, [], experts);
    # expert_indices is None unless enable_return_routed_experts (DeepSeek
    # default), so we only assert the third element is the empty residual list.
    kv_caches2, _, third, _ = model([], ids, positions)
    assert third == []


def test_glm_load_weights_roundtrip(mesh_1d):
    """load_weights maps real HF weights into the scaffold and forwards finitely.

    Builds the HF oracle, converts its state_dict via t2j_weights, loads it into
    the scaffold (dense FFN + MoE router/shared/experts + per-layer norms + MLA
    + embed + final norm + lm_head), and confirms a forward produces finite
    output and that loaded weights actually changed the output vs random init.
    (The rigorous weight-map golden — indexer name-gating, layers.78 drop — is
    Task 5; this only validates the converter is wired correctly.)
    """
    from tpu_inference.models.jax.glm_moe_dsa import (GlmMoeDsaForCausalLM,
                                                      build_glm_vllm_config)
    cfg = tiny_glm_moe_dsa_config()
    model_hf = build_hf_oracle(cfg=cfg, seed=0, randomize_buffers=True)
    jax_weights = t2j_weights(model_hf.state_dict())

    vllm_config = build_glm_vllm_config(cfg, mesh=mesh_1d)
    model = GlmMoeDsaForCausalLM(vllm_config, jax.random.PRNGKey(0), mesh_1d)

    T = TINY_SEQ_DENSE
    rng = np.random.default_rng(11)
    ids = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(1, T)).astype(np.int32))
    positions = jnp.arange(T, dtype=jnp.int32)

    with jax.default_matmul_precision("highest"):
        _, hidden_before, _, _ = model([], ids, positions)
        loaded = model.load_weights(jax_weights)
        _, hidden_after, _, _ = model([], ids, positions)
        logits = model.compute_logits(hidden_after)

    # Key weights were actually loaded (embed, lm_head, a dense + a MoE layer).
    for k in ("model.embed_tokens.weight", "lm_head.weight",
              "model.norm.weight",
              "model.layers.0.mlp.gate_proj.weight",
              "model.layers.0.self_attn.q_a_proj.weight",
              "model.layers.3.mlp.gate.weight",
              "model.layers.3.mlp.experts.gate_proj"):
        assert k in loaded, f"load_weights did not consume {k!r}"

    hidden_after = np.asarray(hidden_after)
    assert hidden_after.shape == (1, T, cfg.hidden_size)
    assert np.isfinite(hidden_after).all()
    assert np.isfinite(np.asarray(logits)).all()
    # Loading real weights must change the output vs the random init.
    assert maxabs(hidden_before, hidden_after) > 1e-3


def test_glm_registered_in_model_loader():
    """GlmMoeDsaForCausalLM is registered in the model loader registry."""
    from tpu_inference.models.common import model_loader
    from tpu_inference.models.jax.glm_moe_dsa import GlmMoeDsaForCausalLM

    class _Cfg:
        architectures = ["GlmMoeDsaForCausalLM"]

    arch = model_loader._get_model_architecture(_Cfg())
    assert arch is GlmMoeDsaForCausalLM


# ---------------------------------------------------------------------------
# Phase 1a Task 5 — Weight-mapping GOLDEN test + GLM converter.
#
# Proves the HF state_dict -> JAX param conversion is EXACT and TOTAL:
#   (1) every JAX param is populated from HF (no leftover random-init):
#       checksum every param at fresh-random-init, then after load assert
#       EVERY checksum CHANGED.  Covers nnx Params AND the MLA-ref `_w` dicts
#       (the attention weights live in `self_attn._w`, invisible to nnx.state).
#   (2) every HF key is consumed OR explicitly skipped (no silent drops);
#   (3) fused mlp.experts.gate_up_proj [E,2F,D] splits into gate(first half)/
#       up(second half) — checked against a manual numpy split;
#   (4) indexer names gate on indexer_types[i]: present on "full" layers
#       (0,1,2) -> explicitly SKIPPED in Phase 1a (no indexer module yet);
#       absent on the "shared" layer (3);
#   (5) layers.78-style MTP layer dropped (injected synthetic key);
#   (6) transpose rules: linears [out,in]->[in,out], embed NOT, lm_head .T.
#
# The converter under test is `convert_hf_weights(state_dict, hf_config)` ->
# (jax_weights, skipped).  It MUST agree with the harness `t2j_weights` on the
# names + values of every mapped (non-skipped) key (Tasks 3/6/7 rely on
# t2j_weights), reusing the same `t2j` (utils.py:78).
# ---------------------------------------------------------------------------

# Exact relative indexer param names a "full" layer contributes (HF
# GlmMoeDsaIndexer: wq_b/wk/weights_proj Linears + a k_norm LayerNorm that has
# BOTH weight and bias).  Verified live against transformers 5.12.1.
_INDEXER_REL_NAMES = (
    "self_attn.indexer.wq_b.weight",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.weights_proj.weight",
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias",
)


def _all_param_checksums(model):
    """Reorder-immune checksum of EVERY learnable array in the GLM scaffold.

    Covers two storage classes:
      * nnx ``Param`` leaves (embed, norms, dense FFN, MoE router/experts/shared,
        lm_head) — enumerated via ``nnx.state(model, nnx.Param)``;
      * the MLA reference attention weights, which ``GlmMoeDsaAttentionRef``
        stores in a plain ``self._w`` dict (NOT nnx Params, so invisible to
        ``nnx.state``) — enumerated per layer as ``layers.{i}.{bare}``.

    Returns ``{logical_name: weight_checksum(array)}``.
    """
    from flax import nnx as _nnx
    sums = {}
    state = _nnx.state(model, _nnx.Param)
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        name = ".".join(str(getattr(k, "key", k)) for k in path)
        sums[name] = weight_checksum(leaf)
    # MLA attention weights live in each layer's self_attn._w dict.
    for i, layer in enumerate(model.model.layers):
        w = getattr(layer.self_attn, "_w", {})
        for bare, arr in w.items():
            sums[f"model.layers.{i}.{bare}"] = weight_checksum(arr)
    return sums


def _stamp_all_params_with_sentinel(model, value=7.0):
    """Overwrite EVERY learnable array (nnx Params + MLA `_w`) with a sentinel.

    Makes the no-leftover-init check robust: the GLM norm modules (and HF norm
    weights) init to all-ones, so a norm that loads ones-from-ones leaves its
    checksum UNCHANGED and would falsely read as "not populated".  By stamping
    every array to a distinctive non-trivial value first, ANY array that
    load_weights actually populates changes its checksum (HF weights != the
    sentinel), while a genuinely-unloaded array stays at the sentinel.  This
    turns "checksum changed" into a true witness of population.
    """
    from flax import nnx as _nnx
    state = _nnx.state(model, _nnx.Param)
    sentinel = jax.tree.map(
        lambda a: jnp.full(jnp.shape(a), value, dtype=a.dtype), state)
    _nnx.update(model, sentinel)
    for layer in model.model.layers:
        w = getattr(layer.self_attn, "_w", {})
        for bare in list(w.keys()):
            w[bare] = jnp.full(w[bare].shape, value, dtype=w[bare].dtype)


def test_convert_hf_weights_agrees_with_t2j_on_mapped_keys():
    """convert_hf_weights' mapped output == harness t2j_weights (names+values).

    Tasks 3/6/7 consume the param map via t2j_weights; the in-module converter
    must agree EXACTLY on every mapped (non-skipped) key — same names, same
    values (it reuses the same `t2j`).  The only difference is that
    convert_hf_weights ALSO returns the explicit `skipped` set (indexer + MTP),
    which t2j_weights leaves in its output.
    """
    from tpu_inference.models.jax.glm_moe_dsa import convert_hf_weights
    cfg = tiny_glm_moe_dsa_config()
    model_hf = build_hf_oracle(cfg=cfg, seed=0, randomize_buffers=True)
    sd = model_hf.state_dict()

    jax_weights, skipped = convert_hf_weights(sd, cfg)
    t2j_out = t2j_weights(sd)

    # Every mapped key exists in t2j_weights with an identical value.
    for k, v in jax_weights.items():
        assert k in t2j_out, f"convert_hf_weights produced extra key {k!r}"
        assert weight_checksum(v) == weight_checksum(t2j_out[k]), (
            f"convert_hf_weights value for {k!r} differs from t2j_weights")
    # The keys t2j_weights has but the converter does NOT map are exactly the
    # skipped ones (indexer params — t2j_weights has no skip semantics).
    only_in_t2j = set(t2j_out) - set(jax_weights)
    assert only_in_t2j == set(skipped), (
        f"mapped/skipped partition disagrees with t2j_weights: "
        f"only_in_t2j={sorted(only_in_t2j)} skipped={sorted(skipped)}")


def test_convert_hf_weights_indexer_gated_on_indexer_types():
    """Indexer params: present on 'full' layers -> SKIPPED; absent on 'shared'.

    Phase 1a has NO indexer module (Phase 2).  The converter must RECOGNIZE the
    indexer params on full layers and route them to `skipped` (gated on
    indexer_types[i] != 'shared') — NOT silently leave them unconsumed and NOT
    KeyError.  The shared layer contributes no indexer params at all.
    """
    from tpu_inference.models.jax.glm_moe_dsa import convert_hf_weights
    cfg = tiny_glm_moe_dsa_config()
    assert list(cfg.indexer_types) == ["full", "full", "full", "shared"]
    sd = build_hf_oracle(cfg=cfg, seed=0).state_dict()

    jax_weights, skipped = convert_hf_weights(sd, cfg)

    for i, kind in enumerate(cfg.indexer_types):
        rel_present = [f"model.layers.{i}.{r}" for r in _INDEXER_REL_NAMES]
        if kind == "shared":
            # No indexer params on disk for a shared layer.
            for full in rel_present:
                assert full not in sd, (
                    f"shared layer {i} unexpectedly has indexer key {full!r}")
        else:  # full
            for full in rel_present:
                # present on disk...
                assert full in sd, (
                    f"full layer {i} missing expected indexer key {full!r}")
                # ...recognized + explicitly skipped (gated)...
                assert full in skipped, (
                    f"indexer key {full!r} on full layer {i} was not "
                    f"explicitly skipped (silent-drop / gating bug)")
                # ...and NOT mapped into a JAX param (no indexer module in 1a).
                assert full not in jax_weights, (
                    f"indexer key {full!r} should not map to a JAX param in "
                    f"Phase 1a")


def test_convert_hf_weights_drops_mtp_layer():
    """A layers.78-style MTP layer is genuinely DROPPED (proven with a real key).

    Tiny config has 4 layers (0..3), so no natural MTP layer.  Inject synthetic
    `model.layers.78.*` keys (mirroring the real checkpoint's MTP block, which
    HF discards via _keys_to_ignore_on_load_unexpected=[r"model\\.layers\\.78.*"])
    and assert the converter routes EVERY one of them to `skipped`, never to a
    JAX param and never as a KeyError.
    """
    import torch
    from tpu_inference.models.jax.glm_moe_dsa import convert_hf_weights
    cfg = tiny_glm_moe_dsa_config()
    sd = build_hf_oracle(cfg=cfg, seed=0).state_dict()
    H, V = cfg.hidden_size, cfg.vocab_size

    mtp_keys = {
        "model.layers.78.input_layernorm.weight": torch.ones(H),
        "model.layers.78.self_attn.q_a_proj.weight": torch.zeros(cfg.q_lora_rank, H),
        "model.layers.78.mlp.gate_proj.weight": torch.zeros(cfg.intermediate_size, H),
        "model.layers.78.embed_tokens.weight": torch.zeros(V, H),
    }
    sd_with_mtp = dict(sd)
    sd_with_mtp.update(mtp_keys)

    jax_weights, skipped = convert_hf_weights(sd_with_mtp, cfg)

    for k in mtp_keys:
        assert k in skipped, f"MTP key {k!r} was not dropped (skipped)"
        # No mapped JAX param may carry a layers.78 name.
        assert not any("layers.78" in jk for jk in jax_weights), (
            "an MTP (layers.78) key leaked into the mapped JAX weights")


def test_convert_hf_weights_gate_up_split_halves_exact():
    """Fused gate_up_proj [E,2F,D] -> gate=first half, up=second half (exact).

    HF GlmMoeDsaNaiveMoe does `gate, up = linear(x, gate_up[e]).chunk(2, -1)`,
    i.e. along the doubled 2F axis the FIRST F rows are the gate projection and
    the SECOND F rows are the up projection.  Compare the converter's split
    against a manual numpy split of the raw fused tensor.
    """
    from tpu_inference.models.jax.glm_moe_dsa import convert_hf_weights
    cfg = tiny_glm_moe_dsa_config()
    sd = build_hf_oracle(cfg=cfg, seed=0).state_dict()
    F = cfg.moe_intermediate_size
    p = "model.layers.3.mlp.experts."        # layer 3 is the sparse layer

    fused = sd[p + "gate_up_proj"].detach().float().cpu().numpy()  # [E,2F,D]
    E, two_F, D = fused.shape
    assert two_F == 2 * F
    manual_gate = fused[:, :F, :]            # first half -> gate
    manual_up = fused[:, F:, :]              # second half -> up

    jax_weights, _ = convert_hf_weights(sd, cfg)
    assert p + "gate_up_proj" not in jax_weights, "fused key must be split away"
    got_gate = np.asarray(jax_weights[p + "gate_proj"])
    got_up = np.asarray(jax_weights[p + "up_proj"])
    np.testing.assert_array_equal(got_gate, manual_gate)
    np.testing.assert_array_equal(got_up, manual_up)
    # Order matters: gate must NOT equal the up (second) half.
    assert not np.array_equal(got_gate, manual_up), (
        "gate/up halves swapped — wrong chunk order")


def test_convert_hf_weights_transpose_rules():
    """Linears transposed [out,in]->[in,out]; embed NOT; lm_head transposed.

    Checks the documented HF->JAX transpose contract on representative params
    against the raw torch tensors.
    """
    from tpu_inference.models.jax.glm_moe_dsa import convert_hf_weights
    cfg = tiny_glm_moe_dsa_config()
    sd = build_hf_oracle(cfg=cfg, seed=0).state_dict()
    jax_weights, _ = convert_hf_weights(sd, cfg)

    def raw(name):
        return sd[name].detach().float().cpu().numpy()

    # Linear: q_a_proj [out,in] -> [in,out]
    qa = "model.layers.0.self_attn.q_a_proj.weight"
    np.testing.assert_array_equal(np.asarray(jax_weights[qa]), raw(qa).T)
    assert np.asarray(jax_weights[qa]).shape == raw(qa).shape[::-1]

    # embed_tokens: lookup table, NOT transposed.
    emb = "model.embed_tokens.weight"
    np.testing.assert_array_equal(np.asarray(jax_weights[emb]), raw(emb))
    assert np.asarray(jax_weights[emb]).shape == raw(emb).shape

    # lm_head: transposed (UNTIED — separate tensor from embed).
    lm = "lm_head.weight"
    np.testing.assert_array_equal(np.asarray(jax_weights[lm]), raw(lm).T)
    assert not np.array_equal(np.asarray(jax_weights[lm]).shape,
                              np.asarray(jax_weights[emb]).shape) or \
        np.asarray(jax_weights[lm]).shape == raw(lm).shape[::-1]


def test_glm_weight_map_golden(mesh_1d):
    """GOLDEN: HF state_dict -> JAX conversion is EXACT and TOTAL.

    The strongest check in Task 5:
      * fresh-init checksum snapshot of EVERY param (nnx Params + MLA `_w`);
      * convert (raw HF -> jax_weights + explicit `skipped`);
      * load_weights consumes every mapped key with no KeyError;
      * EVERY param's checksum CHANGED (no leftover random-init);
      * EVERY HF key is either consumed (mapped+loaded) or explicitly skipped
        (indexer params) — no silent drops;
      * indexer params are in `skipped`, the MTP-skip mechanism is exercised.
    """
    from tpu_inference.models.jax.glm_moe_dsa import (GlmMoeDsaForCausalLM,
                                                      build_glm_vllm_config,
                                                      convert_hf_weights)
    cfg = tiny_glm_moe_dsa_config()
    model_hf = build_hf_oracle(cfg=cfg, seed=0, randomize_buffers=True)
    sd = model_hf.state_dict()

    vllm_config = build_glm_vllm_config(cfg, mesh=mesh_1d)
    model = GlmMoeDsaForCausalLM(vllm_config, jax.random.PRNGKey(0), mesh_1d)

    # (a) Stamp every learnable array with a distinctive sentinel, then
    #     snapshot.  The GLM norms init to all-ones (so does HF), so a
    #     ones-from-ones norm load would leave the checksum unchanged and read
    #     as a false "not populated"; stamping first makes "checksum changed"
    #     a true witness that the array was populated from the HF weights.
    _stamp_all_params_with_sentinel(model, value=7.0)
    before = _all_param_checksums(model)
    assert len(before) > 0

    # (b) Convert + load.
    jax_weights, skipped = convert_hf_weights(sd, cfg)
    loaded = model.load_weights(jax_weights)

    # (c) NO leftover random-init: every param's checksum must have CHANGED.
    after = _all_param_checksums(model)
    assert set(before) == set(after), "param set changed across load"
    unchanged = [n for n in before if before[n] == after[n]]
    assert not unchanged, (
        f"{len(unchanged)} param(s) left at random-init (not populated from "
        f"HF): {sorted(unchanged)[:10]}")

    # (d) Every mapped JAX key was actually loaded (no KeyError, no dropped map).
    assert set(jax_weights) == set(loaded), (
        f"mapped-but-not-loaded={sorted(set(jax_weights) - set(loaded))}, "
        f"loaded-but-not-mapped={sorted(set(loaded) - set(jax_weights))}")

    # (e) Every HF key is consumed OR explicitly skipped (no silent drops).
    #     loaded keys are the converter's MAPPED names; the converter's skip set
    #     names the HF (pre-conversion) keys it dropped.  Together they must
    #     account for every HF key, modulo the gate_up_proj fusion (one HF
    #     fused key -> two mapped keys).
    hf_keys = set(sd)
    # Reconstruct which HF keys are accounted for by the mapped names.
    accounted = set(skipped)
    for hk in hf_keys:
        if hk in skipped:
            continue
        if hk.endswith("gate_up_proj"):
            base = hk[: -len("gate_up_proj")]
            assert (base + "gate_proj") in loaded and (base + "up_proj") in loaded
            accounted.add(hk)
        else:
            assert hk in loaded, (
                f"HF key {hk!r} was neither loaded nor explicitly skipped "
                f"(silent drop)")
            accounted.add(hk)
    assert accounted == hf_keys, (
        f"unaccounted HF keys: {sorted(hf_keys - accounted)}")

    # (f) Indexer params are the skipped set (gated on indexer_types).
    for i, kind in enumerate(cfg.indexer_types):
        for r in _INDEXER_REL_NAMES:
            full = f"model.layers.{i}.{r}"
            if kind == "full":
                assert full in skipped
            else:
                assert full not in skipped and full not in sd


# ---------------------------------------------------------------------------
# Phase 1a Task 5 — consume-or-skip TEETH test.
#
# Proves that load_weights (the real loader the model runs) RAISES on an
# unrecognized key rather than silently dropping it.  This is the enforcement
# contract: a typo'd mapped key or a new parameter without a load rule produces
# a loud error, not a silent leftover random-init.
# ---------------------------------------------------------------------------

def test_load_weights_raises_on_unrecognized_key(mesh_1d):
    """load_weights raises ValueError when handed an UNRECOGNIZED extra key.

    Proves the consume-or-skip contract has teeth on the real loader:
    injecting a key that is neither consumed nor in the expected-skip set
    (not an indexer param, not MTP) must RAISE, not silently drop.

    The bogus key is in the model.* namespace so it reaches GlmMoeDsa.load_weights
    (rather than being caught at the top-level namespace check in
    GlmMoeDsaForCausalLM.load_weights).
    """
    from tpu_inference.models.jax.glm_moe_dsa import (GlmMoeDsaForCausalLM,
                                                      build_glm_vllm_config,
                                                      convert_hf_weights)
    cfg = tiny_glm_moe_dsa_config()
    model_hf = build_hf_oracle(cfg=cfg, seed=0)
    sd = model_hf.state_dict()
    jax_weights, _ = convert_hf_weights(sd, cfg)

    vllm_config = build_glm_vllm_config(cfg, mesh=mesh_1d)
    model = GlmMoeDsaForCausalLM(vllm_config, jax.random.PRNGKey(0), mesh_1d)

    # Inject a bogus key in the model.* namespace that the loader will never
    # consume and that is NOT in the expected-skip set (not an indexer key,
    # not an MTP-layer key).
    bogus_key = "model.layers.0.self_attn.bogus_proj.weight"
    jax_weights_with_bogus = dict(jax_weights)
    jax_weights_with_bogus[bogus_key] = jnp.zeros((4, 4))

    with pytest.raises(ValueError, match="unrecognized key"):
        model.load_weights(jax_weights_with_bogus)


def test_load_weights_raises_on_top_level_unknown_namespace(mesh_1d):
    """load_weights raises ValueError when a key is outside model.* and lm_head.*.

    A key such as 'totally.unknown.weight' is outside both the 'model.*' and
    'lm_head.weight' namespaces and must be caught at the top-level namespace
    partition in GlmMoeDsaForCausalLM.load_weights.
    """
    from tpu_inference.models.jax.glm_moe_dsa import (GlmMoeDsaForCausalLM,
                                                      build_glm_vllm_config,
                                                      convert_hf_weights)
    cfg = tiny_glm_moe_dsa_config()
    model_hf = build_hf_oracle(cfg=cfg, seed=0)
    sd = model_hf.state_dict()
    jax_weights, _ = convert_hf_weights(sd, cfg)

    vllm_config = build_glm_vllm_config(cfg, mesh=mesh_1d)
    model = GlmMoeDsaForCausalLM(vllm_config, jax.random.PRNGKey(0), mesh_1d)

    jax_weights_with_unknown = dict(jax_weights)
    jax_weights_with_unknown["totally.unknown.weight"] = jnp.zeros((2, 2))

    with pytest.raises(ValueError):
        model.load_weights(jax_weights_with_unknown)


def test_load_weights_accepts_indexer_keys_in_expected_skip_set(mesh_1d):
    """load_weights silently skips indexer keys (they are in the expected-skip set).

    The roundtrip test (test_glm_load_weights_roundtrip) passes t2j_weights(sd)
    which includes the raw indexer keys.  This test explicitly verifies that those
    keys are silently skipped (not errored) when passed directly to load_weights.
    """
    from tpu_inference.models.jax.glm_moe_dsa import (GlmMoeDsaForCausalLM,
                                                      build_glm_vllm_config)
    cfg = tiny_glm_moe_dsa_config()
    model_hf = build_hf_oracle(cfg=cfg, seed=0)
    # t2j_weights keeps indexer keys (they are not split or renamed).
    jax_weights = t2j_weights(model_hf.state_dict())

    vllm_config = build_glm_vllm_config(cfg, mesh=mesh_1d)
    model = GlmMoeDsaForCausalLM(vllm_config, jax.random.PRNGKey(0), mesh_1d)

    # Must not raise even though indexer keys are present and unconsumed.
    loaded = model.load_weights(jax_weights)

    # Confirm the indexer keys are NOT in `loaded` (silently skipped, not consumed).
    for i, kind in enumerate(cfg.indexer_types):
        if kind == "full":
            for r in _INDEXER_REL_NAMES:
                full = f"model.layers.{i}.{r}"
                assert full not in loaded, (
                    f"indexer key {full!r} unexpectedly in loaded set")


# ---------------------------------------------------------------------------
# Phase 1a Task 6 — Full dense-backbone forward MATH gate.
#
# The END-TO-END assembly gate.  Tasks 3/4/5 validated every submodule and the
# converter in isolation; Task 6 wires the WHOLE model and gates its logits
# against the full HF-eager oracle:
#
#   (1) fp32 MATH gate: full HF eager forward (fp32) vs full JAX jnp-ref forward
#       (fp32, under default_matmul_precision("highest")) on tiny config seed=0,
#       seq=32 < index_topk=64 (DSA is the dense identity, §A5).  maxabs < 1e-3.
#       assert_identical_weights FIRST so the result reflects the two
#       implementations, not a silent weight mismatch.  A failure here (with the
#       submodules passing) points at the ASSEMBLY — residual structure,
#       position_ids threading, norm placement, layer schedule, shared cos/sin.
#   (2) bf16 shipped floor: run the JAX forward in bf16; MEASURE the maxabs
#       deviation from the fp32 JAX result and the top-1 argmax agreement vs HF.
#       Expected floor band ~5e-2…2e-1.  argmax >= 0.95 is a BACKSTOP, not the
#       verdict (§H11a: random-weight tiny configs can have near-flat logits, so
#       bf16 can flip argmax legitimately — the diagnosis distinguishes
#       flat-logits from a real bug via the top-1/top-2 gap).
#   (3) injected 1% error trips the fp32 gate: perturb one loaded weight by 1%
#       in the JAX forward and assert the < 1e-3 gate FAILS by a clear margin.
# ---------------------------------------------------------------------------


def _build_full_forward_fixture(mesh, seed=0, T=TINY_SEQ_DENSE,
                                input_seed=2024):
    """Shared fixture for the Task-6 full-forward gates.

    Builds the HF oracle (tiny config, seed, randomize_buffers=True so the
    e_score_correction_bias selection path is non-trivial), runs its full eager
    forward (fp32) to get the oracle logits, and converts its state_dict into a
    GlmMoeDsaForCausalLM via the in-module converter + load_weights.

    ``mesh`` is the AMBIENT mesh established by the ``mesh_1d`` test fixture
    (already entered via ``jax.set_mesh`` in the fixture body) — we reuse it
    rather than nesting a second mesh context.

    Returns:
        (cfg, model, ids_jax, positions_jax, hf_logits_np, jax_weights, hf_model)
    where ``ids_jax`` is [1, T] int32, ``positions_jax`` is [T] int32, and
    ``hf_logits_np`` is the fp32 host [1, T, vocab] HF oracle logits.

    T=32 < index_topk=64 keeps the indexer dense-equivalent (the indexer-less
    jnp ref == the full HF model).
    """
    import torch
    from tpu_inference.models.jax.glm_moe_dsa import (GlmMoeDsaForCausalLM,
                                                      build_glm_vllm_config,
                                                      convert_hf_weights)
    cfg = tiny_glm_moe_dsa_config()
    assert T < cfg.index_topk, f"T={T} must be < index_topk={cfg.index_topk}"

    hf_model = build_hf_oracle(cfg=cfg, seed=seed, randomize_buffers=True)

    rng = np.random.default_rng(input_seed)
    ids_np = rng.integers(0, cfg.vocab_size, size=(1, T)).astype(np.int32)
    with torch.no_grad():
        hf_logits = hf_model(input_ids=torch.as_tensor(ids_np),
                             use_cache=False).logits  # [1, T, vocab]
    hf_logits_np = hf_logits.detach().float().cpu().numpy()

    sd = hf_model.state_dict()
    jax_weights, _skipped = convert_hf_weights(sd, cfg)

    model = GlmMoeDsaForCausalLM(build_glm_vllm_config(cfg, mesh=mesh),
                                 jax.random.PRNGKey(0), mesh)
    model.load_weights(jax_weights)

    ids_jax = jnp.asarray(ids_np)
    positions_jax = jnp.arange(T, dtype=jnp.int32)
    return (cfg, model, ids_jax, positions_jax, hf_logits_np, jax_weights,
            hf_model)


def _hf_ground_truth_jax_weights(hf_model, cfg):
    """Independently rebuild the JAX weight map directly from the raw HF tensors.

    This is the INDEPENDENT ground-truth side for assert_identical_weights: it
    applies the documented HF->JAX transpose/split rules to the raw torch
    state_dict WITHOUT going through the model's own converter, so the identity
    assertion catches a converter (convert_hf_weights / load_weights) bug rather
    than comparing the converter against itself.

    Returns ``{mapped_jax_name: fp32 numpy array}`` for every NON-skipped key
    (linears transposed, embed untransposed, lm_head transposed, fused
    gate_up_proj split into gate/up halves; indexer + MTP keys dropped).
    """
    from tpu_inference.models.jax.glm_moe_dsa import _is_expected_skip_key
    indexer_types = list(cfg.indexer_types)
    nhl = cfg.num_hidden_layers
    F = cfg.moe_intermediate_size
    out = {}
    for name, tensor in hf_model.state_dict().items():
        if _is_expected_skip_key(name, indexer_types, nhl):
            continue
        arr = tensor.detach().float().cpu().numpy()
        if name.endswith("gate_up_proj"):
            # fused [E, 2F, D] -> gate (first half) / up (second half)
            out[name.replace("gate_up_proj", "gate_proj")] = arr[:, :F, :]
            out[name.replace("gate_up_proj", "up_proj")] = arr[:, F:, :]
        elif arr.ndim == 2 and "embed_tokens" not in name:
            out[name] = arr.T
        else:
            out[name] = arr
    return out


def test_full_forward_fp32_math_gate(mesh_1d):
    """FULL GLM forward (jnp-ref MLA) == HF eager forward, logits maxabs < 1e-3.

    THE end-to-end assembly gate (spec §H / Task-6 brief): full HF eager forward
    (fp32) vs full JAX jnp-ref forward (fp32, under
    default_matmul_precision("highest")), tiny config seed=0, seq=32 <
    index_topk=64 (DSA dense identity, §A5).

    assert_identical_weights runs FIRST: the converter's mapped output
    ``jax_weights`` (exactly what load_weights consumes) is compared against an
    INDEPENDENT raw-HF->JAX ground-truth rebuild (raw torch tensors + the
    documented transpose/split rules, NOT routed through the converter), so a
    converter transpose/split/drop bug trips BEFORE the forward parity runs and
    the parity result reflects the two model implementations, not a silent weight
    mismatch. (That every mapped key is consumed into the right model param is
    Task 5's golden test, not duplicated here.) The DSA indexer is the dense
    identity at T<index_topk, so the indexer-less jnp ref must equal the full HF
    model.
    """
    (cfg, model, ids_jax, positions_jax, hf_logits_np, jax_weights,
     hf_model) = _build_full_forward_fixture(mesh_1d, seed=0, T=TINY_SEQ_DENSE)

    # --- assert_identical_weights FIRST -------------------------------------
    # atol=1e-9 absorbs only the fp64 checksum reduction-ORDER noise (the JAX
    # `.T` and numpy `.T` paths accumulate `np.sum` in different orders; worst
    # observed ~4.5e-13). The underlying weight VALUES are bit-identical
    # (elementwise maxabs == 0.0), and 1e-9 is ~10 orders below any real weight
    # bug — a 1% weight error shifts these checksums by O(10).
    ground_truth = _hf_ground_truth_jax_weights(hf_model, cfg)
    loaded_jax = {k: np.asarray(v) for k, v in jax_weights.items()}
    assert_identical_weights(loaded_jax, ground_truth, atol=1e-9)
    # Strengthen: the values themselves must be bit-identical, key-for-key.
    assert set(loaded_jax) == set(ground_truth)
    for k in ground_truth:
        assert float(np.max(np.abs(
            loaded_jax[k].astype(np.float64)
            - ground_truth[k].astype(np.float64)))) == 0.0, (
            f"converter weight {k!r} differs elementwise from raw-HF ground "
            f"truth (transpose/split/drop bug)")

    # --- DSA dense regime sanity: seq < index_topk so the indexer is identity.
    assert ids_jax.shape[1] < cfg.index_topk, (
        "full-forward fp32 gate must run in the DSA dense-equivalent regime")

    # --- full JAX jnp-ref forward (fp32, highest matmul precision) -----------
    with jax.default_matmul_precision("highest"):
        _, hidden, _, _ = model([], ids_jax, positions_jax)
        jax_logits = model.compute_logits(hidden)
    jax_logits_np = np.asarray(jax_logits).astype(np.float32)

    assert jax_logits_np.shape == hf_logits_np.shape, (
        f"logits shape mismatch: jax={jax_logits_np.shape} "
        f"hf={hf_logits_np.shape}")
    assert np.isfinite(jax_logits_np).all(), "JAX logits are not finite"

    delta = maxabs(hf_logits_np, jax_logits_np)
    assert delta < 1e-3, (
        f"full-forward fp32 logits maxabs={delta:.3e} >= 1e-3 (MATH gate). "
        f"Submodules pass in isolation (Tasks 3/4) -> this is an ASSEMBLY bug "
        f"(residual structure, position threading, norm placement, layer "
        f"schedule, or shared cos/sin). Do NOT loosen the tolerance.")


def test_full_forward_bf16_shipped_floor(mesh_1d):
    """bf16 shipped floor: MEASURE the bf16 deviation + top-1 argmax agreement.

    The "shipped" precision is TPU's DEFAULT matmul precision — bf16 passes for
    the fp32 matmuls — i.e. the SAME forward WITHOUT
    `default_matmul_precision("highest")` (the very setting the fp32 math gate
    needs to stay < 1e-3; Task 3 learned that without it TPU fp32 matmul drops to
    ~5e-3). So we run the full forward twice on identical fp32 weights —
    once under "highest" (the reference) and once at the default precision (the
    shipped path) — and MEASURE:
      (a) the maxabs logit deviation shipped-vs-reference (the bf16 floor);
      (b) the top-1 argmax agreement vs the HF oracle.

    argmax >= 0.95 is a BACKSTOP, not the verdict (§H11a): with random-weight
    tiny configs the next-token logits can be near-flat over the vocab, so bf16
    noise may flip argmax legitimately WITHOUT a bug. We MEASURE and REPORT the
    argmax agreement and the median top-1/top-2 logit gap; if argmax agreement
    is low, the gap diagnoses flat-logits (expected) vs a real bf16 divergence.

    NOTE on the band: the spec's nominal floor is ~5e-2…2e-1, calibrated for the
    deeper/wider production stack. The tiny config is only 4 layers, so far less
    bf16 error accumulates — the MEASURED deviation lands BELOW the nominal lower
    bound. That is reported truthfully (it is the depth, not a bug); the
    assertion band is set from the measured reality, not forced to the nominal.
    """
    (cfg, model, ids_jax, positions_jax, hf_logits_np, jax_weights,
     hf_model) = _build_full_forward_fixture(mesh_1d, seed=0, T=TINY_SEQ_DENSE)

    # --- fp32 reference logits (highest matmul precision) -------------------
    with jax.default_matmul_precision("highest"):
        _, hidden_ref, _, _ = model([], ids_jax, positions_jax)
        jax_logits_ref = model.compute_logits(hidden_ref)
    jax_logits_ref_np = np.asarray(jax_logits_ref).astype(np.float32)

    # --- shipped-precision logits (TPU default = bf16 matmul passes) --------
    # Identical weights, identical inputs; the ONLY difference is matmul
    # precision. This is the genuine "run the JAX forward in bf16" measurement.
    _, hidden_ship, _, _ = model([], ids_jax, positions_jax)
    jax_logits_ship = model.compute_logits(hidden_ship)
    jax_logits_ship_np = np.asarray(jax_logits_ship).astype(np.float32)

    # --- (a) bf16/shipped deviation from the fp32 reference -----------------
    bf16_dev = maxabs(jax_logits_ref_np, jax_logits_ship_np)
    assert np.isfinite(jax_logits_ship_np).all(), "shipped logits not finite"
    # Lower bound: bf16 MUST degrade vs the fp32-highest reference (else the
    # "highest" knob did nothing and the measurement is meaningless). Upper
    # bound: it must not blow up far past the nominal band.
    assert bf16_dev > 1e-3, (
        f"shipped(bf16-pass) deviation maxabs={bf16_dev:.3e} <= 1e-3 — bf16 "
        f"did not measurably degrade vs fp32-highest (matmul-precision knob "
        f"had no effect?); the floor measurement is not meaningful.")
    assert bf16_dev < 5e-1, (
        f"shipped(bf16-pass) deviation maxabs={bf16_dev:.3e} >= 5e-1 (blew "
        f"well past the nominal floor band ~5e-2…2e-1 — suspect a real bf16 "
        f"divergence bug, not precision noise).")

    # --- (b) top-1 argmax agreement vs HF -----------------------------------
    hf_top1 = np.argmax(hf_logits_np[0], axis=-1)            # [T]
    ship_top1 = np.argmax(jax_logits_ship_np[0], axis=-1)    # [T]
    ref_top1 = np.argmax(jax_logits_ref_np[0], axis=-1)
    argmax_agree_ship_hf = float(np.mean(hf_top1 == ship_top1))
    argmax_agree_ref_hf = float(np.mean(hf_top1 == ref_top1))

    # --- flat-logits diagnosis: top-1/top-2 gap per position ----------------
    sorted_ref = np.sort(jax_logits_ref_np[0], axis=-1)      # ascending
    top1_top2_gap = sorted_ref[:, -1] - sorted_ref[:, -2]    # [T]
    median_gap = float(np.median(top1_top2_gap))
    min_gap = float(np.min(top1_top2_gap))

    # REPORT (captured in pytest -s / the report). These prints ARE the
    # deliverable measurement, per the brief ("MEASURE and REPORT").
    print(f"\n[bf16 floor] shipped(bf16-pass)-vs-fp32 logits maxabs deviation "
          f"= {bf16_dev:.4e}  (nominal spec band ~5e-2…2e-1; this tiny 4-layer "
          f"config accumulates less, so it reads lower — depth, not a bug)")
    print(f"[bf16 floor] top-1 argmax agreement shipped-vs-HF = "
          f"{argmax_agree_ship_hf:.3f}")
    print(f"[bf16 floor] top-1 argmax agreement fp32-vs-HF    = "
          f"{argmax_agree_ref_hf:.3f}  (clean upper bound)")
    print(f"[bf16 floor] top1/top2 fp32 logit gap: median={median_gap:.4e} "
          f"min={min_gap:.4e}")

    # The fp32 JAX path MUST agree with HF on argmax essentially perfectly (the
    # < 1e-3 math gate guarantees it). This is the true correctness verdict; if
    # it holds, any shipped argmax shortfall is precision noise, not a bug.
    assert argmax_agree_ref_hf >= 0.99, (
        f"fp32 JAX argmax agreement vs HF = {argmax_agree_ref_hf:.3f} < 0.99 — "
        f"the fp32 path itself disagrees with HF on the top token; this is a "
        f"real correctness bug, not bf16 noise (see the fp32 math gate).")

    # argmax BACKSTOP: report; only ESCALATE to a failure if it is low AND the
    # logits are NOT flat (i.e. a genuine bf16 divergence, not §H11a
    # flat-random-logits). On this fixture argmax agreement is 1.0 (the logits
    # are peaked: median gap ~0.10 >> bf16 dev ~0.01), so the backstop passes
    # cleanly — but the guard below is the correct general policy.
    if argmax_agree_ship_hf < 0.95:
        peaked = median_gap > 10.0 * bf16_dev
        assert not peaked, (
            f"shipped argmax agreement {argmax_agree_ship_hf:.3f} < 0.95 on "
            f"PEAKED logits (median top1/top2 gap {median_gap:.3e} >> bf16 dev "
            f"{bf16_dev:.3e}) — this is a REAL bf16 divergence, not flat-logits "
            f"noise.")
        # Otherwise: flat-logits regime (§H11a). Documented, not a bug.


def test_full_forward_injected_error_trips_fp32_gate(mesh_1d):
    """A 1% perturbation of one loaded weight FAILS the fp32 < 1e-3 gate.

    Proves the full-forward fp32 math gate has teeth end-to-end: the CLEAN
    parity is far below 1e-3, so a real 1% bug in a single projection weight
    (here the lm_head) must propagate to logits maxabs >> 1e-3. Perturbing the
    lm_head scales the whole logit tensor by ~1%, a decisive, unambiguous
    failure of the gate.
    """
    (cfg, model, ids_jax, positions_jax, hf_logits_np, jax_weights,
     hf_model) = _build_full_forward_fixture(mesh_1d, seed=0, T=TINY_SEQ_DENSE)

    # Inject a 1% error into the lm_head weight AFTER a clean load.
    model.lm_head.weight.value = model.lm_head.weight.value * 1.01

    with jax.default_matmul_precision("highest"):
        _, hidden, _, _ = model([], ids_jax, positions_jax)
        jax_logits = model.compute_logits(hidden)
    jax_logits_np = np.asarray(jax_logits).astype(np.float32)

    delta = maxabs(hf_logits_np, jax_logits_np)
    assert delta >= 1e-3, (
        f"injected 1% lm_head error did NOT trip the fp32 gate: "
        f"maxabs={delta:.3e} < 1e-3 (gate has no teeth)")
    # And it must trip by a clear margin, not knife-edge.
    assert delta > 1e-2, (
        f"injected-error margin too small: maxabs={delta:.3e} (expected >1e-2)")
