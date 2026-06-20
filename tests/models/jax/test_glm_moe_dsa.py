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
