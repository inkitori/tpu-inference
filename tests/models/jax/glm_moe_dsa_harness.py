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
"""Phase 0 test harness for GLM 5.2 (GlmMoeDsa) bring-up.

This module is the shared scaffolding for the GLM/DSA validation ladder
(spec `docs/superpowers/specs/glm5.2-dsa/` — see `phases/phase-0.md` + `core.md` §H):

  * `tiny_glm_moe_dsa_config` - a min-viable `GlmMoeDsaConfig` that still
    exercises every real code path (>=1 dense + >=1 MoE layer, >=1 "full" and
    >=1 "shared" indexer layer).
  * `build_hf_oracle` - the in-tree `transformers GlmMoeDsaForCausalLM` eager
    forward (torch, CPU, random weights) used as the independent math oracle.
  * `t2j_weights` - HF state_dict -> JAX weight converter (transpose rules,
    fused-`gate_up_proj` split). Built on `tpu_inference.utils.t2j`.
  * `maxabs` / `weight_checksum` / `assert_identical_weights` - comparison
    primitives for the triangulated parity gates.
  * `make_glm_mesh` - builds a JAX `Mesh` over N devices on the canonical
    6-axis `MESH_AXIS_NAMES` the DeepSeek/GLM model code keys against.

Pure-jnp helpers avoid importing torch/transformers at module load; the oracle
builders import them lazily so the cheap-helper tests stay fast.
"""
from __future__ import annotations

import jax
import numpy as np
from jax import numpy as jnp

from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.utils import make_optimized_mesh, t2j

# --- tiny-config constants (spec §6 Phase 0) ---------------------------------
TINY_PAGE_SIZE = 16
TINY_SEQ_DENSE = 32   # <= index_topk=64: DSA == dense (spec §2 equivalence)
TINY_SEQ_SPARSE = 128  # > index_topk=64: true sparsity regime

# Min-viable GlmMoeDsaConfig. Verified live against transformers 5.12.1:
# yields mlp_layer_types=[dense,dense,dense,sparse] (first_k_dense_replace=3)
# and indexer_types=[full,full,full,shared] (freq=4/offset=3). The MLA/indexer
# head dims are kept at the *real* checkpoint values (they auto-pad to the same
# tiles, so they cost ~nothing) per spec §6.
_TINY_CONFIG_KWARGS = dict(
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=8,
    index_topk=64,
    n_routed_experts=8,
    num_experts_per_tok=2,
    index_topk_freq=4,        # kwargs-only: drives indexer_types in __post_init__
    index_skip_topk_offset=3,  # (not persisted as an attribute)
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    qk_nope_head_dim=192,
    v_head_dim=256,
    index_head_dim=128,
    q_lora_rank=2048,
    index_n_heads=32,
    vocab_size=1024,
    rope_type="default",       # plain RoPE, no YaRN (spec §3/§9)
)


def tiny_glm_moe_dsa_config(**overrides):
    """Build the min-viable tiny `GlmMoeDsaConfig` (spec §6 Phase 0)."""
    from transformers import GlmMoeDsaConfig
    kwargs = dict(_TINY_CONFIG_KWARGS)
    kwargs.update(overrides)
    return GlmMoeDsaConfig(**kwargs)


def _randomize_selection_bias(model, *, seed, scale):
    """Fill every `e_score_correction_bias` buffer with seeded noise.

    HF zero-initializes this buffer, so without this the router's
    bias-for-selection path (gathered weights use the *bias-free* sigmoid
    scores; selection uses the biased ones) is exercised vacuously.
    """
    import torch
    gen = torch.Generator().manual_seed(int(seed) + 1)
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                noise = torch.randn(buf.shape, generator=gen,
                                    dtype=torch.float32) * scale
                buf.copy_(noise.to(buf.dtype))


def build_hf_oracle(cfg=None, *, seed=0, randomize_buffers=True,
                    buffer_scale=5.0):
    """Build the HF-eager `GlmMoeDsaForCausalLM` oracle (torch, CPU, fp32).

    `experts_implementation="eager"` is mandatory: the default resolves to a
    fused `grouped_mm` with no CPU fallback; "eager" forces the genuine
    per-expert `GlmMoeDsaNaiveMoe` loop. Must go through `_from_config` (there
    is no public `from_config`). Deterministic for a fixed `seed`.
    """
    import torch
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import \
        GlmMoeDsaForCausalLM
    if cfg is None:
        cfg = tiny_glm_moe_dsa_config()
    torch.manual_seed(int(seed))
    model = GlmMoeDsaForCausalLM._from_config(
        cfg, attn_implementation="eager", experts_implementation="eager")
    model = model.to(torch.float32).eval()
    if randomize_buffers:
        _randomize_selection_bias(model, seed=seed, scale=buffer_scale)
    return model


def _to_fp32_numpy(x) -> np.ndarray:
    """Coerce a jax / numpy / torch array to a host fp32 numpy array.

    A bf16 TPU output returns as numpy bf16 (ml_dtypes); upcasting here is
    load-bearing so the comparison itself is done in fp32, not bf16.
    """
    if type(x).__module__.startswith("torch"):
        import torch
        return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x).astype(np.float32)


def maxabs(a, b) -> float:
    """Max abs elementwise difference, upcasting both sides to fp32 first."""
    return float(np.max(np.abs(_to_fp32_numpy(a) - _to_fp32_numpy(b))))


def t2j_weights(state_dict):
    """Convert an HF `state_dict` to JAX weights.

    Conventions (spec §6 / §5 weight-names): every 2-D linear weight is
    transposed (HF `nn.Linear` is `(out, in)`; JAX `x @ kernel` wants
    `(in, out)`) EXCEPT `embed_tokens` (a lookup table, not a matmul);
    `lm_head` transposes like any linear; the fused experts `gate_up_proj`
    `[E, 2*moe_inter, hidden]` is split along the doubled axis into
    `gate_proj`/`up_proj`. 1-D params (norms, biases) pass through.

    The 3-D expert einsum-axis layout is reconciled against the real JAX MoE
    loader in Phase 1 (weight-map golden test); this converter implements the
    documented, model-independent transpose/split rules.
    """
    out = {}
    for name, tensor in state_dict.items():
        arr = t2j(tensor)
        if name.endswith("gate_up_proj") or name.endswith("gate_up_proj.weight"):
            axis = -2 if arr.ndim == 3 else 0
            gate, up = jnp.split(arr, 2, axis=axis)
            out[name.replace("gate_up_proj", "gate_proj")] = gate
            out[name.replace("gate_up_proj", "up_proj")] = up
        elif arr.ndim == 2 and "embed_tokens" not in name:
            out[name] = arr.T
        else:
            out[name] = arr
    return out


def weight_checksum(arr):
    """Reorder-immune `(sum, sqsum, absmax)` checksum of an array.

    Reductions accumulate in fp64 so the triple is permutation-invariant (it
    must agree across 1-dev vs N-dev loads with different reduction orders).
    This is the spec §7 `[gsum, sqsum, absmax]` corruption probe.
    """
    a = _to_fp32_numpy(arr).astype(np.float64)
    return (float(a.sum()), float((a * a).sum()), float(np.abs(a).max()))


def assert_identical_weights(a, b, *, atol=0.0):
    """Assert two name->array weight maps are identical by per-key checksum.

    Asserted before every parity run so a parity result reflects the two
    implementations, not a silent weight mismatch.
    """
    keys_a, keys_b = set(a), set(b)
    assert keys_a == keys_b, (
        f"weight key mismatch: only in a={sorted(keys_a - keys_b)}, "
        f"only in b={sorted(keys_b - keys_a)}")
    for k in a:
        ca, cb = weight_checksum(a[k]), weight_checksum(b[k])
        for stat, x, y in zip(("sum", "sqsum", "absmax"), ca, cb):
            assert abs(x - y) <= atol, (
                f"weight {k!r} {stat} differs: {x} vs {y} (atol={atol})")


# --- medium-config constants (spec §B11) -------------------------------------
MEDIUM_PAGE_SIZE = 128
MEDIUM_SEQ_DENSE = 2040   # <= index_topk=2048: DSA == dense (spec §B11)
MEDIUM_SEQ_SPARSE = 3000  # > index_topk=2048: true sparsity regime

# Medium config: real per-layer dims, reduced depth/experts (spec §B11).
# hidden=6144 matches production; CPU-eager-feasible with layers=6.
_MEDIUM_CONFIG_KWARGS = dict(
    hidden_size=6144,
    num_hidden_layers=6,
    num_attention_heads=64,
    num_key_value_heads=64,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    qk_nope_head_dim=192,
    v_head_dim=256,
    index_head_dim=128,
    index_n_heads=32,
    index_topk=2048,
    index_topk_freq=4,          # kwargs-only: drives indexer_types in __post_init__
    index_skip_topk_offset=3,   # (not persisted as an attribute)
    n_routed_experts=16,
    num_experts_per_tok=8,
    moe_intermediate_size=512,
    vocab_size=2048,
    rope_type="default",        # plain RoPE, no YaRN
)


def medium_glm_moe_dsa_config(**overrides):
    """Build the §B11 medium `GlmMoeDsaConfig` (real per-layer dims, reduced depth)."""
    from transformers import GlmMoeDsaConfig
    kwargs = dict(_MEDIUM_CONFIG_KWARGS)
    kwargs.update(overrides)
    return GlmMoeDsaConfig(**kwargs)


def build_hf_decode_oracle(input_ids, decode_ids, *, cfg=None, seed=0,
                           randomize_buffers=False):
    """Run HF-eager stepped decode; return per-step last-token logits.

    Prefills `input_ids` (shape `[1, prompt_len]`), then steps through each
    token in `decode_ids` (shape `[1, decode_steps]`) one at a time, threading
    a growing `DynamicCache`. Returns a list of `[1, vocab_size]` fp32 tensors
    (one per decode step).

    The decode cursor flows via `position_ids` offset by
    `past_key_values.get_seq_length()` — there is no `cache_position` kwarg in
    this model. `DynamicCache` is constructed with `config=model.config` so it
    auto-wires per-layer cache types (DSA layers get `DynamicIndexedLayer`).
    """
    import torch
    from transformers.cache_utils import DynamicCache

    if cfg is None:
        cfg = tiny_glm_moe_dsa_config()
    model = build_hf_oracle(cfg=cfg, seed=seed,
                            randomize_buffers=randomize_buffers)
    model.eval()

    with torch.no_grad():
        # --- prefill ---
        cache = DynamicCache(config=model.config)
        prompt_len = input_ids.shape[1]
        position_ids = torch.arange(prompt_len, dtype=torch.long).unsqueeze(0)
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values

        # --- decode steps ---
        per_step_logits = []
        decode_steps = decode_ids.shape[1]
        for step in range(decode_steps):
            past_len = cache.get_seq_length()
            tok = decode_ids[:, step : step + 1]           # (1, 1)
            pos = torch.tensor([[past_len]], dtype=torch.long)
            out = model(
                input_ids=tok,
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
            )
            # last-token logits, fp32
            last_logits = out.logits[:, -1, :].to(torch.float32)
            per_step_logits.append(last_logits)
            cache = out.past_key_values

    return per_step_logits


def make_glm_mesh(num_devices=None, axis_shapes=None, *, devices=None):
    """Build a `Mesh` over N devices on the canonical 6-axis layout.

    Uses `MESH_AXIS_NAMES` (`data, attn_dp, attn_dp_expert, expert, model,
    dcp`) so the mesh is forward-compatible with the DeepSeek/GLM model code's
    `ShardingAxisName` specs (unlike conftest.py's reduced 4-axis `mesh`).
    Defaults to placing all devices on the `model` axis; pass `axis_shapes`
    for a specific geometry (e.g. the Phase-1b S1 stress fixture).
    """
    if devices is None:
        devices = jax.local_devices()
    devices = list(devices)
    if num_devices is None:
        num_devices = len(devices)
    devices = devices[:num_devices]
    if axis_shapes is None:
        shapes = [1] * len(MESH_AXIS_NAMES)
        shapes[MESH_AXIS_NAMES.index("model")] = num_devices
        axis_shapes = tuple(shapes)
    assert int(np.prod(axis_shapes)) == num_devices, (
        f"axis_shapes {axis_shapes} product != num_devices {num_devices}")
    return make_optimized_mesh(tuple(axis_shapes), MESH_AXIS_NAMES,
                               devices=devices)
