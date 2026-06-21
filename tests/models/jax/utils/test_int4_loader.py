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
"""Tests for the uint32-aware (MLX packed int4) weight loader path.

Task 6: the base loader must NOT corrupt MLX-packed int4 weights. For an
MLX-packed ``.weight`` (uint32, with a ``.scales`` sibling) it must skip the
bf16 cast, skip the transpose, keep uint32, and leave the leading expert dim
intact. ``.scales``/``.biases`` pass straight through as bf16. All non-int4
behavior must remain identical.
"""

from pathlib import Path
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from tests.utils.mlx_synthetic import build_synthetic_mlx_moe
from tpu_inference.models.jax.utils.weight_utils import (
    MetadataMap, _load_and_shard_weight, model_weights_single_file_generator)

# Where the synthetic MLX checkpoint puts the stacked expert gate_proj weight.
GATE_W = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
GATE_S = "model.layers.0.mlp.switch_mlp.gate_proj.scales"
GATE_B = "model.layers.0.mlp.switch_mlp.gate_proj.biases"
NORM_W = "model.layers.0.input_layernorm.weight"


def _mesh() -> Mesh:
    devices = jax.local_devices()
    return Mesh(np.asarray(devices).reshape(len(devices)),
                axis_names=("model", ))


def _read_synth(tmp_path: Path) -> dict[str, jax.Array]:
    """Build a synthetic MLX MoE checkpoint and read the raw on-disk tensors."""
    build_synthetic_mlx_moe(tmp_path,
                            layers=1,
                            experts=8,
                            hidden=128,
                            moe_inter=64)
    weights_file = str(tmp_path / "model.safetensors")
    return {
        name: w
        for name, w in model_weights_single_file_generator(weights_file,
                                                            framework="flax")
    }


def _vllm_config(*, int4: bool):
    """A minimal vllm_config; ``int4`` toggles whether Int4Config is active.

    Mirrors the dispatch in ``get_tpu_quantization_config``: an MLX int4 repo
    advertises ``hf_config.quantization = {"group_size", "bits"}``.
    """
    hf_config = MagicMock()
    hf_config.text_config = None
    hf_config.quantization = {"group_size": 64, "bits": 4} if int4 else None
    hf_config.quantization_config = {}

    model_config = MagicMock()
    model_config.hf_config = hf_config
    model_config.quantization = None
    model_config.is_multimodal_model = False
    model_config.dtype = jnp.bfloat16
    model_config.get_head_size.return_value = 128  # -> head_dim_pad == 0
    model_config.get_hidden_size.return_value = 128

    vllm_config = MagicMock()
    vllm_config.model_config = model_config
    return vllm_config


def _make_param_model(specs: dict[str, tuple]):
    """Build an nnx model holding params at the exact dotted paths in ``specs``.

    ``specs`` maps "dotted.path" -> (shape, dtype). Returns (params, shardings).
    """

    class _Holder(nnx.Module):
        pass

    root = _Holder()
    for path, (shape, dtype) in specs.items():
        parts = path.split(".")
        node = root
        for p in parts[:-1]:
            if not hasattr(node, p):
                setattr(node, p, _Holder())
            node = getattr(node, p)
        setattr(node, parts[-1], nnx.Param(jnp.zeros(shape, dtype=dtype)))

    params = nnx.state(root)
    mesh = _mesh()
    try:
        shardings = nnx.get_named_sharding(params, mesh)
    except TypeError:
        shardings = params
    return params, shardings


def _load(vllm_config, params, shardings, metadata_map, hf_key, hf_weight):
    _load_and_shard_weight(
        vllm_config,
        params,
        shardings,
        metadata_map,
        _mesh(),
        hf_key,
        hf_weight,
        keep_hf_weight_suffix_when_match=[],
    )


def test_packed_expert_weight_preserved_as_uint32_not_transposed(tmp_path):
    """A stacked expert .weight loads as uint32 [E, out, in/8], not transposed.

    The transpose_map contains "gate_proj": (1, 0); the on-disk tensor is 3D
    uint32 [E, out, in/8]. Without the int4 guard the loader would cast to bf16
    and attempt the transpose (which is wrong for a packed expert tensor).
    """
    weights = _read_synth(tmp_path)
    hf_w = weights[GATE_W]
    hf_s = weights[GATE_S]
    hf_b = weights[GATE_B]
    assert hf_w.dtype == jnp.uint32 and hf_w.shape == (8, 64, 16)

    vllm_config = _vllm_config(int4=True)
    # Target params sized to the *preserved* (uint32, untransposed) shapes.
    specs = {
        "gate_proj.kernel": (hf_w.shape, jnp.uint32),
        "gate_proj.scales": (hf_s.shape, jnp.bfloat16),
        "gate_proj.biases": (hf_b.shape, jnp.bfloat16),
    }
    params, shardings = _make_param_model(specs)
    name_map = {
        "model.layers.*.mlp.switch_mlp.gate_proj": "gate_proj.kernel",
        "model.layers.*.mlp.switch_mlp.gate_proj.scales": "gate_proj.scales",
        "model.layers.*.mlp.switch_mlp.gate_proj.biases": "gate_proj.biases",
    }
    metadata_map = MetadataMap(name_map=name_map,
                               transpose_map={"gate_proj": (1, 0)})

    _load(vllm_config, params, shardings, metadata_map, GATE_W, hf_w)
    _load(vllm_config, params, shardings, metadata_map, GATE_S, hf_s)
    _load(vllm_config, params, shardings, metadata_map, GATE_B, hf_b)

    kernel = params["gate_proj"]["kernel"].value
    scales = params["gate_proj"]["scales"].value
    biases = params["gate_proj"]["biases"].value

    # Packed weight: uint32, shape [E, out, in/8], leading expert dim intact.
    assert kernel.dtype == jnp.uint32, f"expected uint32, got {kernel.dtype}"
    assert kernel.shape == (8, 64, 16), f"shape changed: {kernel.shape}"
    np.testing.assert_array_equal(np.asarray(kernel), np.asarray(hf_w))

    # scales/biases present and bf16, unchanged.
    assert scales.dtype == jnp.bfloat16
    assert biases.dtype == jnp.bfloat16
    assert scales.shape == hf_s.shape
    assert biases.shape == hf_b.shape
    np.testing.assert_array_equal(np.asarray(scales).view(np.uint16),
                                  np.asarray(hf_s).view(np.uint16))
    np.testing.assert_array_equal(np.asarray(biases).view(np.uint16),
                                  np.asarray(hf_b).view(np.uint16))


def test_non_candidate_weight_still_cast_and_transposed_with_int4_active(
        tmp_path):
    """Scoping proof: a non-candidate .weight is cast+transposed even under int4.

    With Int4Config ACTIVE, a plain bf16 2D ``.weight`` (NOT uint32, no
    .scales sibling) is not an MLX-packed tensor, so the guard must NOT fire:
    it still gets cast to model dtype AND transposed by the transpose_map,
    exactly as before. This is the mutation-sensitive control -- if the guard
    over-fired (e.g. on any .weight while int4 is active) the dtype would stay
    bf16 and the tensor would not be transposed, and these asserts would fail.

    We build the bf16 source by dequantizing one expert of the synthetic
    checkpoint, giving a real [in, out] bf16 matrix to transpose.
    """
    weights = _read_synth(tmp_path)
    # A real 2D bf16 weight, deliberately NOT a packed/uint32 tensor.
    src = jnp.asarray(np.asarray(weights[GATE_S][0]))  # [out, g] bf16, 2D
    assert src.dtype == jnp.bfloat16 and src.ndim == 2
    in_dim, out_dim = src.shape

    vllm_config = _vllm_config(int4=True)  # Int4Config IS active
    vllm_config.model_config.dtype = jnp.float32  # observable cast target

    # Param sized to the TRANSPOSED, cast shape -> proves transpose ran.
    specs = {"down_proj.kernel": ((out_dim, in_dim), jnp.float32)}
    params, shardings = _make_param_model(specs)
    name_map = {"model.layers.*.mlp.down_proj": "down_proj.kernel"}
    metadata_map = MetadataMap(name_map=name_map,
                               transpose_map={"down_proj": (1, 0)})

    hf_key = "model.layers.0.mlp.down_proj.weight"
    _load(vllm_config, params, shardings, metadata_map, hf_key, src)

    kernel = params["down_proj"]["kernel"].value
    # Guard did NOT fire: cast to f32 AND transposed (shape [out, in]).
    assert kernel.dtype == jnp.float32, f"non-candidate not cast: {kernel.dtype}"
    assert kernel.shape == (out_dim, in_dim), f"not transposed: {kernel.shape}"
    np.testing.assert_allclose(np.asarray(kernel),
                               np.asarray(src).astype(np.float32).T)


def test_no_int4_config_leaves_packed_path_disabled(tmp_path):
    """When Int4Config is NOT active, the guard must NOT engage on uint32.

    Feeds the packed uint32 expert weight with NO Int4Config active. Because
    the guard keys off an active Int4Config, it must stay off here, so the
    ordinary path runs and casts the uint32 codes to bf16 (the very corruption
    the int4 path exists to prevent). Asserting the cast happened proves the
    guard is scoped to int4 -- if the guard fired regardless of config, the
    dtype would remain uint32 and this would fail. (transpose_map is empty so
    the only observable side effect is the cast.)
    """
    weights = _read_synth(tmp_path)
    hf_w = weights[GATE_W]
    assert hf_w.dtype == jnp.uint32

    vllm_config = _vllm_config(int4=False)  # Int4Config NOT active
    specs = {"gate_proj.kernel": (hf_w.shape, jnp.bfloat16)}
    params, shardings = _make_param_model(specs)
    name_map = {"model.layers.*.mlp.switch_mlp.gate_proj": "gate_proj.kernel"}
    metadata_map = MetadataMap(name_map=name_map)  # no transpose

    _load(vllm_config, params, shardings, metadata_map, GATE_W, hf_w)

    kernel = params["gate_proj"]["kernel"].value
    # Guard stayed off -> ordinary cast to model dtype (bf16) ran.
    assert kernel.dtype == jnp.bfloat16, f"guard wrongly fired: {kernel.dtype}"
    assert kernel.shape == hf_w.shape
