# Copyright 2025 Google LLC
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
"""Fast CPU-only method-level tests for ``VllmMLXLinearMethod``.

These cover what the einsum-only ``test_mlx_linear.py`` and the TPU e2e cannot:
the real ``create_weights`` param shapes/dtypes and the fused-projection
``apply`` slice/concat split path (``process_weights_after_loading`` reorder +
``apply`` slice/concat), end-to-end against a golden ``x @ W.T``.

CPU-only: ``JAX_PLATFORMS=cpu`` is forced before any jax import so ``jax.devices()``
yields a single CPU device; no TPU and no ``LLM(...)``. The method's
``process_weights_after_loading`` and ``apply`` both run JAX ops
(``jax.device_put`` with a NamedSharding, ``mlx_dequantize``, einsum) -- these run
on the single-device CPU mesh, so every assertion below (including the
fused-split apply comparison) runs on CPU.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import tempfile  # noqa: E402
import types  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torchax  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402
from torchax.interop import jax_view, torch_view  # noqa: E402
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed.parallel_state import (  # noqa: E402
    ensure_model_parallel_initialized, init_distributed_environment)

from tests.layers.common import utils as test_utils  # noqa: E402
from tests.utils.mlx_synthetic import _quantize_affine  # noqa: E402
from tpu_inference.layers.common.sharding import ShardingAxisName  # noqa: E402
from tpu_inference.layers.vllm.quantization.mlx import (  # noqa: E402
    VllmMLXConfig, VllmMLXLinearMethod)

GROUP_SIZE = 64
BITS = 4
PACK_FACTOR = 32 // BITS  # 8 nibbles / uint32
IN_FEATURES = 128  # divisible by both group_size (64) and pack_factor (8)


@pytest.fixture(scope="module", autouse=True)
def _dist_init():
    """PackedvLLMParameter reads the TP rank/size, which require an initialized
    (single-rank, gloo) distributed env. A bare VllmConfig() avoids any model
    download; this keeps the test self-contained and fast (no LLM)."""
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{tempfile.mkstemp()[1]}",
            backend="gloo")
        ensure_model_parallel_initialized(1, 1)
        yield


def _bf16_to_torch(arr: np.ndarray) -> torch.Tensor:
    # `_quantize_affine` returns ml_dtypes.bfloat16 numpy arrays that
    # torch.from_numpy cannot ingest; reinterpret the identical 16-bit pattern
    # as torch.bfloat16.
    return torch.from_numpy(np.ascontiguousarray(arr).view(np.uint16)).view(
        torch.bfloat16)


def _make_linear_config(mesh, output_sizes, num_proj, n_shards,
                        weight_sharding):
    """Duck-typed stand-in for VllmQuantLinearConfig exposing only the attrs
    VllmMLXLinearMethod reads (mesh, weight_sharding, output_sizes, n_shards,
    num_proj, fuse_matmuls)."""
    return types.SimpleNamespace(
        mesh=mesh,
        weight_sharding=weight_sharding,
        bias_sharding=P(weight_sharding[0]),
        output_sizes=list(output_sizes),
        n_shards=n_shards,
        num_proj=num_proj,
        fuse_matmuls=True,
    )


def _load_and_process(method, layer, packed, scales, biases):
    layer.weight.data = torch.from_numpy(packed.astype(np.uint32))
    layer.scales.data = _bf16_to_torch(scales)
    layer.biases.data = _bf16_to_torch(biases)
    method.process_weights_after_loading(layer)


def _apply_np(method, layer, x_np):
    with torchax.default_env():
        y = method.apply(layer, torch_view(jnp.asarray(x_np)))
        return np.asarray(jax_view(y)).astype(np.float32)


def test_create_weights_shapes_and_dtypes():
    """create_weights registers exactly the three MLX params with the packed/
    affine shapes and dtypes from the spec: weight uint32 [out, in//8],
    scales/biases bf16 [out, in//64]."""
    out = 96
    mesh = test_utils.get_spmd_mesh(1)
    cfg = _make_linear_config(mesh, [out], num_proj=1, n_shards=1,
                              weight_sharding=P(None,
                                                ShardingAxisName.ATTN_HEAD))
    method = VllmMLXLinearMethod(VllmMLXConfig(group_size=GROUP_SIZE,
                                               bits=BITS), cfg)
    layer = torch.nn.Module()
    method.create_weights(layer,
                          input_size_per_partition=IN_FEATURES,
                          output_partition_sizes=[out],
                          input_size=IN_FEATURES,
                          output_size=out,
                          params_dtype=torch.bfloat16,
                          weight_loader=lambda *a, **k: None)

    assert tuple(layer.weight.shape) == (out, IN_FEATURES // PACK_FACTOR)
    assert layer.weight.dtype == torch.uint32
    for name in ("scales", "biases"):
        p = getattr(layer, name)
        assert tuple(p.shape) == (out, IN_FEATURES // GROUP_SIZE)
        assert p.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "output_sizes, num_proj, n_shards, label",
    [
        ([32], 1, 1, "single"),
        ([64, 64], 2, 1, "fused_gate_up"),
        ([64, 32, 32], 3, 1, "fused_qkv"),
        # n_shards=2 forces a non-trivial interleave-by-shard reorder in
        # process_weights_after_loading that the apply-time slice/concat must
        # invert -- the actual fusion-split round-trip (single most untested
        # branch), exercised on one CPU device.
        ([64, 64], 2, 2, "fused_gate_up_tp2_reorder"),
    ])
def test_apply_matches_golden(output_sizes, num_proj, n_shards, label):
    """Load synthetic 4-bit packs, run process_weights_after_loading + apply,
    and assert the dequant+matmul (with the fused slice/concat split) reproduces
    a golden ``x @ dequant(W).T`` within bf16 tolerance. Proves real behavior:
    the comparison is against the bf16 golden the MLX checkpoint ships, not a
    mock."""
    out = sum(output_sizes)
    mesh = test_utils.get_spmd_mesh(1)
    # n_shards>1 mimics a TP layout on a single device, so use a replicated
    # weight_sharding (P(None, None)); n_shards=1 uses the column-parallel spec
    # (the [in, out] convention: wsh[1] shards the output dim).
    weight_sharding = (P(None, None) if n_shards > 1 else P(
        None, ShardingAxisName.ATTN_HEAD))
    cfg = _make_linear_config(mesh, output_sizes, num_proj, n_shards,
                              weight_sharding)
    method = VllmMLXLinearMethod(VllmMLXConfig(group_size=GROUP_SIZE,
                                               bits=BITS), cfg)
    layer = torch.nn.Module()
    layer.skip_bias_add = False
    method.create_weights(layer,
                          input_size_per_partition=IN_FEATURES,
                          output_partition_sizes=output_sizes,
                          input_size=IN_FEATURES,
                          output_size=out,
                          params_dtype=torch.bfloat16,
                          weight_loader=lambda *a, **k: None)

    rng = np.random.default_rng(out + n_shards)
    w = rng.standard_normal((out, IN_FEATURES)).astype(np.float32)
    # force_negative_scale exercises the affine sign-flip branch (adversarial).
    packed, scales, biases, golden = _quantize_affine(
        w, GROUP_SIZE, force_negative_scale=True)
    _load_and_process(method, layer, packed, scales, biases)

    x = rng.standard_normal((4, IN_FEATURES)).astype(np.float32)
    y = _apply_np(method, layer, x)
    y_ref = x @ golden.astype(np.float32).T

    assert y.shape == (4, out)
    np.testing.assert_allclose(y, y_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("tp, in_features", [(2, 128), (8, 4096)])
def test_rowparallel_input_dim_sharding_dequant_consistency(tp, in_features):
    """Step-3 (Task 8): RowParallelLinear shards the INPUT dim via
    ``weight_sharding = P(ATTN_HEAD, None)`` (the [in, out] spec of the
    transposed jax weight). For MLX that input dim is BOTH
    packed (``weight`` [out, in//8], one uint32 word = 8 nibbles) and grouped
    (``scales``/``biases`` [out, in//64], one affine scale/bias per 64 inputs).
    Splitting axis 1 of the packed weight by ``tp`` and axis 1 of the grouped
    scales/biases by ``tp`` is only correct if each shard owns WHOLE words AND
    WHOLE groups for the SAME contiguous input range -- otherwise a uint32 word
    or a quant group is split across chips and the per-shard dequant is wrong.

    This proves the math at the tensor level WITHOUT a TPU: it checks
    dequant-then-shard == shard-then-dequant for the input-dim split. (in=4096,
    tp=8 is the only Stage-1 packed RowParallel linear in Qwen3-30B: attn
    o_proj, in = 32 heads * 128 head_dim.)"""
    from tpu_inference.layers.common.quantization import mlx_dequantize
    out = 96
    assert in_features % PACK_FACTOR == 0 and in_features % GROUP_SIZE == 0
    # Required for a clean shard: each chip must own whole words and whole groups.
    assert (in_features // PACK_FACTOR) % tp == 0
    assert (in_features // GROUP_SIZE) % tp == 0

    rng = np.random.default_rng(in_features + tp)
    w = rng.standard_normal((out, in_features)).astype(np.float32)
    packed, scales, biases, _ = _quantize_affine(w, GROUP_SIZE,
                                                  force_negative_scale=True)
    packed_j = jnp.asarray(packed.astype(np.uint32))
    scales_j = jnp.asarray(scales.astype(np.float32))
    biases_j = jnp.asarray(biases.astype(np.float32))

    # Full dequant -> [out, in].
    full = np.asarray(
        mlx_dequantize(packed_j, scales_j, biases_j, group_size=GROUP_SIZE,
                       bits=BITS)).astype(np.float32)

    words_per_shard = (in_features // PACK_FACTOR) // tp
    groups_per_shard = (in_features // GROUP_SIZE) // tp
    in_per_shard = in_features // tp
    for s in range(tp):
        # Shard axis 1 of packed weight (words) and of scales/biases (groups),
        # exactly as P(None, ATTN_HEAD) does on the input dim.
        p_s = packed_j[:, s * words_per_shard:(s + 1) * words_per_shard]
        sc_s = scales_j[:, s * groups_per_shard:(s + 1) * groups_per_shard]
        bi_s = biases_j[:, s * groups_per_shard:(s + 1) * groups_per_shard]
        shard_dequant = np.asarray(
            mlx_dequantize(p_s, sc_s, bi_s, group_size=GROUP_SIZE,
                           bits=BITS)).astype(np.float32)
        # Must equal the matching input slice of the full dequant.
        expected = full[:, s * in_per_shard:(s + 1) * in_per_shard]
        assert shard_dequant.shape == (out, in_per_shard)
        np.testing.assert_array_equal(shard_dequant, expected)
