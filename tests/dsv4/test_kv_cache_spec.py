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
"""Task-15 P1 guard: drive the REAL runner KV-cache allocation for the DSV4-MLA
attention layer and assert it matches the ``mla_swa`` kernel contract.

The ``mla_swa`` kernel (``tpu_inference/kernels/experimental/deepseek_v4/
mla_swa.py:1016``) asserts ``cache_kv.dtype == jnp.uint8`` and destructures
``cache_kv.shape`` as ``(num_blocks, page_size_per_kv_packing, kv_packing,
lkv_dim)`` with ``kv_packing == 4`` and ``lkv_dim == 640`` (640 = 448 fp8 nope +
128 bf16 rope + 7 e8m0 scale + 57 pad). The production runner previously
allocated a bf16/packing-32/512 pool (the shared R1/V3 MLA layout) which the
kernel rejects -- this test pins the kernel-contract layout for DSV4-MLA *only*.

The T12/13/14 dsv4 tests HAND-BUILD their pool (``build_mini_model.py``) and so
do NOT exercise the runner allocation -- this test drives the actual
``KVCacheManager.get_kv_cache_spec()`` + ``initialize_kv_cache()`` code path, the
exact code the production runner runs, on the synthetic production
``dsv4_mesh``.

It also carries a REGRESSION GUARD: a real (non-DSV4) ``MLAAttention`` layer --
the shared R1/V3 MLA path -- must still allocate the bf16/packing-32/512 pool, so
the DSV4 gate provably does not leak into R1/V3 KV layout.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
import torch
from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheGroupSpec,
                                        KVCacheTensor, MLAAttentionSpec)

from tests.dsv4.mesh_fixtures import dsv4_mesh  # noqa: F401  (pytest fixture)
from tpu_inference import utils as common_utils
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.runner.kv_cache_manager import KVCacheManager

# --- Real DSV4-flavoured dims (mirror tests/dsv4/mini_config.py) ------------- #
HEAD_DIM = 512          # nope 448 + rope 64
QK_ROPE_HEAD_DIM = 64
KV_LORA_RANK = 512
BLOCK_SIZE = 64
# Kernel contract for the DSV4-MLA fp8_ds_mla pool.
DSV4_PACKED_WIDTH = 640  # align_to(448 + 64*2 + 448//64, 128) = align_to(583,128)
DSV4_PACKING = 4         # 32 // 8 (uint8)
# Shared R1/V3 MLA layout (must stay untouched by the DSV4 gate). For a
# DeepseekV3-style MLA layer the runner derives head_size as
# align_to(kv_lora_rank,128) + align_to(qk_rope_head_dim,128) = 512 + 128 = 640,
# allocated as a FLOATING dtype (bf16) with packing 32. The DSV4-MLA gate must
# leave BOTH the floating dtype and packing-32 untouched (the mla v2 kernel
# asserts a floating dtype; kernel.py:231-232). The last-dim happens to also be
# 640, so the discriminating R1/V3 invariants are dtype + packing.
RV3_PACKING = 32         # envs.MLA_KV_PACKING_SIZE
RV3_LAST_DIM = 640       # align_to(kv_lora_rank,128)+align_to(rope,128)=512+128


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class _FakeDeepseekV4Attention:
    """Stand-in registered as the DSV4-MLA attention layer.

    Must satisfy ``isinstance(module, DeepseekV4Attention)`` (the runner's gate)
    AND ``isinstance(module, AttentionLayerBase)`` (``get_layers_from_vllm_config``
    filter). We therefore build a *real* trivial concrete subclass of the vLLM
    ``DeepseekV4Attention`` ABC at import time below, bypassing its heavy
    ``__init__`` (weights, cuda streams). It only needs to answer
    ``get_kv_cache_spec`` exactly like the real layer's FlashMLA fp8_ds_mla path.
    """


def _make_dsv4_attn_cls():
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    class _ConcreteDSV4Attn(DeepseekV4Attention):
        # Provide the abstractmethods so the class is concrete; never called.
        @classmethod
        def get_padded_num_q_heads(cls, num_heads):  # pragma: no cover
            return num_heads

        def forward_mqa(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

        def _o_proj(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

        def __init__(self):
            # Bypass DeepseekV4Attention.__init__ entirely (no weights/streams).
            torch.nn.Module.__init__(self)
            self.head_dim = HEAD_DIM
            self.compress_ratio = 4              # >1 -> real MLA spec (not None)
            self.kv_cache_dtype = "fp8_ds_mla"   # FlashMLA fp8_ds_mla layout
            self.kv_cache_torch_dtype = torch.uint8
            self.prefix = "layer.1"

        def get_kv_cache_spec(self, vllm_config):
            # Mirror the real FlashMLA fp8_ds_mla branch (attention.py:599-617).
            return MLAAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=self.head_dim,
                dtype=torch.uint8,
                cache_dtype_str="fp8_ds_mla",
            )

    return _ConcreteDSV4Attn


def _make_rv3_mla_attn_cls():
    """A real (non-DSV4) MLAAttention-like layer for the R1/V3 regression guard.

    The runner routes any non-DSV4 MLA DECODER layer through the shared
    ``mla_head_size`` branch (kv_cache_manager.py:650-653). We register a minimal
    ``MLAAttention`` so ``get_layers_from_vllm_config`` and the gate behave
    exactly as for R1/V3, without depending on DSV4 internals.
    """
    from vllm.model_executor.layers.mla import MLAAttention

    class _FakeRV3MLAAttention(MLAAttention):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.num_kv_heads = 1
            self.head_size = HEAD_DIM
            self.sliding_window = None
            self.kv_sharing_target_layer_name = None
            from vllm.v1.attention.backend import AttentionType
            self.attn_type = AttentionType.DECODER

    return _FakeRV3MLAAttention


def _build_runner_stub(mesh, *, architectures):
    """Minimal real-runner stand-in exposing exactly what KVCacheManager reads.

    Drives the production ``get_kv_cache_spec`` + ``initialize_kv_cache`` code
    (the code under fix) without loading any weights. The compilation-config
    branch of ``get_kv_cache_spec`` is taken because we populate
    ``static_forward_context`` (matching production with MODEL_IMPL_TYPE=vllm).
    """
    static_forward_context: dict = {}

    hf_text_config = SimpleNamespace(
        head_dim=HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
    )
    model_config = SimpleNamespace(
        use_mla=True,
        hf_text_config=hf_text_config,
        hf_config=hf_text_config,
        architectures=architectures,
        get_num_layers=lambda pc: 0,
        get_total_num_kv_heads=lambda: 1,
        get_head_size=lambda: HEAD_DIM,
        get_vocab_size=lambda: 1280,
    )
    cache_config = SimpleNamespace(
        block_size=BLOCK_SIZE,
        cache_dtype="fp8_ds_mla",
        num_gpu_blocks_override=None,
        gpu_memory_utilization=0.9,
    )
    compilation_config = SimpleNamespace(
        static_forward_context=static_forward_context)
    parallel_config = SimpleNamespace(decode_context_parallel_size=1)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=architectures),
        cache_config=cache_config,
        compilation_config=compilation_config,
        parallel_config=parallel_config,
        additional_config={},
    )

    runner = SimpleNamespace(
        mesh=mesh,
        # NOTE: deliberately bf16 -- the shared/default runner kv_cache_dtype.
        # The DSV4 fix must override this to uint8 for the DSV4-MLA layer ONLY;
        # if it leaked off runner.kv_cache_dtype the R1/V3 guard would break.
        kv_cache_dtype=torch.bfloat16,
        model_config=model_config,
        vllm_config=vllm_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        speculative_config=None,
        max_num_reqs=8,
        kv_caches=[],
        layer_name_to_kvcache_index={},
    )
    return runner, static_forward_context


def _allocate_pool(runner, manager):
    """Run the real spec build + allocation; return {layer_name: jax.Array}."""
    spec_dict = manager.get_kv_cache_spec()
    assert spec_dict, "expected at least one kv-cache spec"

    layer_names = list(spec_dict.keys())
    groups = [
        KVCacheGroupSpec(layer_names=[ln], kv_cache_spec=spec_dict[ln])
        for ln in layer_names
    ]
    # One physical tensor per layer; size a handful of blocks worth so
    # num_blocks = size // page_size_bytes is positive and divisor-aligned.
    batch_div = max(
        common_utils.get_mesh_shape_product(runner.mesh,
                                            ShardingAxisName.BATCH), 1)
    target_blocks = 4 * batch_div
    tensors = []
    for ln in layer_names:
        page = spec_dict[ln].page_size_bytes
        tensors.append(
            KVCacheTensor(size=target_blocks * page, shared_by=[ln]))
    kv_cache_config = KVCacheConfig(
        num_blocks=target_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )

    # No kv-connector in the synthetic harness: short-circuit the connector
    # layout check (it reads the global current vLLM config, which we don't set
    # up here). This does not touch the allocation path under test.
    with patch(
            "tpu_inference.runner.kv_cache_manager.get_kv_connector_cache_layout",
            return_value=None):
        manager.initialize_kv_cache(kv_cache_config)
    pools = {}
    for ln in layer_names:
        idx = runner.layer_name_to_kvcache_index[ln]
        pools[ln] = runner.kv_caches[idx]
    return pools, spec_dict


# --------------------------------------------------------------------------- #
# The P1 assertion: the runner allocates the kernel-contract pool for DSV4-MLA.
# --------------------------------------------------------------------------- #
def test_runner_allocates_dsv4_mla_kernel_contract_pool(dsv4_mesh):  # noqa: F811
    mesh = dsv4_mesh
    with jax.set_mesh(mesh):
        runner, sfc = _build_runner_stub(
            mesh, architectures=["DeepseekV4ForCausalLM"])
        dsv4_attn = _make_dsv4_attn_cls()()
        sfc["layer.1"] = dsv4_attn
        manager = KVCacheManager(runner)

        pools, spec_dict = _allocate_pool(runner, manager)
        pool = pools["layer.1"]

        # dtype: kernel asserts uint8 (mla_swa.py:1016).
        assert pool.dtype == jnp.uint8, (
            f"DSV4-MLA pool dtype {pool.dtype}, want uint8")
        # shape: (num_blocks, cdiv(block_size, 4), 4, 640).
        assert pool.ndim == 4, f"DSV4-MLA pool ndim {pool.ndim}, want 4"
        num_blocks, page_per_pack, packing, last = pool.shape
        assert packing == DSV4_PACKING, (
            f"DSV4-MLA packing {packing}, want {DSV4_PACKING}")
        assert page_per_pack == _cdiv(BLOCK_SIZE, DSV4_PACKING), (
            f"DSV4-MLA page/packing {page_per_pack}, "
            f"want {_cdiv(BLOCK_SIZE, DSV4_PACKING)}")
        assert last == DSV4_PACKED_WIDTH, (
            f"DSV4-MLA last-dim {last}, want {DSV4_PACKED_WIDTH}")
        assert num_blocks > 0


# --------------------------------------------------------------------------- #
# Regression guard: the shared R1/V3 MLA path stays bf16/packing-32/512.
# --------------------------------------------------------------------------- #
def test_rv3_mla_pool_unchanged_by_dsv4_gate(dsv4_mesh):  # noqa: F811
    mesh = dsv4_mesh
    with jax.set_mesh(mesh):
        runner, sfc = _build_runner_stub(
            mesh, architectures=["DeepseekV3ForCausalLM"])
        sfc["layer.0"] = _make_rv3_mla_attn_cls()()
        manager = KVCacheManager(runner)

        pools, spec_dict = _allocate_pool(runner, manager)
        pool = pools["layer.0"]

        # R1/V3 MLA must remain a floating dtype (mla v2 kernel asserts
        # floating: kernel.py:231-232) -- the DSV4 uint8 gate must NOT leak.
        assert jnp.issubdtype(pool.dtype, jnp.floating), (
            f"R1/V3 MLA pool dtype {pool.dtype} is not floating -- "
            f"DSV4 gate leaked into the shared MLA path!")
        assert pool.dtype == jnp.bfloat16, (
            f"R1/V3 MLA pool dtype {pool.dtype}, want bfloat16")
        assert pool.ndim == 4, f"R1/V3 MLA pool ndim {pool.ndim}, want 4"
        _, page_per_pack, packing, last = pool.shape
        assert packing == RV3_PACKING, (
            f"R1/V3 MLA packing {packing}, want {RV3_PACKING} "
            f"(DSV4 packing-4 override leaked!)")
        assert page_per_pack == _cdiv(BLOCK_SIZE, RV3_PACKING), (
            f"R1/V3 MLA page/packing {page_per_pack}, "
            f"want {_cdiv(BLOCK_SIZE, RV3_PACKING)}")
        assert last == RV3_LAST_DIM, (
            f"R1/V3 MLA last-dim {last}, want {RV3_LAST_DIM} "
            f"(DSV4 640 override leaked!)")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-s"]))
