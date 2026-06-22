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
"""Primary correctness gate for the MLX 4-bit MoE serving path.

Builds a synthetic MLX-layout Qwen3-MoE checkpoint (uint32-packed switch_mlp
experts, quantized embed/lm_head, a ``quantization_config`` block that fires
``is_mlx_quantized``) and a bf16 *reference* checkpoint dequantized from the
EXACT same golden weights. Loads both through the identical vLLM/torchax path
and asserts the MLX path's greedy decode matches the bf16 reference. The MLX
path dequantizes to the same bf16 values, so the two forwards must agree.
"""
import os
import tempfile
import time

import pytest

os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")

from tests.utils.mlx_synthetic import (  # noqa: E402
    build_bf16_reference_moe, build_synthetic_mlx_moe)

REAL_MLX_MODEL = "mlx-community/Qwen3-30B-A3B-4bit"

# NOTE: do NOT import jax or call jax.devices() at module/collection scope.
# Doing so initializes the TPU backend in this (parent) process before vLLM
# forks its EngineCore subprocess, and a fork after JAX's multithreaded TPU
# init deadlocks the child at engine bring-up. The test inherently requires a
# TPU (it serves a model through the vllm/torchax path); the harness runs on
# TPU, so no eager device guard is needed.


# tp values to exercise. tp=1 is the original gate. tp=2 forces the w13 GMM_TP
# path to ACTUALLY SHARD across the model axis (P(...,MLP_TENSOR) on the w13
# out_dim), which is the only way the w13 ``groupbias`` sharding spec -- identical
# to ``w13_weight_scale`` -- is numerically proven against a reference. With the
# synthetic dims (moe_inter=128, group_size=64 -> w2 has 128/64=2 quant blocks),
# the GMM_TP reorder assert ``moe_inter % tp == 0`` holds for tp in {1, 2}, and the
# padded w13 out_dim (= 2*align_to(moe_inter//tp,128)*tp) is divisible by tp by
# construction, so tp=2 hits the real sharded path (not a divisibility error).
#
# moe_inter=128 (not 64) is deliberate: w2 is ALSO kept int4 now, and w2 stays
# int4 only when its block count (moe_inter/group_size) divides the MLP tensor
# degree. 128/64=2 is divisible by tp=2, so at tp=2 the w2 per-group scale AND
# groupbias shard on the block dim (P(None, MLP_TENSOR)) -- this test then also
# proves w2 groupbias sharding, not just w13's. (At moe_inter=64 -> 1 block, w2
# would fall back to bf16 at tp=2 and the w2-int4 sharding path would go
# unexercised.)
#
# tp=8 (the production config) is NOT parametrized here: this synthetic model has
# num_attention_heads=2 (NUM_HEADS, chosen so head_dim=64 dodges TPU head padding
# and q/k/v/o_proj stay [hidden,hidden]), and vLLM hard-rejects tp>num_heads at
# config-validation time ("attention heads (2) must be divisible by tensor
# parallel size") BEFORE any MoE weight is loaded or sharded -- so tp=8 cannot
# exercise the groupbias path on this model at all. Reaching tp=8 would require
# bumping the model to >=8 heads (hidden=512), which adds no NEW groupbias
# coverage: tp=8 shards the identical w13 out_dim axis the identical way as tp=2.
# tp=2 is the real, divisible, sharded numeric gate; the production tp=8 path is
# covered by the RUN_REAL_MLX 30B coherence test below.
TP_VALUES = [1, 2]


@pytest.mark.parametrize("tensor_parallel_size", TP_VALUES)
def test_synthetic_mlx_moe_logits_match_bf16_reference(tensor_parallel_size):
    """MLX-4bit vs bf16-reference EXACT greedy-token match.

    At tp>1 the w13 packed codes + per-group scale + per-group ``groupbias`` are
    sharded across the model axis via ``shard_moe_weights`` (groupbias rides the
    same ``P(None,None,None,MLP_TENSOR)`` rails as ``w13_weight_scale``). The MLX
    path reconstructs ``w = q*scale + groupbias`` IN-KERNEL per shard, so an exact
    token match against the unquantized bf16 reference proves the sharded
    groupbias is applied correctly. If groupbias were replicated or sharded on the
    wrong axis while the weight is sharded on out_dim, the per-shard
    reconstruction would be wrong and this exact-match would FAIL.

    w2 is ALSO kept int4 now (down_proj experts: signed int4 codes + per-group
    scale + ``w2_groupbias``). With moe_inter=128, group_size=64 -> 2 quant
    blocks, so at tp=2 the w2 scale AND groupbias shard on the block dim
    (``P(None, MLP_TENSOR)``); the kernel reconstructs ``w2 = q*scale +
    groupbias`` per shard. So this exact-match ALSO proves w2 groupbias sharding.
    Because the bf16 reference's down_proj golden IS the int4-affine dequant
    (golden = q*scale_bf + bias_bf, see ``_quantize_affine``), both sides see the
    SAME dequantized w2 -- this stays an exact-match test of the SHARDING/kernel,
    not of int4 quant error.
    """
    # Skip rather than spuriously fail when the box has too few chips. Use the
    # JAX-free chip counter (glob over /dev/accel*//dev/vfio) -- calling
    # jax.devices()/jax.device_count() here would initialize the TPU backend in
    # this parent process and then DEADLOCK vLLM's EngineCore fork (see the
    # module NOTE). get_num_chips() never touches JAX.
    from tpu_inference.tpu_info import get_num_chips
    n_dev = get_num_chips()
    if tensor_parallel_size > n_dev:
        pytest.skip(
            f"tensor_parallel_size={tensor_parallel_size} > {n_dev} chips")

    from vllm import LLM, SamplingParams
    with tempfile.TemporaryDirectory() as mlx_dir, \
            tempfile.TemporaryDirectory() as ref_dir:
        # moe_inter=128 (group_size=64 -> w2 has 2 quant blocks) so w2 stays int4
        # and its block-dim shard is divisible at tp=2 (see TP_VALUES comment).
        meta = build_synthetic_mlx_moe(mlx_dir, layers=2, experts=8, hidden=128,
                                       moe_inter=128)
        build_bf16_reference_moe(ref_dir, meta["golden"], layers=2, experts=8,
                                 hidden=128, moe_inter=128)

        sp = SamplingParams(max_tokens=8, temperature=0.0, logprobs=5)
        prompt_ids = [1, 5, 9, 13, 2, 7]

        # The MLX weight-stream transform (Task 6) lives in
        # IncrementalModelLoader.get_all_weights, registered under the
        # "tpu_streaming_loader" load format. vLLM's loader selection is a
        # static name lookup (load_format="auto" -> DefaultModelLoader, which
        # never applies the transform), so the MLX path must request this
        # loader explicitly; otherwise the un-transformed quantized
        # lm_head/switch_mlp tensors reach the model and fail to load.
        mlx = LLM(model=mlx_dir,
                  tensor_parallel_size=tensor_parallel_size, max_model_len=64,
                  enforce_eager=True, dtype="bfloat16",
                  load_format="tpu_streaming_loader")
        out_mlx = mlx.generate({"prompt_token_ids": prompt_ids}, sp)
        del mlx
        time.sleep(10)  # Wait for the TPU to be released before the next LLM.

        ref = LLM(model=ref_dir,
                  tensor_parallel_size=tensor_parallel_size, max_model_len=64,
                  enforce_eager=True, dtype="bfloat16")
        out_ref = ref.generate({"prompt_token_ids": prompt_ids}, sp)
        del ref
        time.sleep(10)  # Wait for the TPU to be released.

        mlx_ids = list(out_mlx[0].outputs[0].token_ids)
        ref_ids = list(out_ref[0].outputs[0].token_ids)
        print(f"[tp={tensor_parallel_size}] MLX  tokens: {mlx_ids}")
        print(f"[tp={tensor_parallel_size}] REF  tokens: {ref_ids}")
        assert mlx_ids == ref_ids, (
            f"tp={tensor_parallel_size}: MLX 4-bit (sharded w13 groupbias) "
            f"diverged from bf16 reference: {mlx_ids} != {ref_ids}")


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_MLX") != "1",
    reason="Real 30B MLX bring-up; set RUN_REAL_MLX=1 to run (needs the "
    "downloaded mlx-community/Qwen3-30B-A3B-4bit weights and a TPU).")
def test_real_mlx_30b_auto_loader_serves_coherently():
    """Task 8 guard: a PLAIN ``LLM(model=...)`` with NO hardcoded load_format
    must serve the real MLX 30B coherently.

    This is the production path the synthetic test cannot exercise: with
    ``load_format="auto"`` (the default), the MLX auto-select wiring in
    ``VllmModelWrapper.__init__`` must force the streaming loader so the MLX
    weight transform runs (otherwise load crashes on lm_head.biases /
    un-unstacked switch_mlp). Greedy decode, one prompt, assert non-empty
    ASCII-coherent output. tp defaults to 8 (override via RUN_REAL_MLX_TP);
    the full multi-prompt perf/coherence run lives in the Task-8 report."""
    from vllm import LLM, SamplingParams
    tp = int(os.environ.get("RUN_REAL_MLX_TP", "8"))
    # NO load_format -> exercises the auto-select wiring.
    llm = LLM(model=REAL_MLX_MODEL, tensor_parallel_size=tp, max_model_len=2048,
              dtype="bfloat16")
    sp = SamplingParams(max_tokens=32, temperature=0.0)
    out = llm.generate(["The capital of France is"], sp)
    text = out[0].outputs[0].text
    print(f"REAL MLX output: {text!r}")
    assert text.strip(), "empty completion"
    # Coherence proxy: output is printable ASCII and not a single repeated token.
    assert all(31 < ord(c) < 127 or c.isspace() for c in text), \
        f"non-ASCII/garbage output: {text!r}"
    toks = text.split()
    assert len(set(toks)) > 1, f"degenerate repeated output: {text!r}"
