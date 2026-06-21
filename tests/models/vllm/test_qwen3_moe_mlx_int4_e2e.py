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

os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")

from tests.utils.mlx_synthetic import (  # noqa: E402
    build_bf16_reference_moe, build_synthetic_mlx_moe)

# NOTE: do NOT import jax or call jax.devices() at module/collection scope.
# Doing so initializes the TPU backend in this (parent) process before vLLM
# forks its EngineCore subprocess, and a fork after JAX's multithreaded TPU
# init deadlocks the child at engine bring-up. The test inherently requires a
# TPU (it serves a model through the vllm/torchax path); the harness runs on
# TPU, so no eager device guard is needed.


def test_synthetic_mlx_moe_logits_match_bf16_reference():
    from vllm import LLM, SamplingParams
    with tempfile.TemporaryDirectory() as mlx_dir, \
            tempfile.TemporaryDirectory() as ref_dir:
        meta = build_synthetic_mlx_moe(mlx_dir, layers=2, experts=8, hidden=128,
                                       moe_inter=64)
        build_bf16_reference_moe(ref_dir, meta["golden"], layers=2, experts=8,
                                 hidden=128, moe_inter=64)

        sp = SamplingParams(max_tokens=8, temperature=0.0, logprobs=5)
        prompt_ids = [1, 5, 9, 13, 2, 7]

        # The MLX weight-stream transform (Task 6) lives in
        # IncrementalModelLoader.get_all_weights, registered under the
        # "tpu_streaming_loader" load format. vLLM's loader selection is a
        # static name lookup (load_format="auto" -> DefaultModelLoader, which
        # never applies the transform), so the MLX path must request this
        # loader explicitly; otherwise the un-transformed quantized
        # lm_head/switch_mlp tensors reach the model and fail to load.
        mlx = LLM(model=mlx_dir, tensor_parallel_size=1, max_model_len=64,
                  enforce_eager=True, dtype="bfloat16",
                  load_format="tpu_streaming_loader")
        out_mlx = mlx.generate({"prompt_token_ids": prompt_ids}, sp)
        del mlx
        time.sleep(10)  # Wait for the TPU to be released before the next LLM.

        ref = LLM(model=ref_dir, tensor_parallel_size=1, max_model_len=64,
                  enforce_eager=True, dtype="bfloat16")
        out_ref = ref.generate({"prompt_token_ids": prompt_ids}, sp)
        del ref
        time.sleep(10)  # Wait for the TPU to be released.

        mlx_ids = list(out_mlx[0].outputs[0].token_ids)
        ref_ids = list(out_ref[0].outputs[0].token_ids)
        print(f"MLX  tokens: {mlx_ids}")
        print(f"REF  tokens: {ref_ids}")
        assert mlx_ids == ref_ids
