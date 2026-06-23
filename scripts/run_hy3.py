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
"""Acceptance bring-up for mlx-community/Hy3-preview-4bit (HYV3ForCausalLM).

Runs the Tencent Hunyuan V3 preview 4-bit MoE checkpoint on TPU v6e-8 at
tensor_parallel_size=8 via the torchax/vLLM path, and greedy-decodes a handful
of prompts through the chat template to verify coherent English output.

The model snapshot lives in a read-only, root-owned gcsfuse mount under the HF
cache layout, so this must be run under ``sudo`` with the env preserved, e.g.:

    cd /home/enyouki/tpu-inference && sudo env \
        HF_HOME=/tmp/gcs/bucket HF_HUB_OFFLINE=1 \
        MODEL_IMPL_TYPE=vllm SKIP_JAX_PRECOMPILE=1 \
        ~/vllm_env/bin/python scripts/run_hy3.py 2>&1 | tee /tmp/hy3_run.log
"""

import os
import sys

# This script lives in ``scripts/``, which contains a ``vllm/`` subdirectory
# (benchmarking/integration helpers, no __init__.py). When run as a script,
# Python puts this script's directory on sys.path[0], so that ``scripts/vllm``
# would shadow the real editable ``vllm`` package as an implicit namespace
# package and break ``from vllm import LLM``. Drop the script directory from
# sys.path before importing vllm so the installed package resolves correctly.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _SCRIPT_DIR]

# Env vars MUST be set before importing vllm/jax so they take effect.
os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")
os.environ.setdefault("HF_HOME", "/tmp/gcs/bucket")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "mlx-community/Hy3-preview-4bit"

PROMPTS = [
    "What is the capital of France?",
    "Write a one-sentence summary of what a neural network is.",
    "List three primary colors.",
    "Explain why the sky is blue in two sentences.",
]


def main() -> None:
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=8,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    # One conversation per prompt; llm.chat applies the model's chat template.
    conversations = [[{"role": "user", "content": p}] for p in PROMPTS]
    outputs = llm.chat(messages=conversations,
                       sampling_params=sampling_params)

    print("\n" + "=" * 80)
    print("HY3 GREEDY DECODE RESULTS")
    print("=" * 80)
    for prompt, output in zip(PROMPTS, outputs):
        completion = output.outputs[0].text
        print("\n" + "-" * 80)
        print(f"PROMPT: {prompt}")
        print(f"COMPLETION: {completion}")
    print("\n" + "=" * 80)
    print("HY3_RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("HY3_RUN_FAILED", flush=True)
        raise
