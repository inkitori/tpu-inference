#!/bin/bash
# Batched DFlash serve for correctness testing (perfect-draft + greedy).
# Usage: serve_dflash.sh <max_num_seqs> <logfile> [extra_env...]
# DFLASH_PERFECT_DRAFT is passed via environment by the caller.
set -euo pipefail
MAXSEQS="${1:-16}"
LOGFILE="${2:-/home/enyouki/tpu-inference/scratchpad/serve.log}"

env HF_HOME=/home/enyouki/local_hf \
    DRAFT_MODEL_IMPL_TYPE=torchax \
    RAGGED_GATHER_VERSION=v1 \
    RAGGED_GATHER_REDUCE_VERSION=v1 \
    DFLASH_PERFECT_DRAFT="${DFLASH_PERFECT_DRAFT:-0}" \
    ~/tpu-tooling/tpu-env.sh vllm serve openai/gpt-oss-20b \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --no-async-scheduling \
      --max-model-len 1024 \
      --max-num-seqs "${MAXSEQS}" \
      --gpu-memory-utilization 0.75 \
      --speculative-config '{"model": "z-lab/gpt-oss-20b-DFlash", "num_speculative_tokens": 7, "method": "dflash"}' \
      > "${LOGFILE}" 2>&1
