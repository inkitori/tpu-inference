#!/bin/bash
# Target-ONLY serve (no speculative-config) for greedy losslessness comparison.
# Usage: serve_target_only.sh <max_num_seqs> <logfile>
set -euo pipefail
MAXSEQS="${1:-8}"
LOGFILE="${2:-/home/enyouki/tpu-inference/scratchpad/serve_target.log}"

env HF_HOME=/home/enyouki/local_hf \
    RAGGED_GATHER_VERSION=v1 \
    RAGGED_GATHER_REDUCE_VERSION=v1 \
    ~/tpu-tooling/tpu-env.sh vllm serve openai/gpt-oss-20b \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --no-async-scheduling \
      --max-model-len 1024 \
      --max-num-seqs "${MAXSEQS}" \
      --gpu-memory-utilization 0.75 \
      > "${LOGFILE}" 2>&1
