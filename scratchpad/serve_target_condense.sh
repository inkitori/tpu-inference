#!/bin/bash
# Target-ONLY serve (no speculative-config) at the CONDENSE config: max-model-len
# 4224 / util 0.75 / max-num-seqs 32 so request shapes MATCH the spec-on serve.
# Usage: serve_target_condense.sh <max_num_seqs> <logfile>
set -euo pipefail
MAXSEQS="${1:-32}"
LOGFILE="${2:-/home/enyouki/tpu-inference/scratchpad/serve_target_condense.log}"

env HF_HOME=/home/enyouki/local_hf \
    RAGGED_GATHER_VERSION=v1 \
    RAGGED_GATHER_REDUCE_VERSION=v1 \
    ~/tpu-tooling/tpu-env.sh vllm serve openai/gpt-oss-20b \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --no-async-scheduling \
      --max-model-len 4224 \
      --max-num-seqs "${MAXSEQS}" \
      --gpu-memory-utilization 0.75 \
      > "${LOGFILE}" 2>&1
