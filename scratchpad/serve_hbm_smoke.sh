#!/bin/bash
# HBM smoke: DFlash serve at high max-num-seqs + long max-model-len.
# Usage: serve_hbm_smoke.sh <max_num_seqs> <max_model_len> <util> <logfile>
set -euo pipefail
MAXSEQS="${1:-32}"
MAXLEN="${2:-4224}"
UTIL="${3:-0.80}"
LOGFILE="${4:-/home/enyouki/tpu-inference/scratchpad/serve_hbm.log}"

env HF_HOME=/home/enyouki/local_hf \
    DRAFT_MODEL_IMPL_TYPE=torchax \
    RAGGED_GATHER_VERSION=v1 \
    RAGGED_GATHER_REDUCE_VERSION=v1 \
    DFLASH_PERFECT_DRAFT=0 \
    ~/tpu-tooling/tpu-env.sh vllm serve openai/gpt-oss-20b \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --no-async-scheduling \
      --max-model-len "${MAXLEN}" \
      --max-num-seqs "${MAXSEQS}" \
      --gpu-memory-utilization "${UTIL}" \
      --speculative-config '{"model": "z-lab/gpt-oss-20b-DFlash", "num_speculative_tokens": 7, "method": "dflash"}' \
      > "${LOGFILE}" 2>&1
