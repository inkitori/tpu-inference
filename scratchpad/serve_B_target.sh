#!/bin/bash
# Config B: target-only (NO speculative-config, NO draft). Same target, same
# max-num-seqs 32, same max-model-len 4224, same util 0.75. Apples-to-apples.
set -x
exec env HF_HOME=/home/enyouki/local_hf \
  RAGGED_GATHER_VERSION=v1 RAGGED_GATHER_REDUCE_VERSION=v1 \
  ~/tpu-tooling/tpu-env.sh vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 8 --enable-expert-parallel --no-async-scheduling \
  --max-model-len 4224 --max-num-seqs 32 --gpu-memory-utilization 0.75
