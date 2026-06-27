#!/bin/bash
# Config A: DFlash ON. c=32, max-model-len 4224, util 0.75.
set -x
exec env HF_HOME=/home/enyouki/local_hf DRAFT_MODEL_IMPL_TYPE=torchax \
  RAGGED_GATHER_VERSION=v1 RAGGED_GATHER_REDUCE_VERSION=v1 \
  ~/tpu-tooling/tpu-env.sh vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 8 --enable-expert-parallel --no-async-scheduling \
  --max-model-len 4224 --max-num-seqs 32 --gpu-memory-utilization 0.75 \
  --speculative-config '{"model": "z-lab/gpt-oss-20b-DFlash", "num_speculative_tokens": 7, "method": "dflash"}'
