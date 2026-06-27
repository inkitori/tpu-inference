#!/bin/bash
# DFlash + KV cache ON, parameterized num_speculative_tokens for the sweep.
# c=32, max-model-len 4224, util 0.6 (flag-ON double-buffer OOMs at 0.75 per 13-test-kvcache).
# Usage: NUMSPEC=7 serve_kv_sweep.sh <logfile>
set -x
LOGFILE="${1:-/home/enyouki/tpu-inference/scratchpad/kv_sweep.log}"
UTIL="${UTIL:-0.6}"
NUMSPEC="${NUMSPEC:-7}"
exec env HF_HOME=/home/enyouki/local_hf DRAFT_MODEL_IMPL_TYPE=torchax \
  RAGGED_GATHER_VERSION=v1 RAGGED_GATHER_REDUCE_VERSION=v1 \
  DFLASH_KV_CACHE=1 \
  ~/tpu-tooling/tpu-env.sh vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 8 --enable-expert-parallel --no-async-scheduling \
  --max-model-len 4224 --max-num-seqs 32 --gpu-memory-utilization "${UTIL}" \
  --speculative-config "{\"model\": \"z-lab/gpt-oss-20b-DFlash\", \"num_speculative_tokens\": ${NUMSPEC}, \"method\": \"dflash\"}" \
  > "${LOGFILE}" 2>&1
