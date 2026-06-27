#!/bin/bash
# Same as serve_condense.sh but lower gpu-memory-utilization to leave HBM
# headroom for the NEW _permute_ctx_rows JIT program (~4.58G program alloc;
# the 4.07GiB ctx buffer gather OOM'd at util 0.75). Used to actually exercise
# the condense path and collect SpecDecoding metrics.
# Usage: serve_condense_lowutil.sh <max_num_seqs> <logfile> [util]
set -euo pipefail
MAXSEQS="${1:-32}"
LOGFILE="${2:-/home/enyouki/tpu-inference/scratchpad/serve_condense_lowutil.log}"
UTIL="${3:-0.60}"

env HF_HOME=/home/enyouki/local_hf \
    DRAFT_MODEL_IMPL_TYPE=torchax \
    RAGGED_GATHER_VERSION=v1 \
    RAGGED_GATHER_REDUCE_VERSION=v1 \
    DFLASH_PERFECT_DRAFT="${DFLASH_PERFECT_DRAFT:-0}" \
    ~/tpu-tooling/tpu-env.sh vllm serve openai/gpt-oss-20b \
      --tensor-parallel-size 8 \
      --enable-expert-parallel \
      --no-async-scheduling \
      --max-model-len 4224 \
      --max-num-seqs "${MAXSEQS}" \
      --gpu-memory-utilization "${UTIL}" \
      --speculative-config '{"model": "z-lab/gpt-oss-20b-DFlash", "num_speculative_tokens": 7, "method": "dflash"}' \
      > "${LOGFILE}" 2>&1
