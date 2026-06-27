#!/bin/bash
# Config A + KV cache ON. c=32, max-model-len 4224. (GOAL config.)
# NOTE: flag-ON path allocates BOTH _ctx_buf (3.96 GiB, cross-check, not yet
# dropped) AND the new K/V cache (2.25 GiB), so util 0.75 OOMs in the
# _batched_ctx_write JIT program. Override util via UTIL env for correctness runs.
# DFLASH_PERFECT_DRAFT passed via environment by the caller (0 default).
# Usage: UTIL=0.6 serve_kvcache.sh <logfile>
set -x
LOGFILE="${1:-/home/enyouki/tpu-inference/scratchpad/kv_serve.log}"
UTIL="${UTIL:-0.6}"
exec env HF_HOME=/home/enyouki/local_hf DRAFT_MODEL_IMPL_TYPE=torchax \
  RAGGED_GATHER_VERSION=v1 RAGGED_GATHER_REDUCE_VERSION=v1 \
  DFLASH_KV_CACHE=1 \
  DFLASH_PERFECT_DRAFT="${DFLASH_PERFECT_DRAFT:-0}" \
  ~/tpu-tooling/tpu-env.sh vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 8 --enable-expert-parallel --no-async-scheduling \
  --max-model-len 4224 --max-num-seqs 32 --gpu-memory-utilization "${UTIL}" \
  --speculative-config '{"model": "z-lab/gpt-oss-20b-DFlash", "num_speculative_tokens": 7, "method": "dflash"}' \
  > "${LOGFILE}" 2>&1
