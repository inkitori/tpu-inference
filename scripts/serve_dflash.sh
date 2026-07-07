#!/usr/bin/env bash
# Serve gpt-oss-20b with the DFlash draft (z-lab/gpt-oss-20b-DFlash) on v6e-8.
#
# Flags that matter:
#  - --enable-expert-parallel: mandatory for gpt-oss mxfp4 MoE on this repo's
#    torchax path (plain TP round-trips through GMM_TP and hits an
#    IndivisibleError on the requantized scales).
#  - RAGGED_GATHER_VERSION=v1: the default v2 SparseCore gather fails to lower
#    on v6e at decode.
#  - async scheduling: pass --async-scheduling / --no-async-scheduling through
#    EXTRA_ARGS (defaults to vLLM's default).
set -ex

MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
NUM_SPEC="${NUM_SPEC:-7}"
PORT="${PORT:-8000}"

RAGGED_GATHER_VERSION=v1 RAGGED_GATHER_REDUCE_VERSION=v1 \
exec ~/tpu-tooling/tpu-env.sh vllm serve openai/gpt-oss-20b \
  --port "$PORT" \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --speculative-config "{\"model\": \"z-lab/gpt-oss-20b-DFlash\", \"num_speculative_tokens\": ${NUM_SPEC}, \"method\": \"dflash\"}" \
  "$@"
