#!/bin/bash
# Usage: run_bench.sh <tag>  -> writes scratch_mlx_int4/bench_<tag>.out
# Measures per-user tok/s = 1000/MeanTPOT at concurrency 1 and 8.
set -u
TAG="${1:-run}"
OUT="scratch_mlx_int4/bench_${TAG}.out"
cd /home/enyouki/tpu-inference
COMMON="--model mlx-community/Hy3-preview-4bit --backend vllm --host 127.0.0.1 --port 8000 \
  --dataset-name random --random-input-len 4096 --random-output-len 1024 --random-range-ratio 0.0 \
  --ignore-eos --request-rate inf --seed 42"
BENCH="env HF_HOME=/tmp/gcs/bucket/vllm HF_HUB_OFFLINE=1 /home/enyouki/vllm_env/bin/vllm bench serve"

{
echo "===== WARMUP (compile graphs) $(date) ====="
$BENCH $COMMON --num-prompts 4 --max-concurrency 1 2>&1 | tail -30
echo "===== CONCURRENCY 1 (target 150 tok/s/user) ====="
$BENCH $COMMON --num-prompts 12 --max-concurrency 1 2>&1 | tail -40
echo "===== CONCURRENCY 8 (target 100 tok/s/user) ====="
$BENCH $COMMON --num-prompts 32 --max-concurrency 8 2>&1 | tail -40
echo "===== DONE $(date) ====="
} > "$OUT" 2>&1
echo "wrote $OUT"
