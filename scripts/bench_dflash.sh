#!/usr/bin/env bash
# ShareGPT serving benchmark against a locally running server.
# Usage: bench_dflash.sh [result-tag] [num-prompts] [concurrency]
set -e

TAG="${1:-run}"
NUM_PROMPTS="${2:-128}"
CONCURRENCY="${3:-32}"
PORT="${PORT:-8000}"
DATASET="${DATASET:-/tmp/claude-2001/-home-enyouki-tpu-inference/1bd2c4bd-b0c8-47be-90dc-deb75bff22d3/scratchpad/sharegpt.json}"
OUT_DIR="${OUT_DIR:-/tmp/claude-2001/-home-enyouki-tpu-inference/1bd2c4bd-b0c8-47be-90dc-deb75bff22d3/scratchpad/bench}"
mkdir -p "$OUT_DIR"

MODEL=$(curl -s localhost:$PORT/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")

exec ~/tpu-tooling/tpu-env.sh vllm bench serve \
  --backend vllm \
  --host localhost --port "$PORT" \
  --model "$MODEL" \
  --dataset-name sharegpt \
  --dataset-path "$DATASET" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$CONCURRENCY" \
  --ignore-eos \
  --save-result \
  --result-dir "$OUT_DIR" \
  --result-filename "${TAG}.json"
