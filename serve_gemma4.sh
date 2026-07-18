#!/usr/bin/env bash
set -euo pipefail
export MODEL_IMPL_TYPE=vllm
export OMP_NUM_THREADS=16
export VLLM_XLA_CHECK_RECOMPILATION=0
# Skip JAX precompilation for faster startup (shapes compile lazily on
# first use instead).
export SKIP_JAX_PRECOMPILE=${SKIP_JAX_PRECOMPILE:-1}
# Hot-reload gemma post-processing (parsers + chat template) on source
# change, per request — no server restart needed for parser iteration.
export VLLM_HOT_RELOAD_PARSERS=${VLLM_HOT_RELOAD_PARSERS:-1}
TEMPLATE_PATH=$(ls /dev/shm/hf/models--google--gemma-4-31B-it/snapshots/*/chat_template.jinja 2>/dev/null | head -1)
[ -n "$TEMPLATE_PATH" ] && export VLLM_HOT_RELOAD_CHAT_TEMPLATE="$TEMPLATE_PATH"
MODEL=google/gemma-4-31B-it
TP=${TP:-8}
PORT=${PORT:-8000}
mkdir -p /dev/shm/hf
exec vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len 32768 \
    --no-enable-prefix-caching \
    --download-dir /dev/shm/hf \
    --disable-chunked-mm-input \
    --exclude-tools-when-tool-choice-none \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --port "$PORT"
