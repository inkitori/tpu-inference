#!/bin/bash
LOG="$1"
i=0
max=160   # ~40 min at 15s
while [ $i -lt $max ]; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8000/health 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "READY after $((i*15))s (health 200)"
    grep -aE "GPU KV cache size|Maximum concurrency|KV cache|Available memory|Application startup complete|Uvicorn running" "$LOG" | tail -8
    exit 0
  fi
  # crash detection
  if grep -aqE "RESOURCE_EXHAUSTED|XlaRuntimeError|AssertionError|Traceback \(most recent call last\)|raise RuntimeError|RuntimeError:|Killed|Out of memory|OOM" "$LOG"; then
    # only treat as crash if process is gone OR a hard fatal marker
    if ! pgrep -f "vllm serve openai/gpt-oss-20b" >/dev/null; then
      echo "CRASHED after $((i*15))s (process died)"
      grep -aE "RESOURCE_EXHAUSTED|XlaRuntimeError|AssertionError|Error|Traceback|Killed|Out of memory|OOM" "$LOG" | tail -25
      exit 1
    fi
  fi
  if ! pgrep -f "vllm serve openai/gpt-oss-20b" >/dev/null; then
    echo "CRASHED after $((i*15))s (process not found)"
    tail -30 "$LOG"
    exit 1
  fi
  sleep 15
  i=$((i+1))
done
echo "TIMEOUT after $((max*15))s"
tail -30 "$LOG"
exit 2
