#!/usr/bin/env bash
# Smoke-test the running server: greedy completions to check coherence.
PORT="${PORT:-8000}"
MODEL=$(curl -s localhost:$PORT/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
echo "model: $MODEL"
for prompt in "The capital of France is" "Q: What is 7 * 8?\nA:" "def fibonacci(n):"; do
  echo "=== PROMPT: $prompt"
  curl -s localhost:$PORT/v1/completions -H 'Content-Type: application/json' -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"$prompt\",
    \"max_tokens\": 64,
    \"temperature\": 0
  }" | python3 -c "import sys,json; r=json.load(sys.stdin); print(json.dumps(r['choices'][0]['text']) if 'choices' in r else r)"
done
