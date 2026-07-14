#!/bin/bash
# run_variant.sh <label> [extra vllm serve args...]
# fast_start serve with extra args -> wait ready -> correctness gate ->
# frozen benchmark -> summary. Artifacts persist in ~/hy3_bench_artifacts.
set -u
LABEL=$1; shift
ART=~/tpu-inference/tools/hy3_mtp/bench_artifacts
FS=~/tpu-inference/tools/hy3_mtp/fast_start.sh

$FS stop >/dev/null 2>&1
sleep 3
echo "[$LABEL] starting serve $*"
nohup $FS serve "$@" > /dev/shm/serve.log 2>&1 &
for i in $(seq 1 120); do
    curl -s -m 2 http://localhost:8000/health >/dev/null 2>&1 && break
    sleep 10
done
curl -s -m 2 http://localhost:8000/health >/dev/null 2>&1 || { echo "[$LABEL] SERVER FAILED TO START"; tail -30 /dev/shm/serve.log; exit 1; }
echo "[$LABEL] server ready"

python3 $ART/correctness_check.py "$LABEL" > $ART/correctness_$LABEL.log 2>&1
tail -8 $ART/correctness_$LABEL.log

~/vllm_env/bin/vllm bench serve --base-url http://localhost:8000 \
    --model /dev/shm/models/Hy3-4bit-mtp-mlx --trust-remote-code \
    --dataset-name random --random-input-len 4096 --random-output-len 2048 \
    --max-concurrency 8 --num-prompts 32 --ignore-eos > $ART/bench_$LABEL.log 2>&1

grep -E "Output token throughput|Median TPOT|Mean TPOT|Acceptance length|Position [01]|Mean TTFT|Benchmark duration|Acceptance rate" $ART/bench_$LABEL.log
grep "SpecDecoding metrics" /dev/shm/serve.log | tail -2
# Persist artifacts off-node via the repo snapshot (preemptions wipe the node).
~/claude-transcripts/snapshot >/dev/null 2>&1 && echo "[$LABEL] snapshot pushed"
echo "[$LABEL] done"
