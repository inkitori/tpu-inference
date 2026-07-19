#!/usr/bin/env bash
# serve_gemma4.sh — sub-5-minute google/gemma-4-31B-it serving on a fresh
# v6e-8 node (cold /dev/shm, cold ~/.cache), following the Hy3 fast_start
# playbook (tools/hy3_mtp/fast_start.sh on this branch).
#
#   $0 [fast]       (default) prefetch + serve: parallel-copy weights + XLA
#                   compile cache from the bucket into /dev/shm, then serve.
#                   This is the command the cold-node startup time is
#                   measured on.
#   $0 prefetch     just the bucket -> /dev/shm copy
#   $0 serve        production serve (assumes weights already in /dev/shm)
#   $0 fill         serve with VLLM_XLA_CHECK_RECOMPILATION=1 — run once per
#                   new (jax/libtpu/flags/geometry) config to write a
#                   complete persistent compile cache (drops the jax
#                   persistent-cache size/time thresholds so the ~hundreds
#                   of sub-1s helper jits persist too)
#   $0 warm         exercise a LIVE server with request variants that JIT
#                   lazily (greedy/sampled/logprobs/top-k/p/n>1/chat) so
#                   save-cache captures them
#   $0 save-cache   upload the local XLA cache to the bucket
#   $0 stop         stop the serve started by this script
#
# New-config workflow: fill -> warm -> save-cache. Every node after that:
# just `$0` (prefetch + serve, zero runtime XLA compiles).
#
# Serving flags are identical across fill/serve on purpose: XLA cache
# entries are keyed by the compiled graph shapes, which depend on
# max-model-len/max-num-seqs/etc. Change a flag => re-run fill/warm/save.
set -euo pipefail

export PATH=/opt/google-cloud-sdk/bin:$PATH

MODEL=google/gemma-4-31B-it
HUB_SLUG="models--${MODEL//\//--}"
GCS_BUCKET=${GCS_BUCKET:-personal-mark-us-east5-b}
CACHE_TAG=${CACHE_TAG:-gemma4-31b-v6e8}
SHM_ROOT=${SHM_ROOT:-/dev/shm}
HF_DIR="$SHM_ROOT/hf"
XLA_DST="$SHM_ROOT/xla-cache-gemma4"
HUB_SRC="gs://$GCS_BUCKET/vllm/hub/$HUB_SLUG"
XLA_SRC="gs://$GCS_BUCKET/vllm/xla-cache/$CACHE_TAG"
TP=${TP:-8}
PORT=${PORT:-8000}

log_ts() { echo "[$(date -u +%FT%T.%3NZ)] $*"; }

prefetch() {
    log_ts "prefetch start: $HUB_SRC + $XLA_SRC -> $SHM_ROOT"
    mkdir -p "$HF_DIR/$HUB_SLUG" "$XLA_DST"
    # Weights and compile cache stream concurrently; gcloud storage rsync
    # does sliced parallel downloads (~8 GiB/s bucket->tmpfs in-region).
    gcloud storage rsync -r -q "$HUB_SRC" "$HF_DIR/$HUB_SLUG" &
    local wpid=$!
    gcloud storage rsync -r -q "$XLA_SRC" "$XLA_DST" &
    local cpid=$!
    wait "$wpid"
    log_ts "weights in $HF_DIR/$HUB_SLUG"
    wait "$cpid"
    log_ts "xla cache in $XLA_DST ($(ls "$XLA_DST" | wc -l) entries)"
}

save_cache() {
    [ -n "$(ls -A "$XLA_DST" 2>/dev/null)" ] \
      || { echo "no local XLA cache at $XLA_DST — run fill first" >&2; exit 1; }
    log_ts "uploading XLA cache $XLA_DST -> $XLA_SRC"
    gcloud storage rsync -r -q "$XLA_DST" "$XLA_SRC"
    log_ts "done: $(gcloud storage ls "$XLA_SRC" | wc -l) entries in bucket"
}

serve() {
    local check=$1; shift
    export MODEL_IMPL_TYPE=vllm
    export OMP_NUM_THREADS=16
    # Weights/tokenizer come from the /dev/shm mirror; HF stays offline so
    # startup never waits on the network. HF_HUB_CACHE (not HF_HOME) is
    # what vllm's offline get_model_path() resolves snapshots from.
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
    export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_DIR}
    export VLLM_XLA_CACHE_PATH="$XLA_DST"
    export VLLM_XLA_CHECK_RECOMPILATION="$check"
    if [ "$check" = 1 ]; then
        # fill: AOT-precompile every shape so the persistent cache is
        # complete, and keep the renderer warmup blocking so save-cache
        # runs against a fully warmed server.
        export SKIP_JAX_PRECOMPILE=${SKIP_JAX_PRECOMPILE:-0}
    else
        # production: skip the AOT sweep (~90s even on cache hits — the
        # Pallas/StableHLO lowering that produces the cache key runs every
        # startup). Shapes compile lazily on first use as persistent-cache
        # reads instead. Same lazy-compile behavior as the original serve
        # script; the cache just makes each lazy compile a fast read.
        export SKIP_JAX_PRECOMPILE=${SKIP_JAX_PRECOMPILE:-1}
        # Renderer/MM-processor warmup is ~43s of CPU for gemma4 at 32k;
        # run it in a background thread (vllm VLLM_MM_WARMUP_ASYNC patch) —
        # same work, identical outputs, off the critical path.
        export VLLM_MM_WARMUP_ASYNC=${VLLM_MM_WARMUP_ASYNC:-1}
    fi
    export NUM_PRECOMPILE_WORKERS=${NUM_PRECOMPILE_WORKERS:-4}
    # Hot-reload gemma post-processing (parsers + chat template) on source
    # change, per request — no server restart needed for parser iteration.
    export VLLM_HOT_RELOAD_PARSERS=${VLLM_HOT_RELOAD_PARSERS:-1}
    TEMPLATE_PATH=$(ls "$HF_DIR/$HUB_SLUG"/snapshots/*/chat_template.jinja 2>/dev/null | head -1)
    [ -n "$TEMPLATE_PATH" ] && export VLLM_HOT_RELOAD_CHAT_TEMPLATE="$TEMPLATE_PATH"
    mkdir -p "$HF_DIR" "$XLA_DST"
    log_ts "vllm serve starting (VLLM_XLA_CHECK_RECOMPILATION=$check)"
    exec vllm serve "$MODEL" \
        --tensor-parallel-size "$TP" \
        --max-model-len 32768 \
        --no-enable-prefix-caching \
        --download-dir "$HF_DIR" \
        --disable-chunked-mm-input \
        --exclude-tools-when-tool-choice-none \
        --enable-auto-tool-choice \
        --tool-call-parser gemma4 \
        --reasoning-parser gemma4 \
        --port "$PORT" \
        "$@"
}

stop() {
    local pat="vllm serve $MODEL" pids
    pids=$(pgrep -f "$pat" | grep -vw "$$" || true)
    [ -n "$pids" ] || { echo "no serve process found"; return 0; }
    kill $pids 2>/dev/null || true
    for _ in $(seq 1 30); do
        pgrep -f "$pat" | grep -vwq "$$" || { echo "stopped."; return 0; }
        sleep 2
    done
    echo "still up after 60s — force-killing"
    kill -9 $pids 2>/dev/null || true
    sleep 2
    echo "stopped."
}

# Exercise the request classes that lazily JIT outside the AOT precompile
# sweep. Run once against a live fill server, then save-cache.
warm() {
    local base=${SERVE_URL:-http://localhost:$PORT}
    local before after
    before=$(ls "$XLA_DST" 2>/dev/null | wc -l)
    log_ts "warming $base with variant traffic (cache: $before entries)"
    python3 - "$base" "$MODEL" <<'PYEOF'
import concurrent.futures as cf
import json, subprocess, sys
base, model = sys.argv[1], sys.argv[2]
def req(path, body):
    body = {"model": model, "max_tokens": 32, **body}
    r = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
         f"{base}{path}", "-H", "Content-Type: application/json",
         "-d", json.dumps(body)], capture_output=True, text=True)
    return r.stdout
comp = lambda body: req("/v1/completions", body)
chat = lambda body: req("/v1/chat/completions", body)
cases = [
    comp({"prompt": "hi", "temperature": 0.7}),
    comp({"prompt": "hi", "temperature": 0.0}),
    comp({"prompt": "hi", "temperature": 0.7, "logprobs": 3}),
    comp({"prompt": "hi", "temperature": 0.0, "logprobs": 3}),
    comp({"prompt": "hi", "temperature": 0.9, "top_p": 0.9, "top_k": 40}),
    comp({"prompt": "hi", "temperature": 0.9, "n": 2}),
    chat({"messages": [{"role": "user", "content": "hi"}], "temperature": 0.0}),
    chat({"messages": [{"role": "user", "content": "hi"}], "temperature": 0.7}),
]
print("cases:", cases)
# max-concurrency mixed batch (hits batched sampler/RNG-split shapes)
mix = [{"prompt": "p " * (50 * i + 5),
        "temperature": 0.0 if i % 2 else 0.7,
        "logprobs": 2 if i % 3 == 0 else None} for i in range(8)]
with cf.ThreadPoolExecutor(8) as ex:
    print("concurrent:", list(ex.map(comp, mix)))
PYEOF
    after=$(ls "$XLA_DST" 2>/dev/null | wc -l)
    log_ts "cache: $before -> $after entries"
    [ "$after" -gt "$before" ] && echo "new variants captured; run: $0 save-cache" \
        || echo "no new compiles — cache already covers this traffic"
}

cmd=${1:-fast}
[ $# -gt 0 ] && shift || true
case "$cmd" in
    fast)       prefetch; serve 0 "$@" ;;
    prefetch)   prefetch ;;
    serve)      serve 0 "$@" ;;
    fill)       serve 1 "$@" ;;
    warm)       warm ;;
    save-cache) save_cache ;;
    stop)       stop ;;
    *) echo "usage: $0 [fast|prefetch|serve|fill|warm|save-cache|stop] [extra vllm serve args]" >&2; exit 1 ;;
esac
