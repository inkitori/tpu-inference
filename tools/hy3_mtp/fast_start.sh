#!/bin/bash
# fast_start.sh — self-contained sub-5-minute Hy3 serving on a fresh v6e-8.
#
# Everything needed to go from empty node to serving lives in this script;
# the only prerequisites are a vLLM/tpu-inference venv (TPU_VENV, default
# ~/vllm_env), gcloud auth for prefetch/save-cache, and the checkpoint
# mirrored in the bucket under vllm/hub/ (HF-cache layout, keyed by
# $HF_REPO — default enyoukai/Hy3-4bit-mtp-mlx, the released Hy3 4-bit
# MLX quant with the MTP head restored).
#
#   $0 prefetch     (default) parallel-download checkpoint + XLA compile
#                   cache into /dev/shm (~45s total when the bucket is warm)
#   $0 serve        PRODUCTION serve, foreground (use & / systemd / tmux to
#                   background). Extra args pass through to vllm serve.
#   $0 fill         like serve, but with VLLM_XLA_CHECK_RECOMPILATION=1 —
#                   run this INSTEAD of serve once per new (jax/libtpu/
#                   flags/geometry) configuration to build a complete cache
#   $0 warm         exercise a LIVE server with the request variants that
#                   JIT lazily (concurrency/greedy/logprobs/top-k/n>1)
#   $0 save-cache   upload the local XLA cache to the bucket
#   $0 stop         stop the serve process started by this script
#
# New-config workflow: prefetch -> fill -> warm -> save-cache. Every node
# after that: prefetch -> serve (~45s + ~220s to ready at 32k/bs256,
# measured 2026-07-13; zero runtime XLA compiles).
#
# Why prefetch instead of serving off the gcsfuse mount: the mount reads
# the 168 GB checkpoint at ~550 MB/s single-stream and the 160 GB file
# cache cap LRU-thrashes on every re-read; `gcloud storage rsync` pulls
# ~6 GiB/s and safetensors load from tmpfs at ~1.2 s/shard vs 8.9.
#
# Notes that earned their place the hard way:
#   * --block-size 256 is REQUIRED for max-model-len > 8192. The platform
#     heuristic (flash_attn.py get_page_size, "temporary fix for vmem OOM")
#     drops to 16-token KV pages above 8k; the RPA kernel statically
#     unrolls one DMA per page per kv block, so Pallas lowering — which
#     runs EVERY startup and cannot be skipped by the persistent compile
#     cache (lowering produces the cache key) — took ~46s per token bucket
#     (~20min startups). At 256-token pages it is ~5s. No VMEM OOM seen at
#     32k/bs256 on v6e-8.
#   * --max-model-len / --block-size change block-table shapes: backbone
#     cache entries are keyed per (context, page size). The bucket cache
#     accumulates every config that ran fill+warm+save-cache; stale
#     entries are harmless.
#   * VLLM_XLA_CHECK_RECOMPILATION=1 (fill) both persists sub-1s compiles
#     (write thresholds -> -1) and turns guarded runtime recompiles into
#     request-failing RuntimeErrors. Perfect for validation, wrong for
#     production: serve uses =0, so a missed shape stalls one request and
#     self-heals. Cache reads are identical either way.
#
# Overridable env: TPU_VENV, GCS_BUCKET, TPU_GCS_MOUNT, HF_REPO,
# CACHE_TAG, SHM_ROOT, SERVE_URL, OMP_NUM_THREADS, NUM_PRECOMPILE_WORKERS.
set -euo pipefail

MOUNT=${TPU_GCS_MOUNT:-/tmp/gcs/bucket}
# HF repo id; the bucket mirrors it under vllm/hub in HF-cache layout
# (models--org--name/{refs,snapshots}) with real files in snapshots/.
HF_REPO=${HF_REPO:-enyoukai/Hy3-4bit-mtp-mlx}
HUB_SLUG="models--${HF_REPO//\//--}"
CACHE_TAG=${CACHE_TAG:-hy3-v6e8}
SHM_ROOT=${SHM_ROOT:-/dev/shm}
MODEL_DST="$SHM_ROOT/models/${HF_REPO##*/}"
XLA_DST="$SHM_ROOT/xla_cache_hy3"

# Bucket: explicit env > the mounted bucket > same-region auto-discovery.
# Only prefetch/save-cache need it; serve/fill/warm/stop never touch GCS.
resolve_bucket() {
    BUCKET=${GCS_BUCKET:-$(findmnt -n -o SOURCE "$MOUNT" 2>/dev/null || true)}
    if [ -z "$BUCKET" ]; then
        local zone region_uc
        zone=$(curl -s -m 5 -H "Metadata-Flavor: Google" \
          "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}')
        region_uc=$(printf '%s' "${zone%-*}" | tr '[:lower:]' '[:upper:]')
        BUCKET=$(gcloud storage buckets list --format="value(name,location)" \
                 | awk -v r="$region_uc" '$2 == r {print $1; exit}')
    fi
    [ -n "$BUCKET" ] || { echo "cannot determine bucket; set GCS_BUCKET" >&2; exit 1; }
    HUB_SRC="gs://$BUCKET/vllm/hub/$HUB_SLUG"
    XLA_SRC="gs://$BUCKET/vllm/xla-cache/$CACHE_TAG"
}

save_cache() {
    resolve_bucket
    [ -d "$XLA_DST" ] && [ -n "$(ls -A "$XLA_DST" 2>/dev/null)" ] \
      || { echo "no local XLA cache at $XLA_DST — run a serve first" >&2; exit 1; }
    echo "uploading XLA cache $XLA_DST -> $XLA_SRC"
    gcloud storage rsync -r -q "$XLA_DST" "$XLA_SRC"
    echo "done: $(gcloud storage ls "$XLA_SRC" | wc -l) entries in bucket"
}

prefetch() {
    resolve_bucket
    # /dev/shm is tmpfs (RAM): checkpoint+cache need ~170 GB, kept for the
    # life of the node. The v6e-8 host has 1.4 TB; if this node is smaller,
    # point SHM_ROOT at local NVMe instead.
    avail_gb=$(df --output=avail -BG "$SHM_ROOT" | tail -1 | tr -dc 0-9)
    [ "$avail_gb" -ge 200 ] || echo "WARN: only ${avail_gb}G free on $SHM_ROOT (need ~170G)" >&2

    mkdir -p "$MODEL_DST" "$XLA_DST" 2>/dev/null || {
        sudo mkdir -p "$MODEL_DST" "$XLA_DST"
        sudo chown -R "$(id -u)" "$SHM_ROOT/models" "$XLA_DST"
    }

    # resolve the snapshot commit via refs/main, then pull only that
    # snapshot (real files; the mirror carries no blobs/ duplicates)
    local sha
    sha=$(gcloud storage cat "$HUB_SRC/refs/main" 2>/dev/null) \
      || { echo "no $HUB_SRC/refs/main in bucket — mirror the HF repo first" >&2; exit 1; }
    echo "prefetching checkpoint $HUB_SRC/snapshots/$sha -> $MODEL_DST (~40s for 168GB)"
    time gcloud storage rsync -r -q "$HUB_SRC/snapshots/$sha" "$MODEL_DST"

    if gcloud storage ls "$XLA_SRC" >/dev/null 2>&1; then
        echo "restoring XLA cache $XLA_SRC -> $XLA_DST"
        time gcloud storage rsync -r -q "$XLA_SRC" "$XLA_DST"
        cat <<EOF

ready. start serving (foreground; use & / systemd / tmux to background):
  $0 serve
extra vllm args pass through (last occurrence wins), e.g.:
  $0 serve --max-num-seqs 32 --additional-config '{"compilation_sizes":[96]}'
EOF
    else
        cat <<EOF
NOTE: no XLA cache in bucket yet ($XLA_SRC).
First serve on this configuration compiles cold — use the fill workflow:
  $0 fill          # serve with recompile-guard + full cache writes
  $0 warm          # then, against the live server
  $0 save-cache    # persist for every future node
EOF
    fi
}

# Production serve (check=0) / cache-fill serve (check=1). Foreground: exec's
# into vllm so signals/systemd work as expected. Extra args go to vllm serve.
serve() {
    local check=$1; shift
    [ -f "$MODEL_DST/config.json" ] \
      || { echo "no model at $MODEL_DST — run: $0 prefetch" >&2; exit 1; }

    # --- environment (inlined; no external wrapper needed) -----------------
    local venv="${TPU_VENV:-$HOME/vllm_env}"
    [ -f "$venv/bin/activate" ] \
      || { echo "no venv at $venv — set TPU_VENV to your vllm env" >&2; exit 1; }
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    # Weights/tokenizer are read from $MODEL_DST; HF stays offline so vllm
    # never hits the network (or takes write locks on a RO gcsfuse mount).
    export HF_HOME="${HF_HOME:-$MOUNT}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export MODEL_IMPL_TYPE="${MODEL_IMPL_TYPE:-vllm}"
    # Single-host TP uses UniProcExecutor, which does not clamp torch
    # threads; unclamped, the MoE weight copy oversubscribes a high-vCPU
    # host and checkpoint load balloons from ~1 min to ~20 min.
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

    export SKIP_JAX_PRECOMPILE=0       # AOT-precompile everything at startup
    export ONEHOT_MOE_PERMUTE_THRESHOLD=1024
    export VLLM_XLA_CACHE_PATH="$XLA_DST"
    export VLLM_XLA_CHECK_RECOMPILATION="$check"
    export NUM_PRECOMPILE_WORKERS="${NUM_PRECOMPILE_WORKERS:-4}"

    # cwd guard: python puts the cwd on sys.path, so launching from a
    # directory that contains a vllm/ checkout (e.g. $HOME) shadows the
    # installed package and vLLM's model-registry subprocess dies with
    # "cannot import name 'SamplingParams'". The script's own dir is safe.
    cd "$(dirname "$(readlink -f "$0")")"

    exec vllm serve "$MODEL_DST" \
      --tensor-parallel-size 8 --max-model-len 32768 --max-num-seqs 8 \
      --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
      --block-size 256 \
      --trust-remote-code --enable-expert-parallel --async-scheduling \
      --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
      "$@"
}

stop() {
    local pat="vllm serve $MODEL_DST"
    pgrep -f "$pat" >/dev/null || { echo "no serve process found"; return 0; }
    pkill -f "$pat" || true
    for _ in $(seq 1 30); do
        pgrep -f "$pat" >/dev/null || { echo "stopped."; return 0; }
        sleep 2
    done
    echo "still up after 60s — force-killing"
    pkill -9 -f "$pat" || true
    sleep 2
    echo "stopped."
}

# Exercise the request classes that lazily JIT outside the AOT precompile
# sweep (batch-shape RNG splits, greedy + sampled mixes, logprobs, top-k/p,
# n>1). Run once against a live server after a cold fill, then save-cache.
warm() {
    local base=${SERVE_URL:-http://localhost:8000}
    local before after
    before=$(ls "$XLA_DST" 2>/dev/null | wc -l)
    echo "warming $base with variant traffic (cache: $before entries)"
    python3 - "$base" "$MODEL_DST" <<'PYEOF'
import concurrent.futures as cf
import json, subprocess, sys
base, model = sys.argv[1], sys.argv[2]
def req(body):
    body = {"model": model, "max_tokens": 24, **body}
    r = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
         f"{base}/v1/completions", "-H", "Content-Type: application/json",
         "-d", json.dumps(body)], capture_output=True, text=True)
    return r.stdout
cases = [
    {"prompt": "hi", "temperature": 0.7},
    {"prompt": "hi", "temperature": 0.0},
    {"prompt": "hi", "temperature": 0.7, "logprobs": 3},
    {"prompt": "hi", "temperature": 0.0, "logprobs": 3},
    {"prompt": "hi", "temperature": 0.9, "top_p": 0.9, "top_k": 40},
    {"prompt": "hi", "temperature": 0.9, "n": 2},
]
for c in cases:
    print(c.get("temperature"), c.get("logprobs"), "->", req(c))
# max-concurrency mixed batch (hits batched sampler/RNG-split shapes)
mix = [{"prompt": "p " * (50 * i + 5),
        "temperature": 0.0 if i % 2 else 0.7,
        "logprobs": 2 if i % 3 == 0 else None} for i in range(8)]
with cf.ThreadPoolExecutor(8) as ex:
    print("concurrent:", list(ex.map(req, mix)))
PYEOF
    after=$(ls "$XLA_DST" 2>/dev/null | wc -l)
    echo "cache: $before -> $after entries"
    [ "$after" -gt "$before" ] && echo "new variants captured; run: $0 save-cache" \
        || echo "no new compiles — cache already covers this traffic"
}

cmd=${1:-prefetch}
[ $# -gt 0 ] && shift || true
case "$cmd" in
    prefetch)   prefetch ;;
    serve)      serve 0 "$@" ;;
    fill)       serve 1 "$@" ;;
    warm)       warm ;;
    save-cache) save_cache ;;
    stop)       stop ;;
    *) echo "usage: $0 [prefetch|serve|fill|warm|save-cache|stop] [extra vllm args for serve/fill]" >&2; exit 1 ;;
esac
