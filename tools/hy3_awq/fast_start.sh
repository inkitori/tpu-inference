#!/bin/bash
# fast_start.sh — self-contained sub-5-minute Hy3-AWQ serving on a fresh v6e-8.
#
# Sibling of tools/hy3_mtp/fast_start.sh (the MLX/MTP deployment), adapted for
# the compressed-tensors W4A16 checkpoint (cyankiwi/Hy3-AWQ-INT4) and the
# decode-tuned env stack measured on this branch (76.7 tok/s/user at
# 32k/conc-8, 2026-07-15). Everything needed to go from empty node to serving
# lives in this script; the only prerequisites are a vLLM/tpu-inference venv
# (TPU_VENV, default ~/vllm_env), gcloud auth for prefetch/mirror/save-cache,
# and the checkpoint mirrored in the bucket under vllm/hub/ (run `$0 mirror`
# once from a node that has the checkpoint locally).
#
#   $0 prefetch     (default) parallel-download checkpoint + XLA compile
#                   cache into /dev/shm
#   $0 serve        PRODUCTION serve, foreground (use & / systemd / tmux to
#                   background). Extra args pass through to vllm serve.
#   $0 fill         like serve, but with VLLM_XLA_CHECK_RECOMPILATION=1 —
#                   run this INSTEAD of serve once per new (jax/libtpu/
#                   flags/geometry) configuration to build a complete cache
#   $0 warm         exercise a LIVE server with the request variants that
#                   JIT lazily (concurrency/greedy/logprobs/top-k/n>1)
#   $0 save-cache   upload the local XLA cache to the bucket
#   $0 mirror       upload the local checkpoint dir (MODEL_SRC, default
#                   /dev/shm/hy3-awq-int4) into the bucket hub layout
#   $0 stop         stop the serve process started by this script
#
# New-config workflow: (mirror once) -> prefetch -> fill -> warm -> save-cache.
# Every node after that: prefetch -> serve.
#
# Notes carried over from the hy3_mtp deployment (measured there, verified
# here):
#   * --block-size 256 is REQUIRED for max-model-len > 8192. The platform
#     heuristic (flash_attn.py get_page_size, "temporary fix for vmem OOM")
#     drops to 16-token KV pages above 8k; Pallas lowering — which runs
#     EVERY startup and is not covered by the XLA persistent cache — blows
#     up per token bucket at 16-token pages, and decode attention runs ~4x
#     slower. No VMEM OOM seen at 32k/bs256 on v6e-8.
#   * RPA_DECODE_BKV_SIZE=2048 caps the decode KV block; the kernel default
#     is one max-model-len-sized block whose masked compute scales with
#     max-model-len, not actual context (57.4 -> 75.0 tok/s/user at 32k).
#     If production contexts trend much longer than ~8k, re-sweep
#     {2048,4096,8192} — 1024 was already slower at ~5k contexts.
#   * VLLM_XLA_CHECK_RECOMPILATION=1 (fill) both persists sub-1s compiles
#     and turns guarded runtime recompiles into request-failing
#     RuntimeErrors: right for cache-building, wrong for production.
#   * No MTP here: this branch does not carry the spec-decode AOT
#     precompile fixes from the hy3 branch; --speculative-config would
#     JIT lazily under traffic. Use tools/hy3_mtp for the MTP deployment.
#
# Overridable env: TPU_VENV, GCS_BUCKET, TPU_GCS_MOUNT, HF_REPO, MODEL_SRC,
# CACHE_TAG, SHM_ROOT, SERVE_URL, SERVED_MODEL_NAME, OMP_NUM_THREADS,
# NUM_PRECOMPILE_WORKERS.
set -euo pipefail

MOUNT=${TPU_GCS_MOUNT:-/tmp/gcs/bucket}
# HF repo id; the bucket mirrors it under vllm/hub in HF-cache layout
# (models--org--name/{refs,snapshots}) with real files in snapshots/.
HF_REPO=${HF_REPO:-cyankiwi/Hy3-AWQ-INT4}
HUB_SLUG="models--${HF_REPO//\//--}"
CACHE_TAG=${CACHE_TAG:-hy3-awq-v6e8}
SHM_ROOT=${SHM_ROOT:-/dev/shm}
MODEL_DST="$SHM_ROOT/models/${HF_REPO##*/}"
XLA_DST="$SHM_ROOT/xla_cache_hy3_awq"
# mirror-only: where the checkpoint lives locally before first upload
MODEL_SRC=${MODEL_SRC:-/dev/shm/hy3-awq-int4}

# Bucket: explicit env > the mounted bucket > same-region auto-discovery.
# Only prefetch/mirror/save-cache need it; serve/fill/warm/stop never touch GCS.
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

mirror() {
    resolve_bucket
    [ -f "$MODEL_SRC/config.json" ] \
      || { echo "no checkpoint at $MODEL_SRC — set MODEL_SRC" >&2; exit 1; }
    # Synthetic snapshot id: the local dir is a flat download, not an HF
    # cache, so key the snapshot by content (size+name manifest hash) to
    # stay stable across re-runs and distinguishable across re-quants.
    local sha
    sha=$(cd "$MODEL_SRC" && find . -type f -printf '%s %p\n' | sort -k2 \
          | sha256sum | cut -c1-16)
    echo "mirroring $MODEL_SRC -> $HUB_SRC/snapshots/$sha"
    gcloud storage rsync -r -q "$MODEL_SRC" "$HUB_SRC/snapshots/$sha"
    printf '%s' "$sha" | gcloud storage cp - "$HUB_SRC/refs/main"
    echo "done; refs/main -> $sha"
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
    # /dev/shm is tmpfs (RAM): checkpoint+cache need ~175 GB, kept for the
    # life of the node. The v6e-8 host has 1.4 TB; if this node is smaller,
    # point SHM_ROOT at local NVMe instead.
    avail_gb=$(df --output=avail -BG "$SHM_ROOT" | tail -1 | tr -dc 0-9)
    [ "$avail_gb" -ge 200 ] || echo "WARN: only ${avail_gb}G free on $SHM_ROOT (need ~175G)" >&2

    mkdir -p "$MODEL_DST" "$XLA_DST" 2>/dev/null || {
        sudo mkdir -p "$MODEL_DST" "$XLA_DST"
        sudo chown -R "$(id -u)" "$SHM_ROOT/models" "$XLA_DST"
    }

    local sha
    sha=$(gcloud storage cat "$HUB_SRC/refs/main" 2>/dev/null) \
      || { echo "no $HUB_SRC/refs/main in bucket — run: $0 mirror (on a node with the checkpoint)" >&2; exit 1; }
    echo "prefetching checkpoint $HUB_SRC/snapshots/$sha -> $MODEL_DST (~170GB)"
    time gcloud storage rsync -r -q "$HUB_SRC/snapshots/$sha" "$MODEL_DST"

    if gcloud storage ls "$XLA_SRC" >/dev/null 2>&1; then
        echo "restoring XLA cache $XLA_SRC -> $XLA_DST"
        time gcloud storage rsync -r -q "$XLA_SRC" "$XLA_DST"
        cat <<EOF

ready. start serving (foreground; use & / systemd / tmux to background):
  $0 serve
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
    export VLLM_XLA_CACHE_PATH="$XLA_DST"
    export VLLM_XLA_CHECK_RECOMPILATION="$check"
    export NUM_PRECOMPILE_WORKERS="${NUM_PRECOMPILE_WORKERS:-4}"

    # Decode-tuned stack for this checkpoint:
    #   * one-hot permute + padding-to-expert0 + 8-token buckets: decode
    #     MoE runs matmul-only routing with no padding rows.
    #   * MOE_REQUANTIZE_BLOCK_SIZE is deliberately NOT set: the default
    #     keeps the checkpoint's group-32 granularity. Setting 256 puts
    #     gmm_v2 on its dequant-after-matmul fast path (~+10 tok/s/user at
    #     32k/conc-8) but coarsens the expert requant 8x, measured at
    #     -6.0pp GSM8K (96.6% -> 90.6%, n=500 paired, p=9e-6, 2026-07-15).
    #     Do not enable without re-running that eval.
    export ONEHOT_MOE_PERMUTE_THRESHOLD="${ONEHOT_MOE_PERMUTE_THRESHOLD:-128}"
    export MOE_ROUTE_PADDING_TO_EXPERT0="${MOE_ROUTE_PADDING_TO_EXPERT0:-1}"
    export MIN_TOKEN_BUCKET="${MIN_TOKEN_BUCKET:-8}"
    export RPA_DECODE_BKV_SIZE="${RPA_DECODE_BKV_SIZE:-2048}"

    # cwd guard: python puts the cwd on sys.path, so launching from a
    # directory that contains a vllm/ checkout (e.g. $HOME) shadows the
    # installed package and vLLM's model-registry subprocess dies with
    # "cannot import name 'SamplingParams'". The script's own dir is safe.
    cd "$(dirname "$(readlink -f "$0")")"

    # Reasoning parser: splits Hy3's <think:opensource> trace into the OpenAI
    # reasoning field (required for OpenRouter reasoning validation). Thinking
    # itself is gated per-request by reasoning_effort (template default:
    # no_think). The plugin lives next to this script; we already cd'd here.
    exec vllm serve "$MODEL_DST" \
      --served-model-name "${SERVED_MODEL_NAME:-tencent/Hy3}" \
      --tensor-parallel-size 8 --max-model-len 32768 --max-num-seqs 8 \
      --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 \
      --block-size 256 --kv-cache-dtype fp8 \
      --trust-remote-code --enable-expert-parallel --async-scheduling \
      --reasoning-parser hy3 \
      --reasoning-parser-plugin "$(pwd)/hy3_reasoning_parser.py" \
      "$@"
}

stop() {
    # Collect exact PIDs once, excluding ourselves: pkill/pgrep -f matches any
    # command line containing the pattern, including a bash -c wrapper that
    # embeds this very command — killing by resolved PID cannot self-match.
    local pat="vllm serve $MODEL_DST" pids
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
    mirror)     mirror ;;
    stop)       stop ;;
    *) echo "usage: $0 [prefetch|serve|fill|warm|save-cache|mirror|stop] [extra vllm args for serve/fill]" >&2; exit 1 ;;
esac
