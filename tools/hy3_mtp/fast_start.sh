#!/bin/bash
# fast_start.sh — sub-5-minute Hy3 serving startup on a fresh v6e-8 node.
#
# Measured 2026-07-13 (warm bucket caches, this repo @ hy3): prefetch ~45s
# (168GB weights ~40s + XLA cache ~5s) + serve-to-ready 202s = ~250s total,
# with zero runtime XLA compiles (VLLM_XLA_CHECK_RECOMPILATION=1 verified).
# Serve breakdown: ~40s imports+TPU init, 46s safetensors read, ~14s MoE
# processing (expert-sharded device_put, 8-way parallel PCIe), 7s draft,
# ~91s AOT precompile+warmup (fully persistent-cache-hit).
#
# Why this exists: serving straight off the gcsfuse mount reads the 168 GB
# checkpoint at ~550 MB/s single-stream (~5 min just for weights, and the
# 160 GB default file-cache cap means the model NEVER stays warm — LRU
# thrashes on every sequential re-read). Parallel `gcloud storage rsync`
# pulls the same bytes at ~6 GiB/s (measured 2026-07-13: 168 GB in ~40 s),
# and safetensors load from tmpfs runs at ~1.2 s/shard vs 8.9 s/shard
# through gcsfuse.
#
# Subcommands:
#   prefetch    (default) parallel-download the checkpoint + XLA compile
#               cache into /dev/shm, then print the serve command.
#   warm        exercise a LIVE server with the request variants that JIT
#               lazily (concurrency/greedy/logprobs/top-k) so they land in
#               the compile cache; reports new-entry count.
#   save-cache  upload the local XLA compile cache to the bucket. Run once
#               after warm on a new (jax/libtpu/flag) configuration;
#               prefetch restores it on the next fresh node.
#
# The XLA cache lives in the bucket under vllm/xla-cache/$CACHE_TAG. Cache
# keys include jax/libtpu versions and compile flags, so stale entries are
# harmless (they just miss); re-run save-cache after upgrading the stack.
#
# Overridable env: GCS_BUCKET, TPU_GCS_MOUNT, MODEL_NAME, CACHE_TAG, SHM_ROOT.
set -euo pipefail

MOUNT=${TPU_GCS_MOUNT:-/tmp/gcs/bucket}
MODEL_NAME=${MODEL_NAME:-Hy3-preview-4bit-mtp}
CACHE_TAG=${CACHE_TAG:-hy3-v6e8}
SHM_ROOT=${SHM_ROOT:-/dev/shm}
MODEL_DST="$SHM_ROOT/models/$MODEL_NAME"
XLA_DST="$SHM_ROOT/xla_cache_hy3"

# Bucket: explicit env > the mounted bucket > same-region auto-discovery.
BUCKET=${GCS_BUCKET:-$(findmnt -n -o SOURCE "$MOUNT" 2>/dev/null || true)}
if [ -z "$BUCKET" ]; then
    zone=$(curl -s -m 5 -H "Metadata-Flavor: Google" \
      "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}')
    region_uc=$(printf '%s' "${zone%-*}" | tr '[:lower:]' '[:upper:]')
    BUCKET=$(gcloud storage buckets list --format="value(name,location)" \
             | awk -v r="$region_uc" '$2 == r {print $1; exit}')
fi
[ -n "$BUCKET" ] || { echo "cannot determine bucket; set GCS_BUCKET" >&2; exit 1; }

MODEL_SRC="gs://$BUCKET/vllm/models/$MODEL_NAME"
XLA_SRC="gs://$BUCKET/vllm/xla-cache/$CACHE_TAG"

save_cache() {
    [ -d "$XLA_DST" ] && [ -n "$(ls -A "$XLA_DST" 2>/dev/null)" ] \
      || { echo "no local XLA cache at $XLA_DST — run a serve first" >&2; exit 1; }
    echo "uploading XLA cache $XLA_DST -> $XLA_SRC"
    gcloud storage rsync -r -q "$XLA_DST" "$XLA_SRC"
    echo "done: $(gcloud storage ls "$XLA_SRC" | wc -l) entries in bucket"
}

prefetch() {
    # /dev/shm is tmpfs (RAM): checkpoint+cache need ~170 GB, kept for the
    # life of the node. The v6e-8 host has 1.4 TB; if this node is smaller,
    # point SHM_ROOT at local NVMe instead.
    avail_gb=$(df --output=avail -BG "$SHM_ROOT" | tail -1 | tr -dc 0-9)
    [ "$avail_gb" -ge 200 ] || echo "WARN: only ${avail_gb}G free on $SHM_ROOT (need ~170G)" >&2

    mkdir -p "$MODEL_DST" "$XLA_DST" 2>/dev/null || {
        sudo mkdir -p "$MODEL_DST" "$XLA_DST"
        sudo chown -R "$(id -u)" "$SHM_ROOT/models" "$XLA_DST"
    }

    echo "prefetching checkpoint $MODEL_SRC -> $MODEL_DST (~40s for 168GB)"
    time gcloud storage rsync -r -q "$MODEL_SRC" "$MODEL_DST"

    if gcloud storage ls "$XLA_SRC" >/dev/null 2>&1; then
        echo "restoring XLA cache $XLA_SRC -> $XLA_DST"
        time gcloud storage rsync -r -q "$XLA_SRC" "$XLA_DST"
    else
        echo "NOTE: no XLA cache in bucket yet ($XLA_SRC)."
        echo "      First serve compiles cold; afterwards run: $0 save-cache"
    fi

    cat <<EOF

ready. serve with (low-concurrency c<=8 MTP k=2 config; for c=16-32 use
--max-num-seqs 32 and add --additional-config '{"compilation_sizes":[96]}'):

  SKIP_JAX_PRECOMPILE=0 ONEHOT_MOE_PERMUTE_THRESHOLD=1024 \\
  VLLM_XLA_CACHE_PATH=$XLA_DST \\
  VLLM_XLA_CHECK_RECOMPILATION=1 NUM_PRECOMPILE_WORKERS=4 \\
  ~/tpu-tooling/tpu-env.sh vllm serve $MODEL_DST \\
    --tensor-parallel-size 8 --max-model-len 4096 --max-num-seqs 8 \\
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \\
    --trust-remote-code --enable-expert-parallel --async-scheduling \\
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'

VLLM_XLA_CHECK_RECOMPILATION=1 matters on BOTH the fill run and warm runs:
it drops jax's persistent-cache thresholds (default: skip compiles <1s) so
the ~300 small helper jits get cached too, and it makes guarded runtime
recompiles raise instead of silently stalling a request.

after the first cold serve on a new jax/libtpu/config: run '$0 warm' against
the live server (some sampler/RNG jit variants only materialize under real
traffic: concurrency, greedy, logprobs), THEN '$0 save-cache'.
EOF
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

case "${1:-prefetch}" in
    prefetch)   prefetch ;;
    warm)       warm ;;
    save-cache) save_cache ;;
    *) echo "usage: $0 [prefetch|warm|save-cache]" >&2; exit 1 ;;
esac
