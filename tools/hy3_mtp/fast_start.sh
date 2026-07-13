#!/bin/bash
# fast_start.sh — sub-5-minute Hy3 serving startup on a fresh v6e-8 node.
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
#   save-cache  upload the local XLA compile cache to the bucket. Run once
#               after the first fully-warm serve on a new (jax/libtpu/flag)
#               configuration; prefetch restores it on the next fresh node.
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
  ~/tpu-tooling/tpu-env.sh vllm serve $MODEL_DST \\
    --tensor-parallel-size 8 --max-model-len 4096 --max-num-seqs 8 \\
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \\
    --trust-remote-code --enable-expert-parallel --async-scheduling \\
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'

after the first cold serve on a new jax/libtpu/config, persist the cache:
  $0 save-cache
EOF
}

case "${1:-prefetch}" in
    prefetch)   prefetch ;;
    save-cache) save_cache ;;
    *) echo "usage: $0 [prefetch|save-cache]" >&2; exit 1 ;;
esac
