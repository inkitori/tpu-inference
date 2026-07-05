#!/bin/bash
# Rebuild the local Hy3-preview-4bit-mtp checkpoint dir on a fresh node.
#
# The MLX 4-bit checkpoint drops the MTP layer (model.layers.80.*). We keep
# the recovered+requantized delta in the model bucket under hy3-mtp/:
#   model-mtp.safetensors        layer-80 weights in MLX int4 (+bf16 extras)
#   config.json                  MLX config + layers.80 router-gate override
#   model.safetensors.index.json merged weight index (35 shards)
#   build_mtp_checkpoint.py      how the delta was built (see doc)
#
# Everything is served from the read-only gcsfuse mount; this script only
# creates a local directory of SYMLINKS (no data copied), so it is safe to
# re-run any time (e.g. after preemption).
#
# Prereqs: the bucket is mounted (~/tpu-tooling/mount-gcs.sh) and the
# mlx-community/Hy3-preview-4bit snapshot exists in the bucket HF cache.
set -euo pipefail

MOUNT=${TPU_GCS_MOUNT:-/tmp/gcs/bucket}
OUT=${1:-$HOME/hy3_mtp/Hy3-preview-4bit-mtp}
DELTA="$MOUNT/hy3-mtp"

SNAP=$(ls -d "$MOUNT"/hub/models--mlx-community--Hy3-preview-4bit/snapshots/*/ | head -1)
[ -d "$SNAP" ] || { echo "MLX snapshot not found under $MOUNT/hub — is the bucket mounted?" >&2; exit 1; }
[ -f "$DELTA/model-mtp.safetensors" ] || { echo "MTP delta not found at $DELTA" >&2; exit 1; }

mkdir -p "$OUT"
# Base MLX shards + tokenizer assets -> symlink from the snapshot.
for f in "$SNAP"model-*.safetensors "$SNAP"tokenizer.json \
         "$SNAP"tokenizer_config.json "$SNAP"chat_template.jinja \
         "$SNAP"generation_config.json; do
    ln -sf "$f" "$OUT/$(basename "$f")"
done
# MTP delta: layer-80 shard + patched config + merged index.
ln -sf "$DELTA/model-mtp.safetensors" "$OUT/model-mtp.safetensors"
cp -f "$DELTA/config.json" "$OUT/config.json"
cp -f "$DELTA/model.safetensors.index.json" "$OUT/model.safetensors.index.json"

echo "checkpoint ready: $OUT"
echo "serve with:"
echo "  SKIP_JAX_PRECOMPILE=0 ONEHOT_MOE_PERMUTE_THRESHOLD=512 ~/tpu-tooling/tpu-env.sh \\"
echo "    vllm serve $OUT \\"
echo "    --tensor-parallel-size 8 --max-model-len 4096 --max-num-seqs 8 \\"
echo "    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 \\"
echo "    --trust-remote-code --enable-expert-parallel \\"
echo "    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'"
