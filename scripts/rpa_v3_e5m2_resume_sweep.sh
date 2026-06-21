#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CHECKPOINT_ROOT=${RPA_E5M2_CHECKPOINT_ROOT:-/home/enyouki/rpa_v3_e5m2_checkpoint}
GCS_URI=${RPA_E5M2_CHECKPOINT_GCS:-gs://personal-mark-eu/tpu-inference-checkpoints/rpa_v3_e5m2_v6e}

cd "$REPO"

if ! command -v gsutil >/dev/null 2>&1; then
  echo "gsutil is required to restore the durable checkpoint from $GCS_URI" >&2
  exit 1
fi

mkdir -p "$CHECKPOINT_ROOT"
gsutil -m rsync -r "$GCS_URI" "$CHECKPOINT_ROOT"

mkdir -p /tmp/rpa_v3_e5m2_v6e_full_db /tmp/jax_rpa_tune_cache
cp -f "$CHECKPOINT_ROOT"/db/*.json /tmp/rpa_v3_e5m2_v6e_full_db/
cp -f "$CHECKPOINT_ROOT"/artifacts/rpa_v3_e5m2_targets.json /tmp/rpa_v3_e5m2_targets.json

rm -rf /tmp/rpa-tuner-tools /tmp/rpa-stubs
tar -C /tmp -xzf "$CHECKPOINT_ROOT/artifacts/rpa-tuner-tools.tar.gz"
tar -C /tmp -xzf "$CHECKPOINT_ROOT/artifacts/rpa-stubs.tar.gz"

if [[ ! -d /tmp/rpa-site311/jax ]]; then
  python3.11 -m pip install --target /tmp/rpa-site311 --upgrade \
    absl-py numpy pytest parameterized requests pyyaml \
    jax==0.9.2 jaxlib==0.9.2 libtpu==0.0.39 \
    --extra-index-url https://pypi.org/simple \
    -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
fi

if [[ "${RPA_E5M2_START_CHECKPOINT_LOOP:-1}" == "1" ]]; then
  mkdir -p "$CHECKPOINT_ROOT"
  nohup bash -lc \
    "while true; do python3.11 '$REPO/scripts/rpa_v3_e5m2_snapshot_checkpoint.py'; sleep 600; done" \
    >> "$CHECKPOINT_ROOT/checkpoint_sync.log" 2>&1 &
  echo "$!" > "$CHECKPOINT_ROOT/checkpoint_sync.pid"
  echo "checkpoint sync loop pid $(cat "$CHECKPOINT_ROOT/checkpoint_sync.pid")"
fi

KERNEL_TUNER_LOCAL_DB_PATH=/tmp/rpa_v3_e5m2_v6e_full_db \
RPA_TARGETS_JSON=/tmp/rpa_v3_e5m2_targets.json \
RPA_KV_DTYPE=float8_e5m2 \
RPA_BKV_P_LIST=1,2,4,8,16,32 \
RPA_BQ_SZ_LIST=8,16,32,64,128 \
KERNEL_TUNER_MEASURE_ITERS=21 \
JAX_COMPILATION_CACHE_DIR=/tmp/jax_rpa_tune_cache \
PYTHONPATH=/tmp/rpa-stubs:/tmp/rpa-site311:/tmp/rpa-tuner-tools:"$REPO" \
python3.11 -m tools.kernel.tuner.v1.kernel_tuner_runner \
  --kernel_tuner_name=rpa_v3_kernel_tuner \
  --run_locally=True \
  --case_set_id=rpa_v3_e5m2_v6e_full \
  --run_id=001 \
  --case_set_desc='RPA v3 v6e q_bfloat16 kv_float8_e5m2 exhaustive block sweep over existing e4m3 shape keys; bkv_p 1,2,4,8,16,32; bq 8,16,32,64,128; median 21 iters' \
  --tpu_version=tpu6e \
  --tpu_cores=8

