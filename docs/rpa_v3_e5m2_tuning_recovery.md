# RPA v3 e5m2 Tuning Recovery

This documents how to resume the v6e `float8_e5m2` RPA v3 block-size sweep if
the preemptible node is lost.

Durable checkpoint:

- `gs://personal-mark-eu/tpu-inference-checkpoints/rpa_v3_e5m2_v6e`

The checkpoint contains:

- `db/*.json`: local tuner DB, including `CaseResults.json`
- `artifacts/rpa_v3_e5m2_targets.json`: the 494 target keys derived from the
  existing v6e e4m3 table surface
- `artifacts/rpa-tuner-tools.tar.gz`: patched historical tuner harness
- `artifacts/rpa-stubs.tar.gz`: minimal local vLLM stubs needed by the harness
- `metadata.json`: latest progress summary

Resume from a fresh clone:

```bash
git clone git@github.com:inkitori/tpu-inference.git
cd tpu-inference
git checkout rpa-tuning
bash scripts/rpa_v3_e5m2_resume_sweep.sh
```

The runner uses the same local DB case set and skips case IDs already present in
`CaseResults.json`, so do not delete the restored DB unless intentionally
starting over.

Exact resumed sweep settings:

- `case_set_id=rpa_v3_e5m2_v6e_full`
- `run_id=001`
- `RPA_KV_DTYPE=float8_e5m2`
- `RPA_BKV_P_LIST=1,2,4,8,16,32`
- `RPA_BQ_SZ_LIST=8,16,32,64,128`
- `KERNEL_TUNER_MEASURE_ITERS=21`
- `KERNEL_TUNER_LOCAL_DB_PATH=/tmp/rpa_v3_e5m2_v6e_full_db`

Check latest durable progress:

```bash
gsutil cat gs://personal-mark-eu/tpu-inference-checkpoints/rpa_v3_e5m2_v6e/metadata.json
```

After the sweep completes, query winners with the restored harness:

```bash
PYTHONPATH=/tmp/rpa-stubs:/tmp/rpa-site311:/tmp/rpa-tuner-tools:$PWD \
python3.11 /tmp/rpa-tuner-tools/tools/kernel/tuner/v1/inspect_result_cli.py \
  --source local \
  --db-path /tmp/rpa_v3_e5m2_v6e_full_db \
  query_min_latency \
  --case_set_id rpa_v3_e5m2_v6e_full \
  --run_id 001 \
  --show page_size \
  --show num_q_heads \
  --show num_kv_heads \
  --show head_dim \
  --show max_model_len \
  --show kv_dtype \
  --show bkv_p \
  --show bq_sz \
  --show latency_us \
  --show warmup_us
```

