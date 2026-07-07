# Project: tpu-inference

IMPORTANT NOTE: This node is preemptible and your work can disappear at any moment, so make sure to commit and push often.

## Local environment paths

- **vLLM venv:** `~/vllm_env` (Python 3.12). Activate with `source ~/vllm_env/bin/activate`, or use the `tpu-env.sh` wrapper below.
- **Upstream vLLM source:** `~/vllm` — the upstream vllm repository
- **TPU helper scripts:** `~/tpu-tooling`

You also have uv installed here, so use that to create venvs and install packages.

## TPU tooling scripts (`~/tpu-tooling`)

### `tpu-env.sh` — load the vLLM/JAX-on-TPU env and run a command in it (no sourcing needed)

```bash
~/tpu-tooling/tpu-env.sh vllm serve <model> --tensor-parallel-size 8 --max-model-len 2048
~/tpu-tooling/tpu-env.sh python my_script.py
~/tpu-tooling/tpu-env.sh                      # interactive shell with env loaded
```

Optional env vars (set inline or export first):

- `TPU_VENV` — venv path (default `$HOME/vllm_env`)
- `HF_HOME` — HF cache dir (default `/tmp/gcs/bucket/vllm`)
- `HF_HUB_OFFLINE` — skip network checks (default `1`)
- `OMP_NUM_THREADS` — thread clamp during checkpoint load (default `16`; keep clamped — avoids 180-thread oversubscription on MoE copy)
- `MODEL_IMPL_TYPE` — vLLM/JAX impl (default `vllm`)
- `SKIP_JAX_PRECOMPILE` — faster startup (default `1`)
- `TPU_GCS_MOUNT` — gcsfuse mount path (default `/tmp/gcs/bucket`)

Example: `OMP_NUM_THREADS=32 TPU_VENV=~/other_env ~/tpu-tooling/tpu-env.sh vllm serve ...`

Note, this sets SKIP_JAX_PRECOMPILE=1 which behaves like --enforce-eager for benchmarking (cold XLA compiles on first use) and will cause JIT XLA compilations. Benchmarks will be affected if the XLA cache isn't warm, so make sure to do a warmup bench first and check if any of the requests took suspiciously long to return.

### `mount-gcs.sh` — (re)mount the GCS model bucket at `/tmp/gcs/bucket`

```bash
~/tpu-tooling/mount-gcs.sh              # mount, or repair a drifted/stale mount (idempotent)
~/tpu-tooling/mount-gcs.sh --force      # always unmount + remount
GCS_BUCKET=other-bkt ~/tpu-tooling/mount-gcs.sh
```

This is a shared serving node: the bucket must be mounted root-owned with
`allow_other` and `--only-dir vllm`, so that both the root serving job and user
runs can read it and `HF_HOME=/tmp/gcs/bucket` finds checkpoints at
`<mount>/hub/models--*`. The script enforces all of that — always use it
instead of running gcsfuse by hand.

- **Bucket selection:** if `GCS_BUCKET` is unset, it auto-discovers a bucket in
  this VM's region (via the metadata server + `gcloud storage buckets list`),
  preferring one with the `vllm/` prefix. Nodes get re-imaged across regions,
  so don't hardcode bucket names anywhere.
- **Region guard:** refuses to mount a bucket in a different region than the VM
  (cross-region reads are slow and cost egress). Override only deliberately
  with `ALLOW_CROSS_REGION=1`.
- If model loads fail with "model not found" despite the model being in the
  bucket (with `HF_HUB_OFFLINE=1`), or `/tmp/gcs/bucket` looks empty/stale, run
  this script first — a wrong/private/off-by-one-dir mount is the usual cause.

### `check-gcs-region.sh` — verify the bucket is in the VM's region

```bash
~/tpu-tooling/check-gcs-region.sh             # auto-discover bucket from live mount
~/tpu-tooling/check-gcs-region.sh my-bucket   # check a specific bucket
```

Exit codes: `0` same region (or covering multi-region), `1` mismatch, `2`
undeterminable. Used as the gate inside `mount-gcs.sh` and `tpu-env.sh`; run it
standalone when picking a bucket for a new node.

### `free-tpu.sh` — fully release the TPU and clean up leftover processes/cruft

```bash
~/tpu-tooling/free-tpu.sh
```

Kills processes holding TPU device nodes (`/dev/vfio/*` on v6e/v5, `/dev/accel*` on v4), removes stale libtpu lockfile + shm segments, verifies with `tpu-info`. Safe anytime — ignores Cloud-TPU host agents, gcsfuse cache, and the current shell. If vLLM runs in Docker, `docker stop` the container first (the script kills the host process but the container may restart it).

