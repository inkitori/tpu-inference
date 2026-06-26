# Project: tpu-inference

## Commit and push often

This is a preemptible node, which means all your work can disappear at any moment. Make sure to commit and push often.

## Local environment paths

- **vLLM venv:** `~/vllm_env` (Python 3.12). Activate with `source ~/vllm_env/bin/activate`, or use the `tpu-env.sh` wrapper below.
- **Upstream vLLM source:** `~/vllm` — the upstream vllm repository
- **TPU helper scripts:** `~/tpu-tooling`

You also have uv installed here, so use that to create venvs and install packages.

## TPU tooling scripts (`~/tpu-tooling`)

### Always confirm the GCS bucket is in this VM's region first

Before any operation that uses the VM with the mounted bucket — **serving a model,
loading a checkpoint, benchmarking** (anything via `tpu-env.sh`, which points
`HF_HOME` at the gcsfuse-mounted bucket) — first make sure the bucket is in the
**same region** as this TPU VM. A cross-region bucket means every checkpoint byte
is a slow, billable cross-region transfer.

```bash
~/tpu-tooling/check-gcs-region.sh   # exit 0 = same region, 1 = mismatch, 2 = undetermined
```

`tpu-env.sh` runs this automatically and **refuses to start on a region mismatch**;
override an intentional cross-region run with `ALLOW_CROSS_REGION=1`.

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
- `ALLOW_CROSS_REGION` — set to `1` to bypass the same-region bucket guard (default unset; the guard hard-blocks on a region mismatch)

Example: `OMP_NUM_THREADS=32 TPU_VENV=~/other_env ~/tpu-tooling/tpu-env.sh vllm serve ...`

Note, this sets SKIP_JAX_PRECOMPILE=1 which behaves like --enforce-eager for benchmarking (cold XLA compiles on first use) and will cause JIT XLA compilations. Benchmarks will be affected if the XLA cache isn't warm, so make sure to do a warmup bench first and check if any of the requests took suspiciously long to return.

### `free-tpu.sh` — fully release the TPU and clean up leftover processes/cruft

```bash
~/tpu-tooling/free-tpu.sh
```

Kills processes holding TPU device nodes (`/dev/vfio/*` on v6e/v5, `/dev/accel*` on v4), removes stale libtpu lockfile + shm segments, verifies with `tpu-info`. Safe anytime — ignores Cloud-TPU host agents, gcsfuse cache, and the current shell. If vLLM runs in Docker, `docker stop` the container first (the script kills the host process but the container may restart it).
