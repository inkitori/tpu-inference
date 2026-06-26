# Project: tpu-inference

## Local environment paths

- **vLLM venv:** `~/vllm_env` (Python 3.12). Activate with `source ~/vllm_env/bin/activate`, or use the `tpu-env.sh` wrapper below.
- **Upstream vLLM source:** `~/vllm` — the upstream vllm repository
- **TPU helper scripts:** `~/tpu-tooling`

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

### `free-tpu.sh` — fully release the TPU and clean up leftover processes/cruft

```bash
~/tpu-tooling/free-tpu.sh
```

Kills processes holding TPU device nodes (`/dev/vfio/*` on v6e/v5, `/dev/accel*` on v4), removes stale libtpu lockfile + shm segments, verifies with `tpu-info`. Safe anytime — ignores Cloud-TPU host agents, gcsfuse cache, and the current shell. If vLLM runs in Docker, `docker stop` the container first (the script kills the host process but the container may restart it).

## Skills

- **`tpu-model-workflow`** (`.claude/skills/tpu-model-workflow/`) — the workflow + testing/benchmarking discipline for adding model / quant / spec-decode support and doing perf work. Invoke it whenever starting that kind of task.
