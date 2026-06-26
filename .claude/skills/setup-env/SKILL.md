---
name: setup-env
description: >-
  How to set up / install the vLLM + tpu-inference dev environment from source,
  and how to spin up a fresh "scratch" vLLM clone + scratch venv pinned to a
  specific tpu-inference commit. Use this whenever the task is installing the
  environment from scratch, or checking out a different tpu-inference commit and
  needing a matching vLLM source + venv that won't disturb the main ~/vllm_env.
---

# Setup: vLLM + tpu-inference from source

tpu-inference is a **vLLM plugin**: both must be installed from source, and vLLM
must be checked out to the exact revision tpu-inference pins in
`.buildkite/vllm_lkg.version`. Wrong vLLM revision → import errors (e.g.
`RoutedExperts` / `FusedMoE`).

## System deps (once per machine)

```bash
sudo apt-get update && sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev
```

## Fresh install from source

```bash
# 1. Clone tpu-inference, read its pinned vLLM revision
git clone https://github.com/vllm-project/tpu-inference.git
export VLLM_COMMIT_HASH="$(cat tpu-inference/.buildkite/vllm_lkg.version)"

# 2. Clone vLLM and check out the pinned revision
git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout "${VLLM_COMMIT_HASH}" && cd ..

# 3. venv (uv, Python 3.12)
uv venv vllm_env --python 3.12
source vllm_env/bin/activate

# 4. Install vLLM (TPU target, CPU torch backend)
cd vllm
uv pip install -r requirements/tpu.txt --torch-backend=cpu
VLLM_TARGET_DEVICE="tpu" uv pip install -e . --no-build-isolation
cd ..

# 5. Install tpu-inference
cd tpu-inference && uv pip install -e . && cd ..
```

## Scratch env for a different tpu-inference commit

When you need to test tpu-inference at commit `<SHA>` **without disturbing the
main `~/vllm` / `~/vllm_env`**, build an isolated scratch tree. The vLLM
revision is whatever *that commit* pins — never assume it matches the current
checkout.

```bash
SHA=<tpu-inference-commit>
SCRATCH=~/scratch-$SHA            # pick any throwaway dir

# tpu-inference at the target commit (worktree avoids re-cloning; or git clone)
git -C ~/tpu-inference worktree add "$SCRATCH/tpu-inference" "$SHA"

# vLLM revision that THIS commit pins
VLLM_COMMIT_HASH="$(cat "$SCRATCH/tpu-inference/.buildkite/vllm_lkg.version")"
git clone https://github.com/vllm-project/vllm.git "$SCRATCH/vllm"
git -C "$SCRATCH/vllm" checkout "$VLLM_COMMIT_HASH"

# separate scratch venv — do NOT reuse ~/vllm_env
uv venv "$SCRATCH/venv" --python 3.12
source "$SCRATCH/venv/bin/activate"

cd "$SCRATCH/vllm"
uv pip install -r requirements/tpu.txt --torch-backend=cpu
VLLM_TARGET_DEVICE="tpu" uv pip install -e . --no-build-isolation
cd "$SCRATCH/tpu-inference" && uv pip install -e . && cd -
```

Cleanup when done: `deactivate; rm -rf "$SCRATCH"; git -C ~/tpu-inference worktree prune`.

## Notes

- Always re-read `.buildkite/vllm_lkg.version` after any tpu-inference checkout —
  the pin moves with the commit.
- `uv` is installed here; prefer it for venvs/installs.
- The repo's main env is `~/vllm_env` (see project CLAUDE.md). Scratch venvs keep
  experiments from corrupting it; the `tpu-env.sh` wrapper honors `TPU_VENV` if
  you want to run a scratch venv through it.
