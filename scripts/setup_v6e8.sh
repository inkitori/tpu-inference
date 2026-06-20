#!/usr/bin/env bash
#
# setup_v6e8.sh — bring up a fresh SINGLE-HOST v6e-8 for the GLM 5.2 (glm_moe_dsa)
# bring-up. Encodes steps 2–5 of the v6e-8 runbook: system deps, uv + py3.12 venv,
# clone + pin vLLM, build vLLM for TPU, install THIS tpu-inference fork editable, then
# a HARD GATE that fails unless jax.devices() returns the expected number of TPU chips
# in a single process (this is exactly what distinguishes a real single-host v6e-8 from
# the v6e-16 slice, where the call hangs forever).
#
# Run it from inside your tpu-inference checkout (the branch you want, e.g. glm5.2-dsa):
#     bash scripts/setup_v6e8.sh
#
# Idempotent: safe to re-run. Env knobs:
#     EXPECTED_DEVICES=8   # chips to require (set 4 for a v6e-4)
#     SKIP_APT=1           # skip the sudo apt-get step (no-sudo boxes)
#     PYTHON_VERSION=3.12
#
set -euo pipefail

EXPECTED_DEVICES="${EXPECTED_DEVICES:-8}"
SKIP_APT="${SKIP_APT:-0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_DIR="$(cd "$REPO_ROOT/.." && pwd)"
VLLM_DIR="$PARENT_DIR/vllm"
VENV_DIR="$REPO_ROOT/.venv"

log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[setup:FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -f "$REPO_ROOT/.buildkite/vllm_lkg.version" ]] \
  || die "run this from inside the tpu-inference checkout (missing .buildkite/vllm_lkg.version)"

log "repo:  $REPO_ROOT"
log "vllm:  $VLLM_DIR (sibling)"
log "venv:  $VENV_DIR"
log "gate:  require $EXPECTED_DEVICES TPU chips in one process"

# --- 1. system deps -----------------------------------------------------------
if [[ "$SKIP_APT" == "1" ]]; then
  log "SKIP_APT=1 — skipping apt-get"
else
  log "installing system deps (sudo apt-get)"
  sudo apt-get update -y
  sudo apt-get install -y git build-essential libopenblas-base libopenmpi-dev libomp-dev
fi

# --- 2. uv --------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
[[ -f "$HOME/.local/bin/env" ]] && source "$HOME/.local/bin/env"
command -v uv >/dev/null 2>&1 || die "uv not on PATH; run: source \$HOME/.local/bin/env  then re-run"

# --- 3. venv ------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  log "creating venv (python $PYTHON_VERSION)"
  uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
else
  log "venv exists — reusing"
fi
PYBIN="$VENV_DIR/bin/python"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# --- 4. clone + pin vLLM ------------------------------------------------------
VLLM_COMMIT_HASH="$(tr -d '[:space:]' < "$REPO_ROOT/.buildkite/vllm_lkg.version")"
log "pinned vLLM commit: $VLLM_COMMIT_HASH"
if [[ ! -d "$VLLM_DIR/.git" ]]; then
  log "cloning vLLM -> $VLLM_DIR"
  git clone https://github.com/vllm-project/vllm.git "$VLLM_DIR"
fi
git -C "$VLLM_DIR" fetch --quiet origin || true
git -C "$VLLM_DIR" checkout --quiet "$VLLM_COMMIT_HASH" \
  || die "could not checkout vLLM commit $VLLM_COMMIT_HASH"

# --- 5. build vLLM for TPU ----------------------------------------------------
log "installing vLLM TPU build requirements"
uv pip install --python "$PYBIN" -r "$VLLM_DIR/requirements/tpu.txt" --torch-backend=cpu
log "building vLLM for TPU (VLLM_TARGET_DEVICE=tpu, --no-build-isolation) — first build is slow"
( cd "$VLLM_DIR" && VLLM_TARGET_DEVICE="tpu" uv pip install --python "$PYBIN" -e . --no-build-isolation )

# --- 6. install THIS tpu-inference fork (editable) — must come AFTER vLLM ------
log "installing tpu-inference (editable) from $REPO_ROOT"
( cd "$REPO_ROOT" && uv pip install --python "$PYBIN" -e . )

# --- 7. HARD GATE: TPU mesh is up, single process, no hang --------------------
log "verifying TPU mesh (hard gate, 180s timeout)"
set +e
EXPECTED_DEVICES="$EXPECTED_DEVICES" timeout 180 "$PYBIN" - <<'PY'
import os, sys, jax
devs = jax.devices()
n = len(devs)
plats = sorted({d.platform for d in devs})
print(f"jax.devices(): {n} device(s); platform(s)={plats}")
expected = int(os.environ.get("EXPECTED_DEVICES", "8"))
if "tpu" not in plats:
    print("FAIL: no TPU devices visible to JAX"); sys.exit(2)
if n != expected:
    print(f"FAIL: expected {expected} TPU chips, got {n}"); sys.exit(3)
print(f"OK: {n} TPU chips up in a single process")
PY
rc=$?
set -e
case "$rc" in
  0)   : ;;
  124) die "jax.devices() TIMED OUT (180s) — this is the multi-host-slice hang. You are NOT on a single-host v6e-8; provision one (host_bounds=1,1,1), or set EXPECTED_DEVICES if intentional." ;;
  *)   die "TPU mesh check failed (rc=$rc) — see output above." ;;
esac

log "DONE ✅  Activate the venv in your shell:  source $VENV_DIR/bin/activate"
log "Next (spec Phase 0): python -c \"from transformers import AutoConfig; print(AutoConfig.from_pretrained('zai-org/GLM-5.2').rope_parameters)\""
