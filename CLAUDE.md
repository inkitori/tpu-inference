# tpu-inference — working context

This fork (`github.com:inkitori/tpu-inference`) is being used to bring up **GLM 5.2
(`GlmMoeDsaForCausalLM`, DeepSeek-V3.2-style MLA + DSA) as a native-JAX model on TPU.**

**Authoritative design spec** (read first for any GLM 5.2 / DSA work):
`docs/superpowers/specs/2026-06-19-glm5.2-dsa-jax-tpu-design.md`
Companion research/explainer under `docs/superpowers/research/` predates the spec and
contains superseded claims — the spec is authoritative.

## Sibling source trees (load-bearing — referenced throughout the spec)

Both live one level up, under `/home/enyouki/`:

- **`/home/enyouki/vllm`** — vLLM source. Two roles: (1) the **build dependency** —
  `tpu-inference` installs against a TPU-built vLLM (`VLLM_TARGET_DEVICE=tpu pip install
  --no-build-isolation -e .`); (2) the **GPU reference implementation** (read-only) for
  the DSA path — `models/deepseek_v2.py` (`is_v32` gate, `Indexer`, YaRN `mscale²`,
  fp8 indexer). vLLM line numbers drift — re-locate by symbol.

- **`/home/enyouki/claude-deepseek-v4`** — the DeepSeek-V4 bring-up fork (real checkout
  under `work/tpu-inference/tpu_inference/`). It is the **DSA port source + TPU-lessons
  blueprint**: the JAX indexer and the already-wired Mosaic `sparse_attn` kernel to
  **adapt V4→V3.2/GLM** (drop compressor/CSA/attn_sink; keep relu + one-hot gather +
  `-1` sentinel), plus the hard-won multi-device fixes (the S1 uninit-HBM-on-reshard
  class: host-stack weights straight into the sharded layout; `shard_map`-wrap Mosaic;
  host-side RoPE-freq precompute). See spec §0, §5, §8.

## Environment

- **venv:** `/home/enyouki/.venv` (Python 3.12.13, created with `uv`; gitignored). This is
  where the validated `transformers==5.12.1` glm_moe_dsa oracle resolves
  (`/home/enyouki/.venv/lib/python3.12/site-packages/transformers/models/glm_moe_dsa/`).
  `uv` is at `~/.local/bin/uv`. System Python is 3.10 and **cannot** install the pinned
  `jax==0.10.1` (needs ≥3.11) — always use the venv.
  Install/refresh deps with `uv pip install --python /home/enyouki/.venv/bin/python ...`
  (absolute path — a cwd-relative `.venv` resolves wrong from the repo root).
- Pins (`requirements.txt`): `jax==0.10.1`, `jaxlib==0.10.1`, `libtpu==0.0.41`,
  `flax==0.12.4`, `transformers==5.12.1` (the validated `glm_moe_dsa` oracle pin — matches
  requirements.txt and spec §6; the spec's line-cited symbols and Phase-0 fixture target 5.12.1), torch.
- The persistent XLA **compilation cache is on by default** — edit→run is near-instant
  after the first cold compile.

## Hardware reality

**Decision (spec §1): develop v1 on a single-host `v6e-8`** (`host_bounds=1,1,1`, one
process, 8 Trillium chips), provisioned separately from the slice below. The multi-device
gate surface is the **8-chip** sharded mesh — it arms the S1 (uninit-HBM-on-reshard) class,
which is a *sharding/reshard* phenomenon, not a host-count one (see spec §0/§1 and
`docs/superpowers/research/2026-06-20-s1-single-host-reproducibility.md`).
`scripts/setup_v6e8.sh` (commit c883aa10) stands this box up and ends in a **hard 8-chip
gate** (`EXPECTED_DEVICES=8`) that fails unless `jax.devices()` returns 8 chips in one process.

**Why a separate v6e-8 is required:** the box this spec was written on (`v6spoteu717`) is
**worker 0 of a 4-host v6e-16 slice** (`--deepsea_host_bounds=2,2,1`; 4 chips/host via
`/dev/vfio/{0,1,2,3}`), *not* an independent host. Verified 2026-06-19 from libtpu init logs
(`--deepsea_slice_builder_worker_addresses=10.164.0.206/.215/.212/.216:8471`). On it, a
single-process `jax.devices()` **hangs forever** — the SliceBuilder blocks waiting for the
other 3 hosts to join (JAX prints *"TPU backend initialization is taking more than 60.0
seconds. Did you run your code on all TPU hosts?"*). This is **not** contention or a wedged
runtime — chips can be free (`sudo fuser /dev/vfio/*` empty) and it still hangs; restarting
`tpu-runtime` does not help. **Fallback if a v6e-8 is unavailable:** drive all 4 hosts of
this slice together (`gcloud compute tpus tpu-vm ssh <name> --worker=all --command=...`, or
Ray/Pathways — what `mark`'s `run_cluster.sh` did), yielding all 16 chips, at the cost of a
heavier multi-host dev loop.

It is also a **shared, multi-tenant pod** (users `mark`, `furka`, `devin*`, …). Read-only
signals without docker/sudo: `/tmp/tmux-*.log`, `ps -eo user,cmd | grep -i python`. **Never
kill another user's job to free a device** — coordinate first.

A single-device pass is **not** TPU validation — the multi-device **8-chip** gate is required
(spec §7).
