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
  System Python is 3.10 and **cannot** install the pinned
  `jax==0.10.1` (needs ≥3.11) — always use the venv.
  Install/refresh deps with `uv pip install --python /home/enyouki/.venv/bin/python ...`
  (absolute path — a cwd-relative `.venv` resolves wrong from the repo root).
- Pins (`requirements.txt`): `jax==0.10.1`, `jaxlib==0.10.1`, `libtpu==0.0.41`,
  `flax==0.12.4`, `transformers==5.12.1` (the validated `glm_moe_dsa` oracle pin; spec §6).
- The persistent XLA **compilation cache is on by default** — edit→run is near-instant
  after the first cold compile.

## Hardware reality

**This box is a single-host `v6e-8`** (`v6spoteu624`): 8 Trillium (v6e) chips on one host, one
process. Single-process `jax.devices()` returns all 8 — **no multi-host init hang**. This is the
spec §1 v1 validation surface. `scripts/setup_v6e8.sh` stands it up and ends in a hard 8-chip
gate (`EXPECTED_DEVICES=8`). The 8-chip sharded mesh arms the S1 (uninit-HBM-on-reshard) class —
a *sharding/reshard* phenomenon, not a host-count one (spec §0/§1,
`docs/superpowers/research/2026-06-20-s1-single-host-reproducibility.md`).

Shared, multi-tenant pod (`mark`, `furka`, `devin*`, …). Read-only signals without docker/sudo:
`/tmp/tmux-*.log`, `ps -eo user,cmd | grep -i python`. **Never kill another user's job** —
coordinate first.

A single-device pass is **not** TPU validation — the multi-device 8-chip gate is required (spec §7).
