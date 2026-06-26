# DFlash spec-decode bringup state (local v6e-8)

Status of serving **target `openai/gpt-oss-20b` + draft `z-lab/gpt-oss-20b-DFlash`**
with DFlash speculative decoding on the local **TPU v6e-8**, torchax path
(`MODEL_IMPL_TYPE=vllm`, `DRAFT_MODEL_IMPL_TYPE=torchax`).

## Verified-working launch

```bash
env HF_HOME=/home/enyouki/local_hf DRAFT_MODEL_IMPL_TYPE=torchax \
  RAGGED_GATHER_VERSION=v1 RAGGED_GATHER_REDUCE_VERSION=v1 \
  ~/tpu-tooling/tpu-env.sh vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 8 --enable-expert-parallel --no-async-scheduling \
  --max-model-len 2048 --max-num-seqs 1 \
  --speculative-config '{"model": "z-lab/gpt-oss-20b-DFlash", "num_speculative_tokens": 7, "method": "dflash"}'
```

Smoke result: correct output, ~2.89 mean acceptance length, ~27% avg draft
acceptance, per-position acceptance `0.571 → 0.071` across the 7 draft slots.

## Why each flag is load-bearing

Every flag fixes a distinct failure hit during bringup:

- **`--enable-expert-parallel`** — the runner builds a 2D `(data, model)` mesh and
  `ShardingAxisName2D.EXPERT == 'model'`. EP routes the MoE through the GMM_EP
  kernel, which shards the 32 experts across `model=8` (4/chip, ~4.5 GiB each).
  Plain TP=8 uses GMM_TP and dies with `IndivisibleError`: the local MXFP4→fp8
  requant (`REQUANTIZED_BLOCK_SIZE=512`, see `mxfp4.py`) produces a
  `(32, 6, 1, 3072)` scale whose axis-1 = 6 cannot divide 8.
  Do **not** instead pass `--additional-config` with `tensor_parallelism=1`/
  `expert_parallelism=8`: on the 2D-mesh path that collapses `model` to 1 device,
  putting all weights on chip 0 → OOM at KV-cache allocation.
- **`RAGGED_GATHER_VERSION=v1 RAGGED_GATHER_REDUCE_VERSION=v1`** — the default v2
  SparseCore gather fails to lower on v6e at decode time
  (`NotImplementedError ... pltpu.async_copy` dynamic indexer shape). The legacy
  v1 kernels lower fine.
- **`--no-async-scheduling`** — async scheduling is ON by default on TPU; DFlash
  asserts no async scheduling and `dp_size == 1` (see the dflash guards in
  `runner/speculative_decoding_manager.py`). Pair with **`--max-num-seqs 1`**
  (DFlash is single-seq only).
- **`num_speculative_tokens: 7`** pairs with the draft's `block_size=8` (the
  proposer fallback is `block_size = num_speculative_tokens + 1`).

## Environment gotchas

- **gcsfuse mount is unusable from this user.** `/tmp/gcs/bucket` is mounted
  root-only (no `allow_other`) and with `--only-dir vllm`, so neither the user
  nor vLLM can read it (and even root's path would be `…/bucket/hub`, not the
  `…/bucket/vllm/hub` that `tpu-env.sh` assumes). Workaround: stage checkpoints
  to a **local HF cache** via `gcloud storage cp` and point `HF_HOME` there
  (this also bypasses `tpu-env.sh`'s same-region guard). Bucket layout is the
  standard HF cache: `gs://<bucket>/vllm/hub/models--<org>--<name>`.
- **Draft remote code needs `datasets`.** `z-lab/gpt-oss-20b-DFlash` ships
  `dflash.py`, which imports `datasets`; install it into the venv with
  `VIRTUAL_ENV=~/vllm_env uv pip install datasets` (the venv has no `pip`).

## Not-yet-done / caveats

- The MXFP4→fp8 requant in `mxfp4.py` is a **local v6e workaround**, not a
  general fix — gate on TPU generation before merging (v7+ has native FP4
  Mosaic lowering).
- The v1 gather fallback is a v6e workaround for the same reason.
