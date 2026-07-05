# Hy3 MTP checkpoint: what it is and how to rebuild it

`mlx-community/Hy3-preview-4bit` dropped the MTP layer (`model.layers.80.*`)
during MLX conversion — the config still says `num_nextn_predict_layers: 1`,
but the weights are gone. MTP speculative decode (the biggest per-user decode
TPS win for this model: 54 → 62 TPS/user on v6e-8) therefore needs a rebuilt
checkpoint that adds layer 80 back.

## Persistent artifacts (survive preemption)

`gs://personal-mark-us-east5-b/vllm/hy3-mtp/` — visible read-only at
`/tmp/gcs/bucket/hy3-mtp/` once `~/tpu-tooling/mount-gcs.sh` has run:

| file | contents |
|---|---|
| `model-mtp.safetensors` | layer-80 weights, requantized to MLX int4 affine (gs=64); `eh_proj`/`enorm`/`hnorm`/norms bf16; router gate 8-bit |
| `config.json` | MLX config + the `model.layers.80.mlp.router.gate` 8-bit override |
| `model.safetensors.index.json` | merged 35-shard weight index |
| `build_mtp_checkpoint.py`, `restore_mtp_checkpoint.sh` | copies of the scripts in this dir |

The 34 base MLX shards come from the bucket HF cache
(`/tmp/gcs/bucket/hub/models--mlx-community--Hy3-preview-4bit`), which also
persists.

## After a preemption (fast path, no downloads)

```bash
~/tpu-tooling/mount-gcs.sh          # if not already mounted
bash tools/hy3_mtp/restore_mtp_checkpoint.sh
```

This recreates `~/hy3_mtp/Hy3-preview-4bit-mtp` as symlinks into the mount
(seconds, no data copied) and prints the serving command. Serve with
`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` — k=1 is
deliberate: k=2 measured position-1 acceptance of only 33% (the single MTP
layer is reused for the second step) and no TPS gain.

## Rebuilding the delta from scratch (only if the bucket copy is lost)

1. Download base shards 111–112 of `tencent/Hy3-preview` (~7.9GB) into
   `~/hy3_mtp/shards/` (command in the header of `build_mtp_checkpoint.py`).
2. `python tools/hy3_mtp/build_mtp_checkpoint.py` — quantizes layer 80 into
   MLX int4 (nibble packing verified bit-exact against real MLX tensors at
   build time), writes the local checkpoint dir.
3. Re-upload the delta:
   `gcloud storage cp ~/hy3_mtp/Hy3-preview-4bit-mtp/{model-mtp.safetensors,config.json,model.safetensors.index.json} gs://personal-mark-us-east5-b/vllm/hy3-mtp/`

## Code prerequisites

The serving-side integration lives on the `hy3` branch of this repo
(vllm_model_wrapper draft kwarg introspection, host-side embed dequant,
draft impl resolution) — commits `82716ce8`, `6388f88c`. Upstream vLLM
already provides the `HYV3MTP` model class.
