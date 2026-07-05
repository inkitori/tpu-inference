# Rebuilding the Hy3 serving node after preemption

Durable state lives in two places: GitHub (`inkitori/tpu-inference` branch
`hy3`, `inkitori/tpu-tooling`) and the model bucket
(`gs://personal-mark-us-east5-b/vllm/` — MLX snapshot in `hub/`, MTP delta in
`hy3-mtp/`). Everything else on the node is rebuildable:

1. Clone tooling + repo:
   ```bash
   git clone git@github.com:inkitori/tpu-tooling.git ~/tpu-tooling
   git clone -b hy3 git@github.com:inkitori/tpu-inference.git ~/tpu-inference
   ```
2. Mount the bucket: `~/tpu-tooling/mount-gcs.sh` (read-only at `/tmp/gcs/bucket`).
3. Rebuild the venv from source per the setup-env flow.
   Upstream vLLM pin used with this branch: `cc7981599` (vllm-project/vllm),
   editable-installed into `~/vllm_env` alongside tpu-inference.
4. Restore the MTP checkpoint (symlink-only, seconds):
   ```bash
   bash ~/tpu-inference/tools/hy3_mtp/restore_mtp_checkpoint.sh
   ```
   It prints the serving command (MTP k=1, `ONEHOT_MOE_PERMUTE_THRESHOLD=512`).
5. Bench dataset: regenerate `/workspace/sharegpt_slices_4k_512.jsonl` with the
   recipe in `git show 4cea8a6e:docs/hy3-decode-profiling.md`.

First serve after a rebuild compiles cold (no XLA cache) — do a warmup pass
before benchmarking.
