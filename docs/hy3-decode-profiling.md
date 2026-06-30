# Hy3-preview-4bit — EP=8 decode profile (runbook)

Profiles a full 8-request decode batch with the phased profiler, gated on batch
size via `PHASED_PROFILER_MIN_BATCH_SIZE` (see `tpu_inference/runner/utils.py`).

## Dataset (rebuild if `/workspace/sharegpt_slices_4k_512.jsonl` is missing)

```bash
sudo mkdir -p /workspace && sudo chown $(id -u):$(id -g) /workspace
curl -sSL "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" -o /dev/shm/sharegpt_v3.json
python - <<'PY'
import json; data=json.load(open("/dev/shm/sharegpt_v3.json")); n=0
out=open("/dev/shm/sharegpt_corpus.txt","w",encoding="utf-8")
for c in data:
    for t in c.get("conversations",[]):
        v=t.get("value","")
        if v: out.write(v+"\n\n"); n+=len(v)+2
    if n>=40*1024*1024: break   # ~40MB is ample
PY
source ~/vllm_env/bin/activate
HF_HOME=/tmp/gcs/bucket HF_HUB_OFFLINE=1 python -m vllm.benchmarks.datasets.create_txt_slices_dataset \
  --input /dev/shm/sharegpt_corpus.txt --output /workspace/sharegpt_slices_4k_512.jsonl \
  --tokenizer mlx-community/Hy3-preview-4bit \
  --num-prompts 1000 --input-len 3000 --output-len 1024 --seed 0 --trust-remote-code
```

3000 in + 1024 out + 15 chat-template tokens = 4039 ≤ 4096, no rejections.

## 1. Server (profiler on)

```bash
~/tpu-tooling/free-tpu.sh
rm -rf /workspace/phased_profile_hy3
SKIP_JAX_PRECOMPILE=0 \
PHASED_PROFILING_DIR=/workspace/phased_profile_hy3 \
PHASED_PROFILER_MIN_BATCH_SIZE=8 \
~/tpu-tooling/tpu-env.sh vllm serve mlx-community/Hy3-preview-4bit \
  --tensor-parallel-size 8 --max-model-len 4096 --max-num-seqs 8 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 \
  --trust-remote-code --enable-expert-parallel
```

`SKIP_JAX_PRECOMPILE=0` warms XLA at startup so the capture is warm.

## 2. Bench (capture is automatic)

```bash
HF_HOME=/dev/shm vllm bench serve \
  --backend openai-chat --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
  --model mlx-community/Hy3-preview-4bit \
  --dataset-name custom --dataset-path /workspace/sharegpt_slices_4k_512.jsonl \
  --skip-chat-template --custom-output-len 1024 --num-prompts 1000 --ignore-eos
```

Server logs `Starting profiling for decode_only phase` → `... finished` (15 steps).
Done within ~30–60s of traffic; the full 1000 prompts are not required.

## 3. View

```bash
uv venv ~/xprof_env --python 3.12 && source ~/xprof_env/bin/activate && uv pip install xprof
xprof --logdir /workspace/phased_profile_hy3 -p 6006
```

## Output

`/workspace/phased_profile_hy3/<phase>/plugins/profile/<ts>/`
- `*.xplane.pb` — xprof/TensorBoard; `*.trace.json.gz` — Perfetto (ui.perfetto.dev)
- `<phase>/dp_rank_0/batch_composition_stats_*.json` — per-step batch stats

Capture fires at decode start (~3k KV). Pin to 4k with
`PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD=4000`.
