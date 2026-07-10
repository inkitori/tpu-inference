# Deploying Hy3-preview-4bit + MTP for max per-user TPS (v6e-8)

Measured 2026-07-10, warm XLA cache, `vllm bench serve` ShareGPT/random,
temperature 0, checkpoint from `restore_mtp_checkpoint.sh`, TP=8 + EP,
`MODEL_IMPL_TYPE=vllm` (default). TPS/user = 1000 / mean TPOT.

## TL;DR: pick the config by concurrency

| concurrency | spec config | TPS/user (measured) |
|---|---|---|
| 1–3 | MTP k=1 | 103–111 (≥100 PASS) |
| 4–8 | MTP **k=2** | ~85 at c=8 (random in=1; k=1 gives ~78) |
| 16–32 | k=1 or k=2 (tie) | 43.9 at c=32 ShareGPT (k=1); k=2 within noise |

**100 TPS/user is achievable only up to concurrency ~3** on this hardware.
At c=32 the hard ceiling is the MoE weight read: 64 verify tokens × top-8 hit
~93% of all 192 experts every step, so each chip reads ~21 GB/step — a
12.7 ms floor at v6e HBM bandwidth (1.64 TB/s) before psums (4.2 ms),
attention, routing, or drafting. The gmm kernel already runs at 85–90% of
HBM peak at this shape; with every remaining software lever landed the step
bottoms out around 28 ms → ~70 TPS/user max at c=32. Reaching 100 at c=32
needs a v6e-16 (halves per-chip expert bytes), a trained drafter with accept
length ≥2.6 at k=1 cost, or a ~3-bit expert quant. See the decode-step
profile breakdown in the repo memory / `docs/hy3-decode-profiling.md` flow.

## Serve commands

Common env (all cases):

```bash
SKIP_JAX_PRECOMPILE=0            # precompile at startup; benches are warm
VLLM_XLA_CACHE_PATH=~/xla_cache_hy3
ONEHOT_MOE_PERMUTE_THRESHOLD=1024
```

`ONEHOT_MOE_PERMUTE_THRESHOLD=1024` is load-bearing at every concurrency:
it swaps the MoE combine's ragged gathers for a one-hot matmul whenever the
routed batch is ≤1024 rows (= 128 padded tokens × top-8, covering decode at
c≤32 for k=1 and k=2). At c=32 it cut 2.3 ms/step: mean TPOT 24.6 → 22.8 ms.

### Low concurrency (interactive, c ≤ 8) — the 100-TPS/user regime

```bash
SKIP_JAX_PRECOMPILE=0 ONEHOT_MOE_PERMUTE_THRESHOLD=1024 \
VLLM_XLA_CACHE_PATH=~/xla_cache_hy3 \
~/tpu-tooling/tpu-env.sh vllm serve ~/hy3_mtp/Hy3-preview-4bit-mtp \
  --tensor-parallel-size 8 --max-model-len 4096 --max-num-seqs 8 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
  --trust-remote-code --enable-expert-parallel --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

- k=2 is the low-concurrency winner: ShareGPT accept length 2.03
  (position-0 ~72%, position-1 ~44% — the old 33% k=2 note predates this
  measurement) and the verify batch stays inside small token buckets.
  Random c=8: TPOT 11.7 ms = 85 TPS/user vs ~78 with k=1.
- `--gpu-memory-utilization 0.90`: k=2's `jit__propose` program needs
  ~920 MB of HBM headroom; 0.95 fails at startup with
  RESOURCE_EXHAUSTED. (k=1 tolerates 0.95.)

### High concurrency (throughput, c = 16–32)

```bash
SKIP_JAX_PRECOMPILE=0 ONEHOT_MOE_PERMUTE_THRESHOLD=1024 \
VLLM_XLA_CACHE_PATH=~/xla_cache_hy3 \
~/tpu-tooling/tpu-env.sh vllm serve ~/hy3_mtp/Hy3-preview-4bit-mtp \
  --tensor-parallel-size 8 --max-model-len 4096 --max-num-seqs 32 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
  --trust-remote-code --enable-expert-parallel --async-scheduling \
  --additional-config '{"compilation_sizes": [96]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

- k=1 and k=2 tie at c=32 (mean TPOT 22.8 vs 23.5 ms; k=2 has the better
  median). Prefer k=2 only if the same server also takes low-concurrency
  traffic; otherwise k=1 with `--gpu-memory-utilization 0.95` and no
  `compilation_sizes` is the simplest config.
- If using k=2, `--additional-config '{"compilation_sizes":[96]}'` is
  REQUIRED for throughput: the 32-req verify batch is 96 tokens, and without
  a 96 bucket it pads to 128 (+4.6 ms/step for phantom tokens). k=1's
  64-token verify already lands exactly on a power-of-2 bucket.

## Knobs that do NOT help (measured; don't re-litigate)

- `SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES=1024` (SparseCore psum offload):
  regression at c=2; leave on auto.
- `--max-num-batched-tokens 1024`: no change — prefill interference is not
  chunk-size-bound.
- `VLLM_TPU_BUCKET_PADDING_GAP=32`: no gain, and novel prompt lengths
  JIT-compile inline unless everything is precompiled (hours of buckets).
  Use `compilation_sizes` for targeted extra buckets instead.
- MTP k=2 without the 96 bucket at c=32: pays 128-token padding, pure wash.
- gmm `tile_m` tuning at c=32 shapes: the kernel is DMA-bound at 85–90% of
  HBM peak there; the tm16 win only applies to the ≤256-row (c≤16 k=1)
  decode shapes and is already wired in via `small_m_tiling`.

## Benchmarking notes

- Always warm up: run the bench once and discard, or judge medians until
  P99 TPOT is within ~1.5× the median. First contact with a novel prompt
  length compiles inline even with `SKIP_JAX_PRECOMPILE=0`.
- `vllm bench serve --dataset-name sharegpt --dataset-path
  /dev/shm/sharegpt_v3.json --temperature 0 --ignore-eos --max-concurrency C
  --num-prompts 3C` approximates steady-state c=C.
- `--num-prompts` below `--max-concurrency` silently caps concurrency at
  num-prompts (e.g. `--num-prompts 8 --max-concurrency 32` is a c=8 bench).
