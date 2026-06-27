# 03 — SPEED BENCH (DFlash gpt-oss-20b, branch `dflash`)

Phase: speed only (correctness PROVEN in 02-test.md). Date: 2026-06-27. Manager: bench.

## HEADLINE BLOCKER: GOAL bench point (concurrency 32) is NOT RUNNABLE.
The DFlash torchax decode path is **hard-coded single-sequence**. It cannot serve
concurrency 32. This is an intentional implementation limit (a feature gap), so a
fresh IMPL manager must add batched/multi-seq DFlash decode before the GOAL
concurrency-32 A-vs-B speed bench can run at all.

### Hard evidence (code-level, conclusive)
`tpu_inference/runner/speculative_decoding_manager.py:102-112`, on the
`method == "dflash"` decode-propose path — runs every decode step, NO flag gates it:
```python
elif self.runner.speculative_config.method == "dflash":
    ...
    assert not async_scheduling, "DFlash does not support async scheduling"
    assert self.runner.dp_size == 1, "DFlash does not support attention data parallelism"
    assert self.runner.input_batch.num_reqs <= 1, \
        "DFlash does not support batched (multi-request) decoding"
```
- `num_reqs <= 1` fires the instant the scheduler batches >=2 requests into one
  decode step — which it WILL at concurrency 32 with `--max-num-seqs 32`. Server
  raises AssertionError "DFlash does not support batched (multi-request) decoding".
- The wiring commit is literally `a03d42e4 "spec-decode: wire DFlash into the
  torchax pipeline (concurrency 1)"`. Single-seq is by design in this commit.
- No scheduler cap auto-downgrades dflash to max_num_seqs=1; the ONLY enforcement
  is this runtime assert. So `--max-num-seqs 32` is not silently clamped — it crashes.
- NOTE: empirical c=32 crash capture was started but not completed (serve time);
  the code + assert message + commit title are decisive on their own. The next
  manager will trivially reproduce by launching config A with `--max-num-seqs 32`
  and firing >=2 concurrent requests.

## What IS confirmed working (config A, the supported regime)
- Config A launches and serves cleanly at **`--max-num-seqs 1`, `--max-model-len 5120`**
  (input=1 + output=4096 fits with headroom). All v6e workarounds applied: EP,
  RAGGED_GATHER v1, HF_HOME=/home/enyouki/local_hf, --no-async-scheduling,
  DRAFT_MODEL_IMPL_TYPE=torchax, num_speculative_tokens 7.
- **HBM @ max-model-len 5120, max-num-seqs 1, DFlash ON: ~28.7–28.9 GiB / 31.25 per
  chip (~2.3 GiB headroom).** Tight but stable. Bumping max-num-seqs to 32 also
  needs the KV cache to fit 32 concurrent 5120-len seqs — HBM pressure is a SECOND
  risk on top of the assert, to watch once the assert is lifted.

## Concurrency-1 per-stream A-vs-B (DFlash's supported regime)
Was being measured (warmup discarded; ~5-8 sequential c=1 requests, out=4096,
temp 0) to tell the next manager whether the per-stream speedup justifies building
batched support. NUMBERS NOT FINALIZED at handoff (run in progress; the c=32
blocker took priority). From prior runs (02-test.md / STATE.md): DFlash real mean
accepted length ~2.35–3.14 (per-pos accept 0.74→0.04). Mean accept ~2.9 implies a
meaningful per-step token multiplier, but whether it nets a wall-clock win depends
on draft+verify overhead — UNMEASURED here.

## VERDICT
- Against the literal GOAL (faster at concurrency 32): **CANNOT VERIFY — BLOCKED.**
  DFlash torchax is single-seq only; c=32 serving is impossible without an impl change.
- Per-stream (c=1): not finalized; mean accept ~2.9 from prior runs suggests upside
  exists but the wall-clock A-vs-B win is unmeasured.

## Handoff -> NEEDS_IMPL
Next manager must add batched/multi-request support to the DFlash torchax decode
path so concurrency-32 serving works, THEN re-run this bench. Concretely:
- Relax/remove `assert num_reqs <= 1` in
  `tpu_inference/runner/speculative_decoding_manager.py:111`.
- Make `propose_dflash_draft_token_ids` + `runner_utils.host_extract_sampled_tokens`
  + the host-side DFlash proposer (`spec_decode/vllm/dflash.py`) handle a real batch
  of sequences (per-request draft-length layout), not just one.
- Check the `dp_size == 1` / `async_scheduling` asserts don't also need to move.
- Then: warm-cache c=32 bench, both configs, report out tok/s + latency + TPOT +
  DFlash mean accepted length @ c=32 + HBM headroom. Watch HBM (only ~2.3 GiB
  headroom at c=1/len5120 already).
