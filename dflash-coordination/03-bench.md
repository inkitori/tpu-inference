# 03 — SPEED BENCH (DFlash gpt-oss-20b, branch `dflash`)

Phase: speed only (correctness already PROVEN in 02-test.md). Date: 2026-06-27.

## HEADLINE: GOAL bench point (concurrency 32) is NOT RUNNABLE as-is.

The DFlash torchax path is **hard-coded to single-sequence decoding**. It cannot
serve concurrency 32. This is an intentional implementation limit, not a config
mistake — so a fresh manager must add batched/multi-seq DFlash support before the
concurrency-32 A-vs-B speed comparison can be run at all.

### Hard evidence (code-level, unambiguous)
`tpu_inference/runner/speculative_decoding_manager.py:102-112`, on the
`method == "dflash"` decode-propose path, runs unconditionally (no flag gates it):
```python
elif self.runner.speculative_config.method == "dflash":
    ...
    assert not async_scheduling, "DFlash does not support async scheduling"
    assert self.runner.dp_size == 1, "DFlash does not support attention data parallelism"
    assert self.runner.input_batch.num_reqs <= 1, \
        "DFlash does not support batched (multi-request) decoding"
```
- The `num_reqs <= 1` assert fires the instant the scheduler batches ≥2 requests
  into one decode step — which it WILL do immediately at concurrency 32 with
  `--max-num-seqs 32`. Result: server crashes with that AssertionError.
- The commit that wired DFlash in is literally titled
  `a03d42e4 "spec-decode: wire DFlash into the torchax pipeline (concurrency 1)"`.
  Single-seq is by design in this commit, not an accident.
- There is NO scheduler-level cap that auto-limits dflash to max_num_seqs=1; the
  only enforcement is this runtime assert. So `--max-num-seqs 32` does not get
  silently downgraded — it crashes.

Conclusion: there is no point launching a concurrency-32 serve to "watch it fail";
the code, the assert message, and the commit title all agree. The required bench
is blocked on an IMPL change (batched multi-seq DFlash decode).

## What CAN be measured now: concurrency-1 A vs B (per-stream speedup)
DFlash's supported regime is concurrency 1. Whether DFlash is faster *per stream*
tells the next manager whether building batched support is even worth it (if it's
not faster at c=1, batching won't save it either).

Bench point used: input≈1, output=4096, temperature=0, warm XLA cache, concurrency 1.
- Config A (DFlash ON):  target + draft, --max-num-seqs 1, --max-model-len 5120,
  num_speculative_tokens 7, all v6e workarounds (EP, v1 gather, local HF cache,
  --no-async-scheduling).
- Config B (target-only): same, no --speculative-config.
Method: cold-compile warmup run DISCARDED (SKIP_JAX_PRECOMPILE=1 ⇒ first use
compiles); then ~5-8 sequential (c=1) measured requests averaged; verified no
single measured request was a hidden compile.

### RESULTS (filled in by L3 bench agent — pending)
- Config A (DFlash, c=1): out tok/s = TBD; mean latency = TBD; TPOT = TBD ms/tok;
  mean accepted length = TBD; per-pos accept = TBD.
- Config B (target-only, c=1): out tok/s = TBD; mean latency = TBD; TPOT = TBD ms/tok.
- HBM headroom @ max-model-len 5120: TBD GiB/32 per chip.
- c=1 verdict: TBD (A faster/slower than B, by X%).

## VERDICT
- Against the literal GOAL (faster at concurrency 32): **CANNOT VERIFY — blocked.**
  DFlash torchax is single-seq only; concurrency-32 serving is impossible without
  an implementation change (batched multi-request DFlash decode).
- Concurrency-1 per-stream A-vs-B: see results above (pending agent).

## Handoff
This phase taps out to L1 with a NEW blocker that a fresh IMPL manager should own:
add batched/multi-sequence support to the DFlash torchax decode path (remove/relax
the `num_reqs <= 1` assert and make `propose_dflash_draft_token_ids` +
`host_extract_sampled_tokens` handle a real batch), so the GOAL concurrency-32
bench becomes runnable. The c=1 number indicates whether that work is worthwhile.
