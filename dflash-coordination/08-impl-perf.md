# 08 — IMPL: DFlash per-step decode cost (PERF round)

Phase: cut DFlash per-step draft cost. Date: 2026-06-27. Manager: impl-perf.
Builds on 06-impl-hbm (serves c=32/len4224/util0.75) and 07-bench-c32 (10.6x slower).

## Isolated microbench (L3, real draft model, N=32, bf16, v6e-8 TP8)
Draft loaded standalone (no 20B target; dummy embed/lm_head [201088,2880]).
_ctx_buf (32,4608,14400) bf16 = 3.96 GiB confirmed.

| C (ctx) | fc + 8 attn draft_forward (ms) |
|--------:|-------------------------------:|
|   512   |  15.1 |
|  1024   |  21.7 |
|  2048   |  41.6 |
|  4096   |  88.3 |
|  4608   |  99.1 |

| _ctx_buf write (32 reqs, 8-row block into 3.96 GiB buf) | ms |
|---|---:|
| EAGER 32x lax.dynamic_update_slice (python loop)        | 234.4 |
| jitted + donate, batched in-place                        | 1.63 |

Sampler ~0.2-0.3 ms (negligible). Target-only step = 8.4 ms; break-even ~46 ms/step.

## Findings (the two bottlenecks, quantified)
1. **The eager write loop is the DOMINANT cost (~234 ms/step), context-INDEPENDENT.**
   Each of 32 eager dynamic_update_slice calls copies the full 3.96 GiB buffer (no
   donation). A jitted donated batched write does the SAME updates in place in 1.63 ms
   (~144x). PURE perf, same values written. THIS IS LEVER #2 (chip task_d0ad4933).
2. **fc/attn forward grows ~linearly with C (O(ctx) recompute).** ~+19 ms/+1024 ctx
   over a ~10 ms floor. Crosses 46 ms break-even at C~2300; at C=4608 it's 99 ms alone.
   This is LEVER #1 (KV cache the context). Numerics-sensitive + architecturally
   non-trivial: the HF draft RECOMPUTES context K/V every call by design (only the
   noise block is cached). Deferred to a fresh manager AFTER the write fix re-benches.

## LANDED THIS ROUND — LEVER #2 (write fix), commit 6b6acd49
Replaced the eager 32x lax.dynamic_update_slice loop in prepare_inputs() with ONE
jitted+donated masked scatter `_batched_ctx_write(ctx_buf, raw_hidden, slot_idx,
dst_row, valid_mask)` (static_argnums=(0,), donate_argnums=(1,)). Host bookkeeping
(_ctx_len/_prev_seq_len/shrink-repair/clamp) UNCHANGED — only the DEVICE write is now
batched+in-place. Masked read-modify-write: invalid rows write `cur` back (no-op);
inactive slots + beyond-n_copy rows + a dedicated dead row are bit-preserved. buf_len
bumped one pad block ONLY when _next_padded_size(max_model_len) <= max_model_len (gives
a guaranteed dead row); precompile warms the scatter per num_tokens_paddings bucket.

## VERIFIED (isolated, two independent L3 passes)
- **Bit-identical**: np.array_equal(old eager loop, new jit) on the FULL buffer across
  mixed prefill+decode+inactive+ragged+buffer-edge+clamp+all-no-op, incl real
  (32,4608,14400). Host _ctx_len state matches too. PASS.
- **Decode-step write cost (real sharded 8-chip mesh)**: ~2.2 ms flat for total_tokens
  32/64/128/256 (decode bucket ≈256 rows), ~3.1 ms even at 4096. So ~234 ms → ~2.2 ms
  (~106x) at decode. The earlier 14.9 ms was an UNSHARDED-layout artifact, not intrinsic.
- **Bucketing risk does NOT materialize**: num_tokens_paddings are per-step-actual pow2;
  decode pads to the small per-step bucket (~256), NOT max. Confirmed in runner.
- **HBM UNCHANGED at GOAL**: max_model_len=4224 → _next_padded_size=4608 > 4224 → bump
  condition FALSE → buf_len stays 4608 (3.96 GiB). The 06-impl-hbm budget is intact.
  (Bump only fires when max_model_len is an exact multiple of 512.)
- **Unit tests 5/5 GREEN** (one assertion legitimately updated: mock _ctx_buf shape
  reflects the dead-row slack at max_model_len=128 → (1,144,32)).

## Per-step picture (N=32)
- BEFORE: write ~234 ms + fc/attn forward 15-99 ms = ~250-334 ms/step
- AFTER : write ~2.2 ms + fc/attn forward 15-99 ms = **~17-101 ms/step**
- Break-even vs target-only is ~46 ms/step. Now dominated by the O(ctx) fc/attn forward
  (LEVER #1, KV cache) which breaks even only for C ≲ ~2k. NEXT ROUND.

## NUMERICS RISK + why NEEDS_TEST
Provably bit-identical in isolation ⇒ risk is ~none. BUT this touched the proven per-slot
_ctx_buf WRITE path, and the GOAL mandates re-running the correctness ladder after any
numerics-touching change. Cheap insurance: re-run perfect-draft batch>1 (the 05-test-batched
ragged b=32 @ GOAL c=32 config) to confirm still 1302/1302 / lossless, THEN the c=32 speed
bench. If TEST passes, the speed bench should show a materially faster DFlash (per-step ~234ms
write term removed); whether it now BEATS target-only depends on the O(ctx) forward — likely
still loses at out=4096 until LEVER #1 (KV cache) lands, but should be far closer.

## REMAINING (fresh manager) — LEVER #1: KV-cache the context
Bottleneck #1 is now dominant: the HF draft recomputes fc (14400→2880) over the FULL
context + 8 dense attn layers EVERY step (O(ctx), grows to 99 ms at C=4608). The HF model
(z-lab/gpt-oss-20b-DFlash, modeling at local_hf .../snapshots/d535.../dflash.py)
RECOMPUTES context K/V every call BY DESIGN — only the noise block is cached. Q is
noise-only; K/V = concat[context | noise]; attention is bidirectional (is_causal=False);
context has no queries. To cache context: store context K/V (post-fc, post-RoPE) once and
reuse, recomputing only the NEW accepted positions' context K/V each step. This is
numerics-sensitive (RoPE/positions, mask) and needs a modeling-path change → a design
decision for a fresh IMPL manager (possibly BLOCKED_USER if it needs editing HF remote code
vs a torchax-side cache). Microbench shows this is THE lever for out=4096.

## Status
- [x] microbench characterize  [x] implement write fix  [x] verify bit-identical vs eager
- [x] verify decode cost ~2.2ms + HBM unchanged  [x] unit tests 5/5  [x] commit+push 6b6acd49
- [ ] re-test (perfect-draft b=32, lossless)  [ ] re-bench c=32  [ ] LEVER #1 KV cache

## Commits
- 6b6acd49 spec-decode: batched jitted+donated _ctx_buf write (kill ~234ms eager loop)

## Scripts
- scratchpad/bench_dflash_step.py (real draft, fc/forward + eager write)
- scratchpad/bench_write_only.py, bench_ctx_write.py (eager-vs-donate write timing)
