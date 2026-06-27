# 16 — IMPL: windowed draft attention (Lever A) + cut fixed overhead (Lever B)

Phase: land the two coupled levers that flip A(DFlash) > B(target-only) at
in=1/out=4096/c=32. Date: 2026-06-27. Manager: impl-window. Builds on
15-feasibility (the two-lever plan + R=1.58<<accept), 12-impl-kvcache (the KV
cache + cached forward this builds on), 10-impl-condense (per-slot state must
follow condense).

## TL;DR / VERDICT — BOTH LEVERS LANDED + MICROBENCH-PROVEN ⇒ NEEDS_TEST
The isolated real-8-chip microbench shows the windowed cached draft forward is
now FLAT in context (~14.2ms @W=256, ~17.5ms @W=512, constant C=1024→4608) vs
the full-context cached forward that scaled 26→86ms — a ~6x cut at C=4608.
Combined projection clears B=3752 across the realistic post-Lever-B overhead
band (W=256: 1.12–1.78x B for FIXED 8–25ms; W=512: 1.04–1.60x B). Lever B
(drop redundant _ctx_buf write + batch device_gets + free 3.96 GiB) is
numerics-NEUTRAL and moves FIXED from ~40ms toward 15–25ms. ⇒ NEEDS_TEST:
lossless re-verify of the WINDOWED path through condense + a real-draft
acceptance check at W (the window changes proposals), then BENCH.

## WHICH LEVERS + HOW (both flag-gated under DFLASH_KV_CACHE=1)

### LEVER A — windowed draft attention (env DFLASH_DRAFT_WINDOW=W, default 0=off)
Slice the cached draft K/V to the LAST W context positions per slot BEFORE the
attention-score matmul, so the O(C·B) score matmul physically shrinks to O(W·B).
LOSSLESS for the target (it still verifies every token); only the draft
proposals change (less context → possibly lower acceptance).
- **prepare_inputs** (spec_decode/vllm/dflash.py): when windowed, padded_ctx =
  fixed W (rounded up to the 512 pad block at load_model); per-slot
  `win_start[i] = max(0, ctx_len_i - W)`; window-valid count = `min(ctx_len_i,W)`.
  Build the (N, W) ctx mask + (N, W+B) position_ids over the window; thread
  `win_start` (N,) into the target_hidden_states tuple (now a 6-tuple). The
  noise positions stay absolute (`ctx_len_i + arange(B)`) — unchanged.
- **propose** (spec_decode/vllm/dflash.py): unpack + pass `win_start` to the
  cached forward.
- **get_draft_forward_cached_fn / draft_forward_cached** (models/vllm/dflash.py):
  new `win_start` arg (static_argnums shifted 7,8 → 8,9). Replace the prefix
  slice `k_cache[:, :, :padded_ctx]` with a per-slot windowed GATHER:
  `jax.vmap(lambda c,s: lax.dynamic_slice_in_dim(c, s, padded_ctx, axis=1),
  in_axes=(1,0), out_axes=1)` over the N axis, for K and V. W is static so XLA
  traces ONE ctx width. win_start all-zeros == the old prefix slice (Lever A off
  is byte-identical to pre-change cached path).
- **precompile**: when windowed, warm the cached forward at the single width W
  (was the full padded_sizes sweep) with the win_start arg.

WHY the gather is correct (RoPE alignment): the cache row index == absolute ctx
position == the RoPE position baked into K at write time. A CONTIGUOUS row gather
[win_start : win_start+W) preserves that — no re-RoPE. In the cached forward the
ctx slice of position_ids is dead (only noise positions drive in-forward RoPE),
so windowing only needs the right cache rows + the right (N,W) valid mask. When
ctx_len < W, win_start=0 and the window degenerates to the [0:W) prefix (rows
[ctx_len:W) are unwritten and masked off) — identical to today.
The cache stays FULL buf_len width; _batched_kv_write / _move_kv_rows /
_kv_project / condense are UNTOUCHED (windowing is read-only at forward time).

### LEVER B — cut the fixed per-step overhead (numerics-NEUTRAL on flag-on path)
On the KV-cache path, _ctx_buf is WRITTEN every step but NEVER READ (kv_project
pulls new rows straight from raw_hidden; the cached forward reads only the K/V
caches — confirmed by L3 code map). So on `self._use_kv_cache`:
- DROP the _ctx_buf ALLOCATION (load_model) → frees ~3.96 GiB → restores util
  0.75 headroom (the 13-test-kvcache HBM blocker that forced util≤0.6). Store
  `self._buf_len` for the dead-row plan.
- DROP the per-step `_batched_ctx_write` (the redundant masked RMW over the
  multi-GiB buffer) and the condense `_move_ctx_rows`. The scatter PLAN
  (slot/dst/valid host loop) is kept — the K/V cache write reuses it.
- SKIP the _ctx_buf write/move/full-forward precompile warms on the flag-on path.
- BATCH the two per-step `device_get`s (seq_lens + query_start_loc) into ONE
  pytree round-trip (was two serialized stalls).
All numerics-neutral on the flag-on path: nothing on that path read _ctx_buf.
Flag-OFF path is completely unchanged (alloc/write/move all still run).

## CONDENSE HANDLING
Unchanged + correct. The K/V cache move (`_move_kv_rows`) already mirrors slot
moves on the flag-on path (10-impl-condense / 12-impl-kvcache); Lever B just
drops the now-dead _ctx_buf move alongside it. Windowing is read-only at forward
time and the cache stays full-width, so condense backfill / InputBatch.condense
mirroring is untouched. _ctx_len stays "absolute written count" (win_start is a
derived per-step host quantity; _ctx_len is never mutated to W), so the next
step's write dst_row = ctx_len + offset still lands at the right absolute row.

## DECISIVE MICROBENCH (REAL 8-chip v6e TP8, N=32, median of 30 warm) — scratchpad/bench_dflash_window_realmesh.py
config: N=32 B=8 L=8 KVH=8 HD=64 raw_dim=14400 hidden=2880 num_spec=7 buf_len=4616
```
     C |   FULL | CACHED full | CACHED W=256 | CACHED W=512   (ms, fwd-only)
  1024 | 21.506 |      26.089 |       14.269 |       17.540
  2048 | 40.868 |      41.904 |       14.267 |       17.497
  4096 | 87.786 |      77.304 |       14.245 |       17.517
  4608 | 98.413 |      85.645 |       14.237 |       17.525
```
Windowed cached STEP total @C=4608 (project+write+fwd+sample): W=256 20.67ms /
W=512 23.98ms. kv_project(B=8) 0.539ms.
**The windowed cached forward is FLAT in C** (the O(C·B) score matmul is gone;
only O(W·B) remains) — ~14.2ms @W=256 regardless of context, vs 85.6ms full-
context cached @C=4608 = **~6x** cut, vs 98.4ms FULL recompute = ~6.9x.

## COMBINED PROJECTION vs B = 3752 tok/s (step = FIXED + device_step; accept=6 assumed)
```
  W=256 (device step @C=4608 = 20.67ms):
    FIXED= 8ms -> 6696 tok/s = 1.78x B   FIXED=15 -> 5382 = 1.43x
    FIXED=25ms -> 4204 tok/s = 1.12x B   FIXED=40 -> 3164 = 0.84x (pre-Lever-B)
  W=512 (device step @C=4608 = 23.98ms):
    FIXED= 8ms -> 6005 = 1.60x   FIXED=15 -> 4926 = 1.31x
    FIXED=25ms -> 3920 = 1.04x   FIXED=40 -> 3001 = 0.80x
```
Lever B moves FIXED from ~40ms (pre) toward 15–25ms (dropped the ~6.5ms device
_ctx_buf write + a device_get + freed HBM; async = Phase 2, NOT done here). So
the realistic post-Lever-B point is ~1.1–1.4x B for W=256. **Both levers
together clear B; Lever A alone (FIXED still 40) does NOT (0.84x) — matches
15-feasibility.** Even if windowing drops accept 6→4, W=256 @ FIXED≤25 still
clears B (~2800–4460 tok/s at accept 4) — but the accept check is required.

## NUMERICS / CORRECTNESS NOTES
- Lever B: bit-NEUTRAL on the flag-on path (nothing read _ctx_buf there).
  Flag-off path byte-unchanged.
- Lever A: win_start all-zeros == old prefix slice (proven CPU-exact + clamp-safe
  by L3: vmap dynamic_slice gather matches numpy ref for prefix/mid/near-end +
  start-clamp when win_start+W>buf_len). Windowing CHANGES draft proposals (less
  context) — this is STRUCTURALLY lossless (target verifies every token) but
  REQUIRES (a) a real-draft acceptance check at W (does accept stay high enough
  that the projection still beats B?) and (b) a lossless greedy re-verify of the
  windowed path through condense, because the draft path changed. ⇒ NEEDS_TEST.
- Unit tests: 17/17 green (test_dflash_torchax + test_dflash) after updating the
  two stale 5-tuple unpacks to the new 6-tuple (win_start).

## HBM
Lever B drops _ctx_buf (3.96 GiB) on the flag-on path; the K/V cache (2.25 GiB)
stays. Net per-impl-math −1.7 GiB ⇒ util 0.75 should fit again (removes the
13-test-kvcache blocker that forced util≤0.6). Not re-measured on a full serve
this round (microbench only) — the BENCH manager should confirm at util 0.75.

## COMMITS (branch dflash, pushed)
- c620d4cd Lever A (windowed draft attn) + Lever B (drop redundant _ctx_buf)
- 28f9a432 update DFlash unit tests for the win_start 6-tuple (17/17)

## WHAT REMAINS
1. NEEDS_TEST (the gate before bench, because the draft path changed):
   - Real-draft ACCEPTANCE at DFLASH_DRAFT_WINDOW=256 (and 512) on the serve
     path c=32 — confirm mean accept len stays high enough (≳4) that the
     projection still beats B. If 256 craters accept, use 512.
   - Lossless greedy re-verify of the WINDOWED path through condense
     (perfect-draft 100% per-slot + greedy-vs-target) — the GOAL ladder, since
     proposals changed. (Lever B alone is bit-neutral; Lever A needs this.)
2. Then NEEDS_BENCH: A(DFLASH_KV_CACHE=1, DFLASH_DRAFT_WINDOW=256, util 0.75)
   vs B(target-only) at in=1/out=4096/c=32, WARM. Confirm A>B and util 0.75 fits.
3. Phase 2 (later, gated): async scheduling to cut FIXED further.

## STATUS
- [x] Lever A implemented (windowed gather, flag DFLASH_DRAFT_WINDOW)
- [x] Lever B implemented (drop _ctx_buf alloc/write/move + batch device_get)
- [x] CPU gather numeric check (prefix/mid/near-end + clamp) PASS
- [x] unit tests 17/17 green
- [x] isolated real-8-chip microbench: windowed fwd FLAT ~14.2ms (~6x), step
      total 20.67ms @W=256, projection clears B at FIXED≤25ms
- [ ] real-draft acceptance at W (NEEDS_TEST)
- [ ] lossless re-verify of windowed path through condense (NEEDS_TEST)
- [ ] full serve A-vs-B bench at util 0.75 (NEEDS_BENCH, the bench manager)
