# 16 — IMPL: windowed draft attention (Lever A) + cut fixed overhead (Lever B)

Phase: land the two coupled levers that flip A(DFlash) > B(target-only) at
in=1/out=4096/c=32. Date: 2026-06-27. Manager: impl-window. Builds on
15-feasibility (the two-lever plan + R=1.58<<accept), 12-impl-kvcache (the KV
cache + cached forward this builds on), 10-impl-condense (per-slot state must
follow condense).

## TL;DR / VERDICT — LEVER A IS A DEAD END (kills acceptance); LEVER B LANDED ⇒ BLOCKED_USER
**LEVER A (windowing) does NOT beat B — it craters acceptance.** The isolated
microbench was beautiful (windowed cached fwd FLAT ~14.2ms@W=256, ~6x cheaper),
BUT the real-draft serve acceptance check at c=32 is DECISIVE: mean accept len
collapses to **~2.5** at BOTH W=256 and W=512 (vs ~6.0–6.7 full-context
baseline). Doubling the window 256→512 recovered NOTHING ⇒ the loss is INTRINSIC
to truncating the DFlash draft's context, not a tunable knob in this range. At
accept 2.5 the ~6x-cheaper draft yields only ~0.6–0.74x B (W=256 @ device step
20.67ms: FIXED=8 → 2790 tok/s = 0.74x; FIXED=15 → 2243 = 0.60x). **The
projection needed accept ≳4 to win; we are at 2.5.** Lever A as implemented
should NOT be shipped at any W tried; if a recovery point exists it is well above
512, which erodes the speedup premise. ⇒ the windowing lever is REJECTED on
evidence; a fresh approach to the O(C·B) draft attn-score matmul is needed (one
that keeps full context). **LEVER B (numerics-NEUTRAL) LANDED + is sound + still
useful** (frees 3.96 GiB → util 0.75 fits, cuts real per-step overhead, does NOT
touch proposals so full-context accept stays ~6) — keep it. ⇒ BLOCKED_USER: the
two-lever feasibility plan's Lever A is invalid; the path to beat B needs a
design decision (see "DEAD END" + "WHAT REMAINS").

The code remains in-tree, flag-gated and OFF by default (DFLASH_DRAFT_WINDOW=0),
so it is harmless and available if a future approach wants the gather machinery.

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

## PROJECTION — was promising at assumed accept=6, but COLLAPSES at the MEASURED accept=2.5
With the ASSUMED accept=6 the windowed projection cleared B (W=256: 1.78x@FIXED8,
1.12x@FIXED25). But the MEASURED real-draft accept at W is **2.5**, so the honest
projection (tput = accept·32/step):
```
  W=256 (device step @C=4608 = 20.67ms), accept=2.5:
    FIXED= 8ms -> step=28.67ms -> 2790 tok/s = 0.74x B   [LOSES]
    FIXED=15ms -> step=35.67ms -> 2243 tok/s = 0.60x B   [LOSES]
    FIXED=25ms -> step=45.67ms -> 1752 tok/s = 0.47x B   [LOSES]
  W=512 (device step @C=4608 = 23.98ms), accept=2.5: ~0.43–0.70x B  [LOSES]
```
**Windowing LOSES at every FIXED because the accept drop (6→2.5, a 2.4x hit)
outweighs the draft-fwd cut (the draft fwd was only ~part of the step; the
target verify + overhead don't shrink, and now yield 2.5 tokens not 6).** The
accept-drop sensitivity note in 15-feasibility was too optimistic — it assumed
accept stayed ≳4; the real drop is to 2.5 and is window-insensitive.

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

## DEAD END — do NOT repeat: windowing the DFlash draft attention
Truncating the DFlash draft's context to a window (W≤512) to cheapen the O(C·B)
attn-score matmul DESTROYS acceptance (6→2.5, window-insensitive 256 vs 512).
The DFlash draft genuinely USES its long context to propose well; a short window
makes it a much worse predictor and the lost tokens dwarf the per-step savings.
Net throughput LOSES at every overhead level. The "two-lever" feasibility plan's
Lever A is therefore INVALID. Any future attempt to cut the O(C·B) draft matmul
must KEEP full context (e.g. a genuinely cheaper full-context attention kernel,
or a structurally different draft) — not window it.

## WHAT REMAINS / OPEN DESIGN QUESTION (BLOCKED_USER)
The economic case (15-feasibility: R=1.58 << accept, target verify NOT the
ceiling) still holds — a CHEAP-ENOUGH draft would win big. But windowing is not
the way to cheapen it. Open options for a fresh manager / user decision:
1. **A cheaper FULL-CONTEXT draft attention** — the O(C·B) score matmul over the
   full C is the cost; a fused/flash-style or RPA Pallas kernel for the draft
   (instead of the eager HF q@k^T+softmax) might cut it WITHOUT dropping context.
   This keeps accept ~6. (Most promising; biggest effort.)
2. **Accept the loss / lower num_spec with full context + Lever B only** — but
   14-bench already showed full-context num_spec sweep PLATEAUS ~1300 tok/s, so
   Lever B alone (no draft-fwd cut) will NOT beat B. Not a path on its own.
3. **Re-examine whether c=32 spec-decode is the right goal** — at c=32 the target
   is already weight-bound + batched (B=3752, TPOT 8.5ms); the draft has to be
   extremely cheap to win. Spec-decode's natural win is LOW concurrency. (User
   goal is fixed at c=32 per GOAL, so this is a flag-for-user note, not a pivot.)
DECISION NEEDED: pursue option 1 (cheaper full-context draft kernel) as the next
impl round, or reassess. Lever B should be KEPT regardless (free HBM + real
overhead cut, numerics-neutral) — it is a prerequisite for any winning config.

## STATUS
- [x] Lever A implemented (windowed gather, flag DFLASH_DRAFT_WINDOW) — works,
      microbench ~6x, but REJECTED on acceptance evidence (kept flag-OFF in tree)
- [x] Lever B implemented (drop _ctx_buf alloc/write/move + batch device_get) —
      SOUND, numerics-neutral, frees 3.96 GiB, KEEP
- [x] CPU gather numeric check (prefix/mid/near-end + clamp) PASS
- [x] unit tests 17/17 green
- [x] isolated real-8-chip microbench: windowed fwd FLAT ~14.2ms (~6x cut)
- [x] real-draft ACCEPTANCE at W=256 AND W=512: ~2.5 (vs ~6 baseline) ⇒ Lever A
      LOSES (0.6–0.74x B). Windowed serve came UP clean at c=32 (path runtime-OK).
- [n/a] lossless re-verify of windowed path: moot (Lever A rejected; not shipping)
- [ ] Lever B-only serve at util 0.75 (confirm HBM win) — for the BENCH manager
- [ ] OPEN: cheaper FULL-CONTEXT draft attention (option 1) — fresh impl round
