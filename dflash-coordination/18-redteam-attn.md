# 18 — RED-TEAM: is the DFlash draft attention REALLY FLOP-bound? (NO — it's REPLICATION-bound)

Phase: adversarially re-test 17-impl-attnkernel's "draft attn is FLOP-bound ⇒ dead end" verdict.
Date: 2026-06-27. Manager: redteam-attn. Method: physics first, then isolated real-8-chip microbenches
(2 L3s). Builds on 17 (the claim), 15-feasibility (R=1.58, verify NOT the ceiling), 12-impl-kvcache
(~52ms attn is ~the whole cached fwd).

## TL;DR / VERDICT — THE CLAIM IS FALSE. THE DRAFT ATTN IS ~60x FIXABLE ⇒ ACHIEVABLE.
17 called the ~48ms draft attention "FLOP-bound." It is NOT. The useful FLOPs are 1.55e11; at v6e
single-chip bf16 peak (918 TFLOP/s) the FLOP floor is **0.17ms**. Measured eager 8L @C=4608 = 48.72ms
= **0.35% of MXU peak**. A FLOP-bound kernel runs at 30-70% of peak, not 0.35%. The cost is
MEMORY-BANDWIDTH + REPLICATION, both fixable:
- **The draft attention is REPLICATED across all 8 chips (`PartitionSpec()`)** — every chip redundantly
  computes all 64 query heads. Sharding the 64 q-heads 8/chip across the mesh ALONE (identical eager
  math) drops 48.78ms → **1.49ms (33x)**.
- **GQA-expanding K/V 8x (KVH=8→Hq=64) + materializing f32 score/prob tensors** is pure HBM traffic.
  GQA-native reads (KVH=8) + bf16 scores on TOP of head-sharding → **0.82ms (~60x)**. Correct to bf16
  (max|diff| vs eager = 1.9e-3).
- The prior FLASH bench was rigged by REPLICATION, not just tiling: replicated flash with sane tiling
  (block_k=512) is 153.7ms (WORSE than the 84ms in 17). Flash is NOT needed — plain eager, head-sharded,
  beats the Pallas flash kernel at this tiny q-block (B=8).

⇒ The draft attention is NOT the wall. With the attention at ~1ms, the draft is no longer the
bottleneck; the only remaining lever is the ~40ms FIXED host overhead (Lever B, already scoped/partly
landed). Projection: attn-fixed + Lever B(FIXED≤25ms) → **1.16–1.88x B=3752**. ACHIEVABLE.

## THE PHYSICS WEDGE (resolved)
| quantity @C=4608 | value |
|---|---|
| useful attn FLOPs (8L, N=32, q@kᵀ + scores@V) | 1.55e11 |
| FLOP floor @ 918 TFLOP/s (1 chip) | 0.17 ms |
| measured EAGER_8L | 48.72 ms |
| ⇒ MXU utilization | **0.35% of peak** |
| eager mem traffic (8x-expanded K/V re-read + f32 scores) | ~26 GB ⇒ HBM floor ~16ms |
| GQA-native fused mem traffic | ~2.5 GB ⇒ HBM floor ~1.5ms (÷8 chips if sharded → sub-ms) |
0.35% of peak is the signature of a memory/utilization-bound kernel, never a FLOP-bound one.

## DECOMPOSITION (L3-A, real 8-chip mesh, replicated, median-of-30) — bench_dflash_attn_decompose.py
| component (8L) | C=2048 | C=4096 | C=4608 |
|---|---:|---:|---:|
| EAGER_8L (baseline) | 18.99 | 43.48 | **48.72** |
| repeat_kv only (GQA expand) | 4.46 | 8.57 | 9.53 |
| score q@kᵀ bf16 (no softmax/value) | 10.27 | 20.08 | 22.40 |
| score + f32 softmax | 11.30 | 27.00 | 30.46 |
| value scores@V | 9.54 | 19.32 | 21.74 |
| **GQA-native (KVH=8 reads, no repeat_kv)** | 5.79 | 11.75 | **13.02** |
- **repeat_kv alone @C=4608 = 9.53ms moving 21.8 GB ⇒ 2285 GB/s = 139% of single-chip HBM peak** —
  exceeding peak = pure-memory smoking gun (it writes the 8x-expanded K/V, then both matmuls RE-READ it).
- EAGER scales LINEARLY in S (flat µs/key) = memory-bound signature (FLOP-bound would scale with the matmul).
- f32 score materialization = +8.1ms (secondary). The 8x K/V expansion (paid 3x: write + 2 re-reads) dominates.
- **GQA-native alone = 3.74x faster** (same FLOPs, ~3x less HBM) — proving the cost is the expansion, not FLOPs.

## PROPERLY-FORMULATED (L3-B, real 8-chip mesh, median-of-30) — bench_dflash_attn_fixed.py
| C | A eager (repl) | D eager HEAD-SHARDED (same math) | E eager GQA-native+bf16 head-sharded | B flash head-sharded bk512 | C flash repl bk512 |
|---|---:|---:|---:|---:|---:|
| 2048 | 18.89 | 0.78 | **0.46** | 9.00 | 77.41 |
| 4096 | 43.48 | 1.35 | **0.59** | 15.96 | 138.43 |
| 4608 | **48.78** | 1.49 | **0.82** | 17.68 | 153.67 |
Correctness vs eager @C=4608 (bf16 tol 5e-2): D 3.9e-3, **E 1.9e-3**, B 2.9e-3 — all correct.
- **Fastest correct = E (GQA-native, bf16 scores, 64 q-heads sharded 8/chip) = 0.82ms @C=4608 = ~60x.**
- **Head-sharding is the dominant fix (~33x): D = identical eager math, just sharded not replicated.**
  48.78/8 ≈ 6.1ms and D is even faster (1.49) ⇒ the baseline was overwhelmingly WASTED REPLICATION.
- GQA-native+bf16 adds ~1.8x on top (1.49 → 0.82).
- Scales SUB-linearly in C (0.46/0.59/0.82) — utilization-bound, confirmed.
- Flash on the replicated path is hopeless (153ms); flash head-sharded (17.68) still loses to eager
  head-sharded — flash's per-block overhead dominates at q-block B=8. **No Pallas kernel needed.**

## WHERE 17 WENT WRONG (the bug in the prior verdict)
17's bench (bench_dflash_attn_kernel.py) ran BOTH eager and flash REPLICATED (`PartitionSpec()`), so it
measured the cost of every chip redundantly doing all 64 heads — then compared two replicated formulations
and concluded "FLOP-bound" because both were slow. It never tested the one thing that matters: SHARDING
the 64 independent heads across the 8-chip mesh (attention is embarrassingly parallel over heads). It also
left the 8x GQA expansion + f32 score materialization in place. The "30ms q@kᵀ + 18ms scores@V" attribution
is real device matmul time, but it's the time of a REPLICATED, GQA-EXPANDED, f32 formulation — not an
irreducible FLOP floor (which is 0.17ms). The reconciliation in 17 (eager 48.8 == 12-impl's 52ms cached fwd)
is CORRECT — the attention IS the whole cached fwd — which is exactly why fixing it (52→~1ms) collapses the
cached forward to ~7-8ms.

## PROJECTION vs B=3752 tok/s (attn fixed 52→~1ms; step = FIXED + verify9.08 + draft_total)
draft_total (conservative) = kv_project 0.56 + write 6.52 + fwd_fixed ~3 (attn ~1ms + MLP/o_proj/norms
over B=8 rows ~2ms) ≈ 10ms. accept=6, verify(k=7)@C4096=9.08ms (from 15).
| FIXED overhead (ms) | step (ms) | tput (tok/s) | vs B |
|---:|---:|---:|---:|
| 40 (today, un-cut) | 59.2 | 3245 | 0.86x |
| 25 (modest Lever B) | 44.2 | 4348 | **1.16x** |
| 15 (Lever B) | 34.2 | 5621 | **1.50x** |
| 8 (Lever B + overlap) | 27.2 | 7069 | **1.88x** |
Sensitivity (accept→5, draft_total→10): FIXED25→0.97x, FIXED15→1.25x, FIXED8→1.57x — still clears B once
overhead is cut. **Crucially head-sharding KEEPS FULL CONTEXT ⇒ accept stays ~6 (no windowing tradeoff).**
This is 15-feasibility's two-lever plan REVALIDATED — but the draft lever is now ~60x (head-shard), not the
~8x windowing that craters accept (16-impl). The two levers (cheap attn + cut overhead) are independent and
both required at FIXED~40; either of them alone at today's numbers loses, together they win with margin.

## VERDICT: ACHIEVABLE ⇒ NEEDS_IMPL
The draft attention is REPLICATION/MEMORY-bound, not FLOP-bound, and is ~60x fixable. The c=32 full-context
DFlash goal is ACHIEVABLE. EXACT FIX TO IMPLEMENT (in priority order):
1. **HEAD-SHARD the draft attention across the mesh (the 33x, do this first).** The draft attn currently
   runs REPLICATED (`PartitionSpec()`) in models/vllm/dflash.py (the cached forward `_draft_forward_cached`
   / draft_forward and the eager_attention_forward call ~L259). Shard the 64 q-heads on the `model` mesh
   axis (8 heads/chip) — attention is independent per head, so this is exact. Repeat_kv'd K/V shard the same
   head axis; mask broadcasts; o_proj input gathers back. Concretely: a shard_map over the per-layer attn
   with in_specs sharding the Hq axis on 'model', or set the q/k/v/out shardings so the head axis lands on
   MLP_TENSOR. Verify lossless (greedy tokens identical; bf16 max|diff| ~2e-3 is the floor).
2. **GQA-native + bf16 scores (the extra ~1.8x).** Don't repeat_kv to Hq=64; group the q-heads and read
   K/V at KVH=8 via lax.dot_general; keep scores bf16 (skip the f32 promotion). (XLA gotcha from L3-A: pure
   bf16 softmax SIGSEGVs the conv-emitter on this v6e build — keep the SOFTMAX in f32, only the score/prob
   matmul inputs bf16; use lax.dot_general not jnp.einsum for the GQA-native form.)
3. (Already scoped) **Lever B — cut the ~40ms FIXED host overhead** (drop redundant _ctx_buf write on the
   KV path, fuse/overlap dispatches, collapse device_gets; async = Phase 2). Required to clear B; with the
   attn fixed, this is now the ONLY remaining gap.
NOTE: this REPLACES 17's "BLOCKED_USER — needs a cheaper draft model / sparse attn / drop c=32." None of
those is needed. No new draft checkpoint, no research-grade sparse kernel, no relaxing c=32. It is a
sharding + formulation fix on the EXISTING draft, full context preserved.

## DEAD ENDS CORRECTED
- 17's "draft attn is FLOP-bound, no kernel helps, BLOCKED_USER" is WRONG — it measured replicated
  formulations only. Head-sharding (not a kernel swap) is the fix. Keep the eager einsum; do NOT pursue flash.
- Windowing (16) is still dead (craters accept). Head-sharding keeps full context so accept is unaffected.

## ARTIFACTS
- scratchpad/bench_dflash_attn_decompose.py (L3-A, commit 0333f8fa) — the 48ms decomposition + roofline.
- scratchpad/bench_dflash_attn_fixed.py (L3-B, commit 32264ef4) — head-sharded/GQA-native/flash table + correctness.
- Projection math reproduced inline (calibrated to 15-feasibility verify/overhead + 12-impl cache I/O).

## STATUS checklist
- [x] PHYSICS: FLOP floor 0.17ms vs measured 48.7ms = 0.35% peak ⇒ NOT FLOP-bound
- [x] DECOMPOSE: repeat_kv 139% HBM peak (memory-bound smoking gun); f32 +8ms; GQA-native 3.74x
- [x] PROPER FORMULATION: head-shard 33x (1.49ms), +GQA-native+bf16 → 60x (0.82ms), correct (1.9e-3)
- [x] flash re-tested: replicated hopeless (153ms), head-sharded loses to eager — no kernel needed
- [x] PROJECTION: attn-fixed + Lever B(FIXED≤25) → 1.16–1.88x B; accept stays ~6 (full context kept)
- [x] VERDICT: ACHIEVABLE — head-shard the draft attn (+ GQA-native) + Lever B; corrects 17's dead end
