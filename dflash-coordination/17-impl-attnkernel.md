# 17 — IMPL: efficient draft attention kernel (the last lever) — REJECTED on evidence

Phase: route the DFlash draft's full-context attention through an efficient TPU
attention kernel (flash / ragged-paged) instead of eager HF einsum, keeping full
context so accept stays ~6. Date: 2026-06-27. Manager: impl-attnkernel. Builds on
16-impl-window (windowing REJECTED — craters accept 6→2.5; Lever B kept), 15-feasibility
(R=1.58 << accept; the O(C·B) attn-SCORE matmul is the wall), 12-impl-kvcache (the
KV cache; cached fwd ~59ms @C=4608, 52ms of it the attention).

## TL;DR / VERDICT — THE KERNEL SWAP IS A DEAD END (flash is ~1.7x SLOWER) ⇒ BLOCKED_USER
The remaining lever was: replace the draft's eager attention (q@k^T + fp32 softmax + @V)
with the efficient Pallas attention kernel this codebase already runs for the TARGET's
decode/verify attention — same math, just a faster kernel, full context preserved.

**It does not work, and we proved it decisively in isolation:**
1. The TARGET's kernel (`ragged_paged_attention_hd64`) is DOUBLY BLOCKED for the draft:
   (a) it is strictly **PAGED** (the draft cache is contiguous per-slot), and (b) it is
   **hard-causal with no override** — it cannot express the draft's BIDIRECTIONAL 8-query
   noise block. Not usable without a new kernel.
2. The contiguous-compatible Pallas kernel (`flash_attention`, the one the JAX-native
   DFlash path already uses) CAN reproduce the exact math, but at the draft's real shape
   it is **~1.7x SLOWER than the eager einsum at every C** (real 8-chip mesh microbench).
3. Root cause (decisive): the draft attention is **FLOP-BOUND on the q@k^T (~30ms) +
   scores@V (~18ms) matmuls over the GQA-expanded 64-head K/V**, NOT memory/softmax-bound.
   Flash has the SAME FLOPs. At query-len B=8 the flash kernel runs N·Hq = 2048 tiny
   one-shot tiles; per-tile launch/VMEM overhead dominates and softmax-fusion buys nothing
   (the (8×C) score intermediate is already small). Plain XLA matmuls pack B=8 better.

⇒ There is NO efficient attention kernel that beats the eager path for this shape, because
the cost is inherent dense-attention FLOPs over full context. The ONLY way to cut that cost
is to cut C (windowing — REJECTED, craters accept) or cut B (num_spec — EXHAUSTED, plateaus
~1300). Both already-failed levers. The kernel-swap lever is therefore exhausted too.
⇒ BLOCKED_USER: the path to beat B at c=32 needs a structurally different idea, not a kernel.

## THE DESIGN PASS (what the existing kernels can / can't do)

### Target's kernel: `ragged_paged_attention_hd64` (kernels/ragged_paged_attention/v3/kernel_hd64.py)
This is what gpt-oss-20b's decode/verify attention uses (head_dim=64 specialized, sinks,
hd64-native). Call-site: layers/jax/attention/gpt_oss_attention.py:192 (shard_map). Signature
takes flat-ragged queries [total_q, Hq, hd], a PAGED kv_cache [pages, page_size, …], kv_lens,
page_indices (block table), cu_q_lens, distribution. TWO hard blockers for the draft:
- **PAGED ONLY** — no contiguous-KV entry point. The draft cache is contiguous
  (L=8, slots=32, buf~4608, KVH=8, hd=64). Would require paging the draft cache + block tables.
- **HARD-CAUSAL, no flag** — causality is hard-wired (q_span < kv_span); `sliding_window` only
  narrows it, never relaxes. The draft needs the 8 noise queries to attend BIDIRECTIONALLY to
  each other (is_causal=False over [ctx|noise]). Impossible to express. (The v3 *default* kernel
  `ragged_paged_attention` does expose use_causal_mask=False, but it requires head_dim%128==0 —
  pads 64→128, doubling KV width/compute — AND is still paged. Not viable.)

### Contiguous-compatible kernel: `flash_attention` (kernels/flash_attention/kernel.py:82)
CAN reproduce the math exactly: q/k/v [batch, heads, seq, hd]; `causal=False` (bidirectional);
additive bias `ab` [batch, heads, q_seq, kv_seq] (encodes the per-slot ragged mask); native
head_dim=64; fp32 online softmax (matches the draft's fp32 softmax to ~1 bf16 ULP — benign,
GOAL ladder #3); does NOT apply RoPE (draft K already post-RoPE — correct). Gotchas: NO internal
GQA (must jnp.repeat K/V 8→64 heads first); `ab` does NOT broadcast (must materialize the full
(N,Hq,B,C+B) bias); the kernel does `(q@k^T + ab)*scale` so additive mask must be passed as
`ab = mask/scale`; pass block_q=B (default 128 > 8 errors); needs a shard_map wrap (Mosaic can't
auto-partition under a mesh). The JAX-native DFlash path (models/jax/dflash.py:225-249) already
uses this exact call — proving the math/wiring is sound. (NOTE: dflash_attention_interface.py's
`dflash_concat_attention` is EAGER einsum, NOT flash — porting it buys nothing.)

## THE DECISIVE MICROBENCH (REAL 8-chip v6e mesh, N=32, median of 30 warm) — scratchpad/bench_dflash_attn_kernel.py
Shape: N=32, Hq=64, KVH=8 (GQA 8x), hd=64, B=8 query block, 8 layers, DISTINCT per-layer K/V
(cross-layer CSE defeated — critical; a first run sharing K/V let XLA hoist repeat_kv and
under-counted eager ~10x). REPLICATED sharding (data=1, model=8) = the draft's actual layout.
```
   C | SCORE_8L (q·kᵀ+softmax) | EAGER_8L | FLASH_8L | speedup (eager/flash)
 512 |                  2.68 |    5.43 |   15.39 |  0.35x
1024 |                  5.36 |   10.08 |   21.48 |  0.47x
2048 |                 11.38 |   18.92 |   39.10 |  0.48x
4096 |                 26.91 |   43.44 |   74.66 |  0.58x
4608 |                 30.43 |   48.78 |   83.96 |  0.58x
```
Per-layer @C=4608: EAGER 1L = 7.22ms, FLASH 1L = 11.25ms (0.64x).
Math sanity @C=512, 1 layer: max|eager−flash| = 1.17e-2 (benign bf16 — confirms flash
reproduces the SAME attention, not a logic mismatch; accept would be unchanged IF we shipped it).

**Flash is SLOWER at every C (0.35–0.58x), never crosses 1.0x.** Projecting flash onto the cached
forward: ~52ms/0.58 + 7ms ≈ **96ms** vs the current ~59ms — ~1.6x WORSE, moving decisively AWAY
from break-even (need step < ~50ms for cached_step/6 < target 8.4ms/tok @accept6).

## RECONCILIATION (numbers cross-check the prior docs)
Isolated EAGER_8L @C=4608 = 48.8ms lands right on 12-impl-kvcache's ~52ms cached forward ⇒ the
attention IS ~the whole cached forward, as that doc said. The q·kᵀ+softmax SCORE half alone is
~30ms; the scores@V VALUE half is the other ~18ms. BOTH are O(N·Hq·B·C·hd) dense FLOPs over the
GQA-expanded 64-head K/V, and BOTH are equally un-helped by flash. The "52ms is the attention-score
matmul" attribution in 14-bench / 12-impl is confirmed; what's NEW is that it is FLOP-bound (not
recoverable by a kernel that keeps the same FLOPs).

## WHY FLOP-BOUND ⇒ NO KERNEL HELPS (the general argument)
Dense full-context attention FLOPs = O(N·Hq·B·C·hd) per matmul (×2 for q@k^T and scores@V), fixed
by the shapes. A flash/ragged/paged kernel changes the MEMORY pattern (no materialized score
matrix), not the FLOPs. The draft's score matrix (B=8 × C) is ALREADY small, so there is no
materialization to remove — flash's classic win does not apply, and its tiny-tile overhead makes
it net slower. The only knobs that reduce the FLOPs are C (context length) and B (noise block) and
Hq — all of which either crater acceptance (C: 16-impl windowing → accept 2.5) or are exhausted
(B/num_spec: 14-bench plateau ~1300). So no attention KERNEL can move this.

## CONDENSE / HBM / NUMERICS NOTES
- No code shipped on the draft forward this round (the swap was REJECTED at the microbench gate
  BEFORE wiring it in) ⇒ condense/HBM behavior is UNCHANGED from 16-impl (Lever B in tree,
  flag-off byte-identical). The flash math is numerically equivalent (~1 bf16 ULP) so IF it had
  been a speedup, accept would have been preserved — but it isn't, so moot.
- The only artifact is the committed microbench script (no source touched). Flag-off path byte-unchanged.

## COMMITS (branch dflash)
- b0b7e042 bench: isolate DFlash eager vs flash_attention draft attn @ real shapes (scratchpad)
- (this doc + SHARED_MEMORY committed alongside)

## DEAD END — do NOT repeat: swapping the draft attention to a flash/ragged/paged kernel
Routing the DFlash draft's full-context attention through flash_attention (or the target's
ragged_paged_attention_hd64) does NOT speed it up. The target kernel is paged + hard-causal
(can't express the draft's contiguous, bidirectional 8-query mask). flash_attention CAN express
it but is ~1.7x SLOWER because the draft attention is FLOP-bound (q=8 tiny, C huge → flash's
tiny-tile overhead dominates, same FLOPs, nothing to fuse). No attention kernel reduces dense-
attention FLOPs over full context. Combined with windowing (REJECTED, craters accept) and num_spec
(EXHAUSTED, plateau ~1300), ALL three attention-cost levers within the c=32/full-context frame are
now exhausted.

## WHAT REMAINS / OPEN (BLOCKED_USER) — the frame must change, not the kernel
Every lever that keeps {full context, c=32, this draft architecture, dense attention} has now been
tried and failed: KV-cache (lands ~7% short), num_spec (plateau), windowing (accept craters),
efficient kernel (FLOP-bound, slower). The economic case (R=1.58 << accept) is real but the draft
step is dominated by FLOP-bound dense attention that nothing in scope can cheapen. Genuinely-new
directions a user/L1 must choose among (all out of THIS phase's "kernel swap" scope):
1. **Reduce the draft's HEADS or layers** — the FLOPs scale with Hq=64 and L=8. A smaller/cheaper
   draft (fewer q-heads, fewer layers, or a distilled draft) cuts FLOPs directly. Changes the draft
   model ⇒ needs a re-trained/different DFlash checkpoint (not a code knob). Biggest lever, biggest cost.
2. **Sparse / top-k context attention** that KEEPS the predictive context (attend to a learned/heuristic
   subset of C, not a recency window) — windowing failed because recency-truncation is dumb; a
   content-selective sparse attention might keep accept ~6 at lower FLOPs. Research-grade, high risk,
   needs a custom kernel anyway (so it's a "new Pallas kernel" fork, not a swap).
3. **Accept that c=32 spec-decode is not the natural regime** — at c=32 the target is weight-bound +
   batched (B=3752, TPOT 8.5ms); spec-decode wins at LOW concurrency where the target step is the
   bottleneck. GOAL fixes c=32, so this is a flag-for-user, not a pivot we can take unilaterally.
DECISION NEEDED FROM USER/L1: pursue a cheaper draft MODEL (option 1), a content-sparse attention
research effort (option 2), or revisit the c=32 requirement (option 3). The pure-engineering levers
on the existing draft are exhausted.

## STATUS checklist
- [x] DESIGN: located target kernel (ragged_paged_attention_hd64) — paged + hard-causal ⇒ can't fit draft
- [x] DESIGN: located contiguous flash_attention — CAN reproduce the draft's exact math
- [x] DESIGN: confirmed flash math == eager (max|d|=1.17e-2 bf16) ⇒ accept would be preserved
- [x] MEASURE: isolated real-8-chip eager-vs-flash @ all C ⇒ flash 0.35–0.58x (SLOWER everywhere)
- [x] ROOT CAUSE: FLOP-bound (q@k^T 30ms + scores@V 18ms dense); flash same FLOPs, tiny-tile overhead
- [x] reconciled with 12-impl 52ms (= eager 48.8ms) ⇒ attribution confirmed, now shown FLOP-bound
- [n/a] IMPLEMENT swap: REJECTED at the microbench gate (would be ~1.6x worse) — nothing wired in
- [n/a] lossless re-verify: moot (not shipping a slower kernel)
- [ ] OPEN (user/L1): cheaper draft model / content-sparse attn / revisit c=32 — all out of scope here
