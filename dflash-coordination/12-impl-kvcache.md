# 12 — IMPL: KV-cache the DFlash draft forward (LEVER #1)

Phase: kill the O(ctx) draft recompute. Date: 2026-06-27. Manager: impl-kvcache.
Builds on 11-bench-c32-v3 (2.90x slower, the gap), 08-impl-perf (per-step map),
10-impl-condense (per-slot state must follow condense), 05-test-batched (perfect-draft verify).

## THE GAP / THE LEVER
DFlash correct + accepts ~6/step but 2.90x SLOWER at in=1/out=4096/c=32 (1310 vs 3793 sys
tok/s). Whole gap = the STATELESS draft recomputes fc (14400->2880) + 8 attn layers' k/v_proj
+ k_norm + RoPE over the ENTIRE context (O(ctx), grows to ~4608) EVERY decode step. TPOT 13.2x.
Need ~2.2x step speedup. KV-caching the context removes the O(ctx) term -> step ~O(1) in ctx.

## ARCHITECTURE (confirmed by L3 mapping)
Draft: 8 attn layers, D=2880, head_dim=64, 64 q heads, **8 KV heads** (GQA), block_size B=8,
raw_hidden_dim 14400 (=5 target layers x 2880). yarn RoPE theta 150000. is_causal=False
(BIDIRECTIONAL over [ctx|noise]). Draft weights REPLICATED (plain nn.Linear, shard_model_to_tpu
replicates) => K/V activations replicated => cache must be replicated.

KEY NUMERICS FACT (the whole basis): K/V at a fixed ctx position p is a deterministic
position-local pipeline: k_p = RoPE_p(k_norm(k_proj(hidden_norm(fc(raw_p))))), v_p =
v_proj(hidden_norm(fc(raw_p))) (no norm/no RoPE). fc+hidden_norm SHARED across 8 layers.
=> caching == recompute, modulo softmax FP-associativity in the attention reduction.

## DESIGN DECISION: Option B2 (separate kv_project jit + cache-consuming attention)
- Add a `_kv_project` path: run fc+hidden_norm + per-layer k_proj/k_norm/RoPE + v_proj on ONLY
  the NEWLY-accepted rows (~B rows/step, NOT C). Reuse the remote model's OWN submodules (fc,
  hidden_norm, k_proj, k_norm, v_proj, rotary_emb, rotate_half) via functional_call => bit-exact,
  NO remote-code edits.
- Per-slot caches `_k_cache`/`_v_cache` shape (L=8, slots=32, buf_len~4608, KVH=8, hd=64) bf16,
  REPLICATED. Write new rows via `_batched_kv_write` (mirror _batched_ctx_write masked-RMW
  scatter, same slot_idx/dst_row/valid plan). Move on condense via `_move_kv_rows` (mirror
  _move_ctx_rows sparse donated scatter, same (dst,src) pairs/buckets).
- Rewrite attention to attend over [cached ctx K/V | freshly computed noise K/V]; fc/k_proj/
  v_proj over context never run on full C again.
- Cache row index = absolute ctx position = the RoPE position baked at write time. K cached
  POST-RoPE. Padding rows stay masked (finfo(bf16).min), same as today.
- DROP `_ctx_buf` once cache is live (raw hidden only needed to project NEW rows = a per-step
  transient). Keep behind `_use_kv_cache` flag during bring-up.

## HBM (corrected — L3 had an 8x arithmetic error; recomputed)
- Full replicated K+V cache @ (8,32,4608,8,64) bf16 = **2.25 GiB** (L3 wrongly said 19.3 GiB).
  Per cache = 8*32*4608*512*2 B = 1.2 GB; x2 (K+V) = 2.25 GiB. CONFIRMED.
- Current `_ctx_buf` = 3.96 GiB. So DROP _ctx_buf + ADD full K/V cache = **NET -1.7 GiB**.
- => NO need to cap slots to 8 (the L3's mitigation is unnecessary). Keep full c=32 at
  max_model_len 4224/buf 4608. HBM is a WIN, not a blocker.

## INCREMENTS (each independently testable + committable)
1. `_kv_project` branch + `get_kv_project_fn` + fetch; bit-identical K/V check vs a remote-
   submodule oracle (isolates the RoPE/k_norm convention risk). NO cache/attention wiring yet.
2. cache tensors (flag off) + `_batched_kv_write` + `_move_kv_rows` + precompile warms;
   numpy-reference bit-identical write/move unit tests.
3. `draft_forward_cached` (attend over [cache|noise]) + whole-step equivalence vs full-recompute
   path; wire `_use_kv_cache`, drop `_ctx_buf` alloc when on.

## NUMERICS NOTE (from increment 1 — important for increment 3)
`_kv_project` is bf16-EXACT for K and V vs the in-forward path WHEN both run in the SAME jit
graph. ACROSS SEPARATE jit graphs, XLA fuses the RoPE FMA (`k*cos + rotate_half(k)*sin`)
differently -> cached K drifts ~1 bf16 ULP (max|d| ~0.03-0.06). V (no RoPE) is always exact.
SINCE the final design caches K from a SEPARATE jit (kv_project in prepare_inputs) and consumes
it in a DIFFERENT jit (draft_forward_cached), cached K will differ from same-graph full-recompute
by ~1 ULP per RoPE. This is genuine bf16 rounding (GOAL ladder #3), NOT a logic bug. MUST verify
end-to-end greedy tokens stay identical (argmax robust to 1-ULP K through softmax) in increment 3.

## STATUS
- [x] design (B2, HBM corrected to a net win)
- [x] increment 1 (kv_project jit + bit-check PASS bf16-exact same-graph) — commit ebb2b44c
- [x] increment 2 (cache write/move + precompile + 17 unit tests, flag DFLASH_KV_CACHE) — 39e2f91f
- [x] increment 3 (cached forward + token-equivalence PASS) — e1fa4115, d7a4300b
- [x] isolated per-step microbench REAL 8-CHIP MESH N=32 — b5d5c7b4 (verdict below)
- [ ] correctness sanity ON REAL SERVE PATH (perfect-draft c=32 + greedy vs target-only, flag on)
- [ ] num_speculative_tokens DOWN to clear the marginal flip (NEXT round, not mine)
- [ ] drop _ctx_buf on flag-on path (HBM win, follow-up; cache proven so cross-check no longer needed)

## DECISIVE MICROBENCH (REAL 8-chip v6e mesh TP8, N=32, median of 30 warm calls) — commit b5d5c7b4
| C    | FULL (ms) | CACHED total (ms) | CACHED fwd-only (ms) | F/Ctotal |
|-----:|----------:|------------------:|---------------------:|---------:|
| 512  | 14.75     | 14.98             | 8.83                 | 0.98x    |
| 1024 | 21.49     | 20.05             | 13.56                | 1.07x    |
| 2048 | 40.89     | 32.49             | 26.01                | 1.26x    |
| 4096 | 87.76     | 53.92             | 47.04                | 1.63x    |
| 4608 | 98.49     | 58.80             | 52.04                | 1.68x    |
Component @C=4608: kv_project(B=8) 0.56ms + write 6.52ms + fwd_cached 52.04ms = 58.80ms.
The cache removes the O(C) PROJECTION recompute (fc 14400->2880 + per-layer k/v/q-proj+norm
over full C) = the 1.6-1.9x win at high C. The O(C) ATTENTION-SCORE matmul (q over C+B keys)
STAYS and now dominates (52 of 59ms @C=4608) -> cached fwd is NOT flat in C.

## FLIP PROJECTION: MARGINAL (closes ~the whole 2.90x gap, lands ~7% short)
Break-even: DFlash step must get under ~target_TPOT x accept = 0.0084 x 6 = 0.050s. At C=4096
(representative of most steps in a 4096-tok gen) cached_step/6 = 8.99 ms/accepted-token vs
target 8.40 ms/token -> ~7% OVER. At C=4608: 9.80 vs 8.40. Lower-C steps already win (C=2048 ->
5.41ms < 8.40). Average sits right ON the line. => the cache cut FULL ~1.7x at high C but not
the ~2x needed to clear. NEXT cheap lever (NEXT round, flagged not done): num_speculative_tokens
DOWN from 7 -> smaller noise block B -> cheaper O(C*B) score matmul AND cheaper target verify.
Given only ~7% over, a modest num_spec cut plausibly flips A>B.

## INCREMENT 3 FINDINGS (important — partial win, NOT a full flat-in-C)
Cache-consuming forward token-EQUIVALENT (tie-free argmax identical all steps; hidden diffs
~1-2 bf16 ULP accumulated over 8 layers = bf16 floor, not a logic bug). BUT isolated
single-device N=8 timing: cached only 1.3-1.6x faster, NOT flat in C. WHY: the cache removes
the O(C) PROJECTION recompute (fc 14400->2880 + 8x k/v_proj/k_norm/RoPE over full C) but the
ATTENTION SCORE matmul (q over C+B keys) is inherently O(C) and STAYS. So step is
proj_cached(O(B)) + attn_scores(O(C)).
- C=512: 4.69->3.62ms; C=1024: 6.66->4.11ms; C=2048: 10.03->7.98ms; C=4096: 19.87->12.55ms.
- CAVEAT: single-device N=8 is NOT the real signal. On the 8-CHIP mesh at N=32, 08-impl-perf
  measured fc+8attn = 88-99ms @ C=4096-4608 where the PROJECTION dominated (the big matmuls).
  Need the real-mesh N=32 number to know the true step speedup + flip projection.
- _ctx_buf STILL written alongside the cache on flag-on (kept for cross-check). Drop it next.

## COMMITS
- ebb2b44c spec-decode: DFlash kv_project (project new ctx rows' K/V) + bit-identical check
- 39e2f91f spec-decode: DFlash per-slot K/V cache write + condense-move (flag-gated, unit-tested)
