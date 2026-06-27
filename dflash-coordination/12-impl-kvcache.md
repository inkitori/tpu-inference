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
- [ ] increment 2 (cache write/move)
- [ ] increment 3 (cached attention + equivalence)
- [ ] isolated per-step microbench (flat in ctx) + flip projection
- [ ] correctness sanity (perfect-draft c=32 + greedy vs target-only)

## COMMITS
- ebb2b44c spec-decode: DFlash kv_project (project new ctx rows' K/V) + bit-identical check
