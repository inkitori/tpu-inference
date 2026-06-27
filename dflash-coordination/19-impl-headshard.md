# 19 — IMPL: head-shard the DFlash draft attention (the 33–60x draft lever)

Phase: land the validated head-shard fix from 18-redteam-attn (the 48ms draft attn
is REPLICATION-bound; shard the 64 q-heads across the 'model' axis). Date:
2026-06-27. Manager: impl-headshard. Builds on 18-redteam-attn (the fix + real-mesh
projections), 16-impl-window (Lever B landed; windowing dead), 12-impl-kvcache (the
cached forward this modifies), 10-impl-condense (per-slot K/V cache + condense).

## TL;DR / VERDICT — HEAD-SHARD LANDED, BIT-EXACT, 8.6x ⇒ NEEDS_IMPL (Lever B)
The full-context cached draft forward drops **83.8ms → 9.7ms (8.6x)** on the real
8-chip mesh, **bit-exact** (max|diff| vs replicated = 0.0, greedy-argmax 100%),
all-gathers **72 → 0**, HBM bytes 54.7GB → 9.9GB. Full context KEPT ⇒ accept stays
~6 (no windowing tradeoff). The draft attention is no longer the bottleneck. The
binding constraint is now the **~40ms FIXED host overhead (Lever B)**, exactly as
18 projected. Projection vs B=3752 (full-context, accept=6): FIXED40 0.78x (today,
loses), FIXED25 **1.01x**, FIXED15 **1.25x**, FIXED8 **1.51x**. ⇒ to beat B in the
serve bench, cut the fixed overhead (Lever B) — now achievable WITHOUT windowing.

## WHAT CHANGED + WHERE
### models/vllm/dflash.py
- **`_sharded_attention(q,k,v,mask,o_weight,scaling,mesh)`** (new, module-level
  ~L40): head-parallel draft attention via `jax.shard_map` over the 'model' axis
  (ShardingAxisName.ATTN_HEAD). Each chip computes ONLY its nh/8 q-heads + kvh/8
  kv-heads: GQA expand (broadcast+reshape WITHIN the local head group), q@k^T *
  scaling + mask, **f32 softmax → bf16**, scores@v, and its slice of the o_proj
  matmul (o_weight sharded P(None,'model') on the contraction axis); per-shard
  o_proj outputs `jax.lax.psum`'d. in_specs q/k/v = P(None,ATTN_HEAD,None,None),
  mask P(), o_weight P(None,ATTN_HEAD); out_specs P(). Numerically IDENTICAL to
  transformers `eager_attention_forward` (same scaling/mask/f32-softmax/GQA). The
  o_proj BIAS is added by the CALLER after (per-shard would multiply it 8x).
- **`_draft_forward_cached`** (~L335): replaced the `eager_attention_forward` +
  `sa.o_proj(...)` call with `_sharded_attention(jax_view(q),...,sa.o_proj.weight,
  sa.scaling, self._mesh)`, then `+ sa.o_proj.bias` (replicated) once. q/k/v
  projection/RoPE/concat unchanged.
- **`load()`**: sets `self.model._mesh = self.mesh` so `_draft_forward_cached` can
  reach the mesh for shard_map.
- **`_reshard_draft_attn_heads()`** (88862fdb, kept): reshards q/k/v_proj.weight →
  P(ATTN_HEAD,None), o_proj.weight → P(None,ATTN_HEAD). Still useful (o_weight goes
  into shard_map sharded; q_proj output feeds the shard_map boundary).
- **`get_kv_project_fn`** out_sharding (88862fdb): KVH axis on KV_CACHE_HEAD so the
  cached K/V the writer produces are head-sharded.
### spec_decode/vllm/dflash.py
- **`_k_cache`/`_v_cache`** allocated with `P(None,None,None,KV_CACHE_HEAD,None)` —
  KVH axis (axis 3) head-sharded. SAFE for `_batched_kv_write` (indexes slot+row,
  broadcasts over KVH) and `_move_kv_rows`/condense (indexes slot axis only); the
  gather in `get_draft_forward_cached_fn` (dynamic_slice buf_len axis, vmap slot
  axis) is untouched by axis-3 sharding. ⇒ condense stays correct.

## ROOT CAUSE — why weight-sharding + with_sharding_constraint did NOT work
(Real-8-chip HLO dumps; the decisive debug.) GSPMD all-gathers q back to all 64
heads at `q_proj(hs).view(N,B,nh,hd)` — it splits the model-sharded nh*hd
projection-output axis and chooses to GATHER rather than keep it sharded (killer
op `all-gather bf16[N,64,B,hd]`, op_name `aten::view/reshape`). `repeat_kv`'s
reshape (merging the sharded KVH axis with n_rep) does the same to the SCORES.
`jax.lax.with_sharding_constraint` (the `_pin` attempt, ea3a4195, reverted
cc735253) CANNOT prevent it — the constraint is satisfied by re-slicing the
already-gathered replicated tensor (adding a downstream constraint even INCREASED
all-gathers 9→12). `shard_map` is the fix: it gives each chip no freedom to gather
— it computes only its local heads. (`.contiguous()` is NOT the problem; torchax
maps aten.clone to a no-op.) Contrast: the pure-jax probe held the shard because it
starts with q already device_put head-sharded and never does the splitting `.view`.

## MEASUREMENTS (real v6e-8, isolated)
- probe_headshard_cached.py: cached full fwd @C=4096 **83.8ms → 9.7ms** (8.6x);
  max|diff| (head-sharded vs fully-replicated, same session) = **0.0**;
  greedy-argmax **100%**.
- probe_hlo.py: all-gather **72→0**, all-reduce 24→8 (one psum/layer), bytes
  accessed 5.47e10→9.92e9.
- Reference: 18's isolated eager head-sharded attn was 1.49ms @C=4608; the 9.7ms
  here is the WHOLE cached fwd (8 layers: attn ~1ms + MLP/o_proj/norms/RoPE over
  B=8 rows + the K/V gather), consistent.

## PROJECTION vs B=3752 (step = FIXED + verify 9.08 + draft_total; draft_total =
kv_proj 0.56 + kv_write 6.52 + cached_fwd 9.7 = 16.78ms; tput = accept·32/step)
| FIXED (ms) | step | accept=6 | accept=5 |
|---:|---:|---:|---:|
| 40 (today) | 65.9 | 2915 (0.78x) | 2429 (0.65x) |
| 25 | 50.9 | 3775 (**1.01x**) | 3146 (0.84x) |
| 15 | 40.9 | 4699 (**1.25x**) | 3916 (**1.04x**) |
| 8  | 33.9 | 5670 (**1.51x**) | 4725 (**1.26x**) |
Device-component sum (verify+draft) = 25.86ms; the ~40ms gap to the measured serve
step (~105ms) is the FIXED host overhead. ⇒ head-shard alone (FIXED~40) still
loses; with Lever B cutting FIXED to ≤25ms it clears B, ≤15ms with margin, AND it
keeps full context so accept stays ~6 (unlike windowing, 16-impl, which craters to
2.5). This is the FULL-CONTEXT win the open question in 16-impl asked for.

## CORRECTNESS / CONDENSE / HBM
- Bit-EXACT (max|diff| 0.0): head-parallel attention is per-head independent
  through q@k^T/softmax/scores@v; o_proj psum reproduces the same sum. Same dtypes
  (f32 softmax preserved — pure-bf16 SIGSEGVs the v6e conv-emitter).
- Condense: the KVH-axis (axis 3) cache sharding does not touch the slot/row axes
  that writes/moves index ⇒ 10-impl/12-impl condense machinery unchanged. (A real
  serve perfect-draft-through-condense re-verify is still recommended next.)
- HBM: cache sharded on KVH ⇒ per-chip cache footprint ~÷8 vs replicated (frees
  HBM); the shard_map intermediates are per-chip small.
- Non-cached `_draft_forward` (first step, before cache fills): still uses the HF
  eager path (NOT head-sharded) — fine for correctness, negligible at out=4096.

## COMMITS (branch dflash, pushed; local==origin)
- 88862fdb head-shard the projection weights + cache (alone: all-gathered, no win)
- ea3a4195 WIP _pin (with_sharding_constraint) — inert on torchax
- cc735253 revert _pin
- **36ecb75c head-parallel draft attn via shard_map — 84ms→9.7ms, 0 all-gathers** ← the fix
- Probes: scratchpad/probe_headshard_cached.py, probe_hlo.py,
  probe_headshard_components.py (pure-jax 9.6x proof), _l3_minirepro.py.

## NOTE — concurrent sibling session
A sibling Claude session worked the identical task on the same single TPU and
independently reached the same root cause + reverted the inert _pin (cc735253);
the shard_map fix builds directly on that clean tree. No work lost. Single-tenant
device ⇒ serialize TPU access across siblings to avoid wasted compiles.

## STATUS / WHAT REMAINS
- [x] head-shard the draft attention (shard_map) — LANDED, bit-exact, 8.6x, 0 all-gathers
- [x] K/V cache + kv_project head-sharded; condense-safe (axis analysis)
- [x] isolated real-mesh microbench before/after + projection vs 3752
- [ ] **LEVER B: cut the ~40ms FIXED host overhead** (drop un-overlapped dispatches,
      batch/minimize device_gets, vectorize the 32-iter host loops). NOW the binding
      constraint; at FIXED≤15ms the full-context path clears B by ~1.25x. (16-impl
      already landed part of Lever B; more remains.) Stay on --no-async (async=Phase2).
- [ ] real serve re-verify of the sharded path: perfect-draft through condense +
      greedy-vs-target lossless (the draft attn formulation changed) — NEEDS_TEST.
- [ ] then the real out=4096/c=32 serve bench (BENCH manager) — the verdict.
