# 06 — IMPL: DFlash decode HBM (fit c=32 @ max-model-len >= 4224)

Phase: HBM optimization. Date: 2026-06-27. Manager: impl-hbm.
GOAL blocker = DFlash proposer's per-slot _ctx_buf OOMs at c=32, only fit at len 1024.
Need it to serve at --max-num-seqs 32, max-model-len >= ~4224 (for out=4096).

## TL;DR — SERVES NOW at c=32 / len 4224 / util 0.75 (32/32 coherent, no OOM)
Two numerics-IDENTICAL fixes landed. The GOAL bench config (in=1, out=4096, c=32) now
fits at max-model-len 4224, util 0.75. NUMERICS UNCHANGED ⇒ NEEDS_BENCH (no re-test).
Remaining headroom is tight (chip-0 ~0.4 GiB); util 0.80 still OOMs on a SEPARATE
write transient (handoff item, chip task_d0ad4933).

## The budget (measured, L3)
- `_raw_hidden_dim = 14400` (= 5 target layers [target_layer_ids len 5] × hidden 2880).
  NOT 13440 (SHARED_MEMORY's earlier estimate) and NOT the model's 2880 hidden.
- `_ctx_buf` = (max_num_reqs=32, buf_len, 14400) bf16, **REPLICATED per chip** (plain
  jnp.zeros, no sharding). buf_len was `_next_padded_size` = NEXT POWER OF TWO over
  max_model_len ⇒ at len 4224 buf_len=**8192** ⇒ **7.03 GiB/chip**. That was the OOM.
- TP axis = `ShardingAxisName.MLP_TENSOR` (size 8). raw_hidden's D axis is currently
  REPLICATED (target activation is PartitionSpec(MLP_DATA, None)) ⇒ D-sharding _ctx_buf
  is NOT free (would need resharding the write + sharding the draft fc input dim).

## Fix chosen (2 parts, both BIT-IDENTICAL numerics)
Picked the lowest-numerics-risk levers per GOAL ("most headroom, least risk to proven
numerics"). REJECTED D-sharding (Option B): 8-partial all-reduce vs single fp32 matmul
sum can flip a near-tie token ⇒ would force a full lossless re-test. Avoided.

1. **Right-size buf_len** (commit 395e90ce): change context padding granularity from
   next-power-of-two to next multiple of a fixed block `_ctx_pad_block=512`, applied
   consistently to buf_len (load_model), padded_ctx (prepare_inputs per-step slice), and
   the precompile warm set. At len 4224 buf_len=**4608** ⇒ **3.96 GiB/chip** (was 7.03).
   Pure shape/memory change: padding rows are attention-masked to ~0 and consumed only by
   per-position ops (verified adversarially + against HF draft model) ⇒ identical tokens.
   Bounds proven: padded_ctx=ceil512(max_ctx) <= ceil512(max_model_len)=buf_len for all
   max_ctx<=max_model_len. Precompile now enumerates exactly the runtime block-multiples
   {512..4608} (9 shapes) so no unwarmed shape under VLLM_XLA_CHECK_RECOMPILATION.

2. **Fuse the read-slice into the jit** (commit 2f62f0f1): the eager
   `ctx_padded = self._ctx_buf[:num_reqs, :padded_ctx]` (old dflash.py:336) materialized a
   FRESH ~3.96 GiB copy of the buffer EVERY step (XLA gather, not a view) — that was the
   1st smoke's crash site (1/32 succeeded). MOVED the slice INSIDE the jitted draft
   forward: pass the FULL buffer + static (num_reqs, padded_ctx); slice
   `target_hidden[:num_reqs, :padded_ctx]` right before the fc matmul so XLA fuses it (no
   standalone copy). Same values, slice point moved, static trace shape unchanged ⇒
   identical tokens. target_hidden_states is now a 5-tuple
   (ctx_full, position_ids, attention_mask, num_reqs, padded_ctx).

## What changed + where
- `tpu_inference/spec_decode/vllm/dflash.py`: add `_ctx_pad_block=512`; `_next_padded_size`
  → ceil-to-block (now an instance method); precompile loop → block-multiples + warms full
  buffer with static args; prepare_inputs → no eager slice, pass full ctx_buf + ints;
  propose() → unpack 5-tuple, pass num_reqs/padded_ctx to forward.
- `tpu_inference/models/vllm/dflash.py`: draft_forward gains static_argnums=(6,7)
  (num_reqs, padded_ctx); slices target_hidden[:num_reqs, :padded_ctx] before fc.
- `tests/spec_decode/test_dflash_torchax.py`: brought up to the batched per-slot contract
  it had drifted from (was 3/5; the 2 failures were PRE-EXISTING batched-era staleness, not
  mine) — 3-D _ctx_buf, per-slot _ctx_len arrays, batched noise/sampler shapes, 5-tuple
  target_hidden_states, real lm_head weight + input_batch.num_reqs in lifecycle mock, small
  test block (16). Now 5/5. JAX-native test_dflash.py UNTOUCHED (5/5).

## Smoke result (L3, warm-ish)
- **util 0.75, len 4224, c=32: 32/32 succeeded** in ~632s, 31 reqs hit full 4000 tokens,
  engine stayed up, coherent greedy (Paris / George Washington / Au / 299,792,458 /
  J.K. Rowling / Jupiter / oxygen / four). NO RESOURCE_EXHAUSTED in the log.
- _ctx_buf logged shape **(32, 4608, 14400)** = 3.96 GiB confirmed.
- KV cache 1,972,451 tokens; chip-0 used 26.9/31.25 GiB ⇒ free ~4.35 GiB (chips 1-7 freer).
- util 0.80: STILL OOMs — but at a DIFFERENT site than before: dflash.py:324, the eager
  per-step `lax.dynamic_update_slice` WRITE, which also copies the whole 3.96 GiB buffer
  (functional update on a non-donated persistent buffer). My read-slice fix removed the
  336 transient; the 324 write transient is the next-largest. At 0.75 there's room for it;
  at 0.80 (bigger KV) chip-0 only has 2.77 GiB free < 3.96 GiB needed.

## Headroom achieved + numerics risk
- Before: only fit at len 1024 (buf 0.88 GiB). After: fits at len 4224 (the GOAL out=4096
  point), util 0.75, chip-0 margin ~0.4 GiB (TIGHT but works).
- **NUMERICS: bit-identical.** Both fixes change only memory layout / where a slice is
  materialized, never produced values. No re-test required ⇒ NEEDS_BENCH.

## Handoff: remaining headroom lever (NOT done — fresh-manager chunk)
The per-step WRITE `self._ctx_buf = lax.dynamic_update_slice(...)` (dflash.py:324) still
copies the full 3.96 GiB buffer each step (eager, non-donated). Fix = jit it with
donate_argnums on the buffer (donation SAFE: input is immediately replaced by output) so
the update is in place — would reclaim the 3.96 GiB transient and unlock util >= 0.80 /
longer context. Needs a static update shape (per-slot n_copy varies → pad or batch-scatter)
and a perfect-draft batch>1 re-verify (could touch the proven write path). Spawned as chip
**task_d0ad4933**.

## Commits
- 395e90ce right-size _ctx_buf (ceil-to-block 512; 7.03→3.96 GiB at len 4224)
- 2f62f0f1 fuse ctx-buffer read-slice into jitted draft forward (kill the 336 transient)

## Status
- [x] measure budget  [x] pick fix (low-risk: right-size + fuse, NOT shard)
- [x] implement + commit (incremental)  [x] unit tests 5/5 (torchax) + 5/5 (jax-native)
- [x] smoke serve c=32/len4224 (util 0.75) = 32/32 coherent, no OOM
- [x] numerics bit-identical (no re-test)  → NEEDS_BENCH
