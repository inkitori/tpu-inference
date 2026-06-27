# 20 — RECONCILE/VERIFY: did the revert clobber the shard_map head-shard? (NO — HEAD is known-good)

Phase: establish git ground-truth + empirically verify HEAD, resolve the
Report-A-vs-Report-B contradiction. Date: 2026-06-27. Manager: reconcile.
Builds on 19-impl-headshard (Report A, the fix), 18-redteam-attn (the physics).

## TL;DR / VERDICT — shard_map IS the real, effective fix. HEAD is known-good.
The revert (cc735253) did NOT clobber the shard_map fix. Git history is LINEAR and
the revert lands BEFORE the fix, so nothing was lost. HEAD's cached draft forward
runs head-sharded: **9.7ms, 0 all-gathers, bit-exact (max|diff| 0.0, greedy-argmax
100%)**. Report A is correct; Report B's pessimism applies ONLY to the
with_sharding_constraint/GSPMD approach it tested (which IS inert) — B's claim that
"only a hand-rolled grouped-attention rewrite works" is FALSE: shard_map is exactly
that head-parallel grouped attention and it works. No restore needed. HEAD =
865c7516 (== origin/dflash, pushed). STATUS: NEEDS_TEST (lossless serve re-verify
of the sharded path + Lever B + bench — the draft attn is settled).

## 1) GIT FORENSICS — commit order + net effect (the contradiction, resolved)
Linear history on `dflash` (oldest → newest):
1. `88862fdb` head-shard the projection WEIGHTS + cache (alone: all-gathered, no win)
2. `02199850` bench/coord note (documents the all-gather problem)
3. `ea3a4195` WIP `_pin` (with_sharding_constraint) — inert on torchax
4. `cc735253` **REVERT the `_pin`** ← Report B's revert
5. `36ecb75c` **head-parallel draft attn via jax.shard_map** ← THE FIX (Report A)
6. `865c7516` coord docs (= CURRENT HEAD)

Net effect: the revert (step 4) only undid the inert `_pin` (step 3). The shard_map
fix (step 5) was committed AFTERWARD, so the revert could not and did not clobber it.
`git diff 88862fdb HEAD -- tpu_inference/models/vllm/dflash.py` = a +79-line SUPERSET
(adds `_sharded_attention` + shard_map wiring) — HEAD is NOT reverted-to-baseline.
`_draft_forward_cached` calls `_sharded_attention` (shard_map), NOT the replicated
`eager_attention_forward` (that import/call was removed). `spec_decode/vllm/dflash.py`
keeps the head-sharded K/V cache (KV_CACHE_HEAD on axis 3) + kv_project intact.
Working tree clean; HEAD == origin/dflash.

⇒ The "revert may have clobbered A's fix" worry is FALSE for this branch.

## 2) EMPIRICAL MICROBENCH ON HEAD (real v6e-8, isolated — source of truth)
probe_headshard_cached.py (cached full fwd @ C=4096, N=32, head-sharded weights):
- q_proj.weight P('model', None); o_proj.weight P(None, 'model') — heads sharded 8/chip.
  Load log: "DFlash draft attention head-sharded: re-sharded 64 projection tensors".
- (A) REPLICATED caches      = 10.07 ms
- (B) HEAD-SHARDED caches    =  9.71 ms   ← serve layout = HEAD's real number
- (C) "replicated-weights"   = 10.30 ms   (baseline in-probe)
- numeric max|diff| (B vs C) = **0.000e+00**;  greedy-argmax agreement = **100.00%**
probe_hlo.py (compiled HLO of `_draft_forward_cached`):
- **all-gather = 0**, all-reduce = 8 (one psum/layer), reduce-scatter 0, coll-permute 0.

INTERPRETING "84ms didn't reproduce" (NOT a problem — expected):
The probe's (C) baseline replicates only the projection WEIGHTS; it still runs the
NEW `_sharded_attention` (shard_map) because that code path is what HEAD's file does.
The old 84ms path was the deleted `eager_attention_forward` REPLICATED algorithm,
which no longer exists in the file — so the probe cannot exercise it, and all of
A/B/C land at ~10ms (all are the shard_map path). The decisive head-shard-active
signals are the ones that DID land: **9.7ms + 0 all-gathers + bit-exact**, exactly
the "fixed" numbers in 19. (The 84→9.7ms before/after was 19's same-session dev
measurement; no agent ever disputed the eager path was ~84ms.) The all-gather count
is the structural proof: a replicated/clobbered path would show ~72 (per 19); HEAD
shows 0 ⇒ heads are genuinely sharded, never gathered back.

## 3) RESTORE — none needed
HEAD already contains the effective, bit-exact shard_map head-shard. Nothing was
clobbered; no cherry-pick / re-apply required. Left HEAD as-is.

## 4) SETTLING THE CONTRADICTION (definitive, in writing)
- Report A (shard_map works, 36ecb75c): **CORRECT and present + effective on HEAD.**
- Report B (revert + "needs a hand-rolled grouped-attention rewrite"): the REVERT
  (cc735253) was of the INERT `_pin` only and is fine; but B's CONCLUSION is WRONG —
  shard_map IS the head-parallel grouped attention and it is effective (0 all-gathers,
  9.7ms, bit-exact). B's pessimism is valid ONLY for the GSPMD/with_sharding_constraint
  approach it actually tested (which repeat_kv defeats by merging the sharded KVH axis).
  Next managers should NOT pursue B's "rewrite needed" path — it is already done.

## 5) UNIT TESTS
`tests/spec_decode/test_dflash.py`, `tests/spec_decode/test_dflash_torchax.py`,
`tests/models/vllm/test_dflash.py`: **19 passed, 1 failed, 0 skipped.** The single
failure is the KNOWN pre-existing `test_dflash_torchax_wrapper_fns` (stale call:
`draft_forward() missing num_reqs/padded_ctx` — test wasn't updated for the batched
signature). NOT a regression. No other failures, no import/collection errors.

## FINAL STATE
- HEAD = `865c7516` (== origin/dflash, pushed). Working tree clean.
- Head-shard: ACTIVE + bit-exact + 0 all-gathers. The draft attention is SETTLED.
- Binding constraint remains **Lever B (~40ms FIXED host overhead)** + a real serve
  lossless re-verify of the sharded path (attn formulation changed since 13-test).
⇒ STATUS: NEEDS_TEST.
