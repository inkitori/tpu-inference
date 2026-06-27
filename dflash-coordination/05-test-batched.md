# 05 — BATCHED CORRECTNESS TEST (DFlash gpt-oss-20b, branch `dflash`)

Phase: correctness of the NEW batched (multi-request) path. Date: 2026-06-27. Manager: test-batched.
Builds on 02-test.md (batch=1 PASS). Reuses DFLASH_PERFECT_DRAFT=1 injection (tpu_runner.py L1666).

## TL;DR
- **PERFECT-DRAFT @ batch=16 (uniform): PASS** — global aggregate 448/448 accepted, mean accept
  len 8.00, per-pos 1.000×7 across 4 windows.
- **PERFECT-DRAFT @ batch=16 (RAGGED, mixed lengths + staggered arrival): PASS** — 672/672 accepted,
  mean 8.00, per-pos 1.000×7 across 4 windows. No query_start_loc / per-slot slicing bug.
- **PERFECT-DRAFT @ batch=32 RAGGED (GOAL concurrency): PASS** — 1302/1302 accepted, mean 8.00,
  per-pos 1.000×7 across 5 windows. HBM FITS at c=32 / len 1024 / util 0.75 (no RESOURCE_EXHAUSTED).
- **L3 CODE AUDIT of per-slot ctx buffers: CLEAN** — no slot cross-contamination. Every per-slot
  read (seq_lens_host[i], qsl_host[i], _ctx_len[i]) + write (dynamic_update_slice start (i,ctx_len_i,0),
  update leading dim ==1) is indexed by the SAME slot i; forward masks everything beyond each row's
  own ctx_len. This covers the one bug class perfect-draft CAN'T (drafter slot-mixing, since pd
  overrides draft tokens). Minor: dead `num_rejected_tokens` param in prepare_inputs; inert
  draft_attn_metadata (propose() never reads attn_metadata on torchax stateless path) — neither a bug.
- **GREEDY LOSSLESSNESS @ batch>1: <PENDING>**

## Why the aggregate metric is a valid PER-SLOT detector
SpecDecoding metrics (vllm/v1/spec_decode/metrics.py:120) are GLOBAL counters summed across all
requests in a logging window. Under DFLASH_PERFECT_DRAFT=1 the draft tokens are forced to the
target's own greedy argmax, so EVERY slot's expected acceptance is exactly 100% (mean accept len
8.00 = num_spec 7 + 1 bonus). If even ONE slot in the batch had a per-slot indexing / slicing bug,
its rejections would drag the global mean below 8.00 and per-pos rates below 1.000. They did not.

## Config used (fits HBM per 04-impl)
max-model-len 1024, --gpu-memory-utilization 0.75, EP, RAGGED_GATHER v1, --no-async,
DRAFT_MODEL_IMPL_TYPE=torchax, num_speculative_tokens 7, HF_HOME=/home/enyouki/local_hf.
Serve script: scratchpad/serve_dflash.sh <max_num_seqs> <log>. Client: scratchpad/fire_concurrent.py.

## Test 1 — perfect-draft, uniform batch=16 (serve_pd16.log, post-line-321)
16/16 requests OK. Correct answer at START of every completion (Paris, George Washington, Au,
299792, J.K. Rowling, Jupiter, oxygen, 12). 4 metrics windows, all:
mean 8.00 | per-pos 1.000,1.000,1.000,1.000,1.000,1.000,1.000 | Avg 100.0% | Accepted==Drafted.
Totals: Accepted 448 / Drafted 448 (perfect 1:1).
NOTE on text: completions degrade into repetition after the correct prefix — EXPECTED. With
perfect-draft the output is pure target greedy; raw greedy gpt-oss-20b (no chat template, short
prompts, len 1024) degenerates. Same would happen target-only. Irrelevant to accept-rate test.

## Test 2 — perfect-draft, RAGGED batch=16 (serve_pd16.log, post-line-434)
16 concurrent, DIFFERENT prompt lengths ("Hi." .. ~50-word paragraphs), STAGGERED arrival
(0.12s × idx) → batch holds requests at different decode positions in the same step. 16/16 OK,
correct prefix each (2,3,5,7,11; Rayleigh scattering; 4; red/blue; Charles Babbage). 4 windows, all
mean 8.00 | per-pos 1.000×7 | 100.0% | Accepted==Drafted. Totals: Accepted 672 / Drafted 672.
⇒ No ragged query_start_loc / per-slot context-buffer slicing regression.

## Test 3 — greedy losslessness @ batch>1 (PENDING)
Plan: spec-on (DFlash) vs target-only, a few concurrent greedy requests, compare token identity /
agreement before first divergence; any divergence must be a genuine bf16 top-2 near-tie.
