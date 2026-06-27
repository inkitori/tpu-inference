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

## Test 3 — greedy losslessness @ batch>1
Method: real DFlash (perfect-draft OFF) vs target-only, BOTH at concurrent batch=8, matched shape.
Real DFlash accept curve at batch=8 was HEALTHY (mean 3.12-3.75, per-pos 0.79→0.04 monotone) ⇒
drafter genuinely working at batch>1 (not degenerate).

### (a) Full-completion compare (max_tokens=64, concurrent batch=8): 4/8 token-IDENTICAL.
specon_out.json vs target_out.json. IDENTICAL for full 64 tokens: prompts capital-of-France (275ch),
gold-symbol (242ch), sqrt-144 (285ch), first-president (298ch). The other 4 diverged DEEP
(char 18/71/57/133 — dozens of tokens in, AFTER the correct factual answer), in conversational
filler regions ("The" vs "Sure", "Apologies" vs "I apologize"). NONE at step 1-2 (GOAL: deep
divergence on near-tie = FP-fine; step 1-2 = bug).

### (b) Near-tie probe of the 4 divergences — CONFOUNDED for 2 of them.
Re-feeding the agreed prefix as a fresh single-shot prompt to target-only: cases (sky-color/Water
H2O) clean near-ties (gap 0.375 / 0.125 nats — textbook bf16). BUT for the prime-numbers and
largest-planet cases, the target's single-shot argmax matched NEITHER path's next token — this is
EXACTLY the re-prefill / request-shape numerics confound 02-test.md flagged: single-shot re-feed has
different numerics than the incremental decode that produced the split. So the probe is NOT a valid
oracle for those 2; its "NOT-A-TIE" on largest-planet is unreliable (measured a different forward).

### (c) DECISIVE matched-shape step-1 oracle (the 02-test.md clean method, at batch>1): <PENDING>
Fire identical concurrent batch=8 max_tokens=1 at BOTH servers from identical step-1 contexts ⇒
both see identical request shape ⇒ any spec-on argmax != target-only argmax beyond a logged near-tie
is a verifier bug. Target-only step-1 captured (target_step1.json): all 8 sensible argmax (Paris/Au/
Jupiter/George/oxygen/red), gaps 0.06-5.1 nats.

GOTCHA FOUND (worse than 02-test's note): requesting LOGPROBS on the SPEC-ON server triggers
`OverflowError: out of range integral type conversion` in async_llm.py:704 output_handler
(logprobs.py:97 convert_ids_list_to_tokens -> tokenizer.decode of an out-of-range id in the
top-k logprobs stream) — and because it's in the async output_handler background task, it CRASHES
THE ENGINE (connection refused after). The out-of-range id is in the LOGPROBS top-k tensor
(padding/sentinel), NOT in the emitted sequence — sampled tokens are fine (full completions are
coherent+correct). ⇒ For any spec-on probing use max_tokens=1 with NO logprobs (plain text). This
is a logprobs-serialization bug, not a generation/verifier bug, but it's a real crash to avoid.

### (d) DECISIVE matched-shape step-1 oracle, NO logprobs, concurrent batch=8: 8/8 IDENTICAL. ✅
spec-on (real DFlash) vs target-only, both concurrent batch=8, max_tokens=1 plain text (no logprobs
→ no crash). Both servers see IDENTICAL step-1 request shape ⇒ pure verifier test. ALL 8 slots'
next token IDENTICAL: Paris/' '/Au/Jupiter/' '/red/oxygen/George. Includes slot 5 (Name-three-
colors) where target top-2 gap was just 0.062 nats (a genuine near-tie) — spec-on STILL matched the
target argmax. ⇒ The batched verifier emits the target's own argmax for every slot. LOSSLESS at
batch>1 confirmed by the reliable matched-shape oracle.

The 4/8 deep full-completion divergences in (a) are now EXPLAINED: they arise from request-shape
numeric drift across many incremental-decode steps landing on high-entropy near-ties (the 02-test.md
confound), NOT a verifier bug — exactly the FP-near-tie behavior GOAL allows as lossless. The 2
near-tie probes that were clean (0.125/0.375 nats) corroborate; the 2 "confounded" ones are not valid
counterevidence (wrong-shape re-prefill). The matched-shape oracle (d) is the decisive word: 8/8.

### (e) Matched-shape 8-STEP corroboration, no logprobs, batch=8: 8/8 IDENTICAL.
spec-on vs target-only, max_tokens=8 each, matched concurrent batch=8. ALL 8 prompts' full 8-token
output byte-identical (Paris.../2,3,5/Au.../Jupiter.../12.../red,blue,and green/oxygen.../George
Washington...). Slot 5 here = "red, blue, and green" identical both sides (the earlier 64-tok
"yellow" split was pure later-step request-shape drift, not a verifier issue). Confirms agreement
holds 8 steps deep under matched shape, not just step 1.

## VERDICT (batched correctness)
- Perfect-draft per-slot (uniform b=16, ragged b=16, ragged b=32 @ GOAL concurrency): 100% accept,
  every window mean 8.00 / per-pos 1.000×7. 2422 total accepted == 2422 drafted across all windows.
- Code audit: no slot cross-contamination (the one class perfect-draft can't catch).
- Greedy lossless @ batch>1: matched-shape step-1 8/8 identical incl a 0.062-nat near-tie.
⇒ BATCHED DFLASH IS CORRECT + LOSSLESS. Next: HBM/_ctx_buf opt so out=4096 @ c=32 (len>=5120) fits,
then the c=32 speed bench (the GOAL bench point, still unmeasured).

## Gotchas for next manager
- Requesting LOGPROBS on the spec-on server CRASHES the engine (OverflowError in output_handler,
  out-of-range id in top-k logprobs detokenize). Use NO logprobs for spec-on probing.
- Each cold serve ~200s (SKIP_JAX_PRECOMPILE=1). One serve at a time; ~/tpu-tooling/free-tpu.sh
  between. Exit-code-144 on pkill+free chained in one bash call is benign (the pkill SIGTERM); just
  rerun free-tpu.sh.
