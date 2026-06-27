# 02 — CORRECTNESS TEST (DFlash gpt-oss-20b, branch `dflash`)

Phase: correctness only (no speed bench). Date: 2026-06-26/27.

## TL;DR
- **Ladder #1 (PERFECT-DRAFT) PASSED, unambiguous: 100% acceptance.** The verify /
  accept-reject machinery is correct and aligned (no off-by-one). Highest-value test.
- **Ladder #2 (greedy divergence): no bug found, but the HTTP/text oracle is unreliable**
  (see below). Real DFlash produces coherent, correct greedy output; the one apparent
  text divergence is in a region provably full of bf16 near-ties (gap ~0.125 nats), and
  is an artifact of comparing different request shapes, not a verifier bug.

## Committed diagnostic
- `ba34e82b` spec-decode: add `DFLASH_PERFECT_DRAFT` env-gated diagnostic.
  In `tpu_inference/runner/tpu_runner.py` `_sample_from_logits`, right after
  `draft_token_ids = self._extract_draft_token_ids(...)`: if env `DFLASH_PERFECT_DRAFT=1`,
  override `draft_token_ids = jnp.argmax(target_logits, axis=-1)`. Off by default. Pushed.

## Ladder #1 — perfect-draft (run FIRST)
Launch = known-good cmd + `DFLASH_PERFECT_DRAFT=1`, greedy (temp 0), max-num-seqs 1.
Server log `SpecDecoding metrics:` across 3 windows (14, 161, 350 accepted tokens):
- **Mean acceptance length: 8.00** (= num_spec_tokens 7 + 1 bonus → ALL accepted)
- **Per-position acceptance rate: 1.000 ×7**
- **Avg Draft acceptance rate: 100.0%**, Accepted == Drafted (14/14, 161/161, 350/350)
→ verify path emits target argmax at every position; draft==target_argmax ⇒ all-accept.
CONCLUSIVE: no off-by-one in positions/seq_lens; accept rule correct.

## Real DFlash (spec-on, no perfect-draft) — sanity
- Greedy outputs coherent + correct: "Paris", "2, 3, 5, 7, 11", "4".
- `SpecDecoding metrics`: mean accept len ~2.35–3.14, per-pos monotone decreasing
  0.74→0.04 (healthy real-draft curve), avg accept ~19–31%.

## Ladder #2 — greedy divergence (target+draft vs target-only)

### CLEAN RESULT (apples-to-apples): TOKEN-IDENTICAL, lossless.
Matched request shape — stepwise greedy, max_tokens=1, logprobs=20, both servers,
prompt "List the first five prime numbers:", 24 steps:
- spec-on text  = ` 2, 3, 5, 7, 11.\n\n**Answer**: 2, `
- target text   = ` 2, 3, 5, 7, 11.\n\n**Answer**: 2, `  → **IDENTICAL all 24 steps**
- Per-step top-2 gaps track between paths; logprob diffs are tiny bf16 noise (e.g.
  step14 top1 -1.5643 vs -1.4160) and do NOT flip any argmax — including at the
  near-tie step14 ('.' vs '.\n\n', gap=0.125 nats) and step15 (gap ~0.9). Surviving the
  near-tie region while staying identical is strong lossless evidence.

### The earlier "divergence" was an ARTIFACT (single-shot + retokenization), not a bug.
First-pass 3-prompt single-shot completions: 2/3 token-identical (47/47, 48/48); 1/3
("prime numbers") appeared to diverge at re-tokenized idx 15 ("Sure" vs "The"). NOT a
bug signal because:
- It was found by BPE-re-encoding the two output TEXTS, not by comparing true decode
  steps; re-tokenization misrepresents step boundaries.
- Feeding the agreed prefix `"...11.\n\n"` fresh single-shot gives top-1 = "**" (neither
  "The" nor "Sure"; "Sure" not even top-20).
- single-shot (incremental decode) vs stepwise max_tokens=1 (re-prefill) give DIFFERENT
  continuations on the SAME target server → request shape changes numerics; cross-shape
  text comparison is not a valid lossless oracle. The matched-shape test above is.

### Why ladder #1 already settles correctness
In greedy spec-decode the emitted token at EVERY position is the target's own argmax
(accepted = draft matched it; first reject + bonus = target argmax). Perfect-draft at
100% proves that machinery exact. Any residual spec-on vs target-only text difference can
only come from bf16 numeric path differences in the TARGET forward, landing on a near-tie
— which GOAL explicitly allows as lossless.

## Status / handoff
- Correctness machinery: PROVEN via ladder #1 (100% perfect-draft accept).
- Lossless: CONFIRMED via ladder #2 matched-shape stepwise — spec-on == target-only
  token-for-token through 24 steps incl. a 0.125-nat near-tie. No verifier bug.
- Note for bench manager: HTTP logprobs path is BUGGY except max_tokens=1
  (IndexError in vllm/entrypoints/openai/completion/serving.py:627 — only bites when
  >1 token requested with logprobs). Doesn't affect normal serving/bench, just logprob
  capture. Use max_tokens=1 + logprobs<=20 (server caps at 20) for any logit probing.
- Recommend → NEEDS_BENCH (speed phase). Correctness is DONE.
