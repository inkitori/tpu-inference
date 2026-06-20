# RoPE, MLA, and DSA/the indexer in GLM-5.2 — from scratch

**Date:** 2026-06-20
**Purpose:** Plain-language explainer of how RoPE, MLA, and the DSA indexer work, and
**why the indexer's RoPE convention differs from MLA's** (the HF-vs-vLLM divergence the
spec flags). Explanation only — no spec/code changes.

---

## 1. The thing everything is built on: attention

A language model reads a sequence of tokens and, for each position, decides *which
earlier tokens to look at*. "The cat sat on the ___" — to fill the blank it leans on
"cat" and "sat." The machinery for "look at earlier tokens" is **attention**. Every
other concept here (MLA, DSA, the indexer) is a modification of plain attention.

Each token emits three vectors (a vector = a list of numbers):
- **Query (Q):** "what I'm looking for"
- **Key (K):** "what I am / what I offer"
- **Value (V):** "the actual content I'll hand over if you attend to me"

To decide how much token A attends to token B:
1. **Dot-product** A's query with B's key → one number measuring *alignment* (the
   **score**). Big number → "these match, look here."
2. Compute that score against **every** past token → a list of scores.
3. **Softmax** the list → percentages summing to 100%.
4. Take a **weighted average of the Values** using those percentages.

Two facts to hold onto:
- **Scores = query·key dot products.**
- **You score against every past token.** That "every" is expensive, and killing it is
  the whole point of DSA later.

---

## 2. RoPE — putting *position* into attention

The dot product Q·K is **position-blind**: "cat" at position 2 and position 50 produce
identical keys. But order is meaning ("not good" vs "good, not bad"). So we inject *where*
each token is into Q and K **before** the dot product. RoPE does that.

### The rotation trick
RoPE **rotates** each query/key vector by an angle proportional to the token's position
(position 1 → rotate a little, position 50 → rotate 50×). Why rotation? Because when you
dot-product two *rotated* vectors, the result depends only on the **difference** of their
angles — i.e. on **how far apart the two tokens are**. Attention gets *relative* position
for free.

### Rotating a long vector = rotating little 2D pairs
You can only rotate in 2D. So RoPE **chops the vector into pairs**, treats each pair as a
2D point (x, y), and spins each:
```
x' = x·cos θ − y·sin θ
y' = x·sin θ + y·cos θ
```
That's the entire mechanism: chop into pairs, spin each pair by an angle set by position.

### THE interleave-vs-rotate-half thing (the crux)
There are **two conventions for which numbers pair up**, and the model *must* use the one
its weights were trained with. Take an 8-dim vector `[d0 d1 d2 d3 d4 d5 d6 d7]`.

**Interleaved** — pair *adjacent* dims:
```
(d0,d1) (d2,d3) (d4,d5) (d6,d7)
```
**Rotate-half (a.k.a. NeoX)** — split in half, pair *across* the halves:
```
first half  [d0 d1 d2 d3]
second half [d4 d5 d6 d7]
(d0,d4) (d1,d5) (d2,d6) (d3,d7)
```

**Identical rotation math; totally different pairings.** `(d0,d1)` spins different numbers
than `(d0,d4)`, so the output differs, so the scores differ.

**Why a wrong choice is a silent catastrophe:** the layer that produced Q/K was *trained*
expecting one pairing. Apply the other and you rotate numbers that were never meant to
pair → position info comes out scrambled. The model doesn't crash; it produces subtly
wrong scores everywhere → degraded/garbage output. You only catch it by comparing
bit-for-bit against a reference. That's why this one detail is the highest porting risk.

So "interleaved vs rotate-half" literally means **"do adjacent dims pair, or do
first-half/second-half dims pair."**

---

## 3. MLA — GLM-5.2's *main* attention

GLM-5.2's primary attention is **MLA (Multi-head Latent Attention)** (from DeepSeek):
plain attention plus a memory-saving twist.

**Why it exists:** during generation you cache past Keys/Values (the **KV cache**) so you
don't recompute them; for long sequences that cache is gigantic. MLA stores **one small
compressed "latent"** per token instead of full K and V, and **re-expands** on demand.
`kv_lora_rank=512` is the size of that latent.

**The RoPE wrinkle:** you can't cleanly apply position-rotation to a thing you compressed
and decompressed — the math doesn't survive the round trip. So MLA **splits each Q and K
into two pieces**:
- a **"nope"** part — *no* position embedding — `qk_nope=192` dims, carries content
- a **"rope"** part — *gets* RoPE — `qk_rope=64` dims, carries position

Only the 64-dim rope slice is rotated; the score is `(nope·nope) + (rope·rope)`. For
GLM-5.2's MLA, that 64-dim slice uses the **interleaved** pairing — and **HF and vLLM
agree** on it. MLA is *not* the contested part.

---

## 4. DSA and the indexer

### The problem DSA solves
Plain/MLA attention scores **every** past token. 100,000 tokens deep → every new token
does 100,000 dot products. But most of those tokens are irrelevant.

### DSA = "only attend to the tokens that matter"
**DSA (DeepSeek Sparse Attention)** makes attention *sparse*: each token attends to only
the **top-2048 most relevant** past tokens (`index_topk=2048`), not all of them.

Chicken-and-egg: to know *which* 2048 are most relevant, don't you have to score all of
them? If you used full MLA to rank, you'd have paid the full cost.

### The indexer = a cheap "bouncer" that picks the 2048
The **indexer** is a **small, cheap, separate attention-like scorer** whose only job is to
quickly rank past tokens and output "these 2048 look most relevant." Then the expensive
MLA runs **only on those 2048**.

- **Indexer = bouncer at the door:** fast rough check on everyone, picks who gets in.
- **MLA = the expensive party inside:** only deals with the 2048 who got selected.

So each token does two passes:
1. **Indexer pass (cheap):** roughly score all past tokens, pick top-2048.
2. **MLA pass (expensive but small):** real attention over just those 2048.

The indexer is allowed to be rough (selection only, not the final blend). It has its
**own tiny weights** (`wq_b`, `wk`, `weights_proj`, `k_norm`, …), its own small head dim
(`index_head_dim=128`), and a simplified formula: dot products → `relu` → weighted sum
across heads → top-2048. It deliberately **skips** softmax and value-averaging.

### "full" vs "shared"
Running the indexer every layer still costs something, so GLM-5.2 cheats: **"full"**
layers recompute the top-2048; **"shared"** layers **reuse the previous layer's
selection**. Real config: layers 0,1,2 "full," then ~every 4th "full," rest "shared" —
so most layers reuse an earlier pick. That's what
`indexer_types`/`index_topk_freq`/`index_skip_topk_offset` control.

---

## 5. Why the indexer has a *different* RoPE than MLA

The crux: **the indexer is a completely separate mini-model with its own
separately-trained weights.** It is *not* MLA — it merely happens to be attention-shaped,
so it *also* needs position info and *also* uses RoPE. Nothing forces it to share MLA's
RoPE convention. GLM-5.2's authors wired:
- MLA's rope slice → **interleaved**
- the indexer's rope → **rotate-half**

Two independent pieces of code, two independent choices. The HF source even carries a
comment to the effect of "the indexer uses non-interleaved (half-split) RoPE, unlike main
MLA," precisely because it's surprising. **Apply the wrong pairing to either and you
scramble that component's position info** → the silent-garbage failure from §2.

### Why this caused the HF-vs-vLLM drama
The disagreement is about the **indexer's** pairing for the *real* checkpoint:
- HF's reference code **hardcodes** the indexer to rotate-half (ignores any config flag).
- The real GLM-5.2 config sets `indexer_rope_interleave: true`, and **vLLM reads that
  flag** → concludes interleaved.

So for the real model the two references **disagree on the indexer**: HF → rotate-half,
vLLM → interleaved. MLA is *not* contested (both say interleaved). Only the indexer is,
and picking wrong reintroduces the silent scramble — which is why it must be resolved
against the real weights before locking it. (This is open-question #5 in the design spec.)

---

## 6. The whole thing in one breath

- **Attention:** each token scores past tokens via query·key dot products, softmaxes,
  then blends their values.
- **RoPE:** inject position by rotating Q/K in 2D pairs; **interleaved vs rotate-half** =
  whether *adjacent* dims pair or *first-half/second-half* dims pair. Wrong choice =
  silent garbage.
- **MLA:** GLM-5.2's main attention; compresses the KV cache and splits Q/K into a 192-dim
  content "nope" part + a 64-dim "rope" slice. That slice is **interleaved** (HF and vLLM
  agree).
- **DSA:** make attention sparse — attend to only the top-2048 relevant past tokens.
- **Indexer:** the cheap "bouncer" that selects those 2048. A separate mini-attention with
  its own weights, hence its *own* RoPE — **rotate-half** in HF — and that indexer RoPE is
  the only thing HF and vLLM dispute for the real checkpoint.
