# GLM 5.2 (DSA) — provenance & rationale

**Not needed to implement any phase.** This is the "why" behind the decisions in `core.md` / `phases/`,
plus the hard-won lessons from the DeepSeek-V4 bring-up that shaped the plan. Read it when a decision looks
arbitrary and you want the evidence; skip it when you're writing code.

---

## Why the plan is shaped the way it is

- **Develop on the real v6e from the start (not CPU).** The prior **DeepSeek-V4** bring-up
  (`claude-deepseek-v4`) was dominated by **multi-device, real-reshard-path bugs CPU structurally cannot
  reproduce** — chiefly **uninitialized-HBM-on-reshard ("S1")**: a device-side weight reshard read uninit
  HBM and baked per-process garbage into the weights, producing **coherent-looking-but-wrong** output at
  temp=0. It took ~30 sessions, with the root-cause hypothesis overturned repeatedly, and was cracked in one
  session only once a per-weight/per-stage checksum probe (`[ckSPLIT]`, fork `c98430d8`) was built — which
  is why the localization procedure (core §H6), not deductive bisection, is mandated. The S1 fix was
  host-stacking weights into the sharded layout (fork `5a3ed435`); `gmm_v2(zero_initialize=True)` was tried
  and **disproven** (fork `65bc1858`/`a0eb101b`).

- **Decode from the start.** V4 validated prefill, shipped, and decode collapsed at step 1 (*"prefill is
  COHERENT; decode collapses"*, fork `e6788e7f`). The KV-write index `[kv_len−q_len:kv_len]` is computed
  in-kernel from `seq_lens`, so in prefill `kv_len−q_len==0` and that arithmetic + the decode mask are
  **never exercised by a forward pass**. A prefill-only gate is structurally blind to it — hence Phase 1c
  (core §H13 guardrail 4). XLA write-elision under `donate_argnums` cost the fork ~6 sessions (→ the
  donated-buffer R1==R2 fixture).

- **Single-host suffices for S1.** The V4 fork isolated S1 with a replicated-vs-sharded experiment on a
  *fixed* mesh, and only ever ran multi-host because its pod couldn't boot smaller. S1 is therefore a
  *sharding/reshard* phenomenon, not a host-count one — a single-host v6e-8 reproduces it (evidence:
  `../research/2026-06-20-s1-single-host-reproducibility.md`). **Caveat carried forward:** genuinely
  multi-host-only modes (cross-host collective ordering, slice-builder/transfer races) were **never
  separated out** in the fork — they are *unvalidated*, not *real-weight-gated*. The fork **reproduced its
  worst multi-device bug on random weights** on the multi-host slice (`s1_mh_repro.py`, "random weights, no
  543 GiB load"), and every genuinely multi-host bug it hit (asymmetric-load wedge `ca016156`; RoPE-precompute
  launch-id core-halt `fb54237b`; KV/`device_put` races `7669894d`/`8b42d7c1`; Ray init `a5d6512f`/`5c13bc4c`)
  was **weight-value-independent**. ⇒ **Phase Mh** validates multi-host orchestration on dummy weights before
  Phase R, so Phase R first-contacts neither.

- **The v2 production-readiness re-scope.** The original finish line was single-sequence *prefill*
  correctness; production (decode, batching, fp8) was deferred past a real-checkpoint load. A
  production-readiness audit (9 adversarial agents over the spec + repo + fork + vLLM,
  `../research/2026-06-20-production-readiness-adversarial-audit.md`) showed this **repeats the V4 failure**
  and **mis-sequences** the giant model ahead of work that needs only small/medium configs. A second
  adversarial pass (2026-06-20) added: the **EP-mode reshard** gap (production's real sharding mode was never
  armed pre-R), the **fp8 UE8M0 first-contact** (the on-disk scale decode wasn't wired into the fp8 path),
  the **DSA-selection oracle-approximation** hole (the HF oracle skips the real Hadamard+fp8 indexer), the
  **multi-host pile-up** (→ Phase Mh), the **1M-context first-contact** (kernel compile + RoPE table), and
  the serving-surface gaps (abort, observability, warmup primers). Those drive the fixes now folded into the
  phase work orders.

## Architecture-fact provenance

- All `core.md` §A–§F facts were derived from real source (transformers `glm_moe_dsa`, vLLM `deepseek_v2.py`,
  the `claude-deepseek-v4` fork) and **re-verified against transformers 5.12.1** in a py3.12 venv. They are
  device-agnostic and unchanged by the TPU-first strategy.
- The indexer-cache layer-set (core §B8) **changed from transformers 5.9.0**, which built the indexer on
  every layer (keys cached on all 78; "shared" was a forward-time `skip_topk` only). The all-78 / 19.5 GB
  figure applies **only** under the default `index_topk_freq=1`; 5.12.1 nulls the indexer on "shared" layers,
  so the real config caches on 21 of 78 → ~5.25 GB.
- **Config availability:** the real `zai-org/GLM-5.2` `config.json` is the basis for the real-checkpoint
  overrides in core §B2 (rope_type=default → no YaRN; `index_topk_freq=4`/`offset=3`;
  `indexer_rope_interleave=true`; `head_dim=192`; `max_position_embeddings=1048576`). The **weights** are not
  available in the dev environment (too large). ⚠️ **Verification-integrity note:** these config values
  should be re-confirmed against the actual artifact and a **checksummed copy committed in-repo**, rather
  than relying on an uncached `~/.cache/huggingface/hub` entry — the values cannot be re-verified from a box
  that lacks the cache.

## Resolved questions (closed; recorded so they aren't re-opened)

- **YaRN `mscale²`** — the real `config.json` uses `rope_type="default"` → YaRN never enabled, the mscale
  branch never taken (and HF 5.12.1 == vLLM on the formula anyway). Not a blocker. (core §B3, §E6)
- **Dense-schedule key** — both keys coexist; `first_k_dense_replace` (default 3) is the HF field that
  generates `mlp_layer_types` in `__post_init__`. Read it directly. (core §B4)
- **Indexer-cache layer-set** — "full" layers only, 21 of 78, driven by `indexer_types`; matches vLLM. (core §B8)
- **vLLM fp8 indexer ReLU location** — relu **is** applied (HF + fork verified at `:247`; vLLM Triton ref
  `triton_fp8_mqa_logits.py:128-129,149-150`, `scores = tl.maximum(scores, 0.0)`); only the compiled CUDA
  DeepGEMM path is opaque. Final confirm against the real vLLM-GPU path in Phase R.

## Open questions deferred to Phase R (real weights required)

- **Indexer RoPE convention on the real checkpoint** — `indexer_rope_interleave=true`; HF runs rotate-half,
  vLLM interleaved. Only trained `wq_b`/`wk` adjudicate. (core §D2, phase-R)
- **Real top-k selection fidelity + the Hadamard/fp8 oracle gap** — random near-ties hide selection bugs in
  dev; the HF oracle skips the real Hadamard transform + fp8 scoring, so the real selected set may differ
  from the HF-eager set even when our impl is correct. (core §H10, §H11a, phase-R)

## Superseded / dropped (recoverable from git)

- The single-file `2026-06-19-glm5.2-dsa-jax-tpu-design.md` (this directory replaces it).
- An earlier wrong-class draft (`Glm4MoeLiteForCausalLM` = GLM-4.7-Flash, MLA-only / no DSA) and a CPU-first
  plan — both recoverable from git.
- The v1→v2 renumbering scaffolding (`(was 4)` etc.) and the "(v2 ADD)" tags — the phase files simply *are*
  the work now.
