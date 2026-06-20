# S1 (uninit-HBM-on-reshard) — does a single-host multi-chip slice reproduce it?

**Date:** 2026-06-20
**Context:** Deciding the GLM 5.2 v1 TPU test surface. Question: does a single-host
v6e-4 / v6e-8 (one process, 4–8 local chips) reproduce the S1 bug class, given the
DeepSeek-V4 fork only ever observed it on an 8-host / 32-chip slice?
**Source investigated:** `/home/enyouki/claude-deepseek-v4` (real checkout under
`work/tpu-inference/tpu_inference/`), git history + handoff/session notes.

---

## Verdict

**S1 requires multi-DEVICE SHARDING, not multiple hosts/processes.** A single-host
multi-chip mesh (one process, real `JaxAutoWeightsLoader` with production-style
attn_dp/expert sharding) exercises the same code path that produced S1. The
"multi-host" framing in the V4 fork is an **environmental accident** — that pod could
only boot as a 32-chip / 8-host slice — not a property of the bug.

---

## The two-part anatomy of S1

Two manifestations of the same uninit-HBM-on-reshard mechanism:

**(a) Forward-path / idle-shard manifestation (Sessions 11–25).** During sharded
forward compute, idle attn_dp ranks (token-axis shards with no real tokens) fed
uninitialized HBM into an implicit collective-matmul. Code comment:
`work/tpu-inference/tpu_inference/layers/jax/moe/deepseek_v4_moe.py:288-293` — *"the
S1-buggy case where idle DP ranks feed uninit HBM into the implicit
collective-matmul."*

**(b) Weight-LOAD reshard manifestation — the confirmed root cause (Sessions 26–28).**
What commit `5a3ed435` fixed. Per `cce33efc` and `HANDOFF_S1.md:9-20`:
- `pick_partition_spec` (`deepseek_v4_loader.py:508`) shards per-expert `w1`/`w3`
  `[2048,4096]` on **axis-1** but `w2` `[4096,2048]` on **axis-0**.
- Consolidation did `device_put(jnp.stack(leaves), P('attn_dp',None,None))` at
  `deepseek_v4.py:1535` — a **device-side reshard** of the axis-1 leaves into the
  expert-sharded layout. On TPU that reshard path **read uninitialized HBM**, baking
  garbage into `w1_stacked`/`w3_stacked`. `w2` (axis-0) went through a clean path and
  was byte-identical.
- **The fix** (`5a3ed435`): `np.stack` the 256 per-expert host numpys and
  `jax.make_array_from_callback` straight into the sharded layout — host→device
  scatter, **no device reshard, no uninit read.**

## Why the divergence read as "per-process"

The fork measured determinism by running **two engine processes** and comparing output
(the "×2 fresh engines" / FIB-md5 gate). Each process independently re-loaded weights
and read whatever garbage was in *its* HBM, so the baked-in garbage differed per
process → engines diverged at temp=0 (`c6c94230`: "same executable + different output =
RUNTIME per-process nondeterminism = uninit HBM read fresh per process").

**"Per-process" describes the detection method and the symptom, not the mechanism's
requirement.** The uninit read happens **inside each process's own intra-process SPMD
reshard across its local devices.** With one process, a single load still reads uninit
HBM and bakes garbage — you just need a different detector (NaN-poison HBM before load,
or N-device-vs-1-device equality) since you can't diff two processes.

## The decisive topology evidence: it's SHARDING, not host-count

PHASE-2 controlled experiment (`CLAUDE.full.md:619-622`), same 32-chip mesh, varying
only the sharding:
- `replicated` (all `P()`, 32 chips, no reshard) → **NO_S1** (bad=0/12)
- `sharded` (attn_dp=8, experts+attn parallel) → **S1_REPRODUCED**

Same hosts, same processes, same chips — the bug toggled purely on **whether weights
were sharded/resharded.** That isolates the trigger to the reshard, an **intra-process
XLA/SPMD reshard across local devices**; nothing requires a cross-process collective.

The 8-host/32-chip framing was forced by the environment (`CLAUDE.full.md:600-603`):
*"The single-host small-TPU loop the prior runbook assumed DOES NOT EXIST: a lone
worker can't boot a v6e-32 (libtpu waits forever for the other 7 hosts)."* There is **no
statement anywhere of a minimum host count**; the only stated minimum is the *sharding
geometry* (attn_dp>1, an expert/token partition that actually reshards).

## CPU / single-device caveat

CPU and single-device do **not** reproduce S1 — uninitialized HBM is a TPU-device
phenomenon. `CLAUDE.full.md:835-861` (PHASE 6): the CPU "peaked-weights" repro was a
**red herring** (bf16 rounding on random-weight near-ties; fp32 is bit-exact). So:
- Single-device **CPU**: cannot reproduce (no uninit HBM, no reshard).
- Single-host **multi-chip TPU with real sharding**: **does** exercise the buggy reshard
  path. The fork never ran this surface (the pod wouldn't boot small), but the mechanism
  is fully present on a 4-/8-chip single-host sharded mesh.

## The max_num_seqs>1 MoE bug

Same mechanism, forward-path flavor (`HANDOFF_QUANT.md:100-104`,
`deepseek_v4_moe.py:290`): with `max_num_seqs>1`, the token axis becomes sharded across
attn_dp and idle/un-owned token rows feed uninit HBM into the implicit collective-matmul,
silently corrupting concurrent requests. Also a **sharding/idle-shard** phenomenon, not
a host-count one — reproduces on a single-host sharded mesh too. (v1 stays single-seq.)

---

## Bottom line for GLM 5.2 bring-up

**A single-host v6e-4 or v6e-8 (one process, 4–8 local chips, real loader with
production-style attn_dp/expert sharding) exercises the same code path that produced
S1.** The trigger is the intra-process device reshard into a sharded layout, present on
any genuinely-sharded multi-chip mesh regardless of host count. Multi-host adds nothing
the single-host sharded mesh lacks **for this bug class**.

**Residual host-count-specific risk:** none identified for S1 specifically. Honest
caveat: the fork could only ever run 8-host, so genuinely multi-host-only phenomena
(cross-host collective ordering, slice-builder/transfer bugs) were never *separated
out*. S1 itself was cleanly isolated to sharding, so it is **not** in that residual
class — but a real multi-host run (Phase 3) remains where those *other* potential issues
would surface. Single-host is sufficient for **S1**, not a blanket certification of
multi-host distribution.

## Two conditions a single-host gate MUST meet (else it passes vacuously)

1. **Real sharding, not just multi-chip.** A trivially-divisible tiny config passes
   vacuously — use a **sharding-geometry stress fixture** (expert count not divisible by
   mesh → ≥1 empty shard; reshard tiling that bites). Spec: `design.md:181, 247`.
2. **A single-process detector** (can't diff two engines): **NaN-poison HBM before load
   + N-device==1-device fp32 equality**. Spec: `design.md:234, 246`. The host-stack fix
   (`5a3ed435`) is the remedy; `gmm_v2(zero_initialize=True)` was tested and **disproven**
   (`65bc1858`/`a0eb101b`) — do not rely on it.

## Key references
- `HANDOFF_S1.md:9-20` — root cause + fix summary
- commit `cce33efc` — root-cause verdict (loader axis-1 vs axis-0, `deepseek_v4.py:1535` reshard)
- commit `5a3ed435` — the fix (host-stack via `make_array_from_callback`)
- `CLAUDE.full.md:619-622` — replicated→NO_S1 vs sharded→S1 (topology-isolating experiment)
- `CLAUDE.full.md:600-603` — why everything ran 8-host (env, not bug)
- `CLAUDE.full.md:835-861` — CPU repro is a red herring; uninit-HBM is TPU-device-specific
- `layers/jax/moe/deepseek_v4_moe.py:288-293` — idle-DP-rank uninit-HBM
- `HANDOFF_QUANT.md:100-104` — max_num_seqs>1 confirmed corruption, same mechanism
- commits `65bc1858`, `a0eb101b` — gmm zero_initialize disproven
