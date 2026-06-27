"""REAL 8-chip mesh DFlash per-step timing: FULL recompute vs CACHED, N=32.

The decisive timing for whether the DFlash KV-cache lever flips A(DFlash)>B
(target-only) at concurrency 32. Mirrors the serve config's draft sharding:
TP8 (all 8 chips on MLP_TENSOR, data=1 so the N=32 batch axis is replicated,
exactly as the draft forward does at DP=1). bf16.

Per DECODE step we time:
  1. FULL path (flag off): draft_forward  (O(ctx) recompute: fc(14400->2880)
     + 8 attn layers over the full [ctx|noise]).
  2. CACHED path (flag on):
       - total: kv_project(new B rows) + _batched_kv_write + draft_forward_cached
                + sample  (the whole steady-state decode step).
       - forward-only: just draft_forward_cached (the dominant cached cost).
  3. target-only step proxy: compute_logits over the N draft hidden rows
     (cheap; the real target decode step is ~8.4ms/step per 08-impl-perf).

At C = 512, 1024, 2048, 4096, 4608. Each jitted shape is warmed (cold compile
discarded) then timed as the median of several warm calls.

Run:
  HF_HOME=/home/enyouki/local_hf \
  ~/tpu-tooling/tpu-env.sh python scratchpad/bench_dflash_step_realmesh.py
"""

import os
import statistics
import time

os.environ.setdefault("HF_HOME", "/home/enyouki/local_hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import jax
import jax.numpy as jnp
import numpy as np

DRAFT_MODEL_PATH = "z-lab/gpt-oss-20b-DFlash"

# Real serve config: TP8. The draft's MLP_TENSOR axis tuple is
# ('attn_dp','attn_dp_expert','expert','model','dcp'); its product is the
# effective tensor-parallel degree for the draft projections. Putting all 8
# chips on 'model' shards fc/k_proj/v_proj/q_proj 8-ways exactly as serve does,
# while 'data' (= MLP_DATA, the N batch axis) stays size 1 -> replicated, which
# is precisely the DP=1 behaviour the draft forward assumes.
MESH_AXIS_NAMES = ("data", "attn_dp", "attn_dp_expert", "expert", "model",
                   "dcp")


def _real_mesh():
    devs = jax.devices()
    n = len(devs)
    assert n == 8, f"expected 8 chips, got {n}"
    # data=1, model=8, rest=1
    shape = (1, 1, 1, 1, 8, 1)
    devices = np.array(devs).reshape(shape)
    return jax.sharding.Mesh(devices, axis_names=MESH_AXIS_NAMES)


def _median_ms(call, n_iter=30):
    # warmup (cold XLA compile) discarded
    jax.block_until_ready(call())
    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        r = call()
        jax.block_until_ready(r)
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def main():
    from tpu_inference.layers.common.sharding import ShardingAxisName  # noqa
    from tpu_inference.models.vllm.dflash import DFlashTorchaxWrapper

    mesh = _real_mesh()
    print(f"mesh: {mesh}")
    with jax.set_mesh(mesh):
        wrapper = DFlashTorchaxWrapper(mesh)
        import transformers
        cfg0 = transformers.AutoConfig.from_pretrained(DRAFT_MODEL_PATH,
                                                       trust_remote_code=True)
        vocab = getattr(cfg0, "vocab_size", 201088)
        hidden = cfg0.hidden_size
        fake = jnp.zeros((vocab, hidden), dtype=jnp.bfloat16)
        wrapper.load(DRAFT_MODEL_PATH, {
            "vllm_model.model.embedding.weight": fake,
            "vllm_model.lm_head.weight": fake,
        })

        cfg = wrapper.model.dflash.config
        L = cfg.num_hidden_layers
        KVH = cfg.num_key_value_heads
        HD = getattr(cfg, "head_dim", hidden // cfg.num_attention_heads)
        B = cfg.block_size
        raw_dim = wrapper.model.dflash.fc.in_features
        num_spec = 7 if B - 1 >= 7 else B - 1

        draft_forward = wrapper.get_draft_forward_fn()
        draft_forward_cached = wrapper.get_draft_forward_cached_fn()
        kv_project = wrapper.get_kv_project_fn()
        compute_logits = wrapper.get_compute_logits_fn()

        N = 32
        params = wrapper.params
        embed_w = fake
        lm_head_w = fake
        Cs = [512, 1024, 2048, 4096, 4608]
        buf_len = max(Cs) + 8

        print(f"config: N={N} B={B} L={L} KVH={KVH} HD={HD} raw_dim={raw_dim} "
              f"hidden={hidden} num_spec={num_spec}")

        # Persistent buffers sized to the largest C (built once, like serve).
        ctx_buf = jnp.zeros((N, buf_len, raw_dim), dtype=jnp.bfloat16)
        k_cache = jnp.zeros((L, N, buf_len, KVH, HD), dtype=jnp.bfloat16)
        v_cache = jnp.zeros((L, N, buf_len, KVH, HD), dtype=jnp.bfloat16)
        noise_ids = jnp.zeros((N, B), dtype=jnp.int32)

        # New rows projected each step in steady-state decode: ~B accepted rows.
        # (kv_project's M is dynamic; we project B fresh rows per step.)
        M_new = B
        raw_new = jnp.zeros((N, M_new, raw_dim), dtype=jnp.bfloat16)

        def make_inputs(c):
            position_ids = jnp.zeros((N, c + B), dtype=jnp.int32)
            attn = jnp.zeros((N, c + B), dtype=jnp.bfloat16)
            new_pos = jnp.zeros((N, M_new), dtype=jnp.int32)
            return position_ids, attn, new_pos

        # ---- the cached steady-state decode step (project + write + fwd + sample) ----
        def cached_step(c, position_ids, attn, new_pos):
            # 1. project K/V for the B new rows.
            k_proj, v_proj = kv_project(params, raw_new, new_pos)
            # 2. batched write into the per-slot cache (the _batched_kv_write
            #    analogue: scatter the L,N,M new rows into the cache at the
            #    write offset). Use a fixed offset (c - M_new) for timing.
            off = max(0, c - M_new)
            k_c = k_cache.at[:, :, off:off + M_new].set(k_proj)
            v_c = v_cache.at[:, :, off:off + M_new].set(v_proj)
            # 3. cache-consuming forward.
            h = draft_forward_cached(params, noise_ids, k_c, v_c, position_ids,
                                     embed_w, attn, N, c)
            # 4. sample draft tokens (rows 1..1+num_spec through lm_head argmax).
            dh = h[:, 1:1 + num_spec]
            logits = compute_logits(params, dh, lm_head_w)
            return jnp.argmax(logits, axis=-1)

        # target-only step proxy: just the lm_head matmul over N rows (the per
        # step "verify" matmul is what we can isolate here). NOTE: real target
        # decode step is ~8.4ms/step (08-impl-perf); this proxy is a lower bound,
        # we report the 08-impl-perf number as the break-even reference.
        target_hidden_proxy = jnp.zeros((N, num_spec, hidden),
                                        dtype=jnp.bfloat16)

        def target_proxy():
            logits = compute_logits(params, target_hidden_proxy, lm_head_w)
            return jnp.argmax(logits, axis=-1)

        rows = []
        print(f"\nDFlash per-step timing (REAL 8-chip mesh, N={N})  ms/step")
        hdr = (f"{'C':>6} | {'FULL':>9} | {'CACHED tot':>11} | "
               f"{'CACHED fwd':>11} | {'speedup(F/Cfwd)':>15} | "
               f"{'speedup(F/Ctot)':>15}")
        print(hdr)
        print("-" * len(hdr))
        for c in Cs:
            position_ids, attn, new_pos = make_inputs(c)

            t_full = _median_ms(lambda: draft_forward(
                params, noise_ids, ctx_buf, position_ids, embed_w, attn, N, c))
            t_cfwd = _median_ms(lambda: draft_forward_cached(
                params, noise_ids, k_cache, v_cache, position_ids, embed_w,
                attn, N, c))
            t_ctot = _median_ms(
                lambda: cached_step(c, position_ids, attn, new_pos))

            sp_fwd = t_full / t_cfwd
            sp_tot = t_full / t_ctot
            print(f"{c:>6} | {t_full:>9.3f} | {t_ctot:>11.3f} | "
                  f"{t_cfwd:>11.3f} | {sp_fwd:>14.2f}x | {sp_tot:>14.2f}x")
            rows.append((c, t_full, t_ctot, t_cfwd, sp_fwd, sp_tot))

        # isolated component timings at the largest C, to see what dominates the
        # cached step (project vs write vs forward vs sample).
        c = Cs[-1]
        position_ids, attn, new_pos = make_inputs(c)
        t_proj = _median_ms(lambda: kv_project(params, raw_new, new_pos))
        off = max(0, c - M_new)

        def write_only():
            k_proj, v_proj = kv_project(params, raw_new, new_pos)
            return (k_cache.at[:, :, off:off + M_new].set(k_proj),
                    v_cache.at[:, :, off:off + M_new].set(v_proj))

        t_projwrite = _median_ms(write_only)
        t_tgt = _median_ms(target_proxy)
        print(f"\ncomponent breakdown @ C={c}:")
        print(f"  kv_project (B={M_new} rows)      : {t_proj:>8.3f} ms")
        print(f"  kv_project + cache write         : {t_projwrite:>8.3f} ms")
        print(f"  draft_forward_cached only        : {rows[-1][3]:>8.3f} ms")
        print(f"  cached step total                : {rows[-1][2]:>8.3f} ms")
        print(f"  target-only proxy (lm_head only) : {t_tgt:>8.3f} ms "
              f"(real target step ~8.4ms per 08-impl-perf)")

        # ---- flip projection ----
        print("\n" + "=" * 60)
        print("FLIP PROJECTION (N=32)")
        print("=" * 60)
        B_TPOT = 0.0084  # target-only per-token-step, s (bench-v3)
        accept = 6       # ~accepted tokens per step
        # effective per-accepted-token cost of the cached DFlash step.
        for (c, t_full, t_ctot, t_cfwd, _, _) in rows:
            per_tok_tot = (t_ctot / 1e3) / accept
            per_tok_full = (t_full / 1e3) / accept
            print(f"  C={c:>4}: FULL/6={per_tok_full*1e3:7.2f}ms  "
                  f"CACHEDtot/6={per_tok_tot*1e3:7.2f}ms  "
                  f"(target/tok={B_TPOT*1e3:.2f}ms)")
        # average step over a realistic generation: context grows monotonically
        # 0->~4224, so most steps sit at LARGE C. Use C=4096 row as the
        # representative high-C average.
        rep = [r for r in rows if r[0] == 4096][0]
        rep_per_tok = (rep[2] / 1e3) / accept
        verdict = ("LIKELY FLIPS" if rep_per_tok < B_TPOT else
                   ("MARGINAL" if rep_per_tok < 1.5 * B_TPOT else
                    "LIKELY STILL SLOWER"))
        print(f"\n  representative high-C (C=4096): cached_step/6 = "
              f"{rep_per_tok*1e3:.2f}ms  vs  target {B_TPOT*1e3:.2f}ms")
        print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
