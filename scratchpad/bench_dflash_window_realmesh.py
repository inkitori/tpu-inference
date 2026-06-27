"""REAL 8-chip mesh DFlash per-step timing: FULL vs CACHED vs WINDOWED, N=32.

Lever A (windowed draft attention) decisive isolated timing. Extends
bench_dflash_step_realmesh.py for the new draft_forward_cached signature (adds
win_start (N,) before the static N/padded_ctx) and measures the windowed cached
forward at W in {256, 512} against the FULL-context cached forward, on the real
TP8 mesh (data=1 so the N=32 batch axis is replicated, exactly like serve at
DP=1). bf16.

We time, per "context" C (the accumulated context length the step would see):
  1. FULL recompute (flag off): draft_forward over full [ctx|noise].
  2. CACHED full-context: draft_forward_cached with win_start=0, padded_ctx=C
     (the Lever-A-OFF baseline -- the O(C*B) score matmul).
  3. CACHED windowed W=256/512: draft_forward_cached with win_start=max(0,C-W),
     padded_ctx=W (the O(W*B) score matmul; the WHOLE point of Lever A).
  4. cached windowed STEP total: kv_project(new B rows) + cache write +
     windowed forward + sample (the steady-state decode step under Lever A).

The window gather is INSIDE the jit (vmap dynamic_slice over the buf_len axis),
so the timing reflects the real per-step cost including the gather.

Run:
  HF_HOME=/home/enyouki/local_hf \
  ~/tpu-tooling/tpu-env.sh python scratchpad/bench_dflash_window_realmesh.py
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

MESH_AXIS_NAMES = ("data", "attn_dp", "attn_dp_expert", "expert", "model",
                   "dcp")


def _real_mesh():
    devs = jax.devices()
    n = len(devs)
    assert n == 8, f"expected 8 chips, got {n}"
    shape = (1, 1, 1, 1, 8, 1)
    devices = np.array(devs).reshape(shape)
    return jax.sharding.Mesh(devices, axis_names=MESH_AXIS_NAMES)


def _median_ms(call, n_iter=30):
    jax.block_until_ready(call())  # discard cold compile
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
        Cs = [1024, 2048, 4096, 4608]
        Ws = [256, 512]
        buf_len = max(Cs) + 8

        print(f"config: N={N} B={B} L={L} KVH={KVH} HD={HD} raw_dim={raw_dim} "
              f"hidden={hidden} num_spec={num_spec}  buf_len={buf_len}")

        ctx_buf = jnp.zeros((N, buf_len, raw_dim), dtype=jnp.bfloat16)
        k_cache = jnp.zeros((L, N, buf_len, KVH, HD), dtype=jnp.bfloat16)
        v_cache = jnp.zeros((L, N, buf_len, KVH, HD), dtype=jnp.bfloat16)
        noise_ids = jnp.zeros((N, B), dtype=jnp.int32)

        M_new = B
        raw_new = jnp.zeros((N, M_new, raw_dim), dtype=jnp.bfloat16)
        zeros_ws = jnp.zeros((N,), dtype=jnp.int32)

        def make_inputs(width):
            position_ids = jnp.zeros((N, width + B), dtype=jnp.int32)
            attn = jnp.zeros((N, width + B), dtype=jnp.bfloat16)
            new_pos = jnp.zeros((N, M_new), dtype=jnp.int32)
            return position_ids, attn, new_pos

        # windowed cached STEP total (project + write + windowed fwd + sample)
        def cached_windowed_step(width, win_start, position_ids, attn, new_pos):
            k_proj, v_proj = kv_project(params, raw_new, new_pos)
            off = max(0, buf_len - M_new - 1)
            k_c = k_cache.at[:, :, off:off + M_new].set(k_proj)
            v_c = v_cache.at[:, :, off:off + M_new].set(v_proj)
            h = draft_forward_cached(params, noise_ids, k_c, v_c, position_ids,
                                     embed_w, attn, win_start, N, width)
            dh = h[:, 1:1 + num_spec]
            logits = compute_logits(params, dh, lm_head_w)
            return jnp.argmax(logits, axis=-1)

        rows = []
        print(f"\nDFlash forward timing (REAL 8-chip mesh, N={N})  ms")
        hdr = (f"{'C':>6} | {'FULL':>9} | {'CACHED full':>11} | "
               + " | ".join(f"CACHED W={w}".rjust(12) for w in Ws))
        print(hdr)
        print("-" * len(hdr))
        for c in Cs:
            pos_full, attn_full, _ = make_inputs(c)
            t_full = _median_ms(lambda: draft_forward(
                params, noise_ids, ctx_buf, pos_full, embed_w, attn_full, N, c))
            # CACHED full-context: win_start=0, width=C
            t_cfull = _median_ms(lambda: draft_forward_cached(
                params, noise_ids, k_cache, v_cache, pos_full, embed_w,
                attn_full, zeros_ws, N, c))
            wvals = {}
            for w in Ws:
                pos_w, attn_w, _ = make_inputs(w)
                # window covers last W of a C-long context.
                ws = jnp.full((N,), max(0, c - w), dtype=jnp.int32)
                t_w = _median_ms(lambda pw=pos_w, aw=attn_w, wsv=ws, ww=w:
                                 draft_forward_cached(
                                     params, noise_ids, k_cache, v_cache, pw,
                                     embed_w, aw, wsv, N, ww))
                wvals[w] = t_w
            print(f"{c:>6} | {t_full:>9.3f} | {t_cfull:>11.3f} | "
                  + " | ".join(f"{wvals[w]:>12.3f}" for w in Ws))
            rows.append((c, t_full, t_cfull, wvals))

        # full windowed STEP total at the highest C for each W (the serve step).
        print(f"\nwindowed cached STEP total (project+write+fwd+sample) @ "
              f"C={Cs[-1]}:")
        c = Cs[-1]
        step_tot = {}
        for w in Ws:
            pos_w, attn_w, new_pos = make_inputs(w)
            ws = jnp.full((N,), max(0, c - w), dtype=jnp.int32)
            t_step = _median_ms(lambda pw=pos_w, aw=attn_w, wsv=ws, np_=new_pos,
                                ww=w: cached_windowed_step(ww, wsv, pw, aw, np_))
            step_tot[w] = t_step
            print(f"  W={w}: windowed step total = {t_step:>7.3f} ms")

        # component breakdown at highest C.
        pos_w, attn_w, new_pos = make_inputs(Ws[0])
        t_proj = _median_ms(lambda: kv_project(params, raw_new, new_pos))
        print(f"\ncomponent @ C={c}:  kv_project(B={M_new}) = {t_proj:.3f} ms")

        # ---- combined throughput projection vs B = 3752 tok/s ----
        print("\n" + "=" * 64)
        print("PROJECTION vs B = 3752 tok/s  (model: step = FIXED + step_device)")
        print("=" * 64)
        accept = 6.0
        # Lever B cut the fixed host overhead. Feasibility put it at ~40ms today;
        # dropping the redundant _ctx_buf write (device ~6.5ms + a redundant RMW)
        # + one fewer device_get + skipped warms moves it toward the 15-25ms band.
        # Report the projection across a band of plausible FIXED values.
        FIXED_BAND = [8.0, 15.0, 25.0, 40.0]  # ms
        for w in Ws:
            dev = step_tot[w]  # device step total for window W @ high C
            print(f"\n  Window W={w}  (device step total @ C={c} = {dev:.2f}ms):")
            for fx in FIXED_BAND:
                step_ms = fx + dev
                tput = accept * N / (step_ms / 1e3)
                ratio = tput / 3752.0
                tag = ("BEATS B" if ratio >= 1.0 else "loses")
                print(f"    FIXED={fx:>5.1f}ms -> step={step_ms:6.2f}ms -> "
                      f"{tput:7.0f} tok/s = {ratio:.2f}x B  [{tag}]")
        print("\nNOTE: device step total already INCLUDES the windowed fwd + "
              "kv_project + write + sample. FIXED is the residual host overhead "
              "AFTER Lever B. accept=6 assumed; rerun with measured accept.")


if __name__ == "__main__":
    main()
