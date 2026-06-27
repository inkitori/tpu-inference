"""Isolated per-step timing: cached draft forward vs full recompute (inc 3).

Proves the lever: draft_forward_cached attends over [cached ctx K/V | noise K/V]
so its per-step cost is O(B) -- FLAT in context length C -- whereas the full
draft_forward recomputes fc + all-layer ctx K/V every step, cost O(C). Times both
at C = 512, 1024, 2048, 4096 (N=8, B=block_size) on the REAL draft model.

Run:
  HF_HOME=/home/enyouki/local_hf \
  ~/tpu-tooling/tpu-env.sh python scratchpad/bench_dflash_step.py
"""

import os
import time

os.environ.setdefault("HF_HOME", "/home/enyouki/local_hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import jax
import jax.numpy as jnp
import numpy as np

DRAFT_MODEL_PATH = "z-lab/gpt-oss-20b-DFlash"


def _mesh():
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    return jax.sharding.Mesh(devices, axis_names=("data", "model"))


def main():
    from tpu_inference.layers.common.sharding import ShardingAxisName  # noqa
    from tpu_inference.models.vllm.dflash import DFlashTorchaxWrapper

    mesh = _mesh()
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

        draft_forward = wrapper.get_draft_forward_fn()
        draft_forward_cached = wrapper.get_draft_forward_cached_fn()

        N = 8
        params = wrapper.params
        embed_w = fake
        Cs = [512, 1024, 2048, 4096]
        buf_len = max(Cs) + 8

        # Persistent buffers sized to the largest C.
        ctx_buf = jnp.zeros((N, buf_len, raw_dim), dtype=jnp.bfloat16)
        k_cache = jnp.zeros((L, N, buf_len, KVH, HD), dtype=jnp.bfloat16)
        v_cache = jnp.zeros((L, N, buf_len, KVH, HD), dtype=jnp.bfloat16)
        noise_ids = jnp.zeros((N, B), dtype=jnp.int32)

        def bench(fn, c, n_iter=30):
            pad = c
            position_ids = jnp.zeros((N, pad + B), dtype=jnp.int32)
            attn = jnp.zeros((N, pad + B), dtype=jnp.bfloat16)
            if fn == "full":
                call = lambda: draft_forward(params, noise_ids, ctx_buf,
                                             position_ids, embed_w, attn, N, pad)
            else:
                call = lambda: draft_forward_cached(params, noise_ids, k_cache,
                                                    v_cache, position_ids,
                                                    embed_w, attn, N, pad)
            # warmup (cold XLA compile)
            jax.block_until_ready(call())
            t0 = time.perf_counter()
            for _ in range(n_iter):
                r = call()
            jax.block_until_ready(r)
            return (time.perf_counter() - t0) / n_iter * 1e3  # ms

        print(f"\nDFlash per-step timing (N={N}, B={B}, L={L})  ms/step")
        print(f"{'C':>6} | {'full (O(C))':>12} | {'cached (O(B))':>13} | "
              f"{'speedup':>8}")
        print("-" * 50)
        for c in Cs:
            t_full = bench("full", c)
            t_cached = bench("cached", c)
            print(f"{c:>6} | {t_full:>12.3f} | {t_cached:>13.3f} | "
                  f"{t_full / t_cached:>7.2f}x")


if __name__ == "__main__":
    main()
