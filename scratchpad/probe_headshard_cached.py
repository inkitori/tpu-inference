"""Decisive head-shard probe for the DFlash CACHED full forward (real TP8 mesh).

The window bench (bench_dflash_window_realmesh.py) allocates k/v_cache as plain
jnp.zeros => REPLICATED. But serve allocates them HEAD-SHARDED on KVH (axis 3).
This probe times the CACHED-full forward (win_start=0) with:
  (A) REPLICATED caches + head-sharded weights  (what the window bench measured)
  (B) HEAD-SHARDED caches + head-sharded weights (what serve does)  <-- AFTER
  (C) REPLICATED caches + REPLICATED weights     (the true replicated baseline)
and reports a same-session numeric max|diff| of (B) vs (C).

Run:
  HF_HOME=/home/enyouki/local_hf \
  ~/tpu-tooling/tpu-env.sh python scratchpad/probe_headshard_cached.py
"""
import os
import statistics
import time

os.environ.setdefault("HF_HOME", "/home/enyouki/local_hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

DRAFT_MODEL_PATH = "z-lab/gpt-oss-20b-DFlash"
MESH_AXIS_NAMES = ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp")


def _real_mesh():
    devs = jax.devices()
    assert len(devs) == 8, f"expected 8 chips, got {len(devs)}"
    devices = np.array(devs).reshape((1, 1, 1, 1, 8, 1))
    return jax.sharding.Mesh(devices, axis_names=MESH_AXIS_NAMES)


def _median_ms(call, n_iter=30):
    jax.block_until_ready(call())
    s = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        jax.block_until_ready(call())
        s.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(s)


def main():
    from tpu_inference.layers.common.sharding import ShardingAxisName  # noqa
    from tpu_inference.models.vllm.dflash import DFlashTorchaxWrapper

    mesh = _real_mesh()
    print(f"mesh: {mesh}")
    print(f"ShardingAxisName.ATTN_HEAD = {ShardingAxisName.ATTN_HEAD!r}  "
          f"KV_CACHE_HEAD = {ShardingAxisName.KV_CACHE_HEAD!r}")
    with jax.set_mesh(mesh):
        import transformers
        cfg0 = transformers.AutoConfig.from_pretrained(DRAFT_MODEL_PATH,
                                                       trust_remote_code=True)
        vocab = getattr(cfg0, "vocab_size", 201088)
        hidden = cfg0.hidden_size
        fake = jnp.zeros((vocab, hidden), dtype=jnp.bfloat16)

        wrapper = DFlashTorchaxWrapper(mesh)
        wrapper.load(DRAFT_MODEL_PATH, {
            "vllm_model.model.embedding.weight": fake,
            "vllm_model.lm_head.weight": fake,
        })
        cfg = wrapper.model.dflash.config
        L = cfg.num_hidden_layers
        KVH = cfg.num_key_value_heads
        nh = cfg.num_attention_heads
        HD = getattr(cfg, "head_dim", hidden // nh)
        B = cfg.block_size
        params = wrapper.params

        # ---- inspect the actual sharding of a q_proj weight after reshard ----
        qk = next(k for k in params if k.endswith(".self_attn.q_proj.weight"))
        ok = next(k for k in params if k.endswith(".self_attn.o_proj.weight"))
        print(f"\nq_proj.weight {qk}: shape {params[qk].shape}  "
              f"sharding {params[qk].sharding}")
        print(f"o_proj.weight {ok}: shape {params[ok].shape}  "
              f"sharding {params[ok].sharding}")

        draft_forward_cached = wrapper.get_draft_forward_cached_fn()

        N = 32
        C = 4096  # full-context CACHED measurement
        buf_len = C + 8
        num_spec = 7

        noise_ids = jnp.zeros((N, B), dtype=jnp.int32)
        pos_full = jnp.zeros((N, C + B), dtype=jnp.int32)
        attn_full = jnp.zeros((N, C + B), dtype=jnp.bfloat16)
        zeros_ws = jnp.zeros((N,), dtype=jnp.int32)

        # deterministic non-trivial caches for the numeric check.
        key = jax.random.PRNGKey(0)
        k_rep = jax.random.normal(key, (L, N, buf_len, KVH, HD),
                                  dtype=jnp.bfloat16) * 0.1
        v_rep = jax.random.normal(jax.random.PRNGKey(1),
                                  (L, N, buf_len, KVH, HD),
                                  dtype=jnp.bfloat16) * 0.1
        embed_w = jax.random.normal(jax.random.PRNGKey(2), (vocab, hidden),
                                    dtype=jnp.bfloat16) * 0.02

        # cache shardings
        rep_spec = NamedSharding(mesh, PartitionSpec())
        hs_spec = NamedSharding(
            mesh, PartitionSpec(None, None, None,
                                ShardingAxisName.KV_CACHE_HEAD, None))
        k_rep = jax.device_put(k_rep, rep_spec)
        v_rep = jax.device_put(v_rep, rep_spec)
        k_hs = jax.device_put(k_rep, hs_spec)
        v_hs = jax.device_put(v_rep, hs_spec)

        def run(k, v):
            return draft_forward_cached(params, noise_ids, k, v, pos_full,
                                        embed_w, attn_full, zeros_ws, N, C)

        # (A) replicated caches + head-sharded weights (window-bench layout)
        tA = _median_ms(lambda: run(k_rep, v_rep))
        # (B) head-sharded caches + head-sharded weights (serve layout)
        tB = _median_ms(lambda: run(k_hs, v_hs))
        print(f"\nCACHED full @ C={C}  (head-sharded weights):")
        print(f"  (A) REPLICATED caches  = {tA:7.3f} ms")
        print(f"  (B) HEAD-SHARDED caches = {tB:7.3f} ms   <-- serve layout")

        # (C) replicated-weights baseline: force ALL attn weights replicated.
        params_rep = dict(params)
        suffixes = (".self_attn.q_proj.weight", ".self_attn.q_proj.bias",
                    ".self_attn.k_proj.weight", ".self_attn.k_proj.bias",
                    ".self_attn.v_proj.weight", ".self_attn.v_proj.bias",
                    ".self_attn.o_proj.weight", ".self_attn.o_proj.bias")
        for kk in list(params_rep):
            if any(kk.endswith(s) for s in suffixes):
                params_rep[kk] = jax.device_put(params_rep[kk], rep_spec)

        def run_rep(k, v):
            return draft_forward_cached(params_rep, noise_ids, k, v, pos_full,
                                        embed_w, attn_full, zeros_ws, N, C)

        tC = _median_ms(lambda: run_rep(k_rep, v_rep))
        print(f"  (C) REPLICATED weights + caches = {tC:7.3f} ms  (baseline)")

        # ---- numeric equivalence: (B) head-sharded vs (C) replicated ----
        out_hs = np.asarray(jax.device_get(run(k_hs, v_hs)).astype(jnp.float32))
        out_rep = np.asarray(
            jax.device_get(run_rep(k_rep, v_rep)).astype(jnp.float32))
        diff = np.abs(out_hs - out_rep)
        print(f"\nnumeric max|diff| (B head-sharded vs C replicated) = "
              f"{diff.max():.3e}  mean = {diff.mean():.3e}")
        # also greedy-argmax agreement over the spec rows used by sampling
        dh_hs = out_hs[:, 1:1 + num_spec]
        dh_rep = out_rep[:, 1:1 + num_spec]
        # project to logits via lm_head (fake/embed); use embed_w as lm_head
        lm = np.asarray(jax.device_get(embed_w).astype(jnp.float32))
        am_hs = (dh_hs @ lm.T).argmax(-1)
        am_rep = (dh_rep @ lm.T).argmax(-1)
        agree = (am_hs == am_rep).mean()
        print(f"greedy-argmax agreement over spec rows = {agree*100:.2f}%")


if __name__ == "__main__":
    main()
