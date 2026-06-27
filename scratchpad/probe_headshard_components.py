"""Pinpoint WHERE the 84ms CACHED-full forward goes + whether attn is sharded.

Reproduces the per-layer math of _draft_forward_cached standalone so we can:
  - print the actual jax sharding of q (post-proj) and attn_weights (scores)
  - time attention-only (q@k^T + softmax + scores@v + o_proj) head-sharded
    vs replicated, isolated from MLP/norms.
At C=4096, N=32, B=8, nh=64, kvh=8, hd=64, L=8.

Run:
  HF_HOME=/home/enyouki/local_hf \
  ~/tpu-tooling/tpu-env.sh python scratchpad/probe_headshard_components.py
"""
import os
import statistics
import time

os.environ.setdefault("HF_HOME", "/home/enyouki/local_hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

MESH_AXIS_NAMES = ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp")


def _real_mesh():
    devs = jax.devices()
    assert len(devs) == 8
    return jax.sharding.Mesh(np.array(devs).reshape((1, 1, 1, 1, 8, 1)),
                             axis_names=MESH_AXIS_NAMES)


def _median_ms(call, n_iter=30):
    jax.block_until_ready(call())
    s = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        jax.block_until_ready(call())
        s.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(s)


def main():
    mesh = _real_mesh()
    N, B, nh, kvh, hd = 32, 8, 64, 8, 64
    C = 4096
    S = C + B
    D = 2880
    L = 8
    n_rep = nh // kvh
    scaling = hd ** -0.5
    H = "model"

    with jax.set_mesh(mesh):
        rep = NamedSharding(mesh, PartitionSpec())
        # q head-sharded: (N, nh, B, hd) shard nh on model
        q_hs = NamedSharding(mesh, PartitionSpec(None, H, None, None))
        # k/v head-sharded on KVH: (N, kvh, S, hd) shard kvh on model
        kv_hs = NamedSharding(mesh, PartitionSpec(None, H, None, None))
        # o_proj weight (D, nh*hd) shard contraction on model
        o_hs = NamedSharding(mesh, PartitionSpec(None, H))

        key = jax.random.PRNGKey(0)
        q = jax.random.normal(key, (N, nh, B, hd), dtype=jnp.bfloat16) * 0.1
        k = jax.random.normal(jax.random.PRNGKey(1), (N, kvh, S, hd),
                              dtype=jnp.bfloat16) * 0.1
        v = jax.random.normal(jax.random.PRNGKey(2), (N, kvh, S, hd),
                              dtype=jnp.bfloat16) * 0.1
        mask = jnp.zeros((N, 1, 1, S), dtype=jnp.bfloat16)
        o_w = jax.random.normal(jax.random.PRNGKey(3), (D, nh * hd),
                                dtype=jnp.bfloat16) * 0.02

        def repeat_kv(x):
            b, kvh_, s, hd_ = x.shape
            x = jnp.broadcast_to(x[:, :, None, :, :], (b, kvh_, n_rep, s, hd_))
            return x.reshape(b, kvh_ * n_rep, s, hd_)

        def attn_layer(q, k, v, mask, o_w):
            ks = repeat_kv(k)
            vs = repeat_kv(v)
            aw = jnp.matmul(q, jnp.swapaxes(ks, 2, 3)) * scaling  # (N,nh,B,S)
            aw = aw + mask
            aw = jax.nn.softmax(aw.astype(jnp.float32), axis=-1).astype(
                jnp.bfloat16)
            out = jnp.matmul(aw, vs)  # (N, nh, B, hd)
            out = jnp.swapaxes(out, 1, 2).reshape(N, B, nh * hd)
            out = jnp.matmul(out, o_w.T)  # (N, B, D)
            return out, aw

        def stack8(q, k, v, mask, o_w):
            acc = jnp.zeros((N, B, D), dtype=jnp.bfloat16)
            for _ in range(L):
                out, _ = attn_layer(q, k, v, mask, o_w)
                acc = acc + out
            return acc

        # --- print intermediate sharding (head-sharded inputs) ---
        qh = jax.device_put(q, q_hs)
        kh = jax.device_put(k, kv_hs)
        vh = jax.device_put(v, kv_hs)
        owh = jax.device_put(o_w, o_hs)

        @jax.jit
        def probe(q, k, v, mask, o_w):
            ks = repeat_kv(k)
            aw = jnp.matmul(q, jnp.swapaxes(ks, 2, 3)) * scaling
            aw = jax.nn.softmax((aw + mask).astype(jnp.float32),
                                axis=-1).astype(jnp.bfloat16)
            return aw

        aw = probe(qh, kh, vh, mask, owh)
        print(f"q (head-sharded) sharding:        {qh.sharding.spec}")
        print(f"attn_weights (scores) sharding:   {aw.sharding.spec}")
        print(f"attn_weights shape:               {aw.shape}  "
              f"dtype {aw.dtype}")

        # --- time attention-only stack8: head-sharded vs replicated ---
        f_hs = jax.jit(stack8,
                       out_shardings=NamedSharding(mesh,
                                                   PartitionSpec(None, None,
                                                                 None)))
        qr = jax.device_put(q, rep)
        kr = jax.device_put(k, rep)
        vr = jax.device_put(v, rep)
        owr = jax.device_put(o_w, rep)
        f_rep = jax.jit(stack8,
                        out_shardings=NamedSharding(mesh,
                                                    PartitionSpec(None, None,
                                                                  None)))

        t_hs = _median_ms(lambda: f_hs(qh, kh, vh, mask, owh))
        t_rep = _median_ms(lambda: f_rep(qr, kr, vr, mask, owr))
        print(f"\nattention-only x{L} layers @ C={C}:")
        print(f"  head-sharded = {t_hs:7.3f} ms")
        print(f"  replicated   = {t_rep:7.3f} ms")
        print(f"  speedup      = {t_rep / t_hs:.2f}x")


if __name__ == "__main__":
    main()
