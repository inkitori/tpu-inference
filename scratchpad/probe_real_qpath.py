"""Trace the REAL q path of _draft_forward_cached to find where the head
sharding is LOST. Mirrors dflash.py lines 236-270 exactly (view/q_norm/
transpose/RoPE/concat/eager_attn/o_proj) with head-sharded weights, printing
the jax PartitionSpec at each step via jax.debug callbacks on a tiny trace.

Run:
  HF_HOME=/home/enyouki/local_hf \
  ~/tpu-tooling/tpu-env.sh python scratchpad/probe_real_qpath.py
"""
import os

os.environ.setdefault("HF_HOME", "/home/enyouki/local_hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

MESH_AXIS_NAMES = ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp")


def _real_mesh():
    devs = jax.devices()
    return jax.sharding.Mesh(np.array(devs).reshape((1, 1, 1, 1, 8, 1)),
                             axis_names=MESH_AXIS_NAMES)


def main():
    mesh = _real_mesh()
    N, B, nh, kvh, hd, D = 32, 8, 64, 8, 64, 2880
    C = 512
    S = C + B
    n_rep = nh // kvh
    H = "model"
    with jax.set_mesh(mesh):
        rep = NamedSharding(mesh, PartitionSpec())
        qw_s = NamedSharding(mesh, PartitionSpec(H, None))   # q/k/v weight [out,in]
        ow_s = NamedSharding(mesh, PartitionSpec(None, H))   # o weight [out,in]

        g = jax.random.PRNGKey(0)
        hs = jax.device_put(
            jax.random.normal(g, (N, B, D), dtype=jnp.bfloat16) * 0.1, rep)
        qw = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(1), (nh * hd, D),
                              dtype=jnp.bfloat16) * 0.02, qw_s)
        kw = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(2), (kvh * hd, D),
                              dtype=jnp.bfloat16) * 0.02, qw_s)
        vw = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(3), (kvh * hd, D),
                              dtype=jnp.bfloat16) * 0.02, qw_s)
        ow = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(4), (D, nh * hd),
                              dtype=jnp.bfloat16) * 0.02, ow_s)
        k_ctx = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(5), (N, kvh, C, hd),
                              dtype=jnp.bfloat16) * 0.1,
            NamedSharding(mesh, PartitionSpec(None, H, None, None)))
        v_ctx = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(6), (N, kvh, C, hd),
                              dtype=jnp.bfloat16) * 0.1,
            NamedSharding(mesh, PartitionSpec(None, H, None, None)))
        mask = jnp.zeros((N, 1, 1, S), dtype=jnp.bfloat16)
        scaling = hd ** -0.5

        def repeat_kv(x):
            b, h_, s, d = x.shape
            x = jnp.broadcast_to(x[:, :, None], (b, h_, n_rep, s, d))
            return x.reshape(b, h_ * n_rep, s, d)

        def layer(hs, qw, kw, vw, ow, k_ctx, v_ctx, mask):
            specs = {}

            def rec(name, x):
                jax.debug.inspect_array_sharding(
                    x, callback=lambda sh, n=name: specs.setdefault(n, sh.spec))
                return x

            q = jnp.matmul(hs, qw.T).reshape(N, B, nh, hd)
            rec("1_q_after_proj", q)
            q = jnp.swapaxes(q, 1, 2)  # (N, nh, B, hd)
            rec("2_q_transposed", q)
            k_n = jnp.matmul(hs, kw.T).reshape(N, B, kvh, hd)
            k_n = jnp.swapaxes(k_n, 1, 2)
            v_n = jnp.matmul(hs, vw.T).reshape(N, B, kvh, hd)
            v_n = jnp.swapaxes(v_n, 1, 2)
            k = jnp.concatenate([k_ctx, k_n], axis=2)  # (N, kvh, S, hd)
            v = jnp.concatenate([v_ctx, v_n], axis=2)
            rec("3_k_concat", k)
            ks = repeat_kv(k)
            rec("4_ks_repeat", ks)
            aw = jnp.matmul(q, jnp.swapaxes(ks, 2, 3)) * scaling
            rec("5_scores", aw)
            aw = jax.nn.softmax((aw + mask).astype(jnp.float32),
                                axis=-1).astype(jnp.bfloat16)
            rec("6_softmax", aw)
            out = jnp.matmul(aw, repeat_kv(v))
            rec("7_attn_out", out)
            out = jnp.swapaxes(out, 1, 2).reshape(N, B, nh * hd)
            out = jnp.matmul(out, ow.T)  # (N, B, D)
            rec("8_o_proj_out", out)
            return out, specs

        f = jax.jit(layer)
        out, specs = f(hs, qw, kw, vw, ow, k_ctx, v_ctx, mask)
        jax.block_until_ready(out)
        print("REAL q-path PartitionSpec at each step (head-sharded weights):")
        for kk in sorted(specs):
            print(f"  {kk:22s} {specs[kk]}")


if __name__ == "__main__":
    main()
