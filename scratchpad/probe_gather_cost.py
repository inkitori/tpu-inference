"""Is the 84ms CACHED-full forward GATHER-bound, not attention-bound?

The cached fwd vmap-dynamic_slices a (L,N,C,KVH,hd) window out of the full
(L,N,buf_len,KVH,hd) caches for BOTH k and v, every step. At C=4096 that's
~1GB x2. This probe times that gather in isolation (head-sharded caches) vs
the attention-only stack, both at C=4096, to attribute the 84ms.

Run:
  HF_HOME=/home/enyouki/local_hf \
  ~/tpu-tooling/tpu-env.sh python scratchpad/probe_gather_cost.py
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

MESH_AXIS_NAMES = ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp")


def _mesh():
    return jax.sharding.Mesh(
        np.array(jax.devices()).reshape((1, 1, 1, 1, 8, 1)),
        axis_names=MESH_AXIS_NAMES)


def _median_ms(call, n=30):
    jax.block_until_ready(call())
    s = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(call())
        s.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(s)


def main():
    mesh = _mesh()
    L, N, KVH, hd = 8, 32, 8, 64
    C = 4096
    buf_len = C + 8
    H = "model"
    with jax.set_mesh(mesh):
        kv_s = NamedSharding(mesh, PartitionSpec(None, None, None, H, None))
        k_cache = jax.device_put(
            jnp.zeros((L, N, buf_len, KVH, hd), dtype=jnp.bfloat16), kv_s)
        v_cache = jax.device_put(
            jnp.zeros((L, N, buf_len, KVH, hd), dtype=jnp.bfloat16), kv_s)
        ws = jnp.zeros((N,), dtype=jnp.int32)

        def gather(k_cache, v_cache, ws):
            def _win(cache_ln, start):
                return jax.lax.dynamic_slice_in_dim(cache_ln, start, C, axis=1)
            k = jax.vmap(_win, in_axes=(1, 0), out_axes=1)(k_cache, ws)
            v = jax.vmap(_win, in_axes=(1, 0), out_axes=1)(v_cache, ws)
            return k, v

        f = jax.jit(gather)
        t_gather = _median_ms(lambda: f(k_cache, v_cache, ws))
        print(f"vmap window gather (k+v, L={L},N={N},C={C}) = "
              f"{t_gather:7.3f} ms")

        # gather + a trivial reduce to force the slice to materialize+move
        def gather_sum(k_cache, v_cache, ws):
            k, v = gather(k_cache, v_cache, ws)
            return k.sum() + v.sum()
        t_gs = _median_ms(lambda: jax.jit(gather_sum)(k_cache, v_cache, ws))
        print(f"gather + sum (forces materialize)            = {t_gs:7.3f} ms")

        # prefix-slice variant (what XLA may fuse when ws==0): k_cache[:,:,:C]
        def pslice(k_cache, v_cache):
            return k_cache[:, :, :C].sum() + v_cache[:, :, :C].sum()
        t_ps = _median_ms(lambda: jax.jit(pslice)(k_cache, v_cache))
        print(f"static prefix slice + sum                    = {t_ps:7.3f} ms")


if __name__ == "__main__":
    main()
