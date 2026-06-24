"""Find a vmem_limit_bytes that lets the K=13312 single-group gmm fit at tp=1."""
import sys
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh
sys.path.insert(0, "/home/enyouki/tpu-inference")
from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2

dev = jax.devices()
import jax.extend
try:
    from jax._src import tpu  # noqa
except Exception:
    pass
in_, out, M, GS = 13312, 4096, 64, 64
rng = np.random.default_rng(0)
x = jnp.asarray(rng.standard_normal((M, in_)), jnp.bfloat16)
codes = jnp.asarray(rng.integers(-8, 8, (1, in_, out)), jnp.int4)
scale = jnp.asarray(rng.standard_normal((1, in_ // GS, 1, out)) * 0.01, jnp.float32)
gb = jnp.asarray(rng.standard_normal((1, in_ // GS, 1, out)) * 0.01, jnp.float32)
gsz = jnp.array([M], jnp.int32)

for vlim in [None, 128 << 20, 96 << 20, 64 << 20, 32 << 20]:
    try:
        y = gmm_v2(lhs=x, rhs=codes, group_sizes=gsz, rhs_scale=scale,
                   rhs_groupbias=gb, maybe_quantize_lhs=False,
                   preferred_element_type=jnp.bfloat16, vmem_limit_bytes=vlim)
        jax.block_until_ready(y)
        print(f"vmem_limit={str(vlim):>12}  -> OK  out{y.shape}")
    except Exception as e:
        msg = str(e).splitlines()[0][:80]
        print(f"vmem_limit={str(vlim):>12}  -> FAIL  {msg}")
