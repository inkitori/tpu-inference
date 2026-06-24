"""Dispatch-floor vs true in-graph device-time for the MLX int4 gmm.

Addresses: are the ~150us single-call timings dispatch-dominated? Chain N gmms
inside ONE jit (output fed back through a fixed projection so XLA can't DCE),
block once, divide by N -> true per-gmm device time. Compare bf16-act
(maybe_quantize_lhs=False) vs int8-act (True, lhs-block=64 fix) at M in {2,64}.
M=2 stands in for batch=1 decode (M=1 hits a pre-existing gmm Mosaic limit).

Run: /home/enyouki/vllm_env/bin/python -u scratch_mlx_int4/devtime_probe.py
"""
import sys
import time

import jax

jax.config.update("jax_compilation_cache_dir",
                  "/home/enyouki/tpu-inference/scratch_mlx_int4/.jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import dataclasses  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

REPO = "/home/enyouki/tpu-inference"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tests.utils.mlx_synthetic import _quantize_affine  # noqa: E402
import tpu_inference.kernels.megablox.gmm_v2 as gmm_mod  # noqa: E402
from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2  # noqa: E402
from tpu_inference.layers.common.quantization import mlx_unpack  # noqa: E402

GS, BITS = 64, 4
VMEM = 128 << 20
IN, OUT = 4096, 10240  # qkv_proj
N_CHAIN = 50

_orig = gmm_mod.make_gmm_configs


def _fix64(*a, **k):
    c = _orig(*a, **k)
    return dataclasses.replace(
        c, lhs_cfgs=dataclasses.replace(c.lhs_cfgs, quant_block_size=GS))


def build():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((OUT, IN)).astype(np.float32)
    packed, s, b, _ = _quantize_affine(w, GS, False)
    codes = mlx_unpack(jnp.asarray(packed), BITS) - 8
    scale = jnp.asarray(s).astype(jnp.float32)
    gb = jnp.asarray(b).astype(jnp.float32) + 8.0 * scale
    codes = jnp.transpose(codes, (1, 0))[None].astype(jnp.int4)
    scale = jnp.transpose(scale, (1, 0))[None, :, None, :]
    gb = jnp.transpose(gb, (1, 0))[None, :, None, :]
    # Fixed projection out->in to feed each gmm output back as next input.
    proj = jnp.asarray(rng.standard_normal((OUT, IN)) * 0.02, dtype=jnp.bfloat16)
    return codes, scale, gb, proj


def chained(quant, codes, scale, gb, proj, M):
    """ONE jit that runs N_CHAIN gmms, carrying activation forward (no DCE)."""
    cfg_fn = _fix64 if quant else _orig
    gsz = jnp.array([M], jnp.int32)

    def body(x, _):
        gmm_mod.make_gmm_configs = cfg_fn
        y = gmm_v2(lhs=x, rhs=codes, group_sizes=gsz, rhs_scale=scale,
                   rhs_groupbias=gb, maybe_quantize_lhs=quant,
                   preferred_element_type=jnp.bfloat16, vmem_limit_bytes=VMEM)
        x2 = (y @ proj).astype(jnp.bfloat16)         # [M,out]@[out,in]->[M,in]
        x2 = x2 / (jnp.linalg.norm(x2) + 1e-6)       # keep magnitude bounded
        return x2, None

    def run(x):
        gmm_mod.make_gmm_configs = cfg_fn
        out, _ = jax.lax.scan(body, x, None, length=N_CHAIN)
        return out

    return jax.jit(run)


def time_call(fn, *args, iters=30, warmup=8):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        t.append(time.perf_counter() - t0)
    return float(np.median(t)) * 1e6  # us


def main():
    devs = jax.devices()
    print(f"devices: {len(devs)} x {devs[0].device_kind}\n")

    # (1) Dispatch floor: trivial jitted add, same block_until_ready loop.
    a = jnp.asarray(np.random.randn(128, 128), jnp.bfloat16)
    b = jnp.asarray(np.random.randn(128, 128), jnp.bfloat16)
    addf = jax.jit(lambda a, b: a + b)
    disp = time_call(addf, a, b, iters=200, warmup=20)
    print(f"(1) DISPATCH FLOOR (jit add, per call): {disp:.1f} us\n")

    codes, scale, gb, proj = build()

    print(f"(2)+(3) chained N={N_CHAIN} in ONE jit, per-gmm device time:")
    print(f"  {'path':9s} {'M':>3} {'total_us':>9} {'per_gmm_us':>11}")
    results = {}
    for quant, name in [(False, "bf16-act"), (True, "int8-act")]:
        for M in [2, 64]:
            gmm_mod.make_gmm_configs = _orig
            x = jnp.asarray(np.random.default_rng(M).standard_normal((M, IN)),
                            dtype=jnp.bfloat16)
            fn = chained(quant, codes, scale, gb, proj, M)
            total = time_call(fn, x, iters=20, warmup=5)
            per = (total - disp) / N_CHAIN  # subtract the single dispatch
            results[(name, M)] = per
            print(f"  {name:9s} {M:>3} {total:>9.1f} {per:>11.2f}")
    gmm_mod.make_gmm_configs = _orig

    print("\nSUMMARY")
    for M in [2, 64]:
        bf, i8 = results[("bf16-act", M)], results[("int8-act", M)]
        print(f"  M={M:>2}: bf16={bf:.2f}us  int8={i8:.2f}us  "
              f"int8/bf16={i8/bf:.2f}x  (speedup {bf/i8:.2f}x)")


if __name__ == "__main__":
    main()
