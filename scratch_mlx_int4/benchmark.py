"""Decode-latency benchmark: in-kernel int4 (gmm_v2) vs old dequant+einsum.

At the real mlx-community/Hy3-preview-4bit dense-linear shapes, times both paths
across decode batch buckets (+ one prefill size to show the compute-bound
regime). Weights are passed as jit RUNTIME args (matching the real model_fn,
where XLA cannot constant-fold the dequant).

Run:  python scratch_mlx_int4/benchmark.py
"""
import sys
import time

import jax

# Persistent compile cache: first run compiles the Pallas kernels, every later
# run reuses them from disk (near-instant). gmm_v2 must be compiled regardless
# of jit/eager, so this -- not eager mode -- is what makes re-runs fast.
jax.config.update("jax_compilation_cache_dir",
                  "/home/enyouki/tpu-inference/scratch_mlx_int4/.jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

REPO = "/home/enyouki/tpu-inference"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tests.utils.mlx_synthetic import _quantize_affine  # noqa: E402
from tpu_inference.layers.common.quantization import mlx_dequantize  # noqa: E402
from tpu_inference.layers.vllm.quantization.mlx import (  # noqa: E402
    _mlx_int4_matmul)
from tpu_inference.layers.common.quantization import mlx_unpack  # noqa: E402

GS, BITS = 64, 4
VMEM = 128 << 20  # full v6e VMEM; needed for unsharded K=13312 at tp=1
SHAPES = {
    "qkv_proj":  (4096, 10240),
    "o_proj":    (8192, 4096),
    "gate_up":   (4096, 26624),
    "down_proj": (13312, 4096),
}
DECODE_M = [16, 32, 64, 128, 256]
PREFILL_M = 2048
WARMUP, ITERS = 5, 100


def build(in_, out, seed):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((out, in_)).astype(np.float32)
    packed, s_bf, b_bf, _ = _quantize_affine(w, GS, force_negative_scale=False)
    packed = jnp.asarray(packed)
    scales = jnp.asarray(s_bf)
    biases = jnp.asarray(b_bf)
    codes = mlx_unpack(packed, BITS) - 8
    scale = scales.astype(jnp.float32)
    groupbias = biases.astype(jnp.float32) + 8.0 * scale
    codes = jnp.transpose(codes, (1, 0))[None].astype(jnp.int4)
    scale = jnp.transpose(scale, (1, 0))[None, :, None, :]
    groupbias = jnp.transpose(groupbias, (1, 0))[None, :, None, :]
    return packed, scales, biases, codes, scale, groupbias


def bench(fn, *args):
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    t = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        t.append(time.perf_counter() - t0)
    return float(np.median(t)) * 1e6  # microseconds


def main():
    devices = jax.devices()
    print(f"jax devices: {len(devices)} x {devices[0].device_kind}")
    mesh = Mesh(np.asarray(devices[:1]).reshape(1, 1), ("data", "model"))

    old_fn = jax.jit(lambda x, p, s, b: jnp.einsum(
        "bd,fd->bf", x, mlx_dequantize(p, s, b, group_size=GS, bits=BITS)))

    for name, (in_, out) in SHAPES.items():
        packed, scales, biases, codes, scale, groupbias = build(in_, out, 0)
        w_int4_mb = in_ * out * 0.5 / 1e6
        w_bf16_mb = in_ * out * 2 / 1e6
        print(f"\n== {name}  in={in_} out={out}  "
              f"(int4 {w_int4_mb:.1f}MB vs bf16 {w_bf16_mb:.1f}MB) ==")
        print(f"  {'M':>5} {'old_us':>9} {'new_us':>9} {'speedup':>8}")
        for M in DECODE_M + [PREFILL_M]:
            x = jnp.asarray(np.random.default_rng(M).standard_normal((M, in_)),
                            dtype=jnp.bfloat16)
            gsz = jnp.array([M], jnp.int32)
            new_fn = jax.jit(lambda x, c, sc, gb: _mlx_int4_matmul(
                x, c, sc, gb, gsz, mesh, None, None, vmem_limit_bytes=VMEM))
            old_us = bench(old_fn, x, packed, scales, biases)
            new_us = bench(new_fn, x, codes, scale, groupbias)
            tag = "  <- prefill" if M == PREFILL_M else ""
            print(f"  {M:>5} {old_us:>9.1f} {new_us:>9.1f} "
                  f"{old_us / new_us:>7.2f}x{tag}")


if __name__ == "__main__":
    main()
