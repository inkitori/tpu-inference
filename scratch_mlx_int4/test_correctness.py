"""Correctness check for the in-kernel int4 dense linear (gmm_v2 single-group).

Validates the new VllmMLXLinearMethod path against the OLD dequant+einsum path,
at the real mlx-community/Hy3-preview-4bit dense-linear shapes.

Checks per shape:
  1. transform algebra: scale*(q-8) + (bias+8*scale) reconstructs the f32 affine
     weight sc*q_unsigned + bias EXACTLY (pure numpy, no kernel) -- catches any
     sign-fold / transpose / layout bug.
  2. gmm_v2 numeric: _mlx_int4_matmul(x, codes, scale, groupbias) == x @ dequant.T
     (the OLD apply) within bf16 tolerance, for column- and row-parallel specs at
     tp=1, across decode batch sizes.
  3. (if >1 device) the same equivalence with REAL sharding at tp=ndev -- row
     parallel (psum) for o_proj, column parallel for qkv.

Run:  /home/enyouki/vllm_env/bin/python scratch_mlx_int4/test_correctness.py
"""
import sys

import jax

# Persistent compile cache so re-runs reuse the compiled Pallas kernels.
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
from tpu_inference.layers.common.quantization import (  # noqa: E402
    mlx_dequantize, mlx_unpack)
from tpu_inference.layers.vllm.quantization.mlx import (  # noqa: E402
    _mlx_int4_matmul)

GS, BITS = 64, 4
# Full v6e VMEM (128MB). Only needed for the unsharded K=13312 down_proj at
# tp=1; production (tp>=2) shards K and uses gmm's default tiling (None).
VMEM = 128 << 20

# Hy3-preview-4bit dense linears as (in, out). Attention q/k/v fused -> qkv;
# layer-0 dense MLP gate+up fused -> gate_up. (MoE layers use the MoE path.)
SHAPES = {
    "qkv_proj":  (4096, 10240),   # ColumnParallel (out sharded)
    "o_proj":    (8192, 4096),    # RowParallel    (in  sharded)
    "gate_up":   (4096, 26624),   # ColumnParallel
    "down_proj": (13312, 4096),   # RowParallel
}
BATCHES = [16, 128]


def build_mlx_weight(in_, out, seed):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((out, in_)).astype(np.float32)
    packed, s_bf, b_bf, golden = _quantize_affine(w, GS,
                                                  force_negative_scale=False)
    return (jnp.asarray(packed), jnp.asarray(s_bf), jnp.asarray(b_bf),
            jnp.asarray(golden))


def transform(packed, scales, biases):
    """Mirror VllmMLXLinearMethod.process_weights_after_loading."""
    codes = mlx_unpack(packed, BITS) - 8                       # int32 [out, in]
    scale = scales.astype(jnp.float32)                         # [out, in//gs]
    groupbias = biases.astype(jnp.float32) + 8.0 * scale
    codes = jnp.transpose(codes, (1, 0))[None].astype(jnp.int4)  # [1, in, out]
    scale = jnp.transpose(scale, (1, 0))[None, :, None, :]       # [1,in//gs,1,out]
    groupbias = jnp.transpose(groupbias, (1, 0))[None, :, None, :]
    return codes, scale, groupbias


def rel_l2(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


def main():
    devices = jax.devices()
    print(f"jax devices: {len(devices)} x {devices[0].device_kind}\n")
    mesh1 = Mesh(np.asarray(devices[:1]).reshape(1, 1), ("data", "model"))
    TOL = 3e-2
    fails = 0

    for name, (in_, out) in SHAPES.items():
        packed, scales, biases, golden = build_mlx_weight(in_, out, hash(name) % 7919)
        codes, scale, groupbias = transform(packed, scales, biases)

        # --- check 1: fold algebra reconstructs the f32 affine weight exactly ---
        q_u = np.asarray(mlx_unpack(packed, BITS), np.float32)          # [out,in]
        sc_f = np.repeat(np.asarray(scales, np.float32), GS, axis=1)    # [out,in]
        b_f = np.repeat(np.asarray(biases, np.float32), GS, axis=1)
        dq_f32 = sc_f * q_u + b_f                                       # [out,in]
        recon = (np.repeat(np.asarray(scale[0, :, 0, :], np.float32), GS, 0)
                 * np.asarray(codes[0], np.float32)
                 + np.repeat(np.asarray(groupbias[0, :, 0, :], np.float32), GS, 0))
        e = rel_l2(recon, dq_f32.T)
        ok = e < 1e-5
        fails += not ok
        print(f"[{name:9s}] fold-algebra rel_l2={e:.2e} {'OK' if ok else 'FAIL'}")

        # --- check 2: gmm_v2 single-group == old dequant+einsum, tp=1 ---
        specs = [("col", None, "model")]            # column at every M
        for M in BATCHES:
            x = jnp.asarray(np.random.default_rng(M).standard_normal((M, in_)),
                            dtype=jnp.bfloat16)
            golden_y = jnp.einsum("bd,fd->bf", x, golden)      # OLD path
            ideal_y = np.asarray(x, np.float32) @ np.asarray(golden, np.float32).T
            gsz = jnp.array([M], jnp.int32)
            cases = specs + ([("row", "model", None)] if M == BATCHES[-1] else [])
            for tag, ia, oa in cases:
                y = _mlx_int4_matmul(x, codes, scale, groupbias, gsz, mesh1, ia,
                                     oa, vmem_limit_bytes=VMEM)
                eg, ei = rel_l2(y, golden_y), rel_l2(y, ideal_y)
                ok = eg < TOL
                fails += not ok
                print(f"[{name:9s}] M={M:4d} {tag} gmm-vs-old={eg:.2e} "
                      f"(vs-f32-ideal={ei:.2e}) {'OK' if ok else 'FAIL'}")

    # --- check 3: real sharding at tp=ndev (row psum + column) ---
    n = len(devices)
    if n > 1:
        meshN = Mesh(np.asarray(devices).reshape(1, n), ("data", "model"))
        print(f"\n-- sharded tp={n} --")
        for name, tag, ia, oa in [("o_proj", "row", "model", None),
                                  ("qkv_proj", "col", None, "model")]:
            in_, out = SHAPES[name]
            packed, scales, biases, golden = build_mlx_weight(in_, out, 99)
            codes, scale, groupbias = transform(packed, scales, biases)
            M = 64
            x = jnp.asarray(np.random.default_rng(1).standard_normal((M, in_)),
                            dtype=jnp.bfloat16)
            golden_y = jnp.einsum("bd,fd->bf", x, golden)
            y = _mlx_int4_matmul(x, codes, scale, groupbias,
                                 jnp.array([M], jnp.int32), meshN, ia, oa,
                                 vmem_limit_bytes=VMEM)
            e = rel_l2(y, golden_y)
            ok = e < TOL
            fails += not ok
            print(f"[{name:9s}] tp={n} {tag} rel_l2={e:.2e} "
                  f"{'OK' if ok else 'FAIL'}")
    else:
        print("\n(skipping multi-device sharding test: single device)")

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
