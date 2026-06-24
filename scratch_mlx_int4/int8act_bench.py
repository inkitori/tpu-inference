"""W4A8 vs W4A16 decode microbenchmark for the MLX int4 dense gmm_v2 path.

Hypothesis: at decode (small M) the current bf16-activation int4 gmm is
VPU-dequant-bound (it unpacks int4->bf16 and applies scale+groupbias to every
K*N weight before a tiny matmul). Quantizing ACTIVATIONS to int8 lets the kernel
do int8xint4->int32 on the MXU and apply the affine scale/groupbias POST-matmul
(O(N) per K-group, not O(K*N)).

The ONLY code difference between the two paths is gmm_v2's `maybe_quantize_lhs`:
  * REFERENCE (current prod, `_mlx_int4_matmul`): maybe_quantize_lhs=False
        -> unquantized path, bf16 lhs, per-weight VPU dequant.
  * CANDIDATE (W4A8):                              maybe_quantize_lhs=True
        -> kernel quantizes lhs to int8 per-512-block (per-row absmax), runs
           int8 x int4 -> int32 on MXU, scales by block_scale*rhs_scale and adds
           groupbias*sum(full_lhs) POST-matmul. Affine math stays exact.

Run:  /home/enyouki/vllm_env/bin/python -u scratch_mlx_int4/int8act_bench.py
"""
import sys
import time

import jax

jax.config.update("jax_compilation_cache_dir",
                  "/home/enyouki/tpu-inference/scratch_mlx_int4/.jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

REPO = "/home/enyouki/tpu-inference"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tests.utils.mlx_synthetic import _quantize_affine  # noqa: E402
import dataclasses  # noqa: E402

import tpu_inference.kernels.megablox.gmm_v2 as gmm_mod  # noqa: E402
from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2  # noqa: E402
from tpu_inference.layers.common.quantization import mlx_unpack  # noqa: E402

GS, BITS = 64, 4
VMEM = 128 << 20  # full v6e VMEM; needed for unsharded K=13312 at tp=1
HBM_BW = 1.6e12   # v6e HBM bandwidth (bytes/s)

# *** CORRECTNESS FIX (see report) ***
# The default lhs int8 quant_block_size is 512, but gmm_v2's quantized-LHS path
# applies only ONE rhs_scale/groupbias block per lhs quant block. When the rhs
# quant group (GS=64) is smaller than the lhs block, a single int8 matmul
# accumulates across many rhs scale-groups but multiplies the whole sum by
# rhs_scale[group 0] only -> ~40-50% error. Forcing the lhs quant block to equal
# GS makes the affine scale/groupbias granularity correct (error drops to ~1.5e-2,
# the expected int8-activation error). We monkeypatch make_gmm_configs to set it.
_orig_make_cfgs = gmm_mod.make_gmm_configs


def _make_cfgs_lhs_qbs64(*a, **k):
    c = _orig_make_cfgs(*a, **k)
    return dataclasses.replace(
        c, lhs_cfgs=dataclasses.replace(c.lhs_cfgs, quant_block_size=GS))

# (in, out) for the real Hy3-preview-4bit dense linears.
SHAPES = {
    "qkv_proj":  (4096, 10240),
    "o_proj":    (8192, 4096),
    "gate_up":   (4096, 26624),
    "down_proj": (13312, 4096),
}
# NOTE: M=1 hits a pre-existing gmm_v2 Mosaic limitation (size_lhs_sublane tiling
# of the [G,blocks,1,N] scale buffer needs the m-dim sublane >=2), and fails for
# BOTH paths -- unrelated to int8 act. Real batch=1 decode would pad M to 2.
DECODE_M = [2, 8, 16, 64]
WARMUP, ITERS = 10, 60


def build(in_, out, seed):
    """Real-weight load identical to benchmark.py / process_weights_after_loading."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((out, in_)).astype(np.float32)
    packed, s_bf, b_bf, golden = _quantize_affine(w, GS, force_negative_scale=False)
    codes = mlx_unpack(jnp.asarray(packed), BITS) - 8
    scale = jnp.asarray(s_bf).astype(jnp.float32)
    groupbias = jnp.asarray(b_bf).astype(jnp.float32) + 8.0 * scale
    codes = jnp.transpose(codes, (1, 0))[None].astype(jnp.int4)          # [1,in,out]
    scale = jnp.transpose(scale, (1, 0))[None, :, None, :]               # [1,in//gs,1,out]
    groupbias = jnp.transpose(groupbias, (1, 0))[None, :, None, :]
    return codes, scale, groupbias, jnp.asarray(golden)  # golden f32 [out,in]


def matmul_fn(quantize_lhs, mesh, in_axis, out_axis, fix_qbs=True):
    """gmm_v2 wrapped in shard_map exactly like _mlx_int4_matmul, with the
    maybe_quantize_lhs flag exposed (False=bf16-act REFERENCE, True=int8-act).
    fix_qbs applies the lhs quant_block_size=GS correctness fix for the int8 path."""
    if quantize_lhs and fix_qbs:
        gmm_mod.make_gmm_configs = _make_cfgs_lhs_qbs64
    else:
        gmm_mod.make_gmm_configs = _orig_make_cfgs

    def _local(lhs, rhs, sc, gb, gs):
        y = gmm_v2(lhs=lhs, rhs=rhs, group_sizes=gs, rhs_scale=sc,
                   rhs_groupbias=gb, maybe_quantize_lhs=quantize_lhs,
                   preferred_element_type=jnp.bfloat16, vmem_limit_bytes=VMEM)
        if in_axis is not None:
            y = jax.lax.psum(y, axis_name=in_axis)
        return y

    fn = jax.shard_map(
        _local, mesh=mesh,
        in_specs=(P(None, in_axis), P(None, in_axis, out_axis),
                  P(None, in_axis, None, out_axis),
                  P(None, in_axis, None, out_axis), P()),
        out_specs=P(None, out_axis), check_vma=False)
    return jax.jit(fn)


def bench(fn, *args):
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    t = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        t.append(time.perf_counter() - t0)
    return float(np.median(t)) * 1e6  # us


def rel_l2(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


def main():
    devs = jax.devices()
    print(f"jax devices: {len(devs)} x {devs[0].device_kind}\n")
    mesh1 = Mesh(np.asarray(devs[:1]).reshape(1, 1), ("data", "model"))

    hdr = (f"{'shape':10s} {'M':>4} {'cand|ref':>9} {'cand|f32':>9} "
           f"{'ref|f32':>9} {'bf16_us':>8} {'int8_us':>8} {'speedup':>7} "
           f"{'hbm_us':>7}")
    print(hdr)
    print("-" * len(hdr))

    for name, (in_, out) in SHAPES.items():
        codes, scale, groupbias, golden = build(in_, out, 0)
        gf32 = np.asarray(golden, np.float32)                 # [out, in]
        hbm_floor_us = (in_ * out * 0.5) / HBM_BW * 1e6       # int4 weight bytes
        ref_fn = matmul_fn(False, mesh1, None, "model")
        cand_fn = matmul_fn(True, mesh1, None, "model")

        for M in DECODE_M:
            x = jnp.asarray(np.random.default_rng(M).standard_normal((M, in_)),
                            dtype=jnp.bfloat16)
            gsz = jnp.array([M], jnp.int32)
            ref = ref_fn(x, codes, scale, groupbias, gsz)
            cand = cand_fn(x, codes, scale, groupbias, gsz)
            ideal = np.asarray(x, np.float32) @ gf32.T        # f32 ground truth
            e_cr = rel_l2(cand, ref)
            e_cf = rel_l2(cand, ideal)
            e_rf = rel_l2(ref, ideal)
            bf16_us = bench(ref_fn, x, codes, scale, groupbias, gsz)
            int8_us = bench(cand_fn, x, codes, scale, groupbias, gsz)
            print(f"{name:10s} {M:>4} {e_cr:>9.2e} {e_cf:>9.2e} {e_rf:>9.2e} "
                  f"{bf16_us:>8.1f} {int8_us:>8.1f} {bf16_us/int8_us:>6.2f}x "
                  f"{hbm_floor_us:>7.2f}")
        print()

    # --- tp=8 sanity: one col-parallel (qkv) + one row-parallel (o) shape ---
    n = len(devs)
    if n > 1:
        meshN = Mesh(np.asarray(devs).reshape(1, n), ("data", "model"))
        print(f"-- tp={n} sharding sanity (M=8) --")
        for name, ia, oa in [("qkv_proj", None, "model"),
                             ("o_proj", "model", None)]:
            in_, out = SHAPES[name]
            codes, scale, groupbias, golden = build(in_, out, 7)
            gf32 = np.asarray(golden, np.float32)
            M = 8
            x = jnp.asarray(np.random.default_rng(1).standard_normal((M, in_)),
                            dtype=jnp.bfloat16)
            gsz = jnp.array([M], jnp.int32)
            ref = matmul_fn(False, meshN, ia, oa)(x, codes, scale, groupbias, gsz)
            cand = matmul_fn(True, meshN, ia, oa)(x, codes, scale, groupbias, gsz)
            ideal = np.asarray(x, np.float32) @ gf32.T
            print(f"{name:10s} tp={n} cand|ref={rel_l2(cand, ref):.2e} "
                  f"cand|f32={rel_l2(cand, ideal):.2e} "
                  f"ref|f32={rel_l2(ref, ideal):.2e}")


if __name__ == "__main__":
    main()
