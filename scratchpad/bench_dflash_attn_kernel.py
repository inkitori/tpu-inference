"""ISOLATED DFlash draft attention: EAGER (einsum+softmax) vs Pallas FLASH.

Settles whether the DFlash draft per-layer attention at its real shape
(tiny query block B=8, big ragged context C up to 4608, GQA 64q/8kv heads,
head_dim 64, N=32 slots, 8 layers) is FLOP-bound (flash ~= eager) or whether
fusing the softmax / avoiding HBM materialization of the (Hq,B,C) score/prob
intermediates gives flash a real speedup.

Mirrors the draft's actual sharding: TP8 (data=1, model=8); the draft K/V/q
are REPLICATED across the 8 chips at DP=1 (PartitionSpec()).

Run:
  HF_HOME=/home/enyouki/local_hf \
  ~/tpu-tooling/tpu-env.sh python scratchpad/bench_dflash_attn_kernel.py
"""

import os
import statistics
import time

os.environ.setdefault("HF_HOME", "/home/enyouki/local_hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec

from tpu_inference.kernels.flash_attention.kernel import (BlockSizes,
                                                          flash_attention)

MESH_AXIS_NAMES = ("data", "attn_dp", "attn_dp_expert", "expert", "model",
                   "dcp")

# Real DFlash draft attention shape.
N = 32       # slots (batch axis, replicated at DP=1)
Hq = 64      # query heads
KVH = 8      # kv heads
GROUPS = Hq // KVH  # 8
HD = 64      # head_dim
B = 8        # query block (noise/spec tokens per slot)
L = 8        # layers
SM_SCALE = HD ** -0.5  # 1/8
VMEM = 128 * 1024 * 1024


def _real_mesh():
    devs = jax.devices()
    assert len(devs) == 8, f"expected 8 chips, got {len(devs)}"
    shape = (1, 1, 1, 1, 8, 1)
    devices = np.array(devs).reshape(shape)
    return jax.sharding.Mesh(devices, axis_names=MESH_AXIS_NAMES)


def _median_ms(call, n_iter=30):
    jax.block_until_ready(call())
    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        r = call()
        jax.block_until_ready(r)
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def repeat_kv(x):
    # x: (N, KVH, S, HD) -> (N, Hq, S, HD)
    return jnp.repeat(x, GROUPS, axis=1)


# ---- EAGER: mirror eager_attention_forward / modeling_qwen3 exactly ----
def eager_one_layer(q, k, v, mask):
    # q: (N, Hq, B, HD)   k,v: (N, KVH, S, HD)   mask: (N,1,B,S) bf16
    k_e = repeat_kv(k)               # (N, Hq, S, HD)
    v_e = repeat_kv(v)               # (N, Hq, S, HD)
    scores = jnp.einsum('nhqd,nhkd->nhqk', q, k_e).astype(jnp.float32)
    scores = scores * SM_SCALE
    scores = scores + mask.astype(jnp.float32)
    probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
    out = jnp.einsum('nhqk,nhkd->nhqd', probs, v_e)  # (N, Hq, B, HD)
    return out


def eager_8layer(q, ks, vs, mask):
    # ks, vs: (L, N, KVH, S, HD) -> distinct K/V per layer (defeats cross-layer
    # CSE of repeat_kv, matching the real forward where each layer has its own
    # cached K/V).
    out = q
    acc = 0.0
    for i in range(L):
        o = eager_one_layer(out, ks[i], vs[i], mask)
        acc = acc + o
    return acc


def score_matmul_8layer(q, ks, mask):
    # Just the q@k^T score matmul + softmax (no value matmul), per layer x8.
    # Isolates the O(C*B) attention-SCORE matmul the doc claims dominates.
    acc = 0.0
    for i in range(L):
        k_e = repeat_kv(ks[i])
        scores = jnp.einsum('nhqd,nhkd->nhqk', q, k_e).astype(jnp.float32)
        scores = scores * SM_SCALE + mask.astype(jnp.float32)
        probs = jax.nn.softmax(scores, axis=-1)
        acc = acc + probs.sum()
    return acc


# ---- FLASH: same math via the Pallas kernel ----
# Mosaic kernels can't be auto-partitioned under a mesh, so we run the flash
# call inside a shard_map with ALL-REPLICATED specs (the draft tensors ARE
# replicated at DP=1). Each chip runs the full kernel over the full tensor,
# which is exactly the replicated draft-attention behaviour.
def flash_block_sizes(S):
    return BlockSizes(block_q=B, block_k_major=S, block_k=S, block_b=1)


def _flash_core(q, k, v, ab, bs):
    # q: (N, Hq, B, HD); k,v: (N, KVH, S, HD); ab: (N, Hq, B, S)
    k_e = repeat_kv(k)
    v_e = repeat_kv(v)
    return flash_attention(q, k_e, v_e, ab=ab, causal=False, sm_scale=SM_SCALE,
                           block_sizes=bs, vmem_limit_bytes=VMEM)


def make_flash_one(mesh, bs):
    repl = PartitionSpec()
    core = lambda q, k, v, ab: _flash_core(q, k, v, ab, bs)
    return shard_map(core, mesh=mesh,
                     in_specs=(repl, repl, repl, repl), out_specs=repl,
                     check_rep=False)


def make_flash_8layer(mesh, bs):
    one = make_flash_one(mesh, bs)

    def f8(q, ks, vs, ab):
        # ks, vs: (L, N, KVH, S, HD) distinct per layer.
        out = q
        acc = 0.0
        for i in range(L):
            o = one(out, ks[i], vs[i], ab)
            acc = acc + o
        return acc

    return f8


def main():
    mesh = _real_mesh()
    print(f"mesh: {mesh}")
    repl = NamedSharding(mesh, PartitionSpec())  # replicated, DP=1
    key = jax.random.PRNGKey(0)

    Cs = [512, 1024, 2048, 4096, 4608]
    bf16min = float(jnp.finfo(jnp.bfloat16).min)

    print(f"shape: N={N} Hq={Hq} KVH={KVH} HD={HD} B={B} L={L} "
          f"sm_scale={SM_SCALE}")

    with jax.set_mesh(mesh):
        # jit fns, replicated in/out.
        eager_jit = jax.jit(eager_8layer, out_shardings=repl)
        eager1_jit = jax.jit(eager_one_layer, out_shardings=repl)

        def make_tensors(C):
            S = C + B
            qk = jax.device_put(
                jax.random.normal(jax.random.fold_in(key, 1),
                                  (N, Hq, B, HD), jnp.bfloat16), repl)
            # per-layer distinct K/V (L,N,KVH,S,HD) -- defeats cross-layer CSE.
            ks = jax.device_put(
                jax.random.normal(jax.random.fold_in(key, 2),
                                  (L, N, KVH, S, HD), jnp.bfloat16), repl)
            vs = jax.device_put(
                jax.random.normal(jax.random.fold_in(key, 3),
                                  (L, N, KVH, S, HD), jnp.bfloat16), repl)
            # mask: (N,1,1,S) -> broadcast. Pad last 64 cols ragged-tail.
            m = np.zeros((N, 1, 1, S), dtype=np.float32)
            m[:, :, :, -64:] = bf16min
            mask_nq = jax.device_put(
                jnp.asarray(np.broadcast_to(m, (N, 1, B, S)).copy(),
                            dtype=jnp.bfloat16), repl)  # eager: (N,1,B,S)
            # flash ab: EXACT (N,Hq,B,S). Pre-mul by 1/sm_scale so
            # (ab*sm_scale) reproduces the eager additive mask.
            ab = jnp.broadcast_to(
                (mask_nq.astype(jnp.float32) / SM_SCALE),
                (N, Hq, B, S)).astype(jnp.bfloat16)
            ab = jax.device_put(ab, repl)
            return qk, ks, vs, mask_nq, ab

        score_jit = jax.jit(score_matmul_8layer, out_shardings=repl)

        # ---- sanity: eager vs flash same math @ C=512, one layer ----
        C0 = 512
        S0 = C0 + B
        q0, ks0, vs0, mask0, ab0 = make_tensors(C0)
        bs0 = flash_block_sizes(S0)
        flash1_jit = jax.jit(make_flash_one(mesh, bs0), out_shardings=repl)
        e_out = jax.block_until_ready(eager1_jit(q0, ks0[0], vs0[0], mask0))
        f_out = jax.block_until_ready(flash1_jit(q0, ks0[0], vs0[0], ab0))
        max_abs = float(jnp.max(jnp.abs(e_out.astype(jnp.float32) -
                                        f_out.astype(jnp.float32))))
        print(f"\n[sanity @C={C0}, 1 layer] max abs diff (eager vs flash) = "
              f"{max_abs:.4e}  (benign bf16 diff expected)")

        # ---- timing table ----
        rows = []
        print(f"\nDFlash isolated attn timing (REAL 8-chip mesh, REPLICATED, "
              f"N={N}, distinct K/V per layer)  ms")
        hdr = (f"{'C':>6} | {'SCORE_8L':>9} | {'EAGER_8L':>9} | "
               f"{'FLASH_8L':>9} | {'spd(E/F)':>9}")
        print(hdr)
        print("-" * len(hdr))
        for C in Cs:
            S = C + B
            q, ks, vs, mask, ab = make_tensors(C)
            bs = flash_block_sizes(S)
            flash_jit = jax.jit(make_flash_8layer(mesh, bs),
                                out_shardings=repl)
            t_score = _median_ms(lambda: score_jit(q, ks, mask))
            t_eager = _median_ms(lambda: eager_jit(q, ks, vs, mask))
            t_flash = _median_ms(lambda: flash_jit(q, ks, vs, ab))
            sp = t_eager / t_flash
            print(f"{C:>6} | {t_score:>9.3f} | {t_eager:>9.3f} | "
                  f"{t_flash:>9.3f} | {sp:>8.2f}x")
            rows.append((C, t_eager, t_flash, sp, t_score))

        # ---- per-layer @ C=4608 ----
        C = 4608
        S = C + B
        q, ks, vs, mask, ab = make_tensors(C)
        bs = flash_block_sizes(S)
        flash1b_jit = jax.jit(make_flash_one(mesh, bs), out_shardings=repl)
        t_e1 = _median_ms(lambda: eager1_jit(q, ks[0], vs[0], mask))
        t_f1 = _median_ms(lambda: flash1b_jit(q, ks[0], vs[0], ab))
        print(f"\nper-layer @C={C}:  EAGER 1L = {t_e1:.3f} ms   "
              f"FLASH 1L = {t_f1:.3f} ms   speedup = {t_e1/t_f1:.2f}x")

        # ---- verdict projection ----
        r4608 = [r for r in rows if r[0] == 4608][0]
        sp = r4608[3]
        attn_now = 52.0   # ms attention at C=4608 (12-impl-kvcache)
        rest = 7.0        # ms non-attn cached fwd
        proj_flash_fwd = attn_now / sp + rest
        print("\n" + "=" * 60)
        print("PROJECTION (cached draft fwd ~59ms @C=4608, ~52ms attn)")
        print("=" * 60)
        print(f"  isolated attn speedup @C=4608 (8L): {sp:.2f}x")
        print(f"  projected flash cached fwd: 52/{sp:.2f} + 7 = "
              f"{proj_flash_fwd:.1f} ms")
        print(f"  break-even: cached_step/6 < 8.4ms/tok => step < ~50ms")
        verdict = ("FLASH HELPS MEANINGFULLY" if sp >= 1.25 else
                   "FLOP-BOUND (flash ~= eager)")
        print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
