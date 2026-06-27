"""PROPERLY-FORMULATED DFlash draft attention bench: prove the 8-layer draft
attention is NOT FLOP-bound and can be made materially cheaper than ~48ms by
head-sharding across the real 8-chip mesh (and/or sane flash tiling / GQA-native
reads), not by replicating all 64 heads on every chip.

Shapes: N=32, Hq=64, KVH=8 (GQA group 8), HD=64, B=8, L=8, C in {2048,4096,4608}.
sm_scale=1/8, is_causal=False, bf16. Additive mask (N,1,1,S) bf16: 0 valid,
finfo(bf16).min for the padded last-64 ctx tail.

Variants (each x8 layers):
  A EAGER baseline (eager_8layer, replicated)          -> confirm ~48.8ms @4608
  B FLASH head-SHARDED across mesh (8 heads/chip) + sane tiling (key test)
  C FLASH replicated + sane tiling (block_k=512)       -> isolate tiling fix
  D EAGER head-SHARDED (same math, 8 heads/chip)       -> isolate sharding fix
  E EAGER GQA-native + bf16 + head-SHARDED (all wins)  -> theoretical floor

Run:
  HF_HOME=/home/enyouki/local_hf \
  ~/tpu-tooling/tpu-env.sh python scratchpad/bench_dflash_attn_fixed.py
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

N = 32       # slots
Hq = 64      # query heads
KVH = 8      # kv heads
GROUPS = Hq // KVH  # 8
HD = 64      # head_dim
B = 8        # query block
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
    # (N, KVH, S, HD) -> (N, Hq, S, HD)
    return jnp.repeat(x, GROUPS, axis=1)


# ---------- A: EAGER baseline (replicated) ----------
def eager_one_layer(q, k, v, mask):
    # q:(N,Hq,B,HD) k,v:(N,KVH,S,HD) mask:(N,1,B,S) bf16
    k_e = repeat_kv(k)
    v_e = repeat_kv(v)
    scores = jnp.einsum('nhqd,nhkd->nhqk', q, k_e).astype(jnp.float32)
    scores = scores * SM_SCALE + mask.astype(jnp.float32)
    probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
    out = jnp.einsum('nhqk,nhkd->nhqd', probs, v_e)
    return out


def eager_8layer(q, ks, vs, mask):
    out = q
    acc = 0.0
    for i in range(L):
        acc = acc + eager_one_layer(out, ks[i], vs[i], mask)
    return acc


# ---------- D: EAGER head-sharded (SAME math as eager baseline) ------------
# Each chip holds 8 q-heads + 1 kv-head (sharded KVH=8 across 8 chips). Repeat
# the 1 kv-head x8 INSIDE the shard (== the baseline's repeat_kv), promote
# scores to f32. This is the eager baseline's exact math, just head-parallel.
def _eager_shard_core(q, ks, vs, mask):
    # q:(N,8,B,HD) ks/vs:(L,N,1,S,HD)  mask:(N,1,B,S)
    out = q
    acc = 0.0
    for i in range(L):
        k_e = jnp.repeat(ks[i], GROUPS, axis=1)  # (N,8,S,HD) local repeat
        v_e = jnp.repeat(vs[i], GROUPS, axis=1)
        scores = jnp.einsum('nhqd,nhkd->nhqk', out, k_e).astype(jnp.float32)
        scores = scores * SM_SCALE + mask.astype(jnp.float32)
        probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
        out_i = jnp.einsum('nhqk,nhkd->nhqd', probs, v_e)
        acc = acc + out_i
    return acc


def make_eager_sharded(mesh):
    # shard Hq (axis 1) of q on 'model'; ks/vs (L,N,KVH,S,HD) shard KVH(axis 2).
    qs = PartitionSpec(None, 'model', None, None)
    kvs = PartitionSpec(None, None, 'model', None, None)
    ms = PartitionSpec()  # mask (N,1,B,S) replicated, broadcasts over heads
    return shard_map(_eager_shard_core, mesh=mesh,
                     in_specs=(qs, kvs, kvs, ms),
                     out_specs=qs, check_rep=False)


# ---------- E: EAGER GQA-native + bf16 + head-sharded ----------
# Each chip holds 8 q-heads (one GQA group) + 1 kv-head. Read kv at KVH=1/chip,
# broadcast inside. Scores kept bf16 (no f32 promotion). Floor variant.
def _eager_gqa_core(q, ks, vs, mask):
    # q:(N,8,B,HD)  ks/vs:(L,N,1,S,HD)  mask:(N,1,B,S)
    out = q
    acc = jnp.zeros_like(q)
    for i in range(L):
        k1 = ks[i]  # (N,1,S,HD)
        v1 = vs[i]
        # GQA-native: all 8 local q-heads attend the single local kv-head.
        # scores (N,8,B,S) = q (N,8,B,HD) . k1 (N,1,S,HD) bcast over head.
        scores = jnp.einsum('nhqd,nokd->nhqk', out, k1)  # bf16
        scores = scores * jnp.bfloat16(SM_SCALE) + mask.astype(jnp.bfloat16)
        probs = jax.nn.softmax(scores.astype(jnp.float32),
                               axis=-1).astype(jnp.bfloat16)
        out_i = jnp.einsum('nhqk,nokd->nhqd', probs, v1)
        acc = acc + out_i
    return acc


def make_eager_gqa_sharded(mesh):
    qs = PartitionSpec(None, 'model', None, None)      # q  (N,Hq,B,HD)
    kvs = PartitionSpec(None, None, 'model', None, None)  # k/v (L,N,KVH,S,HD)
    ms = PartitionSpec()
    return shard_map(_eager_gqa_core, mesh=mesh,
                     in_specs=(qs, kvs, kvs, ms),
                     out_specs=qs, check_rep=False)


# ---------- FLASH cores ----------
def flash_bs(block_q, block_k_major, block_k):
    return BlockSizes(block_q=block_q, block_k_major=block_k_major,
                      block_k=block_k, block_b=1)


# B: FLASH head-sharded. Each chip: q (N,8,B,HD) + 1 kv-head (N,1,Spad,HD),
# repeat the kv-head x8 INSIDE the shard (flash has no internal GQA), ab
# (N,8,B,Spad). Sharded so the 8 chips split the 64 q-heads (8/chip).
def _flash_shard_core(q, k1, v1, ab, bs):
    k_e = jnp.repeat(k1, GROUPS, axis=1)  # (N,8,Spad,HD) local
    v_e = jnp.repeat(v1, GROUPS, axis=1)
    return flash_attention(q, k_e, v_e, ab=ab, causal=False, sm_scale=SM_SCALE,
                           block_sizes=bs, vmem_limit_bytes=VMEM)


def make_flash_sharded(mesh, bs):
    qs = PartitionSpec(None, 'model', None, None)   # q (N,Hq,B,HD) shard Hq
    kvs = PartitionSpec(None, 'model', None, None)  # k1/v1 (N,KVH,Spad,HD) shard
    abs_ = PartitionSpec(None, 'model', None, None)  # ab (N,Hq,B,Spad) shard Hq
    core = lambda q, k, v, ab: _flash_shard_core(q, k, v, ab, bs)
    one = shard_map(core, mesh=mesh,
                    in_specs=(qs, kvs, kvs, abs_),
                    out_specs=qs, check_rep=False)

    def f8(q, ks_p, vs_p, ab):
        # ks_p/vs_p: (L,N,KVH,Spad,HD)
        out = q
        acc = 0.0
        for i in range(L):
            acc = acc + one(out, ks_p[i], vs_p[i], ab)
        return acc

    return f8


# C: FLASH replicated, sane tiling.
def _flash_repl_core(q, k, v, ab, bs):
    k_e = repeat_kv(k)
    v_e = repeat_kv(v)
    return flash_attention(q, k_e, v_e, ab=ab, causal=False, sm_scale=SM_SCALE,
                           block_sizes=bs, vmem_limit_bytes=VMEM)


def make_flash_repl(mesh, bs):
    repl = PartitionSpec()
    core = lambda q, k, v, ab: _flash_repl_core(q, k, v, ab, bs)
    one = shard_map(core, mesh=mesh,
                    in_specs=(repl, repl, repl, repl), out_specs=repl,
                    check_rep=False)

    def f8(q, ks, vs, ab):
        out = q
        acc = 0.0
        for i in range(L):
            acc = acc + one(out, ks[i], vs[i], ab)
        return acc

    return f8


def main():
    mesh = _real_mesh()
    print(f"mesh: {mesh}")
    repl = NamedSharding(mesh, PartitionSpec())
    key = jax.random.PRNGKey(0)
    bf16min = float(jnp.finfo(jnp.bfloat16).min)
    print(f"shape: N={N} Hq={Hq} KVH={KVH} HD={HD} B={B} L={L} "
          f"sm_scale={SM_SCALE}  chips=8 (model axis)")

    Cs = [2048, 4096, 4608]

    with jax.set_mesh(mesh):
        eager_jit = jax.jit(eager_8layer, out_shardings=repl)

        def make_tensors(C):
            S = C + B
            q = jax.device_put(jax.random.normal(
                jax.random.fold_in(key, 1), (N, Hq, B, HD), jnp.bfloat16), repl)
            # KVH-head per-layer distinct k/v (L,N,KVH,S,HD)
            ks = jax.device_put(jax.random.normal(
                jax.random.fold_in(key, 2),
                (L, N, KVH, S, HD), jnp.bfloat16), repl)
            vs = jax.device_put(jax.random.normal(
                jax.random.fold_in(key, 3),
                (L, N, KVH, S, HD), jnp.bfloat16), repl)
            # masks (eager): (N,1,B,S) bf16
            m = np.zeros((N, 1, 1, S), dtype=np.float32)
            m[:, :, :, -64:] = bf16min
            mask_nq = jax.device_put(jnp.asarray(
                np.broadcast_to(m, (N, 1, B, S)).copy(), jnp.bfloat16), repl)

            # ----- FLASH: pad kv axis to a multiple of 512 (kernel requires
            # kv_seq_len % block_k_major == 0). Padded keys masked via ab. KVH
            # heads only (repeat_kv happens inside the shard) -> 8x smaller. -----
            Spad = ((S + 511) // 512) * 512
            padk = Spad - S
            def padkv(x):  # pad axis 3 (S) of (L,N,KVH,S,HD)
                return jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, padk), (0, 0)))
            ks_p = jax.device_put(padkv(ks), repl)      # (L,N,KVH,Spad,HD)
            vs_p = jax.device_put(padkv(vs), repl)
            # ab: (N,Hq,B,Spad); bf16min on ragged-tail(64) AND padding cols.
            mp = np.zeros((N, 1, 1, Spad), dtype=np.float32)
            mp[:, :, :, S - 64:] = bf16min  # tail-64 of real ctx + all padding
            ab = jnp.broadcast_to(
                jnp.asarray(np.broadcast_to(mp, (N, 1, B, Spad)).copy()) / SM_SCALE,
                (N, Hq, B, Spad)).astype(jnp.bfloat16)
            ab = jax.device_put(ab, repl)
            return (q, ks, vs, mask_nq, ks_p, vs_p, ab, Spad)

        # ---- correctness reference @ C=4608 ----
        Cref = 4608
        Sref = Cref + B
        (q, ks, vs, mask_nq, ks_p, vs_p, ab, Spad) = make_tensors(Cref)
        ref = jax.block_until_ready(eager_jit(q, ks, vs, mask_nq))

        def maxdiff(out):
            o = jax.block_until_ready(out)
            return float(jnp.max(jnp.abs(o.astype(jnp.float32) -
                                         ref.astype(jnp.float32))))

        print("\n=== correctness vs eager_8layer (replicated) @C=4608 ===")
        # D eager-sharded
        eager_sh = jax.jit(make_eager_sharded(mesh),
                           out_shardings=NamedSharding(
                               mesh, PartitionSpec(None, 'model', None, None)))
        d_out = eager_sh(q, ks, vs, mask_nq)
        print(f"  D eager head-sharded     max|diff| = {maxdiff(d_out):.4e}")
        # E gqa-native sharded
        eager_gqa = jax.jit(make_eager_gqa_sharded(mesh),
                            out_shardings=NamedSharding(
                                mesh, PartitionSpec(None, 'model', None, None)))
        e_out = eager_gqa(q, ks, vs, mask_nq)
        print(f"  E eager GQA-native sh    max|diff| = {maxdiff(e_out):.4e}")
        # B flash-sharded (block_k_major=512 divides Spad; check correctness)
        bs_chk = flash_bs(B, 512, 512)
        flash_sh = jax.jit(make_flash_sharded(mesh, bs_chk),
                           out_shardings=NamedSharding(
                               mesh, PartitionSpec(None, 'model', None, None)))
        b_out = flash_sh(q, ks_p, vs_p, ab)
        print(f"  B flash head-sharded     max|diff| = {maxdiff(b_out):.4e}  "
              f"(Spad={Spad})")
        # C flash-replicated correctness too
        flash_rp = jax.jit(make_flash_repl(mesh, bs_chk), out_shardings=repl)
        c_out = flash_rp(q, ks_p, vs_p, ab)
        print(f"  C flash replicated       max|diff| = {maxdiff(c_out):.4e}")

        # ---- timing sweep ----
        # block_k_major=512 (divides padded Spad); sweep block_k in {128,256,512}
        # (each divides block_k_major).
        block_ks = [128, 256, 512]
        BKM = 512
        print("\n=== timing (ms, 8 layers, median-of-30, real 8-chip mesh) ===")
        hdr = (f"{'C':>5} | {'A eager(repl)':>13} | {'D eager-shard':>13} | "
               f"{'E gqa-shard':>11} | {'B flash-shard(bk)':>18} | "
               f"{'C flash-repl(bk)':>17}")
        print(hdr)
        print("-" * len(hdr))
        results = {}
        for C in Cs:
            S = C + B
            (q, ks, vs, mask_nq, ks_p, vs_p, ab, Spad) = make_tensors(C)
            t_A = _median_ms(lambda: eager_jit(q, ks, vs, mask_nq))

            eager_sh = jax.jit(make_eager_sharded(mesh),
                               out_shardings=NamedSharding(
                                   mesh,
                                   PartitionSpec(None, 'model', None, None)))
            t_D = _median_ms(lambda: eager_sh(q, ks, vs, mask_nq))

            eager_gqa = jax.jit(make_eager_gqa_sharded(mesh),
                                out_shardings=NamedSharding(
                                    mesh,
                                    PartitionSpec(None, 'model', None, None)))
            t_E = _median_ms(lambda: eager_gqa(q, ks, vs, mask_nq))

            # B flash-sharded: sweep block_k (block_k_major fixed at 512)
            best_B = (None, 1e9)
            for bk in block_ks:
                bs = flash_bs(B, BKM, bk)
                try:
                    fsh = jax.jit(make_flash_sharded(mesh, bs),
                                  out_shardings=NamedSharding(
                                      mesh,
                                      PartitionSpec(None, 'model', None, None)))
                    t = _median_ms(lambda: fsh(q, ks_p, vs_p, ab))
                    if t < best_B[1]:
                        best_B = (bk, t)
                except Exception as ex:
                    print(f"    [B flash-shard bk={bk} C={C}] FAIL {ex}")

            # C flash-replicated: sweep block_k
            best_C = (None, 1e9)
            for bk in block_ks:
                bs = flash_bs(B, BKM, bk)
                try:
                    frep = jax.jit(make_flash_repl(mesh, bs),
                                   out_shardings=repl)
                    t = _median_ms(lambda: frep(q, ks_p, vs_p, ab))
                    if t < best_C[1]:
                        best_C = (bk, t)
                except Exception as ex:
                    print(f"    [C flash-repl bk={bk} C={C}] FAIL {ex}")

            print(f"{C:>5} | {t_A:>13.3f} | {t_D:>13.3f} | {t_E:>11.3f} | "
                  f"{best_B[1]:>11.3f}(bk{best_B[0]}) | "
                  f"{best_C[1]:>10.3f}(bk{best_C[0]})")
            results[C] = dict(A=t_A, D=t_D, E=t_E, B=best_B, C=best_C)

        # ---- verdict ----
        r = results[4608]
        cands = {"A eager(repl)": r['A'], "D eager-shard": r['D'],
                 "E gqa-shard": r['E'], "B flash-shard": r['B'][1],
                 "C flash-repl": r['C'][1]}
        best_name = min(cands, key=cands.get)
        print("\n" + "=" * 64)
        print(f"FASTEST @C=4608: {best_name} = {cands[best_name]:.3f} ms")
        print(f"  vs eager baseline {r['A']:.3f} ms  "
              f"=> {r['A']/cands[best_name]:.2f}x faster")
        print("=" * 64)


if __name__ == "__main__":
    main()
