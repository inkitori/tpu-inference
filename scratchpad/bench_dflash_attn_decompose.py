"""DECOMPOSE the DFlash draft 8-layer attention ~48ms @C=4608.

Settle whether the eager 8-layer draft attention (TPU v6e-8, replicated DP=1,
PartitionSpec()) is HBM-bandwidth-bound or FLOP-bound, and quantify each
contributor: repeat_kv GQA expansion, q@k^T score matmul, f32 softmax
materialization, scores@V value matmul, and a GQA-native variant that AVOIDS
the 8x K/V expansion.

Mirrors bench_dflash_attn_kernel.py harness (real 8-chip mesh, replicated,
distinct K/V per layer to defeat cross-layer CSE).

Run:
  HF_HOME=/home/enyouki/local_hf \
  ~/tpu-tooling/tpu-env.sh python scratchpad/bench_dflash_attn_decompose.py
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

# v6e (Trillium) single-chip peaks.
HBM_PEAK_GBPS = 1640.0   # GB/s
MXU_BF16_TFLOPS = 918.0  # TFLOP/s


def _real_mesh():
    devs = jax.devices()
    assert len(devs) == 8, f"expected 8 chips, got {len(devs)}"
    shape = (1, 1, 1, 1, 8, 1)
    devices = np.array(devs).reshape(shape)
    return jax.sharding.Mesh(devices, axis_names=MESH_AXIS_NAMES)


def _median_ms(call, n_iter=30):
    first = call()
    jax.block_until_ready(first)
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


# ============================ COMPONENTS ============================

# (1) EAGER_8L full baseline -- mirror eager_attention_forward exactly.
def eager_one_layer(q, k, v, mask):
    k_e = repeat_kv(k)
    v_e = repeat_kv(v)
    scores = jnp.einsum('nhqd,nhkd->nhqk', q, k_e).astype(jnp.float32)
    scores = scores * SM_SCALE
    scores = scores + mask.astype(jnp.float32)
    probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
    out = jnp.einsum('nhqk,nhkd->nhqd', probs, v_e)
    return out


def eager_8layer(q, ks, vs, mask):
    out = q
    acc = 0.0
    for i in range(L):
        o = eager_one_layer(out, ks[i], vs[i], mask)
        acc = acc + o
    return acc


# (2) repeat_kv ALONE x8 -- pure GQA-expansion HBM write/read.
def repeat_kv_8layer(ks, vs):
    acc = 0.0
    for i in range(L):
        k_e = repeat_kv(ks[i])
        v_e = repeat_kv(vs[i])
        acc = acc + k_e.sum() + v_e.sum()
    return acc


# (3) q@k^T score matmul ALONE x8 (with repeat_kv, bf16 scores, no softmax/value).
def score_matmul_only_8layer(q, ks):
    acc = 0.0
    for i in range(L):
        k_e = repeat_kv(ks[i])
        scores = jnp.einsum('nhqd,nhkd->nhqk', q, k_e)  # bf16
        acc = acc + scores.sum()
    return acc


# (4) score matmul + f32 softmax x8 (no value matmul). [== score_matmul_8layer]
def score_softmax_f32_8layer(q, ks, mask):
    acc = 0.0
    for i in range(L):
        k_e = repeat_kv(ks[i])
        scores = jnp.einsum('nhqd,nhkd->nhqk', q, k_e).astype(jnp.float32)
        scores = scores * SM_SCALE + mask.astype(jnp.float32)
        probs = jax.nn.softmax(scores, axis=-1)
        acc = acc + probs.sum()
    return acc


# (5) scores@V value matmul ALONE x8 -- synthesize probs (N,Hq,B,S) bf16.
def value_matmul_only_8layer(probs, vs):
    acc = 0.0
    for i in range(L):
        v_e = repeat_kv(vs[i])
        out = jnp.einsum('nhqk,nhkd->nhqd', probs, v_e)
        acc = acc + out.sum()
    return acc


# (6) score+softmax kept fully bf16 (NO f32 astype) x8 -- isolate f32 penalty.
#     dot_general with preferred bf16 (einsum->bf16 trips XLA conv-emit SIGSEGV).
_DN_SCORE = (((3,), (3,)), ((0, 1), (0, 1)))  # contract HD; batch N,Hq


def score_softmax_bf16_8layer(q, ks, mask):
    acc = 0.0
    for i in range(L):
        k_e = repeat_kv(ks[i])
        scores = jax.lax.dot_general(q, k_e, _DN_SCORE,
                                     preferred_element_type=jnp.bfloat16)
        scores = scores * jnp.bfloat16(SM_SCALE) + mask  # bf16
        probs = _softmax_bf16(scores)
        acc = acc + probs.sum()
    return acc


def _softmax_bf16(x):
    # bf16-only softmax (no f32 promotion) to isolate the f32 materialization
    # cost. Numerically worse, but we only time it.
    m = jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp((x - m))
    return e / jnp.sum(e, axis=-1, keepdims=True)


# (7) EAGER full but GQA-NATIVE: read K/V at KVH=8 (NO repeat_kv 8x expand).
#     q reshaped (N, KVH, GROUPS*B, HD); use dot_general (batch dims n,k) so
#     XLA emits a batched matmul (NOT a convolution -- the 5D einsum form
#     'nkgqd,nksd->nkgqs' trips XLA's conv lowering and stack-overflows).
def eager_gqa_native_one(q, k, v, mask):
    # q: (N, Hq, B, HD) -> (N, KVH, GROUPS*B, HD); merge GROUPS into query rows.
    qg = q.reshape(N, KVH, GROUPS, B, HD).reshape(N, KVH, GROUPS * B, HD)
    # scores via dot_general: batch=(N,KVH), contract=HD.
    # qg (N,KVH,GB,HD) x k (N,KVH,S,HD) -> (N,KVH,GB,S). K read at KVH=8.
    dn_s = (((3,), (3,)), ((0, 1), (0, 1)))  # contract HD; batch N,KVH
    scores = jax.lax.dot_general(qg, k, dn_s,
                                 preferred_element_type=jnp.float32)
    scores = scores * SM_SCALE  # (N,KVH,GB,S) f32
    # mask (N,1,B,S) -> (N,KVH,GROUPS*B,S): broadcast over GROUPS and KVH.
    mr = mask.astype(jnp.float32).reshape(N, 1, 1, B, -1)
    mr = jnp.broadcast_to(mr, (N, KVH, GROUPS, B, mr.shape[-1])) \
        .reshape(N, KVH, GROUPS * B, -1)
    scores = scores + mr
    probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
    # out via dot_general: probs (N,KVH,GB,S) x v (N,KVH,S,HD) -> (N,KVH,GB,HD).
    dn_o = (((3,), (2,)), ((0, 1), (0, 1)))  # contract S; batch N,KVH
    out = jax.lax.dot_general(probs, v, dn_o,
                              preferred_element_type=jnp.bfloat16)
    return out.reshape(N, Hq, B, HD)


def eager_gqa_native_8layer(q, ks, vs, mask):
    out = q
    acc = 0.0
    for i in range(L):
        o = eager_gqa_native_one(out, ks[i], vs[i], mask)
        acc = acc + o
    return acc


# ============================ BYTE/FLOP MODELS ============================
def bf16_bytes(*dims):
    n = 1
    for d in dims:
        n *= d
    return n * 2


def fmt_bw(bytes_moved, ms):
    gbps = bytes_moved / (ms * 1e-3) / 1e9
    return gbps, 100.0 * gbps / HBM_PEAK_GBPS


def fmt_flops(flops, ms):
    tflops = flops / (ms * 1e-3) / 1e12
    return tflops, 100.0 * tflops / MXU_BF16_TFLOPS


def main():
    import sys
    # Optional: run ONE component in this process (avoids any cross-component
    # XLA compile interaction; each component gets a clean process).
    only = sys.argv[1] if len(sys.argv) > 1 else None

    mesh = _real_mesh()
    print(f"mesh: {mesh}")
    repl = NamedSharding(mesh, PartitionSpec())
    key = jax.random.PRNGKey(0)
    bf16min = float(jnp.finfo(jnp.bfloat16).min)

    Cs = [2048, 4096, 4608]
    if os.environ.get("BENCH_CS"):
        Cs = [int(x) for x in os.environ["BENCH_CS"].split(",")]
    print(f"shape: N={N} Hq={Hq} KVH={KVH} HD={HD} B={B} L={L} "
          f"GROUPS={GROUPS} sm_scale={SM_SCALE}")
    print(f"v6e single-chip: HBM={HBM_PEAK_GBPS} GB/s, "
          f"MXU bf16={MXU_BF16_TFLOPS} TFLOP/s\n")

    with jax.set_mesh(mesh):
        def make_tensors(C):
            S = C + B
            q = jax.device_put(
                jax.random.normal(jax.random.fold_in(key, 1),
                                  (N, Hq, B, HD), jnp.bfloat16), repl)
            ks = jax.device_put(
                jax.random.normal(jax.random.fold_in(key, 2),
                                  (L, N, KVH, S, HD), jnp.bfloat16), repl)
            vs = jax.device_put(
                jax.random.normal(jax.random.fold_in(key, 3),
                                  (L, N, KVH, S, HD), jnp.bfloat16), repl)
            m = np.zeros((N, 1, 1, S), dtype=np.float32)
            m[:, :, :, -64:] = bf16min
            mask_nq = jax.device_put(
                jnp.asarray(np.broadcast_to(m, (N, 1, B, S)).copy(),
                            dtype=jnp.bfloat16), repl)
            probs = jax.device_put(
                jax.random.normal(jax.random.fold_in(key, 4),
                                  (N, Hq, B, S), jnp.bfloat16), repl)
            return q, ks, vs, mask_nq, probs

        # jit each component, replicated out.
        jit_eager = jax.jit(eager_8layer, out_shardings=repl)
        jit_repkv = jax.jit(repeat_kv_8layer, out_shardings=repl)
        jit_score = jax.jit(score_matmul_only_8layer, out_shardings=repl)
        jit_ssf32 = jax.jit(score_softmax_f32_8layer, out_shardings=repl)
        jit_value = jax.jit(value_matmul_only_8layer, out_shardings=repl)
        jit_gqa = jax.jit(eager_gqa_native_8layer, out_shardings=repl)

        # sanity: eager vs gqa-native same math @C=512, 1 layer.
        qS, ksS, vsS, mS, _ = make_tensors(512)
        e1 = jax.block_until_ready(
            jax.jit(eager_one_layer, out_shardings=repl)(qS, ksS[0], vsS[0],
                                                         mS))
        g1 = jax.block_until_ready(
            jax.jit(eager_gqa_native_one, out_shardings=repl)(qS, ksS[0],
                                                              vsS[0], mS))
        md = float(jnp.max(jnp.abs(e1.astype(jnp.float32) -
                                   g1.astype(jnp.float32))))
        print(f"[sanity @C=512, 1L] max abs diff eager vs GQA-native = "
              f"{md:.4e} (benign bf16)\n")

        # component dispatch: name -> (jitfn, args-builder from tensors)
        comp_fns = {
            'EAGER_8L': lambda t: jit_eager(t[0], t[1], t[2], t[3]),
            'repeat_kv_8L': lambda t: jit_repkv(t[1], t[2]),
            'score_mm_8L': lambda t: jit_score(t[0], t[1]),
            'score+sm(f32)_8L': lambda t: jit_ssf32(t[0], t[1], t[3]),
            'value_mm_8L': lambda t: jit_value(t[4], t[2]),
            'GQA_native_8L': lambda t: jit_gqa(t[0], t[1], t[2], t[3]),
        }

        import json
        json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "decomp_results.json")

        results = {}  # C -> dict of component -> ms
        run_list = [only] if only else list(comp_fns.keys())
        for C in Cs:
            tens = make_tensors(C)
            r = {}
            for name in run_list:
                ms = _median_ms(lambda: comp_fns[name](tens))
                r[name] = ms
                print(f"  [C={C}] {name} = {ms:.3f} ms", flush=True)
            results[C] = r

        # merge into the on-disk JSON (so per-component subprocess runs combine).
        disk = {}
        if os.path.exists(json_path):
            with open(json_path) as f:
                disk = json.load(f)
        for C in Cs:
            d = disk.setdefault(str(C), {})
            d.update(results[C])
        with open(json_path, "w") as f:
            json.dump(disk, f, indent=2)
        print(f"\n[merged results -> {json_path}]")

        if only:
            return  # single-component run: skip the full table/roofline

        # ---- table (load merged disk results so all components show) ----
        with open(json_path) as f:
            disk = json.load(f)
        results = {int(k): v for k, v in disk.items()}

        # NB: 'score+sm(bf16)_8L' is OMITTED -- a fully-bf16 softmax over the
        # (N,Hq,B,S) scores trips an XLA conv-emit SIGSEGV on this v6e build
        # regardless of einsum vs dot_general. The f32 penalty is instead read
        # off as (score+sm f32) - (score_mm bf16, no softmax).
        comps = ['EAGER_8L', 'repeat_kv_8L', 'score_mm_8L',
                 'score+sm(f32)_8L', 'value_mm_8L', 'GQA_native_8L']
        print("\n================ DECOMPOSITION (ms, median-of-30) ============"
              "===")
        hdr = f"{'component':>20} | " + " | ".join(f"{f'C={c}':>9}" for c in Cs)
        print(hdr)
        print("-" * len(hdr))
        for comp in comps:
            row = f"{comp:>20} | " + " | ".join(
                f"{results[c][comp]:>9.3f}" for c in Cs)
            print(row)

        # softmax+f32 cost = (score+sm f32) - (score_mm bf16, no softmax)
        print("\n---- derived ----")
        for c in Cs:
            if 'score+sm(f32)_8L' in results[c] and 'score_mm_8L' in results[c]:
                d = results[c]['score+sm(f32)_8L'] - results[c]['score_mm_8L']
                print(f"  C={c}: softmax+f32 cost (score+sm_f32 - score_mm) = "
                      f"{d:+.3f} ms")

        # ---- bandwidth / flop utilization @C=4608 ----
        C = 4608
        S = C + B
        r = results[C]
        print(f"\n================ ROOFLINE @C={C} (single-chip peak) ========"
              f"=====")

        # repeat_kv: read KVH k+v (N,KVH,S,HD)x2, write Hq k+v (N,Hq,S,HD)x2,
        # per layer x L.
        rd = bf16_bytes(N, KVH, S, HD) * 2
        wr = bf16_bytes(N, Hq, S, HD) * 2
        rep_bytes = (rd + wr) * L
        g, p = fmt_bw(rep_bytes, r['repeat_kv_8L'])
        print(f"  repeat_kv_8L: {r['repeat_kv_8L']:.3f} ms, moves "
              f"{rep_bytes/1e9:.3f} GB -> {g:.1f} GB/s ({p:.1f}% HBM peak)")

        # EAGER_8L full: dominant HBM traffic is read/write of expanded K/V
        # (N,Hq,S,HD)x2 + score/prob f32 (N,Hq,B,S) materialization, per layer.
        # FLOPs: 2 matmuls each 2*N*Hq*B*S*HD, per layer.
        eager_flops = 2 * (2 * N * Hq * B * S * HD) * L
        tf, pf = fmt_flops(eager_flops, r['EAGER_8L'])
        # crude HBM: expanded k_e+v_e read (after expansion) + f32 scores+probs
        score_f32 = (N * Hq * B * S) * 4
        eager_bytes = (bf16_bytes(N, Hq, S, HD) * 2 + score_f32 * 2) * L
        ge, pe = fmt_bw(eager_bytes, r['EAGER_8L'])
        print(f"  EAGER_8L: {r['EAGER_8L']:.3f} ms")
        print(f"    useful FLOPs = {eager_flops:.3e} -> {tf:.2f} TFLOP/s "
              f"({pf:.3f}% MXU peak)")
        print(f"    ~HBM (exp.k/v + f32 score/prob) = {eager_bytes/1e9:.3f} GB"
              f" -> {ge:.1f} GB/s ({pe:.1f}% HBM peak)")

        # GQA-native: K/V read at KVH=8 (no expansion). Same FLOPs.
        gqa_flops = eager_flops
        tfg, pfg = fmt_flops(gqa_flops, r['GQA_native_8L'])
        gqa_bytes = (bf16_bytes(N, KVH, S, HD) * 2 + score_f32 * 2) * L
        gg, pg = fmt_bw(gqa_bytes, r['GQA_native_8L'])
        print(f"  GQA_native_8L: {r['GQA_native_8L']:.3f} ms")
        print(f"    useful FLOPs = {gqa_flops:.3e} -> {tfg:.2f} TFLOP/s "
              f"({pfg:.3f}% MXU peak)")
        print(f"    ~HBM (KVH k/v + f32 score/prob) = {gqa_bytes/1e9:.3f} GB "
              f"-> {gg:.1f} GB/s ({pg:.1f}% HBM peak)")

        speedup = r['EAGER_8L'] / r['GQA_native_8L']
        print(f"\n  GQA-native speedup over eager-repeat_kv @C={C}: "
              f"{speedup:.2f}x")

        # scaling trend: does time scale ~linearly in S (HBM) ?
        print("\n---- EAGER_8L scaling vs C (HBM-bound => ~linear in S) ----")
        for c in Cs:
            print(f"  C={c} (S={c+B}): EAGER_8L={results[c]['EAGER_8L']:.3f} ms"
                  f"  ({results[c]['EAGER_8L']/(c+B)*1000:.4f} us/key)")


if __name__ == "__main__":
    main()
