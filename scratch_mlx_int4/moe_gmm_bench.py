"""Decode MoE expert gmm_v2 microbench: TP vs EP, M-padding, tile sweep.

Replicates the production routed-MoE expert gmm (tpu_inference/layers/common/
fused_moe_gmm.py: tensor_parallel_gmm / expert_parallel_gmm) for the Hy3-preview
-4bit MoE shapes (192 experts, 8 active, hidden=4096, moe_intermediate=1536 ->
gate_up out=3072, down in=1536; int4 affine group_size=64; TP=8).

We build the int4 weights DIRECTLY in the gmm_v2 packed layout that
process_moe_weights produces (codes [E,k,n] int4 + scale/groupbias [E,blocks,1,n]
f32), skipping the MLX uint32 pack/unpack (the layout is what matters here, not
the exact codebook). The TP path replicates the w13 reorder+pad (out 3072 ->
n=512/chip) that yields the profiled gmm_v2-...-n_512 shape; EP keeps experts
whole (n=3072, no psum).

Device time is measured with the SAME chained-in-jit technique as
devtime_probe.py: lax.scan N reps carrying the activation forward (XLA can't DCE,
no per-call 125us dispatch). Correctness is rel_L2 vs an fp32-dequant per-expert
reference.

Run: /home/enyouki/vllm_env/bin/python -u scratch_mlx_int4/moe_gmm_bench.py
"""
import sys

import jax

jax.config.update("jax_compilation_cache_dir",
                  "/home/enyouki/tpu-inference/scratch_mlx_int4/.jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import functools  # noqa: E402
import time  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

REPO = "/home/enyouki/tpu-inference"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tpu_inference.kernels.megablox.gmm_v2 import TileSizes, gmm_v2  # noqa: E402

# ---- Hy3-preview-4bit MoE shapes ----
E = 192          # experts
ACTIVE = 8       # active experts at decode (topk routed, batch=1)
H = 4096         # hidden
I = 1536         # moe_intermediate
GATE_UP = 2 * I  # 3072
GS = 64          # int4 affine group size
TP = 8
LAYERS = 79
N_CHAIN = 40
VMEM = 128 << 20

ALIGN = 128


def align_to(x, a):
    return ((x + a - 1) // a) * a


# ----------------------------------------------------------------------------
# Build int4 weights directly in gmm_v2 packed layout.
#   codes: [E, k, n] int4 in [-8,7];  scale/gbias: [E, k//gs, 1, n] f32
# A reference fp32 weight per expert is dequant = scale*codes + gbias  ([E,k,n]).
# ----------------------------------------------------------------------------
def build_expert_weights(n_experts, k, n, seed):
    rng = np.random.default_rng(seed)
    nblk = k // GS
    codes = rng.integers(-8, 8, size=(n_experts, k, n)).astype(np.int8)
    # scales ~ small positive; gbias ~ small. Keep magnitudes realistic so the
    # dequant weight has unit-ish std.
    scale = (rng.random((n_experts, nblk, 1, n)).astype(np.float32) * 0.02 +
             0.005)
    gbias = (rng.standard_normal((n_experts, nblk, 1, n)).astype(np.float32) *
             0.02)
    codes_j = jnp.asarray(codes).astype(jnp.int4)
    scale_j = jnp.asarray(scale)
    gbias_j = jnp.asarray(gbias)
    # fp32 dequant reference weight [E, k, n]
    sc_full = np.repeat(scale[:, :, 0, :], GS, axis=1)   # [E,k,n]
    gb_full = np.repeat(gbias[:, :, 0, :], GS, axis=1)
    deq = sc_full * codes.astype(np.float32) + gb_full
    return codes_j, scale_j, gbias_j, deq


def reorder_for_shards(t, out_size, n_shards, dim):
    """Replicate reorder_concatenated_tensor_for_sharding for a single fused
    [w1|w3] tensor: split each of w1,w3 into n_shards chunks and interleave so
    chip i holds (w1_chunk_i, w3_chunk_i) contiguously along `dim`."""
    # t has w1 then w3 along `dim`, each of length out_size (already padded).
    parts = []
    w1 = np.take(t, range(0, out_size), axis=dim)
    w3 = np.take(t, range(out_size, 2 * out_size), axis=dim)
    csz = out_size // n_shards
    for s in range(n_shards):
        parts.append(np.take(w1, range(s * csz, (s + 1) * csz), axis=dim))
        parts.append(np.take(w3, range(s * csz, (s + 1) * csz), axis=dim))
    return np.concatenate(parts, axis=dim)


def build_w13_tp(seed):
    """gate_up (w13) in TP layout. Per-chip n = 2*align(I/TP,128) = 512.
    Returns codes[E,H,512*8], scale/gbias[E,H//gs,1,512*8], deq[E,H,3072 (padded
    reordered)]. We build w1,w3 each [E,H,I], pad local-intermediate to 128, then
    reorder+concat so the trailing n splits into TP chips of 512 each."""
    rng = np.random.default_rng(seed)
    nblk = H // GS
    local = I // TP                     # 192
    plocal = align_to(local, ALIGN)     # 256
    pad = plocal - local                # 64
    n_per_chip = 2 * plocal             # 512
    N = n_per_chip * TP                 # 4096

    def mk(n):
        codes = rng.integers(-8, 8, size=(E, H, n)).astype(np.int8)
        scale = rng.random((E, nblk, 1, n)).astype(np.float32) * 0.02 + 0.005
        gbias = rng.standard_normal((E, nblk, 1, n)).astype(np.float32) * 0.02
        return codes, scale, gbias

    c1, s1, g1 = mk(I)
    c3, s3, g3 = mk(I)

    def pad_local(t, dim):
        # reshape last dim I -> (TP, local), pad local to plocal, flatten
        sh = list(t.shape)
        sh[dim:dim + 1] = [TP, local]
        t = t.reshape(sh)
        pw = [(0, 0)] * t.ndim
        pw[dim + 1] = (0, pad)
        t = np.pad(t, pw)
        sh2 = list(t.shape)
        sh2[dim:dim + 2] = [TP * plocal]
        return t.reshape(sh2)

    c1p, c3p = pad_local(c1, 2), pad_local(c3, 2)
    s1p, s3p = pad_local(s1, 3), pad_local(s3, 3)
    g1p, g3p = pad_local(g1, 3), pad_local(g3, 3)

    codes = reorder_for_shards(np.concatenate([c1p, c3p], axis=2), TP * plocal,
                               TP, dim=2)
    scale = reorder_for_shards(np.concatenate([s1p, s3p], axis=3), TP * plocal,
                               TP, dim=3)
    gbias = reorder_for_shards(np.concatenate([g1p, g3p], axis=3), TP * plocal,
                               TP, dim=3)

    sc_full = np.repeat(scale[:, :, 0, :], GS, axis=1)
    gb_full = np.repeat(gbias[:, :, 0, :], GS, axis=1)
    deq = sc_full * codes.astype(np.float32) + gb_full   # [E,H,4096]
    return (jnp.asarray(codes).astype(jnp.int4), jnp.asarray(scale),
            jnp.asarray(gbias), deq, n_per_chip)


# ----------------------------------------------------------------------------
# group_sizes: ACTIVE experts get rows; rest 0. Sum == M (padded token count).
# ----------------------------------------------------------------------------
def make_group_sizes(M, active_ids):
    gs = np.zeros(E, np.int32)
    per = M // len(active_ids)
    for i in active_ids:
        gs[i] = per
    gs[active_ids[0]] += M - per * len(active_ids)
    return jnp.asarray(gs)


# ----------------------------------------------------------------------------
# fp32 reference: per active expert, x_rows @ deq[e]  (matches gmm row grouping).
# gate_up applies silu fusion: silu(gate)*up where out = [.. w1 | w3 ..] per chip.
# For correctness we mirror the kernel's group layout: rows are contiguous by
# expert in the sorted order (group_sizes), each block hits expert e's weight.
# ----------------------------------------------------------------------------
def ref_gmm(x, deq, group_sizes, fuse_silu, n_per_chip=None, tp=False):
    x = np.asarray(x, np.float32)
    gs = np.asarray(group_sizes)
    out_n = deq.shape[2]
    y = np.zeros((x.shape[0], out_n), np.float32)
    row = 0
    for e in range(len(gs)):
        ng = int(gs[e])
        if ng == 0:
            continue
        seg = x[row:row + ng] @ deq[e]    # [ng, n]
        y[row:row + ng] = seg
        row += ng
    if fuse_silu:
        if tp:
            # per chip: [w1(plocal)|w3(plocal)]; silu(w1)*w3
            pl = n_per_chip // 2
            y = y.reshape(y.shape[0], tp_chips(out_n, n_per_chip), 2, pl)
            g, u = y[:, :, 0, :], y[:, :, 1, :]
            y = (g / (1 + np.exp(-g))) * u
            y = y.reshape(y.shape[0], -1)
        else:
            half = out_n // 2
            g, u = y[:, :half], y[:, half:]
            y = (g / (1 + np.exp(-g))) * u
    return y


def tp_chips(out_n, n_per_chip):
    return out_n // n_per_chip


def rel_l2(a, b):
    a, b = np.asarray(a, np.float32).ravel(), np.asarray(b, np.float32).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


# ----------------------------------------------------------------------------
# Device-time: chain N gmms in one jit, carry activation forward.
# For gate_up: out n -> need to feed back to [M,H]; use a fixed proj [n,H].
# For down:    out H -> feed back to [M,I_in]; proj [H, k].
# ----------------------------------------------------------------------------
def time_call(fn, *args, iters=20, warmup=6):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        t.append(time.perf_counter() - t0)
    return float(np.median(t)) * 1e6


def dispatch_floor():
    a = jnp.asarray(np.random.randn(128, 128), jnp.bfloat16)
    b = jnp.asarray(np.random.randn(128, 128), jnp.bfloat16)
    f = jax.jit(lambda a, b: a + b)
    return time_call(f, a, b, iters=200, warmup=20)


# ===========================================================================
# Runners. Weights are passed as ARGUMENTS (device_put with sharding) so XLA
# does NOT bake the 3.6GB int4 constants into the executable (which both bloats
# compile time and breaks the persistent cache). Each make_* returns
# (jitted_fn, args) and devtime() calls fn(*args).
# ===========================================================================
def dput(x, mesh, spec):
    return jax.device_put(x, NamedSharding(mesh, spec))


# ---------------- TP gate_up (w13): n sharded, NO psum ----------------
def make_tp_gateup(mesh, codes, scale, gbias, group_sizes, M, qlhs=False,
                   tiles=None):
    N = codes.shape[2]
    k = codes.shape[1]
    n_out_local = N // TP // 2          # silu halves; per-chip
    proj = dput(jnp.asarray(
        np.random.default_rng(7).standard_normal((n_out_local, k)) * 0.02,
        jnp.bfloat16), mesh, P())
    x = dput(jnp.asarray(np.random.default_rng(M).standard_normal((M, k)),
                         jnp.bfloat16), mesh, P())
    gsz = dput(group_sizes, mesh, P())

    def local(x, c, s, g, proj, gs):
        def body(xx, _):
            y = gmm_v2(lhs=xx, rhs=c, group_sizes=gs, rhs_scale=s,
                       rhs_groupbias=g, maybe_quantize_lhs=qlhs,
                       preferred_element_type=jnp.bfloat16, fuse_act="silu",
                       vmem_limit_bytes=VMEM,
                       **({"tile_info": tiles} if tiles else {}))
            x2 = (y @ proj).astype(jnp.bfloat16)
            x2 = x2 / (jnp.linalg.norm(x2) + 1e-6)
            return x2, None
        out, _ = jax.lax.scan(body, x, None, length=N_CHAIN)
        return out.sum()[None]

    fn = jax.jit(jax.shard_map(
        local, mesh=mesh,
        in_specs=(P(), P(None, None, "model"), P(None, None, None, "model"),
                  P(None, None, None, "model"), P(), P()),
        out_specs=P("model"), check_vma=False))
    cd_ = dput(codes, mesh, P(None, None, "model"))
    sd_ = dput(scale, mesh, P(None, None, None, "model"))
    gd_ = dput(gbias, mesh, P(None, None, None, "model"))
    return fn, (x, cd_, sd_, gd_, proj, gsz)


# ---------------- TP down (w2): k sharded, psum ----------------
def make_tp_down(mesh, codes, scale, gbias, group_sizes, M, qlhs=False):
    k = codes.shape[1]       # I
    Hn = codes.shape[2]      # H
    proj = dput(jnp.asarray(
        np.random.default_rng(8).standard_normal((Hn, k // TP)) * 0.02,
        jnp.bfloat16), mesh, P())
    x = dput(jnp.asarray(np.random.default_rng(M).standard_normal((M, k)),
                         jnp.bfloat16), mesh, P(None, "model"))
    gsz = dput(group_sizes, mesh, P())

    def local(x, c, s, g, proj, gs):
        def body(xx, _):
            y = gmm_v2(lhs=xx, rhs=c, group_sizes=gs, rhs_scale=s,
                       rhs_groupbias=g, maybe_quantize_lhs=qlhs,
                       preferred_element_type=jnp.float32, vmem_limit_bytes=VMEM)
            y = jax.lax.psum(y, axis_name="model").astype(jnp.bfloat16)
            x2 = (y @ proj).astype(jnp.bfloat16)
            x2 = x2 / (jnp.linalg.norm(x2) + 1e-6)
            return x2, None
        out, _ = jax.lax.scan(body, x, None, length=N_CHAIN)
        return out.sum()[None]

    fn = jax.jit(jax.shard_map(
        local, mesh=mesh,
        in_specs=(P(None, "model"), P(None, "model", None),
                  P(None, "model", None, None), P(None, "model", None, None),
                  P(), P()),
        out_specs=P("model"), check_vma=False))
    cd_ = dput(codes, mesh, P(None, "model", None))
    sd_ = dput(scale, mesh, P(None, "model", None, None))
    gd_ = dput(gbias, mesh, P(None, "model", None, None))
    return fn, (x, cd_, sd_, gd_, proj, gsz)


# ---------------- EP (experts whole, no psum) ----------------
def make_ep(mesh, codes, scale, gbias, group_sizes, M, k, n_out, fuse,
            is_down=False, qlhs=False):
    local_E = E // TP
    goff = dput(jnp.arange(0, E, local_E, dtype=jnp.int32), mesh, P("model"))
    n_proj_in = (n_out // 2) if fuse else n_out
    proj = dput(jnp.asarray(
        np.random.default_rng(9).standard_normal((n_proj_in, k)) * 0.02,
        jnp.bfloat16), mesh, P())
    x = dput(jnp.asarray(np.random.default_rng(M + 1).standard_normal((M, k)),
                         jnp.bfloat16), mesh, P())
    gsz = dput(group_sizes, mesh, P())

    def local(x, c, s, g, go, proj, gs):
        def body(xx, _):
            y = gmm_v2(lhs=xx, rhs=c, group_sizes=gs, rhs_scale=s,
                       rhs_groupbias=g, group_offset=go[0],
                       maybe_quantize_lhs=qlhs,
                       preferred_element_type=(jnp.float32 if is_down
                                               else jnp.bfloat16),
                       fuse_act=("silu" if fuse else None),
                       vmem_limit_bytes=VMEM)
            x2 = (y.astype(jnp.bfloat16) @ proj).astype(jnp.bfloat16)
            x2 = x2 / (jnp.linalg.norm(x2) + 1e-6)
            return x2, None
        out, _ = jax.lax.scan(body, x, None, length=N_CHAIN)
        return out.sum()[None]

    fn = jax.jit(jax.shard_map(
        local, mesh=mesh,
        in_specs=(P(), P("model", None, None), P("model", None, None, None),
                  P("model", None, None, None), P("model"), P(), P()),
        out_specs=P("model"), check_vma=False))
    cd_ = dput(codes, mesh, P("model", None, None))
    sd_ = dput(scale, mesh, P("model", None, None, None))
    gd_ = dput(gbias, mesh, P("model", None, None, None))
    return fn, (x, cd_, sd_, gd_, goff, proj, gsz)


# ---------------- single-call correctness helpers (tp=1 mesh) -------------
def gmm_ref_check(codes, scale, gbias, deq, group_sizes, M, k, fuse, is_down):
    """Unsharded single gmm over all E vs fp32 dequant ref (validates math)."""
    mesh1 = Mesh(np.asarray(jax.devices()[:1]).reshape(1), ("model",))
    x = jnp.asarray(np.random.default_rng(M + 3).standard_normal((M, k)),
                    jnp.bfloat16)

    def local(xx, c, s, g):
        return gmm_v2(lhs=xx, rhs=c, group_sizes=group_sizes, rhs_scale=s,
                      rhs_groupbias=g, maybe_quantize_lhs=False,
                      preferred_element_type=(jnp.float32 if is_down
                                              else jnp.bfloat16),
                      fuse_act=("silu" if fuse else None), vmem_limit_bytes=VMEM)
    y = jax.shard_map(local, mesh=mesh1, in_specs=(P(), P(), P(), P()),
                      out_specs=P(), check_vma=False)(x, codes, scale, gbias)
    ref = ref_gmm(x, deq, group_sizes, fuse_silu=fuse)
    return rel_l2(y, ref)


# ----------------------------------------------------------------------------
def main():
    devs = jax.devices()
    print(f"devices: {len(devs)} x {devs[0].device_kind}")
    assert len(devs) == TP, f"need {TP} devices"
    mesh = Mesh(np.asarray(devs).reshape(TP), ("model",))

    disp = dispatch_floor()
    print(f"dispatch floor: {disp:.1f} us\n")

    active = list(range(ACTIVE))                       # TP: experts 0..7
    ep_active = [0, 1, 24, 48, 72, 96, 120, 144]       # EP worst: 2 on chip0

    cu_tp, su_tp, gu_tp, deq_up_tp, npc = build_w13_tp(seed=1)
    cd, sd, gd, deq_down = build_expert_weights(E, I, H, seed=2)
    cu_ep, su_ep, gu_ep, deq_up_ep = build_expert_weights(E, H, GATE_UP, seed=3)
    print(f"shapes: gateupTP{tuple(cu_tp.shape)} npc={npc} "
          f"down{tuple(cd.shape)} gateupEP{tuple(cu_ep.shape)}\n")

    results = []

    def add(variant, M, g_us, d_us, rl2_g, rl2_d):
        per_step = (g_us + d_us) * LAYERS * 2 / 1000.0   # x79 layers x(gate+down)
        # NOTE: per_step already = (gate+down) summed; x2 in prompt is the two
        # gmms which we already add. Keep single (gate+down)*79.
        per_step = (g_us + d_us) * LAYERS / 1000.0
        tpot = per_step + 4.0
        results.append((variant, M, g_us, d_us, per_step, rl2_g, rl2_d, tpot,
                        1000.0 / tpot))

    def dt(fn, args):
        for _ in range(6):
            jax.block_until_ready(fn(*args))
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            ts.append(time.perf_counter() - t0)
        return (float(np.median(ts)) * 1e6 - disp) / N_CHAIN

    # ===== harness validation =====
    # The profiled 139us does NOT correspond to 8 active experts. gmm_v2 cost is
    # driven by the # of DISTINCT active experts (nonzero group_sizes): each adds
    # ~3.7us in TP (every chip runs ALL active experts, n-sharded). 8 active ->
    # ~44us; the profiled 139us == ~32 active experts (a multi-token decode:
    # padded m=128 = many tokens x topk hitting ~32 distinct experts). So we
    # validate by SWEEPING active-expert count and confirming the ~3.7us/expert
    # slope reproduces the 139us at active=32. (qlhs=int8 lhs is FASTER, not the
    # cause.)
    print("=== HARNESS GATE: TP gate_up M=128, sweep #active experts ===")
    print("    (profiled 139us reproduced at ~32 active experts)")
    for nact in [8, 16, 32, 64]:
        ids = list(np.linspace(0, E - 1, nact).astype(int))
        gs = make_group_sizes(128, ids)
        fn, a = make_tp_gateup(mesh, cu_tp, su_tp, gu_tp, gs, 128)
        print(f"  active={nact:3d}: gate_up M=128 = {dt(fn, a):7.1f}us")
    for qlhs in [False, True]:
        gs = make_group_sizes(128, active)
        fn, a = make_tp_gateup(mesh, cu_tp, su_tp, gu_tp, gs, 128, qlhs=qlhs)
        print(f"  qlhs={qlhs!s:5} (8 active) gate_up M=128: {dt(fn, a):7.1f}us")

    # ===== main sweep =====
    print("\n=== TP (n-sharded gate_up, k-sharded down) ===")
    for M in [128, 16, 8]:
        gs = make_group_sizes(M, active)
        fg, ag = make_tp_gateup(mesh, cu_tp, su_tp, gu_tp, gs, M)
        g_us = dt(fg, ag)
        fd, ad = make_tp_down(mesh, cd, sd, gd, gs, M)
        d_us = dt(fd, ad)
        e_g = gmm_ref_check(cu_tp, su_tp, gu_tp, deq_up_tp, gs, M, H, True, False)
        e_d = gmm_ref_check(cd, sd, gd, deq_down, gs, M, I, False, True)
        add("TP", M, g_us, d_us, e_g, e_d)
        print(f"  TP M={M:3d}: gate_up={g_us:7.1f}us down={d_us:7.1f}us "
              f"rl2_g={e_g:.2e} rl2_d={e_d:.2e}")

    print("\n=== EP (experts whole; worst: 2 active experts on chip0) ===")
    for M in [128, 16]:
        gs = make_group_sizes(M, ep_active)
        fg, ag = make_ep(mesh, cu_ep, su_ep, gu_ep, gs, M, H, GATE_UP, True)
        g_us = dt(fg, ag)
        fd, ad = make_ep(mesh, cd, sd, gd, gs, M, I, H, False, is_down=True)
        d_us = dt(fd, ad)
        e_g = gmm_ref_check(cu_ep, su_ep, gu_ep, deq_up_ep, gs, M, H, True, False)
        e_d = gmm_ref_check(cd, sd, gd, deq_down, gs, M, I, False, True)
        add("EP", M, g_us, d_us, e_g, e_d)
        print(f"  EP M={M:3d}: gate_up={g_us:7.1f}us down={d_us:7.1f}us "
              f"rl2_g={e_g:.2e} rl2_d={e_d:.2e}")

    # ===== DECISIVE: TP vs EP as #active experts grows (M=128) =====
    # TP runs ALL active experts on EVERY chip; EP distributes them (busiest chip
    # = ceil(active/8)). So EP's advantage grows with active-expert count.
    print("\n=== TP vs EP vs #active experts (M=128) ===")
    print(f"  {'active':>6} {'TP_g+d':>8} {'EP_g+d':>8} {'EP/TP':>6}")
    for nact in [8, 16, 32, 64]:
        ids = list(np.linspace(0, E - 1, nact).astype(int))
        gst = make_group_sizes(128, ids)
        tg = dt(*make_tp_gateup(mesh, cu_tp, su_tp, gu_tp, gst, 128))
        tdn = dt(*make_tp_down(mesh, cd, sd, gd, gst, 128))
        # EP: round-robin assign active experts across the 8 chips.
        ids_ep = [(i % TP) * (E // TP) + (i // TP) for i in range(nact)]
        gse = make_group_sizes(128, ids_ep)
        eg = dt(*make_ep(mesh, cu_ep, su_ep, gu_ep, gse, 128, H, GATE_UP, True))
        edn = dt(*make_ep(mesh, cd, sd, gd, gse, 128, I, H, False, is_down=True))
        print(f"  {nact:>6} {tg + tdn:>8.1f} {eg + edn:>8.1f} "
              f"{(eg + edn) / (tg + tdn):>6.2f}")

    print("\n=== TP gate_up M=128 tile sweep (tile_m,tile_k,tile_n) ===")
    gs128 = make_group_sizes(128, active)
    for tm, tk, tn in [(64, 512, 256), (128, 512, 256), (32, 512, 512),
                       (64, 1024, 256), (64, 512, 512), (16, 512, 256)]:
        try:
            fn, a = make_tp_gateup(mesh, cu_tp, su_tp, gu_tp, gs128, 128,
                                   tiles=TileSizes(tile_m=tm, tile_k=tk,
                                                   tile_n=tn))
            us = dt(fn, a)
            print(f"  tm={tm:3d} tk={tk:4d} tn={tn:3d}: {us:7.1f}us")
        except Exception as ex:
            print(f"  tm={tm} tk={tk} tn={tn}: FAIL {str(ex)[:70]}")

    # ===== final table =====
    print("\n" + "=" * 96)
    hdr = ("variant", "M", "gate_us", "down_us", "step_ms", "rl2_gu", "rl2_dn",
           "TPOT_ms", "tok/s")
    print("{:7} {:>4} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8}".format(*hdr))
    print("-" * 96)
    for v, M, g, d, st, rg, rd, tp, ts in results:
        print(f"{v:7} {M:>4} {g:>9.1f} {d:>9.1f} {st:>9.2f} {rg:>9.2e} "
              f"{rd:>9.2e} {tp:>9.2f} {ts:>8.1f}")
    print("=" * 96)


if __name__ == "__main__":
    main()
