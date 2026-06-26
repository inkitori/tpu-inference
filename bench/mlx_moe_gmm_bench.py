#!/usr/bin/env python3
"""Microbenchmark for the int4 MLX MoE grouped-matmul (gmm_v2) path.

Isolates the two per-shard MoE grouped-matmuls used in the Hy3 W4A16 (MLX-style
affine int4) MoE under TP=8, and measures per-call latency + correctness.

Run via the TPU env wrapper:
    ~/tpu-tooling/tpu-env.sh python bench/mlx_moe_gmm_bench.py
    ~/tpu-tooling/tpu-env.sh python bench/mlx_moe_gmm_bench.py --quick

The kernel under test:
    from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2

MLX path contract (verified against gmm_v2.py):
  * W4A16 affine dequant: W[k,n] = rhs_scale[k//gs,0,n]*code[k,n] + rhs_groupbias[k//gs,0,n]
  * maybe_quantize_lhs=False (keep activations bf16, weights int4)
  * pass BOTH rhs_scale and rhs_groupbias
  * lhs rows MUST be grouped (sorted) by expert: group g owns a contiguous
    block of group_sizes[g] rows. This is gmm_v2's contract (no internal gather).
  * num_blocks = K // group_size, group_size = 64.
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2

# ----------------------------------------------------------------------------
# Editable config
# ----------------------------------------------------------------------------
M_SEQS = [8, 16, 32, 64, 128]          # full sweep of sequence counts
M_SEQS_QUICK = [8, 64]                  # --quick subset
REPS = 50                               # inner reps fused into one dispatch
OUTER_ITERS = 5                         # outer timing iters; we take the min
TOPK = 8
E = 192                                 # number of experts
GROUP_SIZE = 64
REL_L2_TOL = 1e-2
SEED = 0

# Per-shard production shapes (TP=8), group_size=64.
#   GMM1 (w13 gate+up): rhs [E, K=4096, N=384], scale/gbias [E, 64, 1, 384]
#   GMM2 (w2 down):     rhs [E, K=192,  N=4096], scale/gbias [E, 3,  1, 4096]
GMM1 = dict(K=4096, N=384)
GMM2 = dict(K=192, N=4096)


# ----------------------------------------------------------------------------
# Timing harness (single-dispatch scan-of-reps; min over outer iters)
# ----------------------------------------------------------------------------
def make_timed(call_once, reps=REPS):
    """Fuse `reps` calls into one jitted scan -> one dispatch per timed run.

    The carry (lhs) is threaded through each iteration so XLA cannot hoist the
    call out of the scan via LICM/CSE: each rep's input depends on the prior
    rep's output. We inject a tiny, output-shape-independent perturbation back
    into lhs so the dependency is real but does not change problem size or blow
    up numerically. `call_once` takes lhs and returns out[M, N]; we fold a
    scalar summary of out back into lhs.
    """
    def fused(lhs):
        def body(carry, _):
            out = call_once(carry)
            # carry-dependent, cheap, shape-preserving feedback:
            # add a tiny scalar derived from out so the next rep depends on this
            # rep's result. Magnitude ~1e-9 keeps activations/rel-error stable.
            eps = (jnp.mean(out).astype(carry.dtype) * jnp.asarray(1e-9, carry.dtype))
            return carry + eps, None
        carry, _ = jax.lax.scan(body, lhs, jnp.arange(reps))
        # return one concrete call's output (depends on the final carry)
        return call_once(carry)
    return jax.jit(fused)


def time_call(call_once, args, reps=REPS, outer=OUTER_ITERS):
    """Return per-call ms (min over `outer` timed iters of a reps-scan)."""
    timed = make_timed(call_once, reps)
    # warmup (compile + a second run to settle)
    jax.block_until_ready(timed(*args))
    jax.block_until_ready(timed(*args))
    best = float("inf")
    for _ in range(outer):
        t0 = time.perf_counter()
        out = timed(*args)
        jax.block_until_ready(out)
        dt = time.perf_counter() - t0
        best = min(best, dt / reps)
    return best * 1e3  # ms


# ----------------------------------------------------------------------------
# group_sizes synthesis
# ----------------------------------------------------------------------------
def group_sizes_imbalanced(rng, total_rows, e=E):
    """Realistic imbalanced routing: multinomial of total_rows over e experts."""
    probs = np.ones(e) / e
    gs = rng.multinomial(total_rows, probs).astype(np.int32)
    return gs


def group_sizes_balanced(total_rows, e=E):
    """As balanced as possible: floor split + remainder spread over first experts."""
    base = total_rows // e
    rem = total_rows - base * e
    gs = np.full(e, base, dtype=np.int32)
    gs[:rem] += 1
    return gs


# ----------------------------------------------------------------------------
# Weight + activation synthesis (rows grouped by expert)
# ----------------------------------------------------------------------------
def make_weights(rng, K, N, e=E, gs=GROUP_SIZE):
    num_blocks = K // gs
    assert K % gs == 0, f"K={K} must be divisible by group_size={gs}"
    codes_np = rng.integers(-8, 8, size=(e, K, N), dtype=np.int32)
    rhs = jnp.asarray(codes_np, dtype=jnp.int4)
    scale_np = rng.uniform(0.005, 0.025, size=(e, num_blocks, 1, N)).astype(np.float32)
    gbias_np = (0.02 * rng.standard_normal((e, num_blocks, 1, N))).astype(np.float32)
    rhs_scale = jnp.asarray(scale_np)
    rhs_gbias = jnp.asarray(gbias_np)
    return rhs, rhs_scale, rhs_gbias, codes_np, scale_np, gbias_np


def make_lhs(rng, total_rows, K):
    """Small bf16 activations; rows are already in expert-grouped order
    (caller concatenates per-group blocks contiguously)."""
    lhs_np = (0.05 * rng.standard_normal((total_rows, K))).astype(np.float32)
    return jnp.asarray(lhs_np, dtype=jnp.bfloat16), lhs_np


def golden_fp32(lhs_np, codes_np, scale_np, gbias_np, group_sizes_np, gs=GROUP_SIZE):
    """fp32 reference. Per expert g, dequant W_g = scale*code + gbias, then
    out_rows_of_g = lhs_rows_of_g @ W_g. Rows are expert-grouped (contiguous)."""
    e = codes_np.shape[0]
    N = codes_np.shape[2]
    K = codes_np.shape[1]
    out = np.zeros((lhs_np.shape[0], N), dtype=np.float32)
    row = 0
    for g in range(e):
        n = int(group_sizes_np[g])
        if n == 0:
            continue
        # dequant W_g[k,n] = scale_g[k//gs,0,n]*code_g[k,n] + gbias_g[k//gs,0,n]
        scale_full = np.repeat(scale_np[g, :, 0, :], gs, axis=0)[:K]  # [K, N]
        gbias_full = np.repeat(gbias_np[g, :, 0, :], gs, axis=0)[:K]  # [K, N]
        Wg = scale_full * codes_np[g].astype(np.float32) + gbias_full  # [K, N]
        out[row:row + n] = lhs_np[row:row + n] @ Wg
        row += n
    return out


def rel_l2(out_bf16, golden):
    o = np.asarray(out_bf16, dtype=np.float32)
    num = np.linalg.norm(o - golden)
    den = np.linalg.norm(golden)
    return float(num / den) if den > 0 else float(num)


# ----------------------------------------------------------------------------
# One GMM build + (optional) correctness + timing
# ----------------------------------------------------------------------------
def build_call(rhs, rhs_scale, rhs_gbias, group_sizes, group_offset):
    """Return a closure call_once(lhs) -> out for the MLX W4A16 path."""
    def call_once(lhs):
        return gmm_v2(
            lhs,
            rhs,
            group_sizes,
            rhs_scale,
            rhs_gbias,
            None,                       # rhs_bias
            group_offset,               # group_offset = [0]
            preferred_element_type=jnp.bfloat16,
            maybe_quantize_lhs=False,
            zero_initialize=False,
        )
    return call_once


def run_one_gmm(rng, shape, m_seqs, gs_np):
    """Build weights/lhs for one gmm shape & distribution; return (ms, relL2)."""
    K, N = shape["K"], shape["N"]
    total_rows = int(gs_np.sum())
    rhs, rhs_scale, rhs_gbias, codes_np, scale_np, gbias_np = make_weights(rng, K, N)
    lhs, lhs_np = make_lhs(rng, total_rows, K)
    group_sizes = jnp.asarray(gs_np, dtype=jnp.int32)
    group_offset = jnp.array([0], dtype=jnp.int32)

    call_once = build_call(rhs, rhs_scale, rhs_gbias, group_sizes, group_offset)

    # correctness
    out = call_once(lhs)
    jax.block_until_ready(out)
    golden = golden_fp32(lhs_np, codes_np, scale_np, gbias_np, gs_np)
    r = rel_l2(out, golden)

    # timing
    ms = time_call(call_once, (lhs,))
    return ms, r


def run_sweep(m_seqs_list, label, imbalanced=True):
    rng = np.random.default_rng(SEED)
    rows = []
    for m_seqs in m_seqs_list:
        total_rows = m_seqs * TOPK
        if imbalanced:
            gs_np = group_sizes_imbalanced(rng, total_rows)
        else:
            gs_np = group_sizes_balanced(total_rows)
        assert int(gs_np.sum()) == total_rows
        g1_ms, g1_r = run_one_gmm(rng, GMM1, m_seqs, gs_np)
        g2_ms, g2_r = run_one_gmm(rng, GMM2, m_seqs, gs_np)
        rows.append((m_seqs, total_rows, g1_ms, g2_ms, g1_r, g2_r))
        print(f"  [{label}] M_seqs={m_seqs:>4} rows={total_rows:>5}  "
              f"gmm1={g1_ms:7.4f}ms gmm2={g2_ms:7.4f}ms  "
              f"relL2 g1={g1_r:.2e} g2={g2_r:.2e}", flush=True)
    return rows


def print_table(rows, title):
    print(f"\n### {title}\n")
    print("| M_seqs | routed_rows | gmm1_ms | gmm2_ms | gmm1_relL2 | gmm2_relL2 |")
    print("|-------:|------------:|--------:|--------:|-----------:|-----------:|")
    for (m, tot, g1, g2, r1, r2) in rows:
        print(f"| {m} | {tot} | {g1:.4f} | {g2:.4f} | {r1:.2e} | {r2:.2e} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="run only M_seqs=8,64")
    ap.add_argument("--balanced", action="store_true",
                    help="also print the balanced distribution")
    args = ap.parse_args()

    m_seqs_list = M_SEQS_QUICK if args.quick else M_SEQS

    dev = jax.devices()[0]
    print("=" * 78)
    print("MLX int4 MoE grouped-matmul (gmm_v2) microbenchmark")
    print(f"device={dev.device_kind} platform={jax.default_backend()} "
          f"| E={E} topk={TOPK} group_size={GROUP_SIZE} TP=8 W4A16-affine "
          f"| reps={REPS} outer={OUTER_ITERS}")
    print(f"GMM1 rhs[E={E},K={GMM1['K']},N={GMM1['N']}]  "
          f"GMM2 rhs[E={E},K={GMM2['K']},N={GMM2['N']}]")
    print("=" * 78)

    print("\nImbalanced (multinomial) routing [HEADLINE]:")
    imb = run_sweep(m_seqs_list, "imbal", imbalanced=True)

    bal = None
    if args.balanced:
        print("\nBalanced routing:")
        bal = run_sweep(m_seqs_list, "bal", imbalanced=False)

    print_table(imb, "Imbalanced (headline)")
    if bal is not None:
        print_table(bal, "Balanced")

    # correctness gate
    worst = max(max(r1, r2) for (_, _, _, _, r1, r2) in imb)
    print(f"\nworst rel_L2 (imbalanced) = {worst:.2e}  tol={REL_L2_TOL:.0e}  "
          f"-> {'PASS' if worst < REL_L2_TOL else 'FAIL'}")


if __name__ == "__main__":
    main()
