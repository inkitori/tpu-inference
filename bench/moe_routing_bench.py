#!/usr/bin/env python3
"""Microbenchmark for the MoE routing token-grouping sort (F8 candidate).

Isolates the two `jnp.argsort` calls in `_process_tokens_locally`
(tpu_inference/layers/common/fused_moe_gmm.py, ~L619-654):

  argsort#1: topk_argsort_indices = argsort(topk_indices_flat)   # stable group-by-expert
  argsort#2: argsort(topk_argsort_indices)                       # inverse perm (revert)

and proposes two replacements:

  3a (exact, O(N)): replace argsort#2 with a scatter-based inverse permutation.
  3b (stretch):     replace argsort#1 with a stable counting sort (range E).

Goal: decide whether the sort cost is meaningful vs the gmm1+gmm2 pair
(~400-700us per MoE layer at the same M_seqs) and, if so, optimize the parts
that are both EXACT (integer-permutation match) and faster.

Run via the TPU env wrapper:
    ~/tpu-tooling/tpu-env.sh python bench/moe_routing_bench.py
    ~/tpu-tooling/tpu-env.sh python bench/moe_routing_bench.py --quick
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

# ----------------------------------------------------------------------------
# Config (matches realistic decode-step shapes)
# ----------------------------------------------------------------------------
M_SEQS = [8, 16, 32, 64, 128]
M_SEQS_QUICK = [8, 64]
REPS = 50
OUTER_ITERS = 7
TOPK = 8
E = 192
SEED = 0


# ----------------------------------------------------------------------------
# Timing harness (single-dispatch scan-of-reps; min over outer iters)
#
# The function under test maps int32[N] -> int32[N] (a permutation). To defeat
# LICM/DCE we thread an int32 carry through the scan: each rep perturbs the
# input by a carry-dependent permutation (a roll by 1 element repeatedly is
# cheap and shape-preserving) and folds a scalar of the output back. The roll
# keeps it a valid index array in [0,E) since we re-take mod E.
# ----------------------------------------------------------------------------
def make_timed(call_once, reps=REPS):
    def _scalar_dep(out):
        # Reduce any output (array or tuple of arrays) to a scalar that depends
        # on its values. Always 0 (% 1) so it never changes the input range.
        leaves = jax.tree_util.tree_leaves(out)
        s = sum(jnp.sum(leaf) for leaf in leaves)
        return s % 1

    def fused(x):
        def body(carry, _):
            out = call_once(carry)
            # carry-dependent, cheap, shape/range-preserving feedback:
            # rotate the input by 1 and add a scalar derived from out (always 0),
            # keeping carry a valid expert-id array in [0,E).
            eps = _scalar_dep(out).astype(carry.dtype)
            nxt = jnp.roll(carry, 1) + eps
            return nxt, None

        carry, _ = jax.lax.scan(body, x, jnp.arange(reps))
        return call_once(carry)

    return jax.jit(fused)


def time_call(call_once, x, reps=REPS, outer=OUTER_ITERS):
    """Return per-call microseconds (min over `outer` timed iters)."""
    timed = make_timed(call_once, reps)
    jax.block_until_ready(timed(x))
    jax.block_until_ready(timed(x))
    best = float("inf")
    for _ in range(outer):
        t0 = time.perf_counter()
        out = timed(x)
        jax.block_until_ready(out)
        dt = time.perf_counter() - t0
        best = min(best, dt / reps)
    return best * 1e6  # microseconds


# ----------------------------------------------------------------------------
# The routing block (baseline + optimized variants)
# ----------------------------------------------------------------------------
def baseline_full(num_tokens_local):
    """Replicates the full current logic block; returns the two outputs that
    feed the rest of the function: token_indices_sorted, revert_indices."""
    token_indices = jnp.arange(num_tokens_local, dtype=jnp.int32).repeat(TOPK)

    def fn(topk_indices_flat):
        topk_argsort_indices = jnp.argsort(topk_indices_flat)
        token_indices_sorted = token_indices[topk_argsort_indices]
        group_sizes_local = jax.nn.one_hot(
            topk_indices_flat, E, dtype=jnp.int32).sum(axis=0)
        topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)
        return token_indices_sorted, group_sizes_local, topk_argsort_revert_indices

    return fn


def opt_full_3a(num_tokens_local):
    """argsort#1 kept; argsort#2 replaced by scatter-based inverse perm."""
    token_indices = jnp.arange(num_tokens_local, dtype=jnp.int32).repeat(TOPK)

    def fn(topk_indices_flat):
        N = topk_indices_flat.shape[0]
        topk_argsort_indices = jnp.argsort(topk_indices_flat)
        token_indices_sorted = token_indices[topk_argsort_indices]
        group_sizes_local = jax.nn.one_hot(
            topk_indices_flat, E, dtype=jnp.int32).sum(axis=0)
        topk_argsort_revert_indices = jnp.zeros(N, jnp.int32).at[
            topk_argsort_indices].set(jnp.arange(N, dtype=jnp.int32))
        return token_indices_sorted, group_sizes_local, topk_argsort_revert_indices

    return fn


def opt_full_3ab(num_tokens_local):
    """Both: argsort#1 -> stable counting sort, argsort#2 -> scatter inverse."""
    token_indices = jnp.arange(num_tokens_local, dtype=jnp.int32).repeat(TOPK)

    def fn(topk_indices_flat):
        N = topk_indices_flat.shape[0]
        onehot = jax.nn.one_hot(topk_indices_flat, E, dtype=jnp.int32)  # [N,E]
        group_sizes_local = onehot.sum(axis=0)
        offsets = jnp.cumsum(group_sizes_local) - group_sizes_local  # [E]
        rank_within = (jnp.cumsum(onehot, axis=0) - onehot)[
            jnp.arange(N), topk_indices_flat]  # exclusive prefix per element
        dest = offsets[topk_indices_flat] + rank_within  # [N]
        # stable sort order = inverse of dest
        topk_argsort_indices = jnp.zeros(N, jnp.int32).at[dest].set(
            jnp.arange(N, dtype=jnp.int32))
        token_indices_sorted = token_indices[topk_argsort_indices]
        topk_argsort_revert_indices = jnp.zeros(N, jnp.int32).at[
            topk_argsort_indices].set(jnp.arange(N, dtype=jnp.int32))
        return token_indices_sorted, group_sizes_local, topk_argsort_revert_indices

    return fn


# Isolated single-sort closures (return int array so scan carry stays int).
def only_argsort1():
    def fn(topk_indices_flat):
        return jnp.argsort(topk_indices_flat).astype(jnp.int32)

    return fn


def only_argsort2():
    def fn(topk_indices_flat):
        # cost of argsort-of-argsort: feed argsort#1 then argsort it.
        a = jnp.argsort(topk_indices_flat)
        return jnp.argsort(a).astype(jnp.int32)

    return fn


def only_scatter_inverse():
    def fn(topk_indices_flat):
        N = topk_indices_flat.shape[0]
        a = jnp.argsort(topk_indices_flat)
        return jnp.zeros(N, jnp.int32).at[a].set(jnp.arange(N, dtype=jnp.int32))

    return fn


# ----------------------------------------------------------------------------
# Correctness: compare optimized outputs to golden EXACTLY (int perms).
# ----------------------------------------------------------------------------
def check_exact(num_tokens_local, x):
    base = jax.jit(baseline_full(num_tokens_local))
    g_tis, g_gs, g_rev = (np.asarray(v) for v in base(x))
    results = {}
    for name, builder in [("3a", opt_full_3a), ("3ab", opt_full_3ab)]:
        f = jax.jit(builder(num_tokens_local))
        tis, gs, rev = (np.asarray(v) for v in f(x))
        ok = (np.array_equal(tis, g_tis) and np.array_equal(gs, g_gs)
              and np.array_equal(rev, g_rev))
        results[name] = ok
    return results


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------
def run_sweep(m_seqs_list):
    rng = np.random.default_rng(SEED)
    rows = []
    correctness = {}
    for m_seqs in m_seqs_list:
        N = m_seqs * TOPK
        x_np = rng.integers(0, E, size=N, dtype=np.int32)
        x = jnp.asarray(x_np)

        full_us = time_call(baseline_full(m_seqs), x)
        as1_us = time_call(only_argsort1(), x)
        as2_us = time_call(only_argsort2(), x)
        opt3a_us = time_call(opt_full_3a(m_seqs), x)
        opt3ab_us = time_call(opt_full_3ab(m_seqs), x)
        scat_us = time_call(only_scatter_inverse(), x)

        correctness[m_seqs] = check_exact(m_seqs, x)
        rows.append(dict(m_seqs=m_seqs, N=N, full=full_us, as1=as1_us,
                         as2=as2_us, scat=scat_us, opt3a=opt3a_us,
                         opt3ab=opt3ab_us))
        print(f"  M_seqs={m_seqs:>4} N={N:>5}  full={full_us:8.2f}us  "
              f"as1={as1_us:7.2f}  as2={as2_us:7.2f}  scat={scat_us:7.2f}  "
              f"3a={opt3a_us:7.2f}  3ab={opt3ab_us:7.2f}  "
              f"exact={correctness[m_seqs]}", flush=True)
    return rows, correctness


def print_table(rows):
    print("\n### Baseline + variants (per-call microseconds)\n")
    print("| M_seqs | N | full_block | argsort1 | argsort2 | scatter_inv | "
          "opt_3a(full) | opt_3ab(full) |")
    print("|-------:|--:|-----------:|---------:|---------:|------------:|"
          "-------------:|--------------:|")
    for r in rows:
        print(f"| {r['m_seqs']} | {r['N']} | {r['full']:.2f} | {r['as1']:.2f} "
              f"| {r['as2']:.2f} | {r['scat']:.2f} | {r['opt3a']:.2f} | "
              f"{r['opt3ab']:.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    m_seqs_list = M_SEQS_QUICK if args.quick else M_SEQS

    dev = jax.devices()[0]
    print("=" * 78)
    print("MoE routing token-grouping sort microbenchmark (F8 candidate)")
    print(f"device={dev.device_kind} platform={jax.default_backend()} "
          f"| E={E} topk={TOPK} | reps={REPS} outer={OUTER_ITERS}")
    print("=" * 78)

    rows, correctness = run_sweep(m_seqs_list)
    print_table(rows)

    all_exact = all(c["3a"] for c in correctness.values())
    all_exact_3ab = all(c["3ab"] for c in correctness.values())
    print(f"\n3a exact-match all shapes:  {all_exact}")
    print(f"3ab exact-match all shapes: {all_exact_3ab}")


if __name__ == "__main__":
    main()
