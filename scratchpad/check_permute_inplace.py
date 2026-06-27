"""Numeric correctness check for the low-transient in-place ctx move.

Verifies the new masked-scatter move op is BIT-IDENTICAL to the old
full-gather `ctx_buf[gather_src]` for: identity, single move, swap, 3-cycle.

Runs on CPU/host with tiny shapes -- no TPU / model needed.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

# ---- OLD reference: full leading-axis gather --------------------------------


def old_permute(ctx_buf, gather_src):
    return np.asarray(ctx_buf)[np.asarray(gather_src)]


# ---- NEW op: gather ONLY the K padded moved rows, scatter in place ----------
# Signature mirrors what prepare_inputs will pass: dst_slots/src_slots are
# (K,) int32 padded with identity no-op pairs (dst==src==some fixed slot).


@functools.partial(jax.jit, static_argnums=(0, ), donate_argnums=(1, ))
def _move_ctx_rows(self_unused, ctx_buf, dst_slots, src_slots):
    # RHS gather reads the OLD buffer (only K rows); .at[].set writes them to
    # dst. JAX functional semantics: RHS value computed from input array before
    # the in-place store, so swaps/cycles are safe even with donation.
    moved_rows = ctx_buf[src_slots]  # (K, buf_len, D) -- only K rows
    return ctx_buf.at[dst_slots].set(moved_rows)


def build_pairs(gather_src, K):
    """Host-side: build padded (dst_slots, src_slots) from a full gather_src.

    Moved slot i (gather_src[i] != i) -> real pair (dst=i, src=gather_src[i]).
    Pad to fixed K with self-copies of an UNMOVED slot p (gather_src[p]==p):
    pair (dst=p, src=p) gathers ctx_buf[p] (its CURRENT value) and writes it
    back -- a true no-op. Crucially p is NOT a real dst, so the padded dst
    indices never collide with a real dst -> no duplicate-index scatter
    ambiguity. An unmoved slot is guaranteed to exist whenever there are fewer
    than N moves; the K bucket is chosen >= the real move count, and the FULL-
    permutation worst case uses the K==N bucket where no padding is needed.
    """
    N = len(gather_src)
    dst, src = [], []
    for i in range(N):
        if gather_src[i] != i:
            dst.append(i)
            src.append(int(gather_src[i]))
    assert len(dst) <= K, f"more moves ({len(dst)}) than bucket K={K}"
    n_pad = K - len(dst)
    if n_pad > 0:
        # find an unmoved slot to use as the idempotent pad target
        pad_slot = None
        for i in range(N):
            if gather_src[i] == i:
                pad_slot = i
                break
        assert pad_slot is not None, "no unmoved slot but padding requested"
        dst.extend([pad_slot] * n_pad)
        src.extend([pad_slot] * n_pad)
    return np.asarray(dst, dtype=np.int32), np.asarray(src, dtype=np.int32)


def run_case(name, N, buf_len, D, gather_src, K):
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    base = rng.standard_normal((N, buf_len, D)).astype(np.float32)
    ctx = jnp.asarray(base, dtype=jnp.bfloat16)

    expected = old_permute(np.asarray(ctx), gather_src)  # bf16 values

    dst_slots, src_slots = build_pairs(gather_src, K)
    out = _move_ctx_rows(None, ctx, jnp.asarray(dst_slots),
                         jnp.asarray(src_slots))
    out_np = np.asarray(out)

    ok = np.array_equal(out_np, np.asarray(expected))
    # also confirm bit-identical view (bf16 raw bits)
    bits_ok = np.array_equal(
        np.asarray(out).view(np.uint16) if out_np.dtype == np.uint16 else
        jax.device_get(out.view(jnp.uint16)),
        jax.device_get(jnp.asarray(expected).view(jnp.uint16)))
    print(f"[{'PASS' if ok and bits_ok else 'FAIL'}] {name}: "
          f"value_equal={ok} bits_equal={bits_ok} "
          f"(N={N}, K={K}, moves={int((gather_src != np.arange(N)).sum())})")
    return ok and bits_ok


def main():
    N, buf_len, D = 6, 4, 3
    K = N  # worst-case bucket for the test
    results = []

    # identity
    gs = np.arange(N, dtype=np.int32)
    results.append(run_case("identity", N, buf_len, D, gs, K))

    # single move: slot 2 now holds what was in slot 5 (5 untouched elsewhere)
    gs = np.arange(N, dtype=np.int32)
    gs[2] = 5
    results.append(run_case("single_move", N, buf_len, D, gs, K))

    # pure swap i<->j: dst1=1<-src=4, dst=4<-src=1
    gs = np.arange(N, dtype=np.int32)
    gs[1], gs[4] = 4, 1
    results.append(run_case("swap", N, buf_len, D, gs, K))

    # 3-cycle a->b->c->a : slot0 gets slot2, slot2 gets slot3, slot3 gets slot0
    gs = np.arange(N, dtype=np.int32)
    gs[0], gs[2], gs[3] = 2, 3, 0
    results.append(run_case("three_cycle", N, buf_len, D, gs, K))

    # cycle that touches slot 0 as a padding-collision stress (slot0 moved AND
    # padding pairs are (0,0)): ensure trailing identity pad does not corrupt.
    gs = np.arange(N, dtype=np.int32)
    gs[0], gs[1] = 1, 0  # swap 0<->1, plus K padding (0,0) no-ops after
    results.append(run_case("swap_touch_slot0_with_pad", N, buf_len, D, gs, K))

    # smaller bucket than N, few moves padded up to K=3
    gs = np.arange(N, dtype=np.int32)
    gs[2] = 5
    results.append(run_case("single_move_smallK", N, buf_len, D, gs, K=3))

    # FULL permutation (every slot moves) -> no unmoved slot; K must == N so no
    # padding is needed. Worst-case condense.
    perm = np.array([1, 2, 3, 4, 5, 0], dtype=np.int32)  # full 6-cycle
    results.append(run_case("full_permutation_6cycle", N, buf_len, D, perm,
                            K=N))

    # condense-like: low slots finished, high slots shift down (many moves,
    # but a couple of high slots become unmoved/identity tails)
    gs = np.array([3, 4, 5, 3, 4, 5], dtype=np.int32)
    # make slots 3,4,5 identity tails would duplicate srcs; use a realistic
    # compaction: req at old slot 3->0, 4->1, 5->2, and 3,4,5 are new (identity)
    gs = np.array([3, 4, 5, 3, 4, 5], dtype=np.int32)
    gs[3], gs[4], gs[5] = 3, 4, 5  # tails identity
    results.append(run_case("condense_shift_down", N, buf_len, D, gs, K=N))

    print()
    if all(results):
        print("ALL CASES PASS")
    else:
        print("SOME CASES FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
