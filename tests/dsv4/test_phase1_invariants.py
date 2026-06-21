"""Task 14 -- Phase-1 reference-free INVARIANT tests (test-only, no prod code).

Four correctness gates on the Task-12 synthetic mini-model e2e forward
(``build_mini_model`` / ``run_mini_forward``) running on the REAL v6e-8
production mesh (TP=8 = 2-way attn-DP x 4-way model, + EP). No mocks, no
reference model -- these pin self-contained mathematical invariants:

1. RoPE per-pair L2-norm preservation (``apply_rope_gptj_last_k``). A rotation
   preserves the magnitude of each (even, odd) frequency pair. YaRN-safe: for
   this config YaRN is rotation x mscale with mscale==1.0, so the plain
   ``build_cos_sin_cache`` is internally consistent for this norm test (no YaRN
   cache needed -- the invariant is about the rotation alone).

2. Causality within the sliding window (THE load-bearing correctness gate).
   With <=128 tokens the SWA window (128) covers the whole sequence, so the
   model is fully causal: perturbing the LAST token's id must NOT change any
   earlier token's logits. Each forward uses a FRESH KV-cache pool
   (``run_mini_forward`` -> ``build_mini_forward_args`` allocates a new
   ``jnp.zeros`` pool every call), so the two forwards are independent. A
   determinism sanity check (same inputs + fresh cache -> bit-identical logits)
   precedes the causality assert so a real causal/SWA-masking failure is not
   masked by nondeterminism.

3. MoE top-k: exactly ``num_experts_per_tok`` DISTINCT experts selected per
   token. Pinned via the validated Task-7 router reference
   (``moe_ref.topk_softplus_sqrt`` = sqrtsoftplus + noaux_tc), the same router
   the real model uses. The count is READ from ``make_dsv4_mini_config()`` --
   never hardcoded.

4. Logits output sharding: assert the REAL committed jax spec off the
   ``compute_logits_fn`` output. Read (not guessed) from
   ``vllm_model_wrapper.jit_compute_logits_func`` -> the lm-head/logits
   ``out_shardings`` is ``PartitionSpec(MLP_DATA, MLP_TENSOR)``. Fully-ranked
   (logits is 2D ``[T, vocab]`` so the 2-element spec is already fully ranked --
   no trailing ``None`` needed) to avoid the Task-2 oracle footgun.
"""
import jax
import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from tests.dsv4 import moe_ref, torch_ref
from tests.dsv4.build_mini_model import (build_mini_forward_args,
                                         build_mini_model, run_mini_forward)
from tests.dsv4.mesh_fixtures import (assert_sharded_like,  # noqa
                                      assert_threefry_partitionable, dsv4_mesh)
from tests.dsv4.mini_config import make_dsv4_mini_config
from tpu_inference.layers.common.sharding import ShardingAxisName


# --------------------------------------------------------------------------- #
# 1. RoPE per-pair L2-norm preservation (reference-free, CPU-only; YaRN-safe).
# --------------------------------------------------------------------------- #
def test_rope_preserves_pair_l2_norm():
    rope_dim = 64
    cache = torch_ref.build_cos_sin_cache(rope_dim, 256, theta=10000.0)
    x = torch.randn(7, 2, 512)
    pos = torch.arange(7)
    rot = torch_ref.apply_rope_gptj_last_k(x, pos, cache, rope_dim).float()
    # per even/odd pair magnitude is preserved by rotation
    a = x[..., -rope_dim:].float()
    pa = (a[..., ::2]**2 + a[..., 1::2]**2)
    pb = (rot[..., -rope_dim:][..., ::2]**2 + rot[..., -rope_dim:][..., 1::2]**2)
    np.testing.assert_allclose(pa.numpy(), pb.numpy(), rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- #
# 2. Causality within the SWA window (the load-bearing gate; real forward).
# --------------------------------------------------------------------------- #
def test_causality_within_window(dsv4_mesh):
    assert_threefry_partitionable()
    with jax.set_mesh(dsv4_mesh):
        model, cfg = build_mini_model(dsv4_mesh)
        n = 32  # <=128 -> the SWA window covers the whole sequence (fully causal)
        ids = torch.arange(n, dtype=torch.int32) % 1280
        pos = torch.arange(n, dtype=torch.int32)

        # Determinism sanity check FIRST: two forwards with identical inputs +
        # fresh KV caches (run_mini_forward allocates a fresh pool per call) must
        # produce bit-identical logits. If this fails, the causality comparison
        # below is meaningless -- so gate on it before asserting causality.
        base = run_mini_forward(model, cfg, ids, pos).float()
        base_again = run_mini_forward(model, cfg, ids, pos).float()
        np.testing.assert_array_equal(
            base.numpy(), base_again.numpy(),
            err_msg="run_mini_forward is non-deterministic across calls with "
                    "identical inputs + fresh caches -- the causality test is "
                    "meaningless until this is fixed (shared/leaking KV cache?).")

        # Perturb the LAST token's id; earlier logits must be UNCHANGED.
        ids2 = ids.clone()
        ids2[-1] = (ids2[-1] + 1) % 1280
        pert = run_mini_forward(model, cfg, ids2, pos).float()

    # tokens 0..n-2 must be byte-for-byte identical aside from numerical noise;
    # do NOT relax this tolerance -- a failure here is a real causal/SWA masking
    # bug (debug via the Task-11 dense parity + systematic-debugging).
    np.testing.assert_allclose(base[:-1].numpy(), pert[:-1].numpy(),
                               rtol=1e-3, atol=1e-3)
    # And perturbing the last token MUST change the last token's own logits
    # (otherwise the test could pass trivially on a dead forward).
    assert not np.allclose(base[-1].numpy(), pert[-1].numpy(), rtol=1e-3,
                           atol=1e-3), "last-token logits unchanged by its own " \
                                       "id perturbation -- forward is degenerate"


# --------------------------------------------------------------------------- #
# 3. MoE top-k: exactly num_experts_per_tok distinct experts per token.
# --------------------------------------------------------------------------- #
def test_moe_router_selects_exactly_topk():
    cfg = make_dsv4_mini_config()
    topk = cfg["num_experts_per_tok"]          # READ from config -- not hardcoded
    n_exp = cfg["n_routed_experts"]
    rsf = cfg["routed_scaling_factor"]
    assert topk <= n_exp, "num_experts_per_tok must not exceed n_routed_experts"

    torch.manual_seed(0)
    n_tok = 5
    gating = torch.randn(n_tok, n_exp)
    bias = torch.randn(n_exp)  # noaux_tc e_score_correction_bias (selection only)
    weights, indices = moe_ref.topk_softplus_sqrt(
        gating, bias, topk=topk, renormalize=True, routed_scaling_factor=rsf)

    # Shapes: exactly `topk` experts selected per token.
    assert indices.shape == (n_tok, topk)
    assert weights.shape == (n_tok, topk)
    # Exactly `topk` DISTINCT experts per token, each a valid expert id, and a
    # nonzero combine weight for each of the `topk` slots.
    for r in range(n_tok):
        row = indices[r].tolist()
        assert len(set(row)) == topk, (
            f"token {r} selected {len(set(row))} distinct experts, want {topk}")
        assert all(0 <= e < n_exp for e in row), f"expert id OOR: {row}"
        assert int((weights[r] != 0).sum()) == topk, (
            f"token {r} has {(weights[r] != 0).sum()} nonzero weights, want "
            f"{topk}")


# --------------------------------------------------------------------------- #
# 4. Logits output sharding: assert the REAL committed spec.
# --------------------------------------------------------------------------- #
def test_logits_sharding_spec(dsv4_mesh):
    assert_threefry_partitionable()
    with jax.set_mesh(dsv4_mesh):
        model, cfg = build_mini_model(dsv4_mesh)
        n = 16
        ids = torch.arange(n, dtype=torch.int32) % 1280
        pos = torch.arange(n, dtype=torch.int32)
        # run_mini_forward returns a HOST torch.Tensor (torch.from_numpy) with no
        # sharding, so we drive the same forward inline to capture the COMMITTED
        # jax logits array straight off compute_logits_fn (before device_get).
        model_fn, args, _ = build_mini_forward_args(model, cfg, ids, pos)
        _, hidden_states, *_ = model_fn(*args)
        logits = model.compute_logits_fn(model.state_leaves, hidden_states, None)
        assert isinstance(logits, jax.Array), (
            f"expected a committed jax.Array off compute_logits_fn, got "
            f"{type(logits)}")
        # Print the REAL committed sharding ONCE for the record, then assert it.
        print(f"\n[task-14] real logits.sharding = {logits.sharding!r}")
    # REAL spec read (not guessed) from vllm_model_wrapper.jit_compute_logits_func
    # out_shardings = PartitionSpec(MLP_DATA, MLP_TENSOR). Logits is 2D
    # [T, vocab] so this 2-element spec is fully ranked (no trailing None).
    assert_sharded_like(
        logits, dsv4_mesh,
        P(ShardingAxisName.MLP_DATA, ShardingAxisName.MLP_TENSOR))
