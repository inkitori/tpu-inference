import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tpu_inference.layers.common.fused_moe_gmm import (  # noqa: E402
    apply_scoring_fn, compute_moe_routing)


def _np_scores(logits, scoring_fn):
    """Independent transcription of apply_scoring_fn in numpy/float64."""
    x = logits.astype(np.float64)
    if scoring_fn == "softmax":
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)
    elif scoring_fn == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    raise ValueError(scoring_fn)


def _reference_routing(logits, topk, *, scoring_fn, renormalize, bias,
                       routed_scaling_factor):
    """Independent reference replicating vLLM fused_topk_bias / grouped_topk:

      s   = scoring_fn(logits)
      sel = s + bias                 (bias affects selection only)
      idx = topk(sel, k)             (top-k of biased scores)
      w   = gather(s, idx)           (weights from UNBIASED scores)
      if renormalize: w = w / w.sum(-1)   (no 1e-20 epsilon here)
      w = w * routed_scaling_factor
    """
    s = _np_scores(logits, scoring_fn)
    sel = s if bias is None else s + bias[None, :]
    # Largest-k along last axis; argsort descending then take first k.
    order = np.argsort(-sel, axis=-1, kind="stable")
    idx = order[:, :topk]
    w = np.take_along_axis(s, idx, axis=-1)
    if renormalize:
        w = w / w.sum(axis=-1, keepdims=True)
    w = w * routed_scaling_factor
    return w, idx


# Use experts-per-token so that num_tokens*topk is a multiple of 16 is not
# required here (compute_moe_routing has no such assert); keep sizes small.
NUM_TOKENS = 16
NUM_EXPERTS = 64
TOPK = 8


@pytest.mark.parametrize("scoring_fn", ["sigmoid", "softmax"])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.826])
def test_compute_moe_routing_matches_reference(scoring_fn, use_bias,
                                               renormalize,
                                               routed_scaling_factor):
    rng = np.random.default_rng(1234)
    # Continuous random floats so exact ties are practically impossible; this
    # removes the jax-vs-reference top_k tie-break ambiguity.
    logits = rng.standard_normal((NUM_TOKENS, NUM_EXPERTS)).astype(np.float32)
    bias = (rng.standard_normal(NUM_EXPERTS).astype(np.float32)
            if use_bias else None)

    e_bias = None if bias is None else jnp.asarray(bias)
    got_w, got_idx = compute_moe_routing(
        jnp.asarray(logits),
        TOPK,
        scoring_fn=scoring_fn,
        renormalize=renormalize,
        e_score_correction_bias=e_bias,
        routed_scaling_factor=routed_scaling_factor)
    got_w = np.asarray(got_w)
    got_idx = np.asarray(got_idx)

    ref_w, ref_idx = _reference_routing(
        logits,
        TOPK,
        scoring_fn=scoring_fn,
        renormalize=renormalize,
        bias=bias,
        routed_scaling_factor=routed_scaling_factor)

    # Selected experts must match exactly (continuous logits => no ties).
    np.testing.assert_array_equal(got_idx, ref_idx)
    # Weights match within tolerance (allows for the +1e-20 epsilon and f32).
    np.testing.assert_allclose(got_w, ref_w, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("scoring_fn", ["sigmoid", "softmax"])
def test_compute_moe_routing_noop_exact_against_old_path(scoring_fn):
    """Regression: with bias=None and routed_scaling_factor=1.0 the helper is
    bit-identical to the previous plain top_k path (e.g. Qwen3 routing).

    renormalize=False makes the equality exact (no 1e-20 epsilon involved)."""
    rng = np.random.default_rng(7)
    logits = jnp.asarray(
        rng.standard_normal((NUM_TOKENS, NUM_EXPERTS)).astype(np.float32))

    # Old inline path (pre-refactor), bit-for-bit.
    old_scores = apply_scoring_fn(scoring_fn, logits)
    old_weights, old_indices = jax.lax.top_k(old_scores, k=TOPK)

    new_weights, new_indices = compute_moe_routing(
        logits,
        TOPK,
        scoring_fn=scoring_fn,
        renormalize=False,
        e_score_correction_bias=None,
        routed_scaling_factor=1.0)

    # Bit-exact: indices and weights must be identical.
    assert np.array_equal(np.asarray(new_indices), np.asarray(old_indices))
    assert np.array_equal(np.asarray(new_weights), np.asarray(old_weights))


@pytest.mark.parametrize("scoring_fn", ["sigmoid", "softmax"])
def test_compute_moe_routing_noop_with_renormalize(scoring_fn):
    """No-op path with renormalize=True: matches old path up to the 1e-20
    epsilon (tiny tolerance)."""
    rng = np.random.default_rng(11)
    logits = jnp.asarray(
        rng.standard_normal((NUM_TOKENS, NUM_EXPERTS)).astype(np.float32))

    old_scores = apply_scoring_fn(scoring_fn, logits)
    old_weights, old_indices = jax.lax.top_k(old_scores, k=TOPK)
    old_weights = old_weights / (
        old_weights.sum(axis=-1, keepdims=True) + 1e-20)

    new_weights, new_indices = compute_moe_routing(
        logits,
        TOPK,
        scoring_fn=scoring_fn,
        renormalize=True,
        e_score_correction_bias=None,
        routed_scaling_factor=1.0)

    assert np.array_equal(np.asarray(new_indices), np.asarray(old_indices))
    np.testing.assert_allclose(np.asarray(new_weights),
                               np.asarray(old_weights),
                               atol=0,
                               rtol=0)
