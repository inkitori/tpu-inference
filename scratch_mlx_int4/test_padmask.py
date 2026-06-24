"""CPU logic test for the MoE padding-row routing mask.

Run: JAX_PLATFORMS=cpu /home/enyouki/vllm_env/bin/python scratch_mlx_int4/test_padmask.py

Verifies, for a batch=1 decode shape (num_tokens=16 padded, 1 real token, top_k=8,
n_experts=192):
  (a) the real row [0]'s topk_indices/topk_weights are IDENTICAL with vs without mask,
  (b) distinct active experts WITH mask <= top_k + 1, WITHOUT mask >> top_k.
"""
import jax
import jax.numpy as jnp

from tpu_inference.layers.common.fused_moe_gmm import compute_moe_routing

NUM_TOKENS = 16
NUM_ACTUAL = 1
TOPK = 8
N_EXPERTS = 192


def group_sizes_from_indices(topk_indices):
    flat = topk_indices.flatten()
    return jax.nn.one_hot(flat, N_EXPERTS, dtype=jnp.int32).sum(axis=0)


def distinct_active(topk_indices):
    gs = group_sizes_from_indices(topk_indices)
    return int((gs > 0).sum())


def main():
    key = jax.random.PRNGKey(0)
    # Random gating logits standing in for garbage padding-row routing.
    gating = jax.random.normal(key, (NUM_TOKENS, N_EXPERTS), dtype=jnp.float32)

    common = dict(scoring_fn="softmax", renormalize=True,
                  e_score_correction_bias=None, routed_scaling_factor=1.0)

    w_no, idx_no = compute_moe_routing(gating, TOPK, num_actual_tokens=None,
                                       **common)
    w_yes, idx_yes = compute_moe_routing(
        gating, TOPK, num_actual_tokens=jnp.asarray(NUM_ACTUAL, dtype=jnp.int32),
        **common)

    da_no = distinct_active(idx_no)
    da_yes = distinct_active(idx_yes)

    print(f"distinct active experts WITHOUT mask: {da_no}")
    print(f"distinct active experts WITH    mask: {da_yes}")
    print(f"row0 idx (no mask):  {idx_no[0].tolist()}")
    print(f"row0 idx (mask):     {idx_yes[0].tolist()}")
    print(f"padding row1 idx (no mask): {idx_no[1].tolist()}")
    print(f"padding row1 idx (mask):    {idx_yes[1].tolist()}")
    print(f"padding row1 w (mask) sum:  {float(w_yes[1].sum())}")

    # (a) real rows [0:NUM_ACTUAL) are byte-identical
    assert jnp.array_equal(idx_no[:NUM_ACTUAL], idx_yes[:NUM_ACTUAL]), \
        "real-row topk_indices changed!"
    assert jnp.array_equal(w_no[:NUM_ACTUAL], w_yes[:NUM_ACTUAL]), \
        "real-row topk_weights changed!"

    # padding rows forced to expert 0 with zero weight
    assert jnp.all(idx_yes[NUM_ACTUAL:] == 0), "padding rows not forced to expert 0"
    assert jnp.all(w_yes[NUM_ACTUAL:] == 0.0), "padding rows weight not zeroed"

    # (b) active-expert reduction
    assert da_yes <= TOPK + 1, f"masked distinct active {da_yes} > {TOPK + 1}"
    assert da_no > TOPK, f"unmasked distinct active {da_no} not >> {TOPK}"

    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
