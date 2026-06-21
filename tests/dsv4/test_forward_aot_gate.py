"""AOT compile gate: the full synthetic mini-forward must COMPILE on v6e.

WHAT THIS PROVES (read carefully -- the original brief's premise was WRONG):
  Task 4 set ``REQUANTIZED_BLOCK_SIZE = 32`` so the FP4 experts route to
  ``gmm_v2``'s dequant-in-VMEM path (``should_dequantize_before_matmul``: 32 < the
  v6e ``mxu_column_size`` 256 -> True). BUT that in-Mosaic path ITSELF does NOT
  lower on v6e / jax 0.10.1 / libtpu 0.0.41: the in-kernel ``float4_e2m1fn -> bf16``
  convert (``gmm_v2.py:402``) dies in Mosaic (no test ever validated float4 on
  pre-v7 -- ``gmm_test.py`` skips "Expect TPUv7+"). So the block-size fix ALONE
  does NOT make the GMM compile on v6e.

  What makes the forward compile on v6e is the Task-12 WORKAROUND in
  ``fused_moe_gmm.gmm_wrapper``: for native ``float4_e2m1fn`` experts on a pre-v7
  TPU it dequantizes the rhs to bf16 in XLA *outside* the kernel
  (``_dequantize_fp4_rhs_to_bf16``) and takes ``gmm_v2``'s unquantized bf16 x bf16
  path -- NO float4 op ever reaches Mosaic.

  Therefore this AOT gate proves: **the full mini-forward (WITH the Task-12 XLA
  dequant workaround in place) compiles on v6e.** It does NOT prove the in-Mosaic
  FP4 path compiles -- that path remains v7+-only.

The gate uses ``jax.jit(model_fn).lower(*args).compile()`` -> a real
``jax.stages.Compiled`` (run-free; ``.compile()`` is what runs the Mosaic backend,
``.lower()`` alone only serializes). The model_fn args / KV pool / AttentionMetadata
are exactly the ones Task-12's ``run_mini_forward`` builds (the KV pool is a
REPLICATED ``P()`` uint8/640 pool, NOT ``BATCH`` -- the brief was wrong on that).

This file also carries two pinned-here Minors from earlier tasks:
  * (T4) Assert the REAL ``gmm_v2.InputConfigs.should_dequantize_before_matmul``
    property for the FP4 experts (not a mirror of ``32 < 256``), and that block-32
    divides the real expert K dims.
  * (T12) Numeric parity for ``_dequantize_fp4_rhs_to_bf16`` against an INDEPENDENT
    reference dequant -- the only un-pinned numeric surface of the servability
    workaround.

No full-model load: synthetic small-config weights in the real quant formats only.
"""
import jax
import pytest
import torch

from tests.dsv4.build_mini_model import (build_mini_forward_args,
                                         build_mini_model)
from tests.dsv4.mesh_fixtures import dsv4_mesh  # noqa: F401


def test_full_mini_forward_compiles_on_v6e(dsv4_mesh):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("Mosaic compile is TPU-only")
    with jax.set_mesh(dsv4_mesh):
        model, vllm_config = build_mini_model(dsv4_mesh)
        compiled = _aot_compile_mini_forward(model, vllm_config, num_tokens=64)
    assert isinstance(compiled, jax.stages.Compiled), (
        f"expected a real jax.stages.Compiled, got {type(compiled)}")


def _aot_compile_mini_forward(model, vllm_config, num_tokens):
    """AOT-compile (run-free) the full mini-forward on the v6e host.

    ``model.model_fn`` for the vllm/torchax path IS already a top-level
    ``jax.jit`` object (``VllmModelWrapper.jit_step_func`` ->
    ``step_fun_with_options``, jitted with ``compiler_options`` +
    ``static_argnames=(layer_name_to_kvcache_index, is_first_rank,
    is_last_rank)``). Task-12 already drives it to FINITE logits, so a real
    ``Compiled`` IS reachable. So we call ``.lower(*args).compile()`` on it
    DIRECTLY against the EXACT concrete args Task-12 builds -- NO outer ``jax.jit``
    wrap. (An outer wrap would make ``step_fun_impl`` a NESTED jit and JAX forbids
    ``compiler_options`` on a nested jit: ``ValueError: compiler_options can only
    be passed to top-level jax.jit``. Lowering the real top-level entrypoint
    directly both avoids that AND exercises the real serving compiler_options.)

    ``.compile()`` is what runs the Mosaic backend (``.lower()`` alone only
    serializes); the returned ``jax.stages.Compiled`` is the run-free proof that
    the full forward -- WITH the Task-12 FP4 XLA-dequant workaround -- compiles on
    v6e. The static args (the kvcache-index tuple + the two rank bools) are passed
    concretely in ``args``; ``jax.jit``'s ``static_argnames`` picks them up.
    """
    n = num_tokens
    input_ids = torch.arange(n, dtype=torch.int32) % 1280
    positions = torch.arange(n, dtype=torch.int32)
    model_fn, args, _ = build_mini_forward_args(model, vllm_config, input_ids,
                                                positions)
    # model_fn is step_fun_with_options: a top-level jax.jit -> lower+compile it.
    lowered = model_fn.lower(*args)
    return lowered.compile()


# --------------------------------------------------------------------------- #
# Minor (from T4): assert the REAL gmm_v2 dispatch property, not a mirror.
# --------------------------------------------------------------------------- #
def test_fp4_experts_dispatch_to_dequantize_before_matmul(dsv4_mesh):
    """The FP4 experts MUST route to gmm_v2's dequant-in-VMEM branch on v6e.

    T4 only MIRRORED the condition (`32 < 256`). Here we invoke the REAL
    ``gmm_v2.InputConfigs.should_dequantize_before_matmul`` property (which reads
    the live ``pltpu.get_tpu_info().mxu_column_size`` off the hardware) on an
    rhs-config built with the REAL requant block size (``mxfp4.REQUANTIZED_BLOCK_SIZE``)
    and the native FP4 dtype, and assert it is True. We also assert block-32
    divides the REAL expert K dims read off the built MoE module (so the requant
    actually tiles cleanly). This is the property the workaround depends on: the
    block size routes to dequant-in-VMEM, which on v6e is exactly the path the
    Task-12 XLA workaround stands in for.
    """
    if jax.devices()[0].platform != "tpu":
        pytest.skip("gmm_v2 / get_tpu_info is TPU-only")
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    import tpu_inference.layers.vllm.quantization.mxfp4 as mxfp4
    from tpu_inference.kernels.megablox import gmm_v2

    block = mxfp4.REQUANTIZED_BLOCK_SIZE
    mxu = pltpu.get_tpu_info().mxu_column_size
    assert mxu == 256, f"expected v6e mxu_column_size 256, got {mxu}"

    # REAL property: build the actual FP4-rhs InputConfigs the GMM would use for
    # the experts (native float4_e2m1fn rhs, block-32 quant, with a scale) and
    # query the genuine dispatch decision -- NOT a hand-written `block < mxu`.
    rhs_cfgs = gmm_v2.InputConfigs(
        quant_dtype=jnp.float4_e2m1fn,
        quant_block_size=block,
        dtype=jnp.bfloat16,
        has_bias=False,
        has_scale=True,
    )
    assert rhs_cfgs.should_dequantize_before_matmul is True, (
        "FP4 experts (block 32 < mxu 256) must dispatch to the dequant-in-VMEM "
        "path that the Task-12 workaround replaces on v6e")

    # Block-32 must divide the REAL expert K dims (read off the built MoE module).
    with jax.set_mesh(dsv4_mesh):
        model, _ = build_mini_model(dsv4_mesh)
    k_dims = _real_expert_k_dims(model)
    assert k_dims, "found no RoutedExperts FP4 weights on the built model"
    for name, k in k_dims:
        assert k % block == 0, (
            f"expert K dim {k} ({name}) not divisible by requant block {block}; "
            "requant would not tile cleanly")


def _real_expert_k_dims(model):
    """Read the contracting (K) dim of every native-FP4 expert weight on the
    built MoE modules (`w13_weight`/`w2_weight` are (E, K, N) float4_e2m1fn after
    the mxfp4 PWAL)."""
    import jax.numpy as jnp
    from torchax.interop import jax_view
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    nn_module = model.model.model
    out = []
    for mod_name, mod in nn_module.named_modules():
        if not isinstance(mod, RoutedExperts):
            continue
        for wname in ("w13_weight", "w2_weight"):
            w = getattr(mod, wname, None)
            if w is None:
                continue
            arr = jax_view(w.data if hasattr(w, "data") else w)
            if arr.dtype == jnp.float4_e2m1fn and arr.ndim == 3:
                out.append((f"{mod_name}.{wname}", int(arr.shape[1])))
    return out


# --------------------------------------------------------------------------- #
# Minor (from T12): NUMERIC parity for the FP4 XLA-dequant workaround helper.
# --------------------------------------------------------------------------- #
def test_dequantize_fp4_rhs_to_bf16_matches_independent_reference():
    """`_dequantize_fp4_rhs_to_bf16` (the only un-pinned numeric surface of the
    servability workaround) must match an INDEPENDENT reference dequant.

    Construction: we build the native ``float4_e2m1fn`` rhs from KNOWN host-side
    e2m1 values (all 16 codes, each exactly representable in both float4 and bf16,
    so the cast is lossless) and an arbitrary float32 block scale. The helper does
    ``rhs.astype(bf16).reshape(E, num_blocks, block, N) * scale.astype(bf16)``.

    The reference is INDEPENDENT: it does NOT re-decode the float4 array via the
    same helper path. Instead it (a) decodes the packed FP4 BYTES through Task-7's
    validated ``break_fp4_e2m1`` (a different e2m1 decoder) and cross-checks they
    equal the known host values, then (b) reproduces the workaround's *bf16*
    multiply from those independently-decoded values. We compare in bf16 (the
    helper multiplies in bf16 to match the kernel), so the tolerance only needs to
    cover bf16 rounding of the scale multiply -- the e2m1 magnitudes themselves are
    exact in bf16.
    """
    import jax.numpy as jnp
    import numpy as np

    from tests.dsv4.torch_ref import break_fp4_e2m1
    from tpu_inference.layers.common.fused_moe_gmm import \
        _dequantize_fp4_rhs_to_bf16

    if jax.devices()[0].platform != "tpu":
        # float4_e2m1fn arrays only materialize on TPU in this stack.
        pytest.skip("float4_e2m1fn is TPU-only here")

    # All 16 e2m1 codes (low nibble first within each packed byte): codes 0..7 are
    # +[0,.5,1,1.5,2,3,4,6]; codes 8..15 are the sign-flipped negatives.
    e2m1_pos = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
    e2m1_table = np.concatenate([e2m1_pos, -e2m1_pos])  # index by 4-bit code

    rng = np.random.default_rng(0)
    # block (8) is a small per-block group (N is even so 2 codes pack per byte).
    E, num_blocks, block, N = 2, 4, 8, 8
    K = num_blocks * block

    # Known float values per (E, K, N), each an exact e2m1 magnitude.
    codes = rng.integers(0, 16, size=(E, K, N)).astype(np.uint8)
    known_vals = e2m1_table[codes].astype(np.float32)        # exact in float4/bf16

    # Native float4_e2m1fn rhs (lossless cast of the known exact values).
    rhs = jnp.asarray(known_vals, dtype=jnp.float32).astype(jnp.float4_e2m1fn)
    # Round-trip must be exact (else the test premise is broken).
    assert np.array_equal(np.asarray(rhs.astype(jnp.float32)), known_vals)

    # float32 block scale (E, num_blocks, 1, N) -- arbitrary positive magnitudes.
    scale = jnp.asarray(rng.uniform(0.3, 3.0, size=(E, num_blocks, 1, N)),
                        dtype=jnp.float32)

    got = _dequantize_fp4_rhs_to_bf16(rhs, scale)          # bf16, shape (E, K, N)
    assert got.dtype == jnp.bfloat16 and got.shape == (E, K, N)

    # --- INDEPENDENT reference --------------------------------------------------
    # (a) Decode the PACKED FP4 bytes through Task-7's break_fp4_e2m1 (a separate
    #     e2m1 decoder) and confirm it agrees with our known host values -- this
    #     pins the e2m1 decode independently of jnp's float4 astype. Pack two
    #     codes/byte, low nibble first, along a flattened (E*K, N) view.
    flat = codes.reshape(E * K, N)
    pairs = flat.reshape(E * K, N // 2, 2)
    packed = (pairs[..., 0] | (pairs[..., 1] << 4)).astype(np.uint8)
    decoded = break_fp4_e2m1(torch.from_numpy(packed),
                             torch.float32).numpy().reshape(E * K, N)
    assert np.array_equal(decoded, known_vals.reshape(E * K, N)), (
        "break_fp4_e2m1 disagrees with the known e2m1 host values")

    # (b) Reproduce the workaround's bf16 multiply from the independently-known
    #     values (NOT via the helper): cast known vals to bf16, multiply by the
    #     bf16 scale broadcast over the block axis, in bf16.
    vals_bf16 = jnp.asarray(known_vals, jnp.float32).astype(jnp.bfloat16)
    vals_bf16 = vals_bf16.reshape(E, num_blocks, block, N)
    scale_bf16 = scale.astype(jnp.bfloat16)
    ref = (vals_bf16 * scale_bf16).reshape(E, K, N)

    got_f = np.asarray(got.astype(jnp.float32))
    ref_f = np.asarray(ref.astype(jnp.float32))
    # Both sides multiply the same bf16 operands -> bit-identical (atol=rtol=0).
    np.testing.assert_allclose(got_f, ref_f, rtol=0.0, atol=0.0)
