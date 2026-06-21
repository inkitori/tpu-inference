"""Independent cross-check of the production MLX dequant primitive against a
standalone NumPy transcription of MLX's *published* affine ``mx.dequantize``
algorithm, run on the REAL ``mlx-community/Qwen3-30B-A3B-4bit`` checkpoint shards.

Why this test exists
--------------------
The only prior correctness gate compared ``mlx_dequantize`` against a synthetic
oracle (``mlx_synthetic._quantize_affine``) written by the same author sharing
the same affine math -- a shared bug would pass both while producing wrong real
outputs. This test breaks that circularity:

  * ``_independent_mlx_dequant`` below was written *from the published MLX spec*
    (https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html
    and .../mlx.core.quantize.html, cross-checked against the DeepWiki
    "Quantization | ml-explore/mlx" page), NOT copied from the production code.
  * The production primitive ``mlx_dequantize`` is imported and called purely as
    a BLACK BOX -- its implementation was not read while authoring this file.
  * The inputs are the genuine quantized weight/scale/bias triples straight from
    the real safetensors shards -- no synthetic oracle anywhere in the loop.

Published MLX affine 4-bit layout (verbatim from the docs):
  * dequant formula:  ``w_i = scale * q_i + bias``
  * 4-bit packing:    8 elements per uint32 along the INPUT dim, the 1st element
                      in the 4 least-significant bits ("low nibble first"),
                      the 2nd in bits 4-7, etc.
  * grouping:         one (scale, bias) pair per ``group_size`` consecutive
                      elements along the input (last) dim.
"""

import glob
import os

import numpy as np
import pytest

# Keep everything on CPU: the production primitive runs through JAX.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

GROUP_SIZE = 64
BITS = 4
PACK_FACTOR = 32 // BITS  # 8 nibbles per uint32

_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3-30B-A3B-4bit/"
    "snapshots/d388dead1515f5e085ef7a0431dd8fadf0886c57"
)

# Representative triples: one attention linear (2D) and one MoE/expert tensor
# (3D, experts stacked on axis 0) so both layouts are exercised.
_ATTN = "model.layers.0.self_attn.q_proj"
_MOE = "model.layers.0.mlp.switch_mlp.gate_proj"


def _independent_mlx_dequant(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
) -> np.ndarray:
    """Standalone NumPy dequant transcribed from MLX's published affine spec.

    ``packed`` has the input dimension packed: 32//bits low-nibble-first values
    per uint32 along the last axis. ``scales``/``biases`` carry one value per
    ``group_size`` input elements along the last axis. Returns the float
    reconstruction with the input dim fully expanded.

    Deliberately written without reference to the production implementation.
    """
    assert packed.dtype == np.uint32, packed.dtype
    pack_factor = 32 // bits
    mask = (1 << bits) - 1  # 0xF for 4 bits

    # --- unpack: low nibble first ---
    # For each uint32 the element at nibble position p lives in bits
    # [p*bits, (p+1)*bits). Build the expanded last axis by right-shifting
    # by p*bits and masking, then interleaving the nibble positions so the
    # original element order is preserved (element 0 = lowest nibble).
    shifts = (np.arange(pack_factor, dtype=np.uint32) * bits)  # [0,4,8,...,28]
    # packed[..., :, None] -> [..., n_words, pack_factor]
    q = (packed[..., None] >> shifts) & np.uint32(mask)
    # Flatten the (word, nibble) pair back into a single input axis so that
    # logical index = word_index * pack_factor + nibble_index.
    q = q.reshape(*packed.shape[:-1], packed.shape[-1] * pack_factor)
    q = q.astype(np.float32)

    in_dim = q.shape[-1]
    assert in_dim % group_size == 0, (in_dim, group_size)
    n_groups = in_dim // group_size
    assert scales.shape[-1] == n_groups, (scales.shape, n_groups)
    assert biases.shape[-1] == n_groups, (biases.shape, n_groups)

    # --- broadcast (scale, bias) across each group of group_size elements ---
    s = np.repeat(scales.astype(np.float32), group_size, axis=-1)
    b = np.repeat(biases.astype(np.float32), group_size, axis=-1)

    # --- affine reconstruction: w = scale * q + bias ---
    return s * q + b


def _shard_for(name: str) -> str:
    """Locate the shard file containing ``name.weight`` via the index map."""
    import json

    idx_path = os.path.join(_SNAPSHOT, "model.safetensors.index.json")
    weight_map = json.load(open(idx_path))["weight_map"]
    return os.path.join(_SNAPSHOT, weight_map[name + ".weight"])


def _load_triple(name: str):
    """Load (packed uint32, scales, biases). scales/biases are bf16 in the
    checkpoint, which NumPy cannot represent, so they are read through torch and
    upcast to float32 -- the SAME bf16 bit pattern is handed to both dequant
    paths, so the upcast introduces no asymmetry."""
    import torch
    from safetensors import safe_open

    shard = _shard_for(name)
    with safe_open(shard, framework="pt") as f:
        packed = f.get_tensor(name + ".weight").numpy()  # uint32, exact
        scales = f.get_tensor(name + ".scales")  # torch bf16
        biases = f.get_tensor(name + ".biases")  # torch bf16
    assert scales.dtype == torch.bfloat16, scales.dtype
    assert biases.dtype == torch.bfloat16, biases.dtype
    scales = scales.to(torch.float32).numpy()
    biases = biases.to(torch.float32).numpy()
    return packed, scales, biases


_HAVE_WEIGHTS = (
    os.path.isdir(_SNAPSHOT)
    and len(glob.glob(os.path.join(_SNAPSHOT, "*.safetensors"))) > 0
)

skip_no_weights = pytest.mark.skipif(
    not _HAVE_WEIGHTS,
    reason=(
        "real MLX checkpoint snapshot not present at %s; skipping independent "
        "real-shard cross-check (CI without weights)" % _SNAPSHOT
    ),
)


def _production_dequant(packed, scales, biases):
    """Call the production primitive as a pure black box (CPU JAX)."""
    import jax.numpy as jnp

    from tpu_inference.layers.common.quantization import mlx_dequantize

    out = mlx_dequantize(
        jnp.asarray(packed),
        jnp.asarray(scales),
        jnp.asarray(biases),
        group_size=GROUP_SIZE,
        bits=BITS,
    )
    # The primitive emits bfloat16 (the model compute dtype). Upcast to fp32 for
    # comparison WITHOUT changing any value: bf16 -> fp32 is exact and lossless.
    return np.asarray(jnp.asarray(out, dtype=jnp.float32))


def _to_bf16_grid(arr: np.ndarray) -> np.ndarray:
    """Round an fp32 array onto the bfloat16 value grid (returned as fp32)."""
    import torch

    # .copy() guarantees a writable buffer (jax-backed arrays are read-only,
    # which otherwise triggers a torch.from_numpy UserWarning).
    return (
        torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))
        .to(torch.bfloat16)
        .to(torch.float32)
        .numpy()
    )


def _compare(name, packed, scales, biases):
    mine = _independent_mlx_dequant(packed, scales, biases)  # fp32 reference
    prod = _production_dequant(packed, scales, biases)  # production output

    assert mine.shape == prod.shape, (name, mine.shape, prod.shape)

    # The production primitive emits the model compute dtype (bfloat16); the
    # independent reference is fp32. The only legitimate difference between a
    # CORRECT dequant and this reference is the final bf16 round-off, so the
    # faithful equivalence test is: production must equal the fp32 reference
    # ROUNDED TO THE bf16 GRID, exactly. A real layout / nibble-order / group-
    # axis / sign bug perturbs whole quantized levels (errors of order 1) and
    # would never collapse onto the bf16 grid -- so this is a strict check, not
    # a loosened tolerance.
    assert prod.dtype == np.float32  # _production_dequant already upcast for us
    prod_was_bf16 = np.array_equal(prod, _to_bf16_grid(prod))

    mine_bf16 = _to_bf16_grid(mine)

    # Primary (strict) assertion: bit-exact on the bf16 grid.
    exact_bf16 = np.array_equal(mine_bf16, prod)

    # Secondary diagnostics: raw fp32-vs-bf16 spread (should be ~bf16 ulp), and
    # a tolerance that a layout bug could never satisfy (bf16 has ~2^-8 rel
    # precision, so 1e-2 rel is generous slack yet catches any structural bug).
    abs_err = np.abs(mine - prod)
    max_abs = float(abs_err.max())
    denom = np.maximum(np.abs(mine), 1e-12)
    max_rel = float((abs_err / denom).max())
    within_quant_tol = bool(np.allclose(mine, prod, atol=1e-2, rtol=1e-2))

    print(
        f"[{name}] shape={mine.shape} prod_dtype=bf16(prod_on_bf16_grid="
        f"{prod_was_bf16}) bf16_exact={exact_bf16} "
        f"raw_max_abs_err={max_abs:.3e} raw_max_rel_err={max_rel:.3e} "
        f"(<= bf16 round-off) within_1e-2={within_quant_tol}"
    )

    assert exact_bf16, (
        f"{name}: production dequant != independent reference rounded to bf16 "
        f"(raw max_abs={max_abs:.3e}, max_rel={max_rel:.3e}). The discrepancy "
        f"is NOT explained by bf16 round-off, indicating a real nibble-order / "
        f"group-axis / sign / dtype bug in the production primitive."
    )


@skip_no_weights
def test_attention_linear_q_proj():
    """2D attention linear: weight [out, in//8]."""
    packed, scales, biases = _load_triple(_ATTN)
    assert packed.ndim == 2 and packed.dtype == np.uint32
    assert packed.shape[1] * PACK_FACTOR == scales.shape[1] * GROUP_SIZE
    _compare(_ATTN, packed, scales, biases)


@skip_no_weights
def test_moe_expert_gate_proj():
    """3D MoE/expert tensor: weight [E, out, in//8] (experts stacked)."""
    packed, scales, biases = _load_triple(_MOE)
    assert packed.ndim == 3 and packed.dtype == np.uint32
    assert packed.shape[2] * PACK_FACTOR == scales.shape[2] * GROUP_SIZE
    _compare(_MOE, packed, scales, biases)


def test_independent_dequant_self_consistency():
    """Sanity check the independent transcription on a tiny hand-built example
    with NO checkpoint and NO production code: verifies low-nibble-first unpack
    and the affine formula directly. Runs everywhere (no weights needed)."""
    # One row, one group of group_size=2, bits=4: pack q=[1,2,...] etc.
    # Build a uint32 packing 8 nibbles [3,1,4,1,5,9,2,6] low-nibble-first.
    nibbles = [3, 1, 4, 1, 5, 9, 2, 6]
    word = 0
    for p, v in enumerate(nibbles):
        word |= (v & 0xF) << (4 * p)
    packed = np.array([[word]], dtype=np.uint32)  # [1, 1] -> in_dim 8
    # group_size=8 here: one (scale,bias) per row-group of 8 elements.
    scale = np.array([[2.0]], dtype=np.float32)
    bias = np.array([[-1.0]], dtype=np.float32)
    out = _independent_mlx_dequant(packed, scale, bias, group_size=8, bits=4)
    expected = np.array([[2.0 * q - 1.0 for q in nibbles]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)
