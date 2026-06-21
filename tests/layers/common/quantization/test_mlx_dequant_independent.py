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
  * The nibble unpack uses a DELIBERATELY DIFFERENT construction than production
    (byte view + ``np.unpackbits`` + manual 4-bit regrouping, NOT the canonical
    ``packed[..., None] >> shifts & mask`` broadcast/reshape idiom). This matters:
    two transliterations of the same idiom would make the same ordering/reshape
    choices, so their agreement could never catch a shared nibble-order/reshape
    bug -- the very failure mode this test exists to detect. See
    :func:`_unpack_nibbles_via_unpackbits` for the full rationale.
  * The production primitive ``mlx_dequantize`` is imported and called purely as
    a BLACK BOX -- its implementation was not read while authoring this file.
  * The inputs are the genuine quantized weight/scale/bias triples straight from
    the real safetensors shards -- no synthetic oracle anywhere in the loop.
  * A committed, WEIGHTS-FREE negative control
    (``test_negative_control_reversed_nibble_order_is_detected``) proves the
    comparison has teeth: a deliberately WRONG (reversed-nibble) unpack is shown
    to be DETECTED by the same ``_compare`` machinery, in CI without the
    checkpoint.

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


def _unpack_nibbles_via_unpackbits(packed: np.ndarray, bits: int) -> np.ndarray:
    """Re-derive the 4-bit unpack using a DELIBERATELY DIFFERENT construction.

    *** Why this is structurally distinct from production (read before editing) ***
    The production ``mlx_unpack`` (a black box here) uses the canonical vectorized
    idiom ``(packed[..., None] >> (arange(pack_factor)*bits)) & mask`` followed by
    a ``reshape``. Two transliterations of that SAME idiom necessarily make the
    same nibble-ordering and reshape choices, so bit-exact agreement between them
    could NOT catch a shared nibble-order / reshape bug -- the exact failure mode
    Task 13 exists to detect.

    To make agreement *meaningful*, this construction never uses the
    shift-by-arange / broadcast / reshape trick at all. Instead it:

      1. views the uint32 words as their LITTLE-ENDIAN raw bytes (4 bytes/word),
      2. expands every byte to its 8 bits with ``np.unpackbits`` (MSB-first
         within each byte),
      3. reassembles 4-bit values from those bits *by hand*, encoding the MLX
         layout from first principles: within a uint32, nibble position p (the
         p-th logical element, low-nibble-first) occupies bits [4p, 4p+4). On a
         little-endian machine, byte b of the word holds bits [8b, 8b+8), so
         nibble p lives in byte ``p // 2``: the LOW nibble (p even) is bits
         [8b, 8b+4), the HIGH nibble (p odd) is bits [8b+4, 8b+8). Because
         ``np.unpackbits`` lays a byte out MSB-first, those are array columns
         [4:8] (low nibble) and [0:4] (high nibble) respectively.

    The result is the SAME logical "8 nibbles/uint32 along input, low-nibble-
    first, logical index = word*8 + nibble" mapping, derived through a completely
    different code path (byte view + bit expansion + manual regrouping) so that
    agreement with production is independent evidence of correct ordering.
    """
    assert bits == 4, "this construction is specialised for the 4-bit layout"
    assert packed.dtype == np.uint32, packed.dtype

    *lead, n_words = packed.shape

    # (1) little-endian byte view: each uint32 -> 4 bytes, low byte first.
    le = packed.astype("<u4")
    raw_bytes = le.view(np.uint8).reshape(*lead, n_words, 4)  # [..., word, byte]

    # (2) expand each byte to 8 bits, MSB-first (numpy default).
    #     shape -> [..., word, byte, 8]
    bit = np.unpackbits(raw_bytes, axis=-1).reshape(*lead, n_words, 4, 8)

    # weights to fold a 4-bit MSB-first slice back into an integer.
    nib_weight = np.array([8, 4, 2, 1], dtype=np.uint32)

    # (3) reassemble the 8 nibbles per word, one logical position at a time,
    #     and place each directly into its logical slot (no reshape trick).
    out = np.empty((*lead, n_words * 8), dtype=np.uint32)
    for p in range(8):  # logical nibble position within the word
        byte_idx = p // 2
        # p even -> low nibble of the byte -> MSB-first columns [4:8]
        # p odd  -> high nibble of the byte -> MSB-first columns [0:4]
        cols = slice(4, 8) if (p % 2 == 0) else slice(0, 4)
        nib_bits = bit[..., byte_idx, cols]           # [..., word, 4]
        value = (nib_bits.astype(np.uint32) * nib_weight).sum(axis=-1)  # [..., word]
        out[..., p::8] = value  # logical index = word*8 + p
    return out


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

    Deliberately written without reference to the production implementation; the
    nibble unpack uses :func:`_unpack_nibbles_via_unpackbits`, a construction
    chosen specifically so it does NOT mirror production's vectorized
    shift/mask/reshape idiom (see that function's docstring).
    """
    assert packed.dtype == np.uint32, packed.dtype

    # --- unpack via the deliberately-distinct byte-view/unpackbits path ---
    q = _unpack_nibbles_via_unpackbits(packed, bits).astype(np.float32)

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


def _build_synthetic_triple(group_size: int = GROUP_SIZE):
    """Hand-craft a small (packed uint32, scales, biases) triple with values we
    fully control, sized so the production primitive accepts it (one group of
    ``group_size`` along the input dim, requiring group_size//8 uint32 words).

    No checkpoint and no synthetic-oracle math: nibbles are picked by hand and
    packed low-nibble-first directly. Returns numpy arrays.
    """
    assert group_size % PACK_FACTOR == 0
    n_words = group_size // PACK_FACTOR  # words to cover one group along input
    rows = 3

    # Deterministic but irregular nibble pattern so a reversed order is visible.
    rng = np.random.default_rng(20250621)
    q = rng.integers(0, 16, size=(rows, group_size), dtype=np.uint32)

    # Pack low-nibble-first by hand (NOT via the production idiom): for each
    # word, fold its 8 logical nibbles into bits [0,4),[4,8),...,[28,32).
    packed = np.zeros((rows, n_words), dtype=np.uint32)
    for w in range(n_words):
        for p in range(PACK_FACTOR):
            logical = w * PACK_FACTOR + p
            packed[:, w] |= (q[:, logical] & np.uint32(0xF)) << np.uint32(4 * p)

    # One (scale, bias) per group of group_size along the last axis.
    scales = np.array([[0.5], [1.25], [-0.75]], dtype=np.float32)
    biases = np.array([[-2.0], [0.25], [3.5]], dtype=np.float32)
    return packed, scales, biases, q


def _wrong_dequant_reversed_nibbles(packed, scales, biases,
                                    group_size=GROUP_SIZE, bits=BITS):
    """A DELIBERATELY WRONG unpack: reverses the nibble order within each word
    (logical index = word*8 + (7 - nibble) instead of word*8 + nibble), then
    applies the correct affine formula. Used only by the negative control to
    prove the comparison detects a layout bug."""
    assert bits == 4
    correct = _unpack_nibbles_via_unpackbits(packed, bits)
    *lead, in_dim = correct.shape
    n_words = in_dim // PACK_FACTOR
    # reverse the 8 nibbles within each word
    rev = correct.reshape(*lead, n_words, PACK_FACTOR)[..., ::-1]
    rev = rev.reshape(*lead, in_dim).astype(np.float32)
    s = np.repeat(scales.astype(np.float32), group_size, axis=-1)
    b = np.repeat(biases.astype(np.float32), group_size, axis=-1)
    return s * rev + b


def test_negative_control_reversed_nibble_order_is_detected():
    """Committed, WEIGHTS-FREE proof the methodology has teeth.

    We build a hand-crafted packed triple, obtain the reference from the
    production ``mlx_dequantize`` (black box), then feed a DELIBERATELY WRONG
    (reversed-nibble) unpack into the same comparison and assert it is DETECTED.
    Always runs in CI -- no 17GB checkpoint required."""
    packed, scales, biases, _q = _build_synthetic_triple(GROUP_SIZE)

    # Reference straight from production (black box).
    prod = _production_dequant(packed, scales, biases)
    prod_bf16 = _to_bf16_grid(prod)

    # 1) The CORRECT independent unpack must AGREE bit-exactly on the bf16 grid.
    correct = _to_bf16_grid(_independent_mlx_dequant(packed, scales, biases))
    assert np.array_equal(correct, prod), (
        "synthetic sanity: correct independent unpack should match production "
        "on the bf16 grid"
    )

    # 2) The WRONG (reversed-nibble) unpack must be DETECTED -- the comparison
    #    is not vacuous. A real layout bug perturbs whole quantized levels, which
    #    can never collapse onto the bf16 grid.
    wrong = _to_bf16_grid(
        _wrong_dequant_reversed_nibbles(packed, scales, biases))
    assert not np.array_equal(wrong, prod_bf16), (
        "NEGATIVE CONTROL FAILED: reversed-nibble unpack was NOT detected -- the "
        "comparison cannot distinguish a layout bug, so passing the real-shard "
        "tests would be meaningless."
    )

    # 3) And the magnitude of the wrong-order error is order ~1 (whole quantized
    #    levels), >> any bf16 round-off, confirming the strict bf16-grid check
    #    catches it rather than a loosened tolerance.
    max_abs_wrong = float(np.abs(wrong - prod_bf16).max())
    print(
        f"[negative-control] reversed-nibble unpack detected: "
        f"max_abs_err={max_abs_wrong:.3e} (>> bf16 round-off); correct unpack "
        f"matches production bit-exactly on the bf16 grid."
    )
    assert max_abs_wrong > 1e-2, max_abs_wrong
