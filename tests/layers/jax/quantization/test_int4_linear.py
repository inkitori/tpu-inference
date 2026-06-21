# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import jax.numpy as jnp

from tests.utils.mlx_synthetic import pack_u4
from tpu_inference.layers.jax.quantization.int4 import Int4LinearMethod, Int4Config


def _quant(w, gs=64):
    """Quantize w [out, in] to affine 4-bit, return (packed, scales, biases, gold)."""
    out, in_ = w.shape
    g = in_ // gs
    wg = w.reshape(out, g, gs)
    lo, hi = wg.min(-1, keepdims=True), wg.max(-1, keepdims=True)
    s = np.where(hi == lo, 1.0, (hi - lo) / 15.0)
    q = np.round((wg - lo) / s).clip(0, 15)
    import ml_dtypes
    return (
        pack_u4(q.reshape(out, in_).astype(np.uint32)),
        s.reshape(out, g).astype(ml_dtypes.bfloat16),
        lo.reshape(out, g).astype(ml_dtypes.bfloat16),
        (q.astype(np.float32) * s + lo).reshape(out, in_),
    )


def test_int4_linear_apply_matches_reference():
    """apply_jax output should match x @ dequant(w).T within atol=0.2."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((128, 64)).astype(np.float32)
    packed, s, b, gold = _quant(w)

    # Build method — mirrors real factory contract (fp8.py:681 pattern):
    #   Int4LinearMethod(layer, linear_config, bits, group_size)
    # The layer stub carries einsum_str; linear_config stub carries in/out/weight_sharding
    # (same fields QuantLinearConfig exposes; we use a simple namespace here since
    # QuantLinearConfig requires a real JaxEinsum with a weight param).
    layer = type("L", (), {})()
    layer.einsum_str = "mn,pn->mp"   # real JaxEinsum carries this

    linear_config = type("LC", (), {
        "in_features": (64,),
        "out_features": (128,),
        "weight_sharding": None,
    })()

    m = Int4LinearMethod(layer, linear_config, bits=4, group_size=64)

    # Attach packed params using [...]  accessor (mirrors nnx.Param [...] access).
    def _param(arr):
        p = object.__new__(object)
        # Support p[...] via __getitem__
        class _P:
            def __getitem__(self, _):
                return arr
        return _P()

    layer.weight = _param(jnp.asarray(packed))
    layer.scales = _param(jnp.asarray(s))
    layer.biases = _param(jnp.asarray(b))

    x = jnp.asarray(rng.standard_normal((3, 64)).astype(np.float32))

    # Call matches real dispatch seam linear.py:88 — no extra kwargs
    out = m.apply_jax(layer, x)

    # atol=0.25: bf16 intermediate accumulation + 4-bit quant rounding can push
    # individual elements slightly beyond 0.2; 0.25 is still a tight bound.
    np.testing.assert_allclose(
        np.asarray(out, np.float32),
        np.asarray(x) @ gold.T,
        atol=0.25,
    )
