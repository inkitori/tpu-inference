# Copyright 2025 Google LLC
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

import jax.numpy as jnp
import numpy as np

from tests.utils.mlx_synthetic import _quantize_affine
from tpu_inference.layers.common.quantization import mlx_dequantize


def test_mlx_linear_dequant_then_matmul_matches_golden():
    """The apply() math: y = x @ dequant(weight).T must match x @ golden.T."""
    rng = np.random.default_rng(1)
    out_f, in_f, gs = 32, 128, 64
    w = rng.standard_normal((out_f, in_f)).astype(np.float32)
    packed, scales, biases, golden = _quantize_affine(w, gs, force_negative_scale=True)
    x = rng.standard_normal((4, in_f)).astype(np.float32)

    weight = mlx_dequantize(jnp.asarray(packed), jnp.asarray(scales),
                            jnp.asarray(biases), group_size=gs, bits=4)  # [out, in]
    y = np.asarray(jnp.einsum("bd,fd->bf", jnp.asarray(x), weight)).astype(np.float32)
    y_ref = x @ golden.astype(np.float32).T
    np.testing.assert_allclose(y, y_ref, atol=2e-2, rtol=2e-2)
