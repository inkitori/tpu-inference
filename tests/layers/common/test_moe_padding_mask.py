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
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from tpu_inference.layers.common.fused_moe_gmm import \
    mask_padding_rows  # noqa: E402

NUM_TOKENS = 16
NUM_EXPERTS = 64
TOPK = 8


def _routed(seed=0):
    rng = np.random.default_rng(seed)
    logits = jnp.asarray(
        rng.standard_normal((NUM_TOKENS, NUM_EXPERTS)).astype(np.float32))
    weights, indices = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), k=TOPK)
    return weights, indices


def test_real_rows_untouched_padding_rows_collapsed():
    weights, indices = _routed()
    num_actual = 5

    got_w, got_i = mask_padding_rows(weights, indices, num_actual)

    # Real rows are bit-identical.
    assert np.array_equal(np.asarray(got_w[:num_actual]),
                          np.asarray(weights[:num_actual]))
    assert np.array_equal(np.asarray(got_i[:num_actual]),
                          np.asarray(indices[:num_actual]))
    # Padding rows all route to expert 0 with zero weight.
    assert np.all(np.asarray(got_i[num_actual:]) == 0)
    assert np.all(np.asarray(got_w[num_actual:]) == 0.0)


def test_full_batch_is_noop():
    weights, indices = _routed(seed=1)
    got_w, got_i = mask_padding_rows(weights, indices, NUM_TOKENS)
    assert np.array_equal(np.asarray(got_w), np.asarray(weights))
    assert np.array_equal(np.asarray(got_i), np.asarray(indices))


def test_traced_num_actual_tokens_does_not_recompile():
    """num_actual_tokens is a dynamic scalar: one trace serves all values."""
    weights, indices = _routed(seed=2)

    traces = 0

    @jax.jit
    def fn(w, i, n):
        nonlocal traces
        traces += 1
        return mask_padding_rows(w, i, n)

    for n in (1, 7, NUM_TOKENS):
        got_w, got_i = fn(weights, indices, jnp.int32(n))
        assert np.all(np.asarray(got_i[n:]) == 0)
        assert np.array_equal(np.asarray(got_w[:n]), np.asarray(weights[:n]))
    assert traces == 1
