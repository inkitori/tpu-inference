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
"""CPU unit tests for the MoE decode padding-row routing mask.

These run without a TPU. They lock the two invariants that matter:

  1. Real (non-padding) tokens are routed EXACTLY as before (the change is
     numerically transparent to actual outputs).
  2. When there is NO padding (num_actual_tokens == num_tokens) the mask is a
     bitwise no-op -- this is the structural guarantee that full / large batches
     cannot regress, since the optimization simply does nothing there.

Plus a check that padding rows collapse onto a single expert (the mechanism that
cuts the distinct-active-expert count at decode).
"""
import unittest

import jax.numpy as jnp
import numpy as np

from tpu_inference.layers.common.fused_moe_gmm import \
    mask_padding_token_routing


class MaskPaddingRoutingTest(unittest.TestCase):

    def _decode_case(self):
        # 8 padded rows, topk=2; row 0 is the only real token, rows 1-7 are
        # padding that (pre-mask) routed to a spread of distinct experts.
        topk_indices = jnp.array(
            [[3, 5],   # real token
             [1, 2], [4, 6], [7, 0], [2, 4],
             [6, 1], [5, 7], [3, 6]],  # padding -> many distinct experts
            dtype=jnp.int32)
        topk_weights = jnp.arange(1, 17, dtype=jnp.float32).reshape(8, 2) / 16.0
        num_actual_tokens = jnp.array(1, dtype=jnp.int32)
        return topk_weights, topk_indices, num_actual_tokens

    def test_real_rows_unchanged(self):
        w, idx, n = self._decode_case()
        mw, mi = mask_padding_token_routing(w, idx, n)
        np.testing.assert_array_equal(np.asarray(mi[0]), np.asarray(idx[0]))
        np.testing.assert_array_equal(np.asarray(mw[0]), np.asarray(w[0]))

    def test_padding_rows_collapse_to_expert_zero(self):
        w, idx, n = self._decode_case()
        mw, mi = mask_padding_token_routing(w, idx, n)
        self.assertTrue(bool(jnp.all(mi[1:] == 0)))
        self.assertTrue(bool(jnp.all(mw[1:] == 0.0)))

    def test_cuts_distinct_active_experts(self):
        w, idx, n = self._decode_case()
        _, mi = mask_padding_token_routing(w, idx, n)
        # Masked active set is exactly the real token's experts plus expert 0.
        real_experts = set(np.asarray(idx[0]).tolist())
        self.assertEqual(set(np.asarray(mi).ravel().tolist()),
                         real_experts | {0})
        # And it is strictly smaller than the unmasked active set here.
        self.assertLess(int(jnp.unique(mi).size), int(jnp.unique(idx).size))

    def test_no_padding_is_exact_noop(self):
        # The regression guard: a full batch (no padding) must be untouched, so
        # larger batch sizes can never be slowed down by the mask.
        w, idx, _ = self._decode_case()
        num_tokens = idx.shape[0]
        mw, mi = mask_padding_token_routing(
            w, idx, jnp.array(num_tokens, dtype=jnp.int32))
        np.testing.assert_array_equal(np.asarray(mi), np.asarray(idx))
        np.testing.assert_array_equal(np.asarray(mw), np.asarray(w))

    def test_partial_batch_masks_only_the_tail(self):
        # num_actual_tokens=3 -> rows 0,1,2 kept; rows 3.. collapsed.
        w, idx, _ = self._decode_case()
        mw, mi = mask_padding_token_routing(w, idx, jnp.array(3, jnp.int32))
        np.testing.assert_array_equal(np.asarray(mi[:3]), np.asarray(idx[:3]))
        np.testing.assert_array_equal(np.asarray(mw[:3]), np.asarray(w[:3]))
        self.assertTrue(bool(jnp.all(mi[3:] == 0)))
        self.assertTrue(bool(jnp.all(mw[3:] == 0.0)))


if __name__ == "__main__":
    unittest.main()
