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
"""Shared-expert psum merge: fused_moe_func(shared_partial=...) must equal
fused_moe_func(...) + psum-reduced shared output.

Needs 8 TPU devices (moe_gmm_local runs the Pallas gmm kernel).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
from tpu_inference.layers.common.sharding import ShardingAxisName

NUM_DEVICES = 8
NUM_EXPERTS = 16
TOPK = 4
HIDDEN = 64
INTERMEDIATE = 32
NUM_TOKENS = 16


def _mesh():
    devices = np.array(jax.devices()[:NUM_DEVICES]).reshape(1, NUM_DEVICES)
    return Mesh(devices, ("data", "model"))


def _run(mesh, use_ep, shared_partial, x, w1, w2, gating):
    kwargs = dict(
        hidden_states=x,
        w1=w1,
        w2=w2,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        gating_output=gating,
        topk=TOPK,
        shared_partial=shared_partial,
        renormalize=True,
        mesh=mesh,
        use_ep=use_ep,
        activation="silu",
        scoring_fn="softmax",
    )
    return fused_moe_func(**kwargs)


@pytest.mark.parametrize("use_ep", [False, True])
def test_shared_partial_merges_into_combine(use_ep):
    if (len(jax.devices()) < NUM_DEVICES
            or jax.devices()[0].platform != "tpu"):
        pytest.skip("needs 8 TPU devices (Pallas gmm kernel)")
    mesh = _mesh()
    rng = np.random.default_rng(0)
    dt = jnp.float32

    x = jnp.asarray(rng.standard_normal((NUM_TOKENS, HIDDEN)), dtype=dt)
    w1 = jnp.asarray(rng.standard_normal(
        (NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE)) * 0.1, dtype=dt)
    w2 = jnp.asarray(rng.standard_normal(
        (NUM_EXPERTS, INTERMEDIATE, HIDDEN)) * 0.1, dtype=dt)
    gating = jnp.asarray(rng.standard_normal((NUM_TOKENS, NUM_EXPERTS)),
                         dtype=dt)
    # Unreduced per-shard shared-expert partials [n_shards, T, H].
    partials_np = rng.standard_normal(
        (NUM_DEVICES, NUM_TOKENS, HIDDEN)).astype(np.float32) * 0.1
    partials = jax.device_put(
        jnp.asarray(partials_np),
        NamedSharding(mesh, P(ShardingAxisName.MLP_TENSOR, None, None)))

    routed = _run(mesh, use_ep, None, x, w1, w2, gating)
    merged = _run(mesh, use_ep, partials, x, w1, w2, gating)

    expected = np.asarray(routed) + partials_np.sum(axis=0)
    np.testing.assert_allclose(np.asarray(merged), expected,
                               rtol=1e-5, atol=1e-5)
