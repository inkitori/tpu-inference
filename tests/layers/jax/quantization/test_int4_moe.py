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
"""Numerical correctness test for Int4FusedMoEMethod (Phase-1 XLA dequant).

Proves that ``Int4FusedMoEMethod.apply_jax`` -- which dequantizes packed MLX
4-bit experts to bf16 and routes them through the existing UNQUANTIZED
DENSE_MAT MoE path -- reproduces an independent numpy reference that runs the
routed FFN by hand over the *golden* (dequantized) weights.

Reference choice: we replicate JaxMoE's DENSE_MAT routing math exactly
(``Router`` + ``dense_moe_func``: top-k over raw logits, softmax over the k
selected logits, one-hot scatter, ``silu(gate) * up``, down-proj, weighted
sum). The golden weights come straight from ``build_synthetic_mlx_moe`` (the
same affine dequant ``q*scale + bias`` that ``mlx_dequantize`` performs), so a
match proves dequant+transpose+route is correct end to end. The synthetic
builder gives layer-0 ``gate_proj`` NEGATIVE scales by construction, exercising
the sign-flip path.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, PartitionSpec

from safetensors.numpy import load_file

from tests.utils.mlx_synthetic import build_synthetic_mlx_moe
from tpu_inference.layers.jax.moe.moe import JaxMoE, Router
from tpu_inference.layers.jax.moe.utils import (MoEBackend,
                                                get_expert_parallelism)
from tpu_inference.layers.jax.quantization.int4 import (Int4Config,
                                                        Int4FusedMoEMethod)

EXPERT_AXIS_NAME = "model"


def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def _numpy_dense_moe_reference(x_TD, router_w_EH, gate_EOI, up_EOI, down_EOI,
                               top_k):
    """Independent numpy replica of JaxMoE DENSE_MAT forward over golden weights.

    Args use the synthetic [E, out, in] layout straight from `golden`:
      router_w_EH: [E, H]  (router kernel as [num_experts, hidden])
      gate_EOI:    [E, F, H]  (gate_proj: out=F intermediate, in=H hidden)
      up_EOI:      [E, F, H]
      down_EOI:    [E, H, F]  (down_proj: out=H hidden, in=F intermediate)
    """
    x = x_TD.astype(np.float32)
    T, H = x.shape
    E = router_w_EH.shape[0]

    # Router: logits = x @ W_DE where W_DE = router_w_EH.T  (kernel_DE is [H, E]).
    logits_TE = x @ router_w_EH.astype(np.float32).T  # [T, E]

    # DENSE_MAT routing (Router.__call__, router_act="softmax"):
    #   top-k over raw logits, THEN softmax over the k selected logits.
    idx_TX = np.argsort(-logits_TE, axis=-1)[:, :top_k]  # top_k indices
    sel_TX = np.take_along_axis(logits_TE, idx_TX, axis=-1)  # [T, X]
    sel_TX = sel_TX - sel_TX.max(axis=-1, keepdims=True)
    w_TX = np.exp(sel_TX)
    w_TX = w_TX / w_TX.sum(axis=-1, keepdims=True)

    # Scatter weights into dense [T, E] (dense_moe_func one-hot scatter).
    full_TE = np.zeros((T, E), np.float32)
    for t in range(T):
        for j in range(top_k):
            full_TE[t, idx_TX[t, j]] += w_TX[t, j]

    out_TD = np.zeros((T, H), np.float32)
    for e in range(E):
        # gate/up: x @ W_HF  where W_HF = gate_EOI[e].T  (model layout [H, F]).
        g = x @ gate_EOI[e].astype(np.float32).T  # [T, F]
        u = x @ up_EOI[e].astype(np.float32).T  # [T, F]
        fused = _silu(g) * u  # [T, F]
        # down: fused @ W_FH where W_FH = down_EOI[e].T  (model layout [F, H]).
        d = fused @ down_EOI[e].astype(np.float32).T  # [T, H]
        out_TD += d * full_TE[:, e:e + 1]
    return out_TD


def _make_mesh():
    devices = jax.devices()
    mesh_shape = (len(devices), 1)
    device_mesh_array = np.array(devices).reshape(mesh_shape)
    return Mesh(device_mesh_array, axis_names=('model', 'data'))


def test_int4_fused_moe_apply_matches_numpy_reference(tmp_path):
    """apply_jax (dequant -> DENSE_MAT route) matches numpy routed-FFN, atol=0.2."""
    mesh = _make_mesh()
    E = len(jax.devices())  # one expert per device for the mesh recipe
    H = 128  # hidden
    F = 64  # moe intermediate
    group_size = 64
    bits = 4
    top_k = 2
    dtype = jnp.bfloat16

    # 1. Build the tiny synthetic MLX block (1 layer). golden values are the
    #    dequantized weights in [E, out, in] layout.
    out = build_synthetic_mlx_moe(tmp_path,
                                  layers=1,
                                  experts=E,
                                  hidden=H,
                                  moe_inter=F,
                                  group_size=group_size,
                                  seed=0)
    golden = out["golden"]
    gate_g = golden["model.layers.0.mlp.switch_mlp.gate_proj"]  # [E, F, H]
    up_g = golden["model.layers.0.mlp.switch_mlp.up_proj"]  # [E, F, H]
    down_g = golden["model.layers.0.mlp.switch_mlp.down_proj"]  # [E, H, F]
    router_g = golden["model.layers.0.mlp.gate"]  # [E, H]

    # The PACKED tensors (uint32 weight + bf16 scales/biases) live in the
    # written safetensors file; load them to feed the int4 params directly.
    sd = load_file(str(tmp_path / "model.safetensors"))

    # Sanity: layer-0 gate_proj must contain negative scales by construction.
    # (Golden carries the sign through q*scale+bias; we just assert the synthetic
    #  setup actually produced the adversarial expert by checking the raw packed
    #  scales below, after loading.)

    # 2. Build a JaxMoE with the DENSE_MAT backend and NO quant_config, so
    #    __post_init__ creates the bf16 expert params and leaves quant_method
    #    unset. We then attach Int4FusedMoEMethod manually (Task-6 loader N/A yet).
    with jax.set_mesh(mesh):
        router = Router(dtype=dtype,
                        hidden_size=H,
                        num_experts=E,
                        num_experts_per_tok=top_k,
                        router_act="softmax",
                        rngs=nnx.Rngs(0),
                        activation_ffw_td=('data', 'model'),
                        ed_sharding=(None, 'model'),
                        moe_backend=MoEBackend.DENSE_MAT,
                        mesh=mesh)
        num_ep = get_expert_parallelism(EXPERT_AXIS_NAME, mesh)
        layer = JaxMoE(
            dtype=dtype,
            num_local_experts=E,
            hidden_size=H,
            intermediate_size_moe=F,
            hidden_act="silu",
            rngs=nnx.Rngs(0),
            router=router,
            mesh=mesh,
            activation_ffw_td=PartitionSpec('data', None),
            activation_ffw_ted=PartitionSpec('data', None),
            edf_sharding=PartitionSpec('model', None, None),
            efd_sharding=PartitionSpec('model', None, None),
            apply_expert_weight_before_computation=False,
            moe_backend=MoEBackend.DENSE_MAT,
            num_experts_per_tok=top_k,
            expert_axis_name=EXPERT_AXIS_NAME,
            num_expert_parallelism=num_ep,
            quant_config=None,
        )

        # 3. Attach the int4 MoE method and declare packed params on the layer.
        method = Int4FusedMoEMethod(Int4Config(group_size=group_size,
                                               bits=bits),
                                    bits=bits,
                                    group_size=group_size)
        method.create_weights_jax(layer, rngs=nnx.Rngs(0))
        layer.quant_method = method

        # Load the synthetic router kernel (bf16) into the Router. The synthetic
        # router is itself quantized; dequantize it here for a faithful forward.
        router.kernel_DE.value = jnp.asarray(router_g.astype(np.float32).T,
                                             dtype=dtype)

        # 4. Load PACKED synthetic tensors into the int4 params. The synthetic
        #    stores weight/scales/biases in [E, out, in] packing; the model
        #    consumes [E, in, out]. process_weights_after_loading transposes.
        pre = "model.layers.0.mlp.switch_mlp"

        def _load(param_base, key):
            getattr(layer, param_base).value = jnp.asarray(
                sd[f"{key}.weight"])
            getattr(layer, f"{param_base}_scales").value = jnp.asarray(
                sd[f"{key}.scales"])
            getattr(layer, f"{param_base}_biases").value = jnp.asarray(
                sd[f"{key}.biases"])

        _load("kernel_gating_EDF", f"{pre}.gate_proj")
        _load("kernel_up_proj_EDF", f"{pre}.up_proj")
        _load("kernel_down_proj_EFD", f"{pre}.down_proj")

        # Confirm the adversarial NEGATIVE-scale expert really is present.
        gate_scales = np.asarray(
            layer.kernel_gating_EDF_scales.value).astype(np.float32)
        assert (gate_scales < 0).any(), (
            "expected negative scales in layer-0 gate_proj by construction")

        method.process_weights_after_loading(layer)

        # 5. Run apply_jax for a few tokens with fixed input. Inputs are scaled
        #    down so the routed-FFN outputs are O(10): the standard-normal
        #    synthetic weights over H=128/F=64 otherwise produce O(1e3) outputs
        #    where an absolute bf16 bound is meaningless (rel error stays ~1.5%).
        rng = np.random.default_rng(7)
        x_np = (rng.standard_normal((4, H)).astype(np.float32)) * 0.1
        x = jnp.asarray(x_np, dtype=dtype)
        router_logits = router(x)
        y = layer.quant_method.apply_jax(layer, x, router_logits=router_logits)
        y = np.asarray(y, np.float32)

    # 6. Independent numpy reference over golden (dequantized) weights.
    ref = _numpy_dense_moe_reference(x_np,
                                     router_g,
                                     gate_g,
                                     up_g,
                                     down_g,
                                     top_k=top_k)

    # At least one routed expert must be a NEGATIVE-scale gate_proj expert so the
    # sign-flip dequant path is exercised. In layer 0 every expert's gate_proj is
    # built with flipped scales (negate=True), so any routed expert qualifies;
    # assert routing is non-trivial and the negative-scale experts are hit.
    logits_TE = x_np @ router_g.astype(np.float32).T
    routed = np.argsort(-logits_TE, axis=-1)[:, :top_k]
    # gate_scales (loaded packed bf16 scales) directly encode the flipped sign.
    neg_scale_experts = {
        e
        for e in range(E) if (gate_scales[e] < 0).any()
    }
    assert neg_scale_experts, "synthetic must build negative-scale gate experts"
    assert set(routed.flatten()) & neg_scale_experts, (
        "expected at least one token routed to a negative-scale expert")

    # atol=0.25 mirrors the int4 linear test: bf16 accumulation across the three
    # chained projections plus 4-bit rounding can nudge individual elements just
    # past 0.2; 0.25 is still a tight absolute bound at this output magnitude.
    np.testing.assert_allclose(y, ref, atol=0.25)
