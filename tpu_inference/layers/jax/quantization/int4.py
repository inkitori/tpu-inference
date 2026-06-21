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

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.moe_weights import \
    UnfusedMoEWeights
from tpu_inference.layers.common.quantization import mlx_dequantize
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.quantization import QuantizeMethodBase


@dataclass
class Int4Config:
    group_size: int = 64
    bits: int = 4
    overrides: dict = field(default_factory=dict)  # module-name -> {"bits", "group_size"}

    @classmethod
    def from_hf_quant_config(cls, q: dict) -> "Int4Config":
        gs = int(q.get("group_size", 64))
        bits = int(q.get("bits", 4))
        overrides = {k: v for k, v in q.items()
                     if isinstance(v, dict) and "bits" in v}
        return cls(group_size=gs, bits=bits, overrides=overrides)

    def bits_for(self, module_name: str) -> tuple[int, int]:
        for k, v in self.overrides.items():
            if module_name == k or module_name.endswith("." + k):
                return int(v["bits"]), int(v.get("group_size", self.group_size))
        return self.bits, self.group_size


class Int4LinearMethod(QuantizeMethodBase):
    """Phase-1 int4 linear method: dequantize in XLA (on-device), then einsum.

    Mirrors Fp8TensorwiseLinearMethod structure:
    - __init__(layer, linear_config, bits, group_size) — reads dims from linear_config
    - create_weights_jax(layer, *weight_args, rngs, **extra_weight_attrs) — no required kwargs
    - apply_jax(layer, x) calls mlx_dequantize then jnp.einsum with self.einsum_str
    """

    def __init__(self, layer: JaxModule, linear_config, bits: int,
                 group_size: int):
        # Mirror FP8 (fp8.py:86-104): read einsum_str and dims from layer/linear_config
        # at construction time so create_weights_jax and apply_jax need no extra kwargs.
        # Matches the real dispatch seam linear.py:82: create_weights_jax(self, rngs=rngs)
        # and linear.py:88: apply_jax(self, inputs).
        self.einsum_str = layer.einsum_str
        self.linear_config = linear_config
        self.bits = bits
        self.group_size = group_size

        # Flatten in_features and out_features from linear_config (same as FP8 fp8.py:94-95).
        self.in_features = math.prod(linear_config.in_features)
        self.out_features = math.prod(linear_config.out_features)
        self.weight_sharding = linear_config.weight_sharding

    def create_weights_jax(self, layer: JaxModule, *weight_args, rngs,
                           **extra_weight_attrs):
        """Declare packed weight, scales, and biases parameters on layer.

        Shapes (MLX layout, output-major):
          weight : [out, in // (32 // bits)]  uint32 (8 u4 values per word)
          scales : [out, in // group_size]    bf16
          biases : [out, in // group_size]    bf16

        Partition spec mirrors FP8 tensorwise: shard the output dim (axis 0),
        leave the contraction dim (axis 1) unsharded — we must not shard finer
        than group_size along the contraction axis.

        Dims are read from self (set at __init__ from linear_config), not from kwargs.
        """
        in_features = self.in_features
        out_features = self.out_features
        weight_sharding = self.weight_sharding

        assert in_features % self.group_size == 0, (
            f"in_features={in_features} must be divisible by group_size={self.group_size}")
        per_word = 32 // self.bits
        assert in_features % per_word == 0, (
            f"in_features={in_features} must be divisible by per_word={per_word} "
            f"(bits={self.bits})")
        n_words = in_features // per_word
        n_groups = in_features // self.group_size

        # Output-dim sharding mirrors Fp8TensorwiseLinearMethod.
        if isinstance(weight_sharding, P) and len(weight_sharding) > 0:
            w_sharding = P(weight_sharding[0], None)
            scale_sharding = P(weight_sharding[0])
        elif isinstance(weight_sharding,
                        (tuple, list)) and len(weight_sharding) > 0:
            w_sharding = (weight_sharding[0], None)
            scale_sharding = (weight_sharding[0], )
        else:
            w_sharding = None
            scale_sharding = None

        layer.weight = create_param(rngs,
                                    shape=(out_features, n_words),
                                    dtype=jnp.uint32,
                                    sharding=w_sharding)
        layer.scales = create_param(rngs,
                                    shape=(out_features, n_groups),
                                    dtype=jnp.bfloat16,
                                    sharding=scale_sharding)
        layer.biases = create_param(rngs,
                                    shape=(out_features, n_groups),
                                    dtype=jnp.bfloat16,
                                    sharding=scale_sharding)

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        """Dequantize packed weight, cast to input dtype, then einsum."""
        packed = layer.weight[...]
        scales = layer.scales[...]
        biases = layer.biases[...]

        # mlx_dequantize returns bfloat16 [out, in]; cast to match x.
        w = mlx_dequantize(packed, scales, biases,
                           self.group_size, self.bits).astype(x.dtype)

        return jnp.einsum(self.einsum_str, x, w)


class Int4FusedMoEMethod(QuantizeMethodBase):
    """Phase-1 int4 MoE method: dequantize all experts in XLA, then route through
    the EXISTING UNQUANTIZED MoE path.

    Mirrors ``Fp8FusedMoEMethod`` lifecycle (fp8.py:345-628) but stores packed
    uint32 weights + bf16 scales/biases per stacked expert. ``apply_jax``
    dequantizes each projection to bf16, transposes the MLX ``[E, out, in]``
    layout into the model's ``[E, in, out]`` layout, assembles the SAME
    ``UnfusedMoEWeights`` struct the unquantized DENSE_MAT / MEGABLX_GMM path
    builds (unquantized.py:189-199), and calls the shared ``moe_apply``. Packing
    stays packed until ``apply_jax``. Phase-1 targets the correctness rung via
    the unfused path; fused / packed-GMM backends are Task 9 (Phase-2).

    Param names (leading expert dim E, MLX layout ``[E, out, in]``):
      ``kernel_gating_EDF{,_scales,_biases}``  gate_proj  (out=F, in=D)
      ``kernel_up_proj_EDF{,_scales,_biases}`` up_proj    (out=F, in=D)
      ``kernel_down_proj_EFD{,_scales,_biases}`` down_proj (out=D, in=F)
    """

    def __init__(self, config, bits: int, group_size: int):
        # Construction contract mirrors Fp8FusedMoEMethod (fp8.py:352): config-
        # carried args plus int4 extras. The factory dispatches on
        # isinstance(layer, JaxMoE) and constructs with config + bits/group_size.
        super().__init__()
        self.config = config
        self.bits = bits
        self.group_size = group_size
        self.extra_backend_kwargs = {}

    def create_weights_jax(self, layer: JaxModule, *weight_args, rngs,
                           **extra_weight_attrs) -> None:
        """Declare packed uint32 weights + bf16 scales/biases for the 3 MoE
        projections, replacing the bf16 expert kernels JaxMoE created.

        Dims are read from the layer (num_local_experts / hidden_size /
        intermediate_size_moe) -- the base signature takes no dim kwargs.
        Shapes (MLX layout, output-major, leading expert dim E):
          gate/up : weight [E, F, D // per_word] uint32, scales/biases [E, F, D // gs]
          down    : weight [E, D, F // per_word] uint32, scales/biases [E, D, F // gs]
        Partition specs mirror FP8/unquantized MoE: shard the expert axis (axis 0)
        from edf/efd_sharding; never shard the contraction finer than group_size.
        """
        E = layer.num_local_experts
        D = layer.hidden_size
        F = layer.intermediate_size_moe

        per_word = 32 // self.bits
        for in_dim, name in ((D, "D"), (F, "F")):
            assert in_dim % self.group_size == 0, (
                f"{name}={in_dim} must be divisible by group_size="
                f"{self.group_size}")
            assert in_dim % per_word == 0, (
                f"{name}={in_dim} must be divisible by per_word={per_word} "
                f"(bits={self.bits})")

        # Expert-axis sharding only; weight/scale inner dims replicated so we
        # never split the contraction below group_size granularity.
        def _expert_spec(ndim, base):
            if isinstance(base, P) and len(base) > 0:
                return P(base[0], *([None] * (ndim - 1)))
            if isinstance(base, (tuple, list)) and len(base) > 0:
                return (base[0], ) + (None, ) * (ndim - 1)
            return None

        w_edf = _expert_spec(3, layer.edf_sharding)
        w_efd = _expert_spec(3, layer.efd_sharding)

        def _param(shape, dtype, w_spec):
            # Packed uint32 has no float initializer; zero-init the storage
            # directly. Real values are filled by the loader / process step.
            return nnx.Param(jnp.zeros(shape, dtype), sharding=w_spec)

        def _decl(name, out, in_, w_spec):
            n_words = in_ // per_word
            n_groups = in_ // self.group_size
            setattr(layer, name,
                    _param((E, out, n_words), jnp.uint32, w_spec))
            setattr(layer, f"{name}_scales",
                    _param((E, out, n_groups), jnp.bfloat16, w_spec))
            setattr(layer, f"{name}_biases",
                    _param((E, out, n_groups), jnp.bfloat16, w_spec))

        # MLX layout [E, out, in]: gate/up out=F in=D; down out=D in=F.
        _decl("kernel_gating_EDF", F, D, w_edf)
        _decl("kernel_up_proj_EDF", F, D, w_edf)
        _decl("kernel_down_proj_EFD", D, F, w_efd)

    def process_weights_after_loading(self, layer: JaxModule) -> bool:
        """Experts arrive PRE-STACKED from MLX (leading dim E already), packed.

        There is no per-expert concat or fusion to do here, and we must NOT
        dequantize (Phase-1 keeps weights packed until apply_jax). We only assert
        the staged params are present. Returns True (done) per the base contract.
        """
        for name in ("kernel_gating_EDF", "kernel_up_proj_EDF",
                     "kernel_down_proj_EFD"):
            for suffix in ("", "_scales", "_biases"):
                assert hasattr(layer, name + suffix), (
                    f"missing int4 MoE param {name + suffix}")
        return True

    def _dequantize(self, layer: JaxModule, name: str) -> jax.Array:
        """Dequantize one projection to bf16 and transpose MLX [E, out, in] into
        the model's [E, in, out] kernel layout."""
        packed = getattr(layer, name)[...]
        scales = getattr(layer, f"{name}_scales")[...]
        biases = getattr(layer, f"{name}_biases")[...]
        w_E_out_in = mlx_dequantize(packed, scales, biases, self.group_size,
                                    self.bits)
        # MLX stores [E, out, in]; JaxMoE kernels are [E, in, out].
        return jnp.swapaxes(w_E_out_in, -1, -2)

    def apply_jax(self, layer: JaxModule, x: jax.Array, *,
                  router_logits: jax.Array) -> jax.Array:
        """Dequantize all experts to bf16, build the unquantized weights struct
        for the active backend, and route through the shared moe_apply."""
        x_TD = jnp.asarray(x, layer.dtype)
        x_TD = jax.lax.with_sharding_constraint(
            x_TD,
            jax.sharding.NamedSharding(layer.mesh,
                                       P(*layer.activation_ffw_td)))

        # [E, D, F] gate/up ; [E, F, D] down -- the same per-expert layouts
        # JaxMoE.__post_init__ creates and the unquantized path consumes
        # (moe.py:195-217). DENSE_MAT / MEGABLX_GMM take them UNFUSED, mapped
        # w1=gate, w2=up, w3=down (unquantized.py:189-199).
        if layer.moe_backend in (MoEBackend.DENSE_MAT,
                                 MoEBackend.MEGABLX_GMM):
            gate = self._dequantize(layer, "kernel_gating_EDF")
            up = self._dequantize(layer, "kernel_up_proj_EDF")
            down = self._dequantize(layer, "kernel_down_proj_EFD")
            weights = UnfusedMoEWeights(
                w1_weight=gate.astype(layer.dtype),
                w1_weight_scale=None,
                w1_bias=None,
                w2_weight=up.astype(layer.dtype),
                w2_weight_scale=None,
                w2_bias=None,
                w3_weight=down.astype(layer.dtype),
                w3_weight_scale=None,
                w3_bias=None,
            )
        else:
            # Fused GMM/FUSED_MOE backends concat gate+up and run them through
            # process_unquantized_moe_weights to produce a backend-specific
            # fused layout (unquantized.py:81-153). Phase-1 only needs to prove
            # dequant correctness via the unfused path; the fused/packed path is
            # Task 9. Refuse rather than ship an unverified fused layout.
            raise NotImplementedError(
                f"Int4FusedMoEMethod (phase 1) supports DENSE_MAT / "
                f"MEGABLX_GMM; got moe_backend={layer.moe_backend}. Fused "
                f"backends arrive with the packed GMM path in phase 2.")

        return moe_apply(layer, x_TD, router_logits, weights,
                         layer.moe_backend, layer.mesh,
                         self.extra_backend_kwargs)
