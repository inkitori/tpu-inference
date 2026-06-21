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
from jax.sharding import PartitionSpec as P

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
