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
    - stores einsum_str from layer at construction
    - create_weights_jax declares weight (uint32), scales (bf16), biases (bf16)
    - apply_jax calls mlx_dequantize then jnp.einsum with self.einsum_str
    """

    def __init__(self, layer: JaxModule, config: Int4Config, bits: int,
                 group_size: int):
        # Mirror FP8: read einsum_str from the layer at construction time so
        # apply_jax(layer, x) needs no extra kwargs — matching the real dispatch
        # seam at linear.py:88: self.quant_method.apply_jax(self, inputs)
        self.einsum_str = layer.einsum_str
        self.config = config
        self.bits = bits
        self.group_size = group_size

    def create_weights_jax(self, layer: JaxModule, *weight_args, rngs,
                           in_features: int, out_features: int,
                           weight_sharding=None, **extra_weight_attrs):
        """Declare packed weight, scales, and biases parameters on layer.

        Shapes (MLX layout, output-major):
          weight : [out, in // (32 // bits)]  uint32 (8 u4 values per word)
          scales : [out, in // group_size]    bf16
          biases : [out, in // group_size]    bf16

        Partition spec mirrors FP8 tensorwise: shard the output dim (axis 0),
        leave the contraction dim (axis 1) unsharded — we must not shard finer
        than group_size along the contraction axis.
        """
        per_word = 32 // self.bits
        n_words = math.ceil(in_features / per_word)
        n_groups = math.ceil(in_features / self.group_size)

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
