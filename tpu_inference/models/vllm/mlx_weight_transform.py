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
"""Load-time weight-stream transform bridging MLX checkpoints to vLLM's MoE
parameter layout (``Qwen3MoeForCausalLM``, ``HYV3ForCausalLM``).

MLX-stacked checkpoints differ from what vLLM expects in a few ways this
transform fixes, while leaving everything else untouched:

  * ``...mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`` arrive as a
    single STACKED ``[E, out, in*]`` tensor. vLLM's MoE expects PER-EXPERT keys
    ``...mlp.experts.{e}.{gate,up,down}_proj.{...}``; we slice axis 0. The
    ``weight`` stays ``uint32``-packed (downstream ``VllmMLXMoEMethod`` keeps
    4-bit and dequantizes in the forward).
  * ``model.embed_tokens.{weight,scales,biases}`` and ``lm_head.{...}`` are
    quantized in MLX, but the unquantized embedding/head modules want a single
    plain ``bf16`` ``.weight``. We buffer the triplet and emit one dequantized
    ``bf16`` weight (via ``mlx_dequantize``), dropping scales/biases.

Hy3-specific (``HYV3ForCausalLM``) fixes:

  * ``...mlp.router.gate.{weight,scales,biases}`` is an 8-bit affine triplet
    (per-module quant override), but vLLM's ``GateLinear`` is forced unquantized
    (fp32). We dequant the tiny ``[num_experts, hidden]`` gate to ``bf16`` at
    load (vLLM later strips ``router.`` -> ``mlp.gate``). Triplet integrity is
    enforced (incomplete -> raise).
  * ``...mlp.router.expert_bias`` (plain fp32 ``[num_experts]``) is renamed to
    ``...mlp.expert_bias`` -- vLLM registers the selection-bias parameter there
    (and shares it with ``FusedMoE.e_score_correction_bias``) but only remaps
    ``router.gate``, so without this rename the bias would KeyError at load.
  * ``...mlp.shared_mlp.{gate,up,down}_proj.{...}`` (the single shared expert,
    4-bit) keeps its ``shared_mlp.`` infix -- vLLM's ``HYV3MoEFused`` registers
    the shared expert as ``self.shared_mlp`` (an ``HYV3FeedForward``), so the
    param path is ``...mlp.shared_mlp.{gate,up,down}_proj.*`` (gate/up then merge
    into ``gate_up_proj`` via vLLM's stacked-params mapping). Stays 4-bit.

Everything else (attention q/k/v/o, qk_norm, dense-layer mlp, norms) passes
through unchanged.
"""

import re
from typing import Iterable, Iterator

import jax.numpy as jnp
import torch
from torchax.ops.mappings import j2t

from tpu_inference.layers.common.quantization import mlx_dequantize
from tpu_inference.utils import t2j

_SWITCH = re.compile(
    r"^(.*)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)
# Shared expert (Hy3): strip the ``shared_mlp.`` infix to the bare ``mlp.`` name
# vLLM's HYV3FeedForward registers; weight stays 4-bit-packed.
_SHARED = re.compile(
    r"^(.*)\.mlp\.shared_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)
# Router gate (Hy3): 8-bit affine triplet, dequantized to bf16 at load.
_ROUTER_GATE = re.compile(r"^(.*\.mlp\.router\.gate)\.(weight|scales|biases)$")
# The router gate is the only per-module 8-bit override (VllmMLXConfig.from_config
# validates this fail-fast), so its dequant bit-width is fixed at 8.
_ROUTER_GATE_BITS = 8
_DEQUANT_PREFIXES = ("model.embed_tokens", "lm_head")


def _dequant_to_bf16(weight: torch.Tensor, scales: torch.Tensor,
                     biases: torch.Tensor, group_size: int,
                     bits: int) -> torch.Tensor:
    # The weights arrive as PLAIN CPU torch.Tensor straight off the checkpoint
    # stream (load device is "cpu"; torchax env is NOT active yet), so we use the
    # repo's t2j() — the same idiom AWQ/FP8/unquantized use at load time — to
    # cross into JAX, NOT jax_view() (which asserts an already-torchax tensor).
    # The packed weight crosses as uint32; mlx_dequantize unpacks the 4-bit
    # nibbles and applies the affine (w = scale * q + bias) in XLA, returning
    # bf16. The .astype(bf16) guards the contract.
    #
    # We must materialize back to a PLAIN CPU torch.Tensor via j2t() (the mirror
    # of t2j), NOT torch_view(): torch_view yields a torchax-wrapped tensor, but
    # this weight is yielded into vLLM's load-time weight stream where the
    # torchax env is DISABLED. The consuming weight_loaders end in
    # param.data.copy_(loaded_weight) against a plain CPU param; a torchax src
    # would dispatch into __torch_dispatch__ and assert "torchax Tensors can only
    # do math within the torchax environment". j2t returns a real torch.Tensor.
    packed = weight if weight.dtype == torch.uint32 else weight.to(torch.uint32)
    w = mlx_dequantize(t2j(packed, use_dlpack=False),
                       t2j(scales, use_dlpack=False),
                       t2j(biases, use_dlpack=False),
                       group_size=group_size,
                       bits=bits)
    return j2t(w.astype(jnp.bfloat16))


def transform_mlx_weights(weights: Iterable[tuple[str, torch.Tensor]], *,
                          group_size: int, bits: int,
                          num_experts: int) -> Iterator[tuple[str, torch.Tensor]]:
    """Bridge an MLX weight stream to vLLM's Qwen3-MoE parameter names/layout.

    Args:
      weights: the raw ``(name, tensor)`` stream from the checkpoint.
      group_size: MLX affine quant group size (from hf_config quant block).
      bits: MLX quant bit-width (4).
      num_experts: number of experts to un-stack the ``switch_mlp`` tensors into.

    Yields:
      ``(name, tensor)`` pairs in vLLM's expected layout.
    """
    # Buffer embed/lm_head triplets so we can dequant once all three parts
    # (weight + scales + biases) have arrived.
    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        m = _SWITCH.match(name)
        if m is not None:
            prefix, proj, suffix = m.group(1), m.group(2), m.group(3)
            # Slice axis 0 into per-expert tensors; .contiguous() preserves the
            # uint32 dtype and the [out, in*] per-expert shape.
            for e in range(num_experts):
                yield (f"{prefix}.mlp.experts.{e}.{proj}.{suffix}",
                       tensor[e].contiguous())
            continue

        sh = _SHARED.match(name)
        if sh is not None:
            prefix, proj, suffix = sh.group(1), sh.group(2), sh.group(3)
            # Shared expert: KEEP the ``shared_mlp.`` infix. vLLM's HYV3MoEFused
            # registers the shared expert as ``self.shared_mlp`` (an
            # HYV3FeedForward), so the parameter path is
            # ``...mlp.shared_mlp.{gate,up,down}_proj.{weight,scales,biases}``.
            # (The bare ``...mlp.{proj}`` name only exists on the dense layers
            # < first_k_dense_replace, which never carry a ``shared_mlp`` infix.)
            # Stays 4-bit-packed; vLLM applies the MLX quant method and merges
            # gate/up into gate_up_proj via its stacked-params mapping.
            yield (f"{prefix}.mlp.shared_mlp.{proj}.{suffix}", tensor)
            continue

        rg = _ROUTER_GATE.match(name)
        if rg is not None:
            rg_base, suffix = rg.group(1), rg.group(2)
            slot = pending.setdefault(rg_base, {})
            slot[suffix] = tensor
            if {"weight", "scales", "biases"} <= slot.keys():
                # 8-bit affine triplet -> bf16 (the gate is unquantized in vLLM).
                yield (f"{rg_base}.weight",
                       _dequant_to_bf16(slot["weight"], slot["scales"],
                                        slot["biases"], group_size,
                                        _ROUTER_GATE_BITS))
                del pending[rg_base]
            continue

        if name.endswith(".mlp.router.expert_bias"):
            # Rename to the param name vLLM registers (it only remaps router.gate).
            yield (name[:-len(".router.expert_bias")] + ".expert_bias", tensor)
            continue

        base = next((p for p in _DEQUANT_PREFIXES
                     if name.startswith(p)
                     and name[len(p):] in (".weight", ".scales", ".biases")),
                    None)
        if base is not None:
            slot = pending.setdefault(base, {})
            slot[name[len(base) + 1:]] = tensor
            if {"weight", "scales", "biases"} <= slot.keys():
                yield (f"{base}.weight",
                       _dequant_to_bf16(slot["weight"], slot["scales"],
                                        slot["biases"], group_size, bits))
                del pending[base]
            continue

        yield (name, tensor)

    # Any embed/lm_head that was already plain bf16 (no scales/biases shipped)
    # never completed a triplet; pass its buffered parts through unchanged. A
    # router gate, however, always ships a full triplet -- an incomplete one is a
    # corrupt checkpoint, so fail loudly rather than emit a packed uint32 weight
    # into the unquantized GateLinear.
    for base, slot in pending.items():
        if _ROUTER_GATE.match(f"{base}.weight"):
            raise ValueError(
                f"MLX router gate {base!r} has an incomplete quant triplet "
                f"(present: {sorted(slot)}); expected weight+scales+biases.")
        for suffix, tensor in slot.items():
            yield (f"{base}.{suffix}", tensor)
