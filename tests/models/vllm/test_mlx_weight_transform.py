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

import numpy as np
import torch

from tests.utils.mlx_synthetic import _quantize_affine
from tpu_inference.models.vllm.mlx_weight_transform import transform_mlx_weights


def _bf16_to_torch(arr: np.ndarray) -> torch.Tensor:
    # ADAPTATION vs brief: `_quantize_affine` returns scales/biases as
    # `ml_dtypes.bfloat16` numpy arrays, which `torch.from_numpy` cannot ingest
    # directly. Reinterpret the raw 16-bit pattern as `torch.bfloat16` (bf16 and
    # uint16 share an identical bit layout). Same values, true to the MLX
    # checkpoint dtype the transform sees in production.
    return torch.from_numpy(np.ascontiguousarray(arr).view(np.uint16)).view(
        torch.bfloat16)


def _packed_t(out, in_, gs):
    w = np.random.default_rng(0).standard_normal((out, in_)).astype(np.float32)
    packed, scales, biases, golden = _quantize_affine(w, gs, False)
    return (torch.from_numpy(packed.view(np.uint32).astype(np.uint32)),
            _bf16_to_torch(scales), _bf16_to_torch(biases), golden)


def test_experts_unstacked_and_renamed_kept_packed():
    E, out, in_, gs = 4, 16, 64, 64
    w = np.random.default_rng(0).standard_normal((E, out, in_)).astype(np.float32)
    # stack per-expert affine packs
    packs = [_quantize_affine(w[e], gs, False) for e in range(E)]
    stk_w = torch.from_numpy(np.stack([p[0] for p in packs]).astype(np.uint32))
    stk_s = _bf16_to_torch(np.stack([p[1] for p in packs]))
    stk_b = _bf16_to_torch(np.stack([p[2] for p in packs]))
    stream = [
        ("model.layers.0.mlp.switch_mlp.gate_proj.weight", stk_w),
        ("model.layers.0.mlp.switch_mlp.gate_proj.scales", stk_s),
        ("model.layers.0.mlp.switch_mlp.gate_proj.biases", stk_b),
    ]
    out_map = dict(transform_mlx_weights(stream, group_size=gs, bits=4, num_experts=E))
    for e in range(E):
        k = f"model.layers.0.mlp.experts.{e}.gate_proj.weight"
        assert k in out_map and out_map[k].dtype == torch.uint32
        assert f"model.layers.0.mlp.experts.{e}.gate_proj.scales" in out_map
    assert "switch_mlp" not in " ".join(out_map)


def test_embed_and_lm_head_dequantized_to_bf16_weight():
    out, in_, gs = 32, 64, 64
    pw, ps, pb, golden = _packed_t(out, in_, gs)
    stream = [
        ("model.embed_tokens.weight", pw),
        ("model.embed_tokens.scales", ps),
        ("model.embed_tokens.biases", pb),
    ]
    out_map = dict(transform_mlx_weights(stream, group_size=gs, bits=4, num_experts=1))
    assert set(out_map) == {"model.embed_tokens.weight"}
    w = out_map["model.embed_tokens.weight"]
    assert w.dtype == torch.bfloat16 and tuple(w.shape) == (out, in_)
    # The dequantized weight must be a PLAIN torch.Tensor, not a torchax-wrapped
    # subclass: it is yielded into vLLM's load-time weight stream (torchax env
    # DISABLED) where weight_loaders end in param.data.copy_(loaded_weight). A
    # torchax src would dispatch into __torch_dispatch__ and assert outside the
    # torchax env. exact-type check (isinstance passes for the subclass).
    assert type(w) is torch.Tensor
    # Exercise the exact failing path: copy_ into a plain CPU bf16 param.
    dst = torch.empty(w.shape, dtype=torch.bfloat16)
    dst.copy_(w)
    assert torch.equal(dst, w)


def test_attention_and_norms_pass_through():
    t = torch.zeros(4, 4, dtype=torch.uint32)
    stream = [("model.layers.0.self_attn.q_proj.weight", t),
              ("model.layers.0.input_layernorm.weight", torch.zeros(4))]
    out_map = dict(transform_mlx_weights(stream, group_size=64, bits=4, num_experts=1))
    assert "model.layers.0.self_attn.q_proj.weight" in out_map
    assert "model.layers.0.input_layernorm.weight" in out_map
