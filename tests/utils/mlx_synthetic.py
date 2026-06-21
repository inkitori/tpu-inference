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

import json
import numpy as np
from pathlib import Path
from safetensors.numpy import save_file


def pack_u4(vals: np.ndarray) -> np.ndarray:
    vals = vals.astype(np.uint32)
    *lead, n = vals.shape
    v = vals.reshape(*lead, n // 8, 8)
    word = np.zeros((*lead, n // 8), dtype=np.uint32)
    for k in range(8):
        word |= (v[..., k] & 0xF) << np.uint32(4 * k)
    return word


def _quantize_affine(w: np.ndarray, group_size: int, force_negative_scale: bool):
    # w: [out, in] -> packed uint32 [out, in/8], scales/biases bf16 [out, in/gs], plus bf16 golden
    out, in_ = w.shape
    g = in_ // group_size
    wg = w.reshape(out, g, group_size)
    lo = wg.min(-1, keepdims=True)
    hi = wg.max(-1, keepdims=True)
    scale = (hi - lo) / 15.0
    scale = np.where(scale == 0, 1.0, scale)
    bias = lo
    if force_negative_scale:               # adversarial: flip sign on half the groups
        flip = (np.arange(g) % 2 == 0)
        scale = scale.copy()
        scale[:, flip, :] *= -1.0
        bias = np.where(flip[None, :, None], hi, lo)
    q = np.round((wg - bias) / scale).clip(0, 15).astype(np.uint32)
    golden = (q.astype(np.float32) * scale + bias).reshape(out, in_)
    packed = pack_u4(q.reshape(out, in_))
    # store scales/biases as bf16 via ml_dtypes
    import ml_dtypes
    s_bf = scale.reshape(out, g).astype(ml_dtypes.bfloat16)
    b_bf = bias.reshape(out, g).astype(ml_dtypes.bfloat16)
    return packed, s_bf, b_bf, golden.astype(ml_dtypes.bfloat16)


def build_synthetic_mlx_moe(dir: Path, *, layers=2, experts=8, hidden=128,
                            moe_inter=64, group_size=64, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    tensors, golden = {}, {}

    def add_quant(name, w, negate):
        p, s, b, gold = _quantize_affine(w, group_size, negate)
        tensors[name + ".weight"] = p
        tensors[name + ".scales"] = s
        tensors[name + ".biases"] = b
        golden[name] = np.asarray(gold).astype(np.float32)

    def add_quant_stacked(name, w_stack, negate):  # w_stack: [E, out, in]
        ps, ss, bs, gs = [], [], [], []
        for e in range(w_stack.shape[0]):
            p, s, b, gold = _quantize_affine(w_stack[e], group_size, negate)
            ps.append(p); ss.append(s); bs.append(b); gs.append(np.asarray(gold).astype(np.float32))
        tensors[name + ".weight"] = np.stack(ps)
        tensors[name + ".scales"] = np.stack(ss)
        tensors[name + ".biases"] = np.stack(bs)
        golden[name] = np.stack(gs)

    import ml_dtypes
    for L in range(layers):
        pre = f"model.layers.{L}"
        # attention (dense linears)
        add_quant(f"{pre}.self_attn.q_proj", rng.standard_normal((hidden, hidden)).astype(np.float32), L == 0)
        add_quant(f"{pre}.self_attn.k_proj", rng.standard_normal((hidden, hidden)).astype(np.float32), False)
        add_quant(f"{pre}.self_attn.v_proj", rng.standard_normal((hidden, hidden)).astype(np.float32), False)
        add_quant(f"{pre}.self_attn.o_proj", rng.standard_normal((hidden, hidden)).astype(np.float32), False)
        # norms (bf16, not quantized)
        for nm in ["input_layernorm", "post_attention_layernorm"]:
            tensors[f"{pre}.{nm}.weight"] = np.ones(hidden, dtype=ml_dtypes.bfloat16)
        # router gate (quantized) + stacked experts
        add_quant(f"{pre}.mlp.gate", rng.standard_normal((experts, hidden)).astype(np.float32), False)
        add_quant_stacked(f"{pre}.mlp.switch_mlp.gate_proj",
                          rng.standard_normal((experts, moe_inter, hidden)).astype(np.float32), L == 0)
        add_quant_stacked(f"{pre}.mlp.switch_mlp.up_proj",
                          rng.standard_normal((experts, moe_inter, hidden)).astype(np.float32), False)
        add_quant_stacked(f"{pre}.mlp.switch_mlp.down_proj",
                          rng.standard_normal((experts, hidden, moe_inter)).astype(np.float32), False)

    tensors["model.norm.weight"] = np.ones(hidden, dtype=ml_dtypes.bfloat16)

    save_file(tensors, str(dir / "model.safetensors"), metadata={"format": "mlx"})
    cfg = {
        "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
        "hidden_size": hidden, "num_hidden_layers": layers, "num_experts": experts,
        "num_experts_per_tok": min(2, experts), "moe_intermediate_size": moe_inter,
        "vocab_size": 256, "tie_word_embeddings": False,
        "quantization": {"group_size": group_size, "bits": 4},
        "quantization_config": {"group_size": group_size, "bits": 4},
    }
    (dir / "config.json").write_text(json.dumps(cfg))
    return {"golden": golden}
