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
    packed = pack_u4(q.reshape(out, in_))
    # store scales/biases as bf16 via ml_dtypes
    import ml_dtypes
    s_bf = scale.reshape(out, g).astype(ml_dtypes.bfloat16)
    b_bf = bias.reshape(out, g).astype(ml_dtypes.bfloat16)
    # Golden is the dequant of the *stored* (bf16) scale/bias -- this is what a real
    # MLX checkpoint ships and what any correct dequantizer must reproduce. Computing it
    # from the full-precision scale/bias would make golden inconsistent with the bf16
    # params returned here (off by a bf16 ulp).
    scale_bf = s_bf.astype(np.float32).reshape(out, g, 1)
    bias_bf = b_bf.astype(np.float32).reshape(out, g, 1)
    golden = (q.astype(np.float32) * scale_bf + bias_bf).reshape(out, in_)
    return packed, s_bf, b_bf, golden.astype(ml_dtypes.bfloat16)


# A small head layout that keeps q/k/v/o_proj == [hidden, hidden] and avoids
# TPU head-dim/num-head padding on a single-device "model" axis:
#   num_attention_heads == num_key_value_heads == 2 (no GQA), head_dim == 64
#   (get_padded_head_dim(64) == 64, get_padded_num_heads(2, 1) == 2).
# So num_heads * head_dim == hidden == 128 for the default hidden.
NUM_HEADS = 2
HEAD_DIM = 64
VOCAB = 256


def build_synthetic_mlx_moe(dir: Path, *, layers=2, experts=8, hidden=128,
                            moe_inter=64, group_size=64, seed=0) -> dict:
    dir = Path(dir)
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
    head_dim = HEAD_DIM

    # Embedding: quantized in real MLX repos; JaxEmbed has no int4 method in
    # phase 1, so it must load as plain bf16. We emit a plain bf16 embedding and
    # record golden so the bf16-reference builder reuses the EXACT same values.
    emb = rng.standard_normal((VOCAB, hidden)).astype(np.float32)
    tensors["model.embed_tokens.weight"] = emb.astype(ml_dtypes.bfloat16)
    golden["model.embed_tokens"] = np.asarray(
        tensors["model.embed_tokens.weight"]).astype(np.float32)

    for L in range(layers):
        pre = f"model.layers.{L}"
        # attention (dense linears, quantized)
        add_quant(f"{pre}.self_attn.q_proj", rng.standard_normal((hidden, hidden)).astype(np.float32), L == 0)
        add_quant(f"{pre}.self_attn.k_proj", rng.standard_normal((hidden, hidden)).astype(np.float32), False)
        add_quant(f"{pre}.self_attn.v_proj", rng.standard_normal((hidden, hidden)).astype(np.float32), False)
        add_quant(f"{pre}.self_attn.o_proj", rng.standard_normal((hidden, hidden)).astype(np.float32), False)
        # q/k norms (bf16, per-head_dim) + layer norms (bf16) -- not quantized
        for nm in ["self_attn.q_norm", "self_attn.k_norm"]:
            w = rng.standard_normal(head_dim).astype(np.float32)
            tensors[f"{pre}.{nm}.weight"] = w.astype(ml_dtypes.bfloat16)
            golden[f"{pre}.{nm}"] = np.asarray(
                tensors[f"{pre}.{nm}.weight"]).astype(np.float32)
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

    # lm_head: quantized (JaxEinsum -> Int4LinearMethod). HF/MLX store [V, D]
    # (out=V, in=D); the model kernel is "TD,DV->TV" (D,V), loaded with a
    # (1,0) transpose on the standard path, but the int4 method keeps the
    # packed [out=V, in=D] layout and contracts in=D at apply-time.
    add_quant("lm_head", rng.standard_normal((VOCAB, hidden)).astype(np.float32), False)

    save_file(tensors, str(dir / "model.safetensors"), metadata={"format": "mlx"})
    cfg = {
        "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
        "hidden_size": hidden, "num_hidden_layers": layers, "num_experts": experts,
        "num_experts_per_tok": min(2, experts), "moe_intermediate_size": moe_inter,
        "num_attention_heads": NUM_HEADS, "num_key_value_heads": NUM_HEADS,
        "vocab_size": VOCAB, "tie_word_embeddings": False,
        "decoder_sparse_step": 1, "mlp_only_layers": [],
        "quantization": {"group_size": group_size, "bits": 4},
        "quantization_config": {"group_size": group_size, "bits": 4},
    }
    (dir / "config.json").write_text(json.dumps(cfg))
    return {"golden": golden}


def build_bf16_reference_moe(dir: Path, golden: dict, *, layers=2, experts=8,
                             hidden=128, moe_inter=64) -> None:
    """Write a PLAIN bf16 (unquantized) Qwen3-MoE checkpoint from the SAME golden
    weights that ``build_synthetic_mlx_moe`` produced.

    This is the reference oracle for the e2e test: loading these bf16 tensors
    into an unquantized ``Qwen3MoeForCausalLM`` and running the same forward
    gives the logits the int4 path must match (the int4 path dequantizes to the
    exact same bf16 values, so the two forwards should agree to ~bf16 atol).

    Standard HF Qwen3-MoE layout (NOT MLX):
      * attention/router/lm_head/embed weights: plain bf16 ``.weight`` [out, in].
      * experts: PER-EXPERT keys ``mlp.experts.{i}.{gate,up,down}_proj.weight``,
        stored in standard HF ``[out, in]`` orientation (NOT transposed).

    Both this bf16 reference and the MLX 4-bit checkpoint are served, under
    ``MODEL_IMPL_TYPE=vllm``, by vLLM's own torch ``Qwen3MoeForCausalLM`` (it is
    a vLLM-preferred architecture), whose ``FusedMoE.weight_loader`` consumes
    per-expert ``gate/up/down_proj.weight`` in HF ``[out, in]`` and packs them
    into the stacked ``w13``/``w2`` buffers. The MLX path feeds exactly the same
    ``[out, in]`` golden values (un-transposed, just uint32-packed along ``in``)
    via ``transform_mlx_weights``. So the reference MUST store the golden weights
    in their native ``[out, in]`` orientation -- transposing them to ``[in, out]``
    only half-fills vLLM's ``torch.empty`` expert buffers (the loader silently
    narrow-clips to the loaded shape), leaving the other half uninitialized
    garbage that NaNs the whole forward.
    """
    dir = Path(dir)
    import ml_dtypes
    tensors = {}

    def _bf16(arr):
        return np.asarray(arr).astype(ml_dtypes.bfloat16)

    # Embedding -- plain bf16, identical values to the MLX checkpoint's embed.
    tensors["model.embed_tokens.weight"] = _bf16(golden["model.embed_tokens"])

    for L in range(layers):
        pre = f"model.layers.{L}"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"{pre}.self_attn.{proj}.weight"] = _bf16(
                golden[f"{pre}.self_attn.{proj}"])
        for nm in ("q_norm", "k_norm"):
            tensors[f"{pre}.self_attn.{nm}.weight"] = _bf16(
                golden[f"{pre}.self_attn.{nm}"])
        for nm in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"{pre}.{nm}.weight"] = np.ones(hidden,
                                                    dtype=ml_dtypes.bfloat16)
        tensors[f"{pre}.mlp.gate.weight"] = _bf16(golden[f"{pre}.mlp.gate"])
        for proj in ("gate_proj", "up_proj", "down_proj"):
            stacked = golden[f"{pre}.mlp.switch_mlp.{proj}"]  # [E, out, in]
            for e in range(experts):
                # Store HF-standard [out, in] (un-transposed). vLLM's FusedMoE
                # loader and the MLX path both consume per-expert weights in this
                # orientation; the MLX golden is already [out, in].
                tensors[f"{pre}.mlp.experts.{e}.{proj}.weight"] = _bf16(
                    stacked[e])

    tensors["model.norm.weight"] = np.ones(hidden, dtype=ml_dtypes.bfloat16)
    # lm_head: HF [V, D]; the standard loader transposes (1,0) to the (D, V)
    # kernel. Golden lm_head is the dequantized [out=V, in=D] = [V, D] matrix.
    tensors["lm_head.weight"] = _bf16(golden["lm_head"])

    save_file(tensors, str(dir / "model.safetensors"))
    cfg = {
        "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
        "hidden_size": hidden, "num_hidden_layers": layers,
        "num_experts": experts, "num_experts_per_tok": min(2, experts),
        "moe_intermediate_size": moe_inter,
        "num_attention_heads": NUM_HEADS, "num_key_value_heads": NUM_HEADS,
        "vocab_size": VOCAB, "tie_word_embeddings": False,
        "decoder_sparse_step": 1, "mlp_only_layers": [],
    }
    (dir / "config.json").write_text(json.dumps(cfg))
