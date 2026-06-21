"""Synthetic DeepSeek-V4-Flash *mini* config for routine/parity testing.

Tiny dims so the model instantiates in seconds, but keeps:
  * all three attention regimes (dense / CSA ratio-4 / HCA ratio-128),
  * the real quant formats (FP4 e2m1 experts, FP8 e4m3 block linears),
  * mesh-divisible dims for the DP-attention production mesh, whose head/group
    parallel `model` axis is size 4 (NOT 8): num_attention_heads % 4 == 0,
    n_routed_experts % 4 == 0, o_groups % 4 == 0 and o_groups | num_attention_heads,
  * head_dim == 512 (nope 448 + rope 64) — mla_swa.quantize_kv_inputs asserts this.
"""
from __future__ import annotations

# 4 layers: dense, CSA(ratio 4), HCA(ratio 128), dense.
MINI_COMPRESS_RATIOS = [1, 4, 128, 1]
MINI_NUM_LAYERS = len(MINI_COMPRESS_RATIOS)
MINI_NUM_HEADS = 8          # % 4 (model axis) == 0 -> 2 heads/shard
MINI_NUM_EXPERTS = 8        # % 4 == 0 and divisible by the EP axis
MINI_HIDDEN = 256
MINI_O_GROUPS = 4           # % 4 == 0 -> n_local_groups = 1/shard on the 4-way model axis


def make_dsv4_mini_config() -> dict:
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "hidden_size": MINI_HIDDEN,
        "intermediate_size": 512,
        "num_hidden_layers": MINI_NUM_LAYERS,
        "num_attention_heads": MINI_NUM_HEADS,
        "num_key_value_heads": 1,
        # Attention latent / MLA dims (head_dim must stay 512 for mla_swa).
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 128,
        "o_lora_rank": 128,
        "kv_lora_rank": 512,
        "o_groups": MINI_O_GROUPS,
        "sliding_window": 128,
        # Per-layer attention regime selector (base clamps to max(1, ratio)).
        "compress_ratios": list(MINI_COMPRESS_RATIOS),
        # MoE.
        "n_routed_experts": MINI_NUM_EXPERTS,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 256,
        "first_k_dense_replace": 0,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        # Norm / RoPE.
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "compress_rope_theta": 160000,
        "max_position_embeddings": 4096,
        "rope_scaling": {
            "type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 256,
        },
        # Quant: FP4 experts, FP8 block linears.
        "expert_dtype": "fp4",
        "moe_quant_algo": "MXFP4",
        "vocab_size": 1280,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
