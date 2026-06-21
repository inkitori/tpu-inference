"""Synthetic DeepSeek-V4-Flash *mini* config for routine/parity testing.

Tiny dims so the model instantiates in seconds, but keeps:
  * all three attention regimes (dense / CSA ratio-4 / HCA ratio-128),
  * the real quant formats (FP4 e2m1fn + MXFP4 experts) AND the FP8 block-scaled
    linear quant config (the nested ``quantization_config`` block below), which
    makes the linears block-quantized FP8 with ue8m0 ``weight_scale_inv`` -- this
    is what the MLA ``wo_a`` PWAL dequant reads (Task 12 / Task 10). Without it,
    linears would be per-tensor FP8 (no ``weight_scale_inv``) and the wo_a dequant
    diverges,
  * mesh-divisible dims for the DP-attention production mesh, whose head/group
    parallel `model` axis is size 4 (NOT 8): num_attention_heads % 4 == 0,
    n_routed_experts % 4 == 0, o_groups % 4 == 0 and o_groups | num_attention_heads,
  * head_dim == 512 (nope 448 + rope 64) — mla_swa.quantize_kv_inputs asserts this.

The returned dict is fed to vLLM as ``hf_overrides`` (Task 12), which REPLACES the
hub config's fields verbatim -- so it must carry every key vLLM reads directly
(e.g. ``rope_scaling.rope_type``: vLLM's ``_get_and_verify_max_len`` indexes
``rope_type`` and the transformers ``type``->``rope_type`` normalization
(``patch_rope_parameters``) does NOT run on an override dict) and
``quantization:"deepseek_v4_fp8"`` (selects ``VllmDeepseekV4Fp8Config`` =
FP8 linears + FP4/mxfp4 experts).
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
            # vLLM reads "rope_type" directly (the "type" alias is normalized by
            # transformers' patch_rope_parameters, which does NOT run on an
            # hf_overrides dict). Keep both so the config works as an override and
            # via the normal HF path.
            "type": "yarn",
            "rope_type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 256,
        },
        # Quant: FP4/MXFP4 experts + FP8 block-scaled linears.
        # Top-level expert quant selectors:
        "expert_dtype": "fp4",
        "moe_quant_algo": "MXFP4",
        # quantization="deepseek_v4_fp8" -> VllmDeepseekV4Fp8Config (FP8 linears +
        # FP4/mxfp4 experts). Required so get_tpu_quantization_config picks the
        # DSV4 quant method (Task 12 / quantization/__init__.py).
        "quantization": "deepseek_v4_fp8",
        # Nested FP8-linear quant config. Mirrors the real DeepSeek-V4-Flash
        # quantization_config: block-quantized (weight_block_size) FP8 e4m3 with
        # ue8m0 scale_fmt -> creates a 2D weight_scale_inv per linear, which the
        # MLA wo_a PWAL dequant consumes. "activation_scheme" is REQUIRED
        # (get_from_keys raises if absent). "moe_quant_algo" is read by the
        # moe_quant_algo property of the quant config.
        "quantization_config": {
            "quant_method": "fp8",            # -> is_checkpoint_fp8_serialized
            "fmt": "e4m3",
            "activation_scheme": "dynamic",   # REQUIRED key
            "weight_block_size": [128, 128],  # -> block_quant=True -> weight_scale_inv
            "scale_fmt": "ue8m0",
            "moe_quant_algo": "MXFP4",
        },
        "vocab_size": 1280,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
