import numpy as np

from tests.dsv4.mini_config import (MINI_NUM_EXPERTS, MINI_NUM_HEADS,
                                     MINI_NUM_LAYERS, make_dsv4_mini_config)


def test_mini_config_has_all_three_regimes():
    cfg = make_dsv4_mini_config()
    ratios = cfg["compress_ratios"]
    assert len(ratios) == cfg["num_hidden_layers"] == MINI_NUM_LAYERS
    # base clamps to max(1, ratio): dense->1, CSA->4, HCA->128
    assert 1 in ratios, "need >=1 dense layer (ratio 1)"
    assert 4 in ratios, "need >=1 CSA layer (ratio 4)"
    assert 128 in ratios, "need >=1 HCA layer (ratio 128)"


def test_mini_config_is_mesh_divisible():
    # Production DP-attention mesh: model axis = 4 (NOT 8). Heads/groups shard
    # over the 4-way `model` axis; experts over expert/model. So dims must be
    # divisible by 4, and o_groups must be a multiple of 4 to keep n_local_groups>=1.
    cfg = make_dsv4_mini_config()
    assert cfg["num_attention_heads"] % 4 == 0
    assert cfg["n_routed_experts"] % 4 == 0
    assert cfg["o_groups"] % 4 == 0
    # o_groups must also divide num_attention_heads (heads_per_group integer).
    assert cfg["num_attention_heads"] % cfg["o_groups"] == 0
    assert MINI_NUM_HEADS == cfg["num_attention_heads"]
    assert MINI_NUM_EXPERTS == cfg["n_routed_experts"]


def test_mini_config_preserves_real_attention_dims():
    cfg = make_dsv4_mini_config()
    # mla_swa.quantize_kv_inputs asserts actual_head_dim == 512.
    assert cfg["head_dim"] == 512
    assert cfg["qk_rope_head_dim"] == 64
    assert cfg["head_dim"] - cfg["qk_rope_head_dim"] == 448  # nope dim
    assert cfg["sliding_window"] == 128
    assert cfg["rms_norm_eps"] == 1e-6
    assert cfg["num_experts_per_tok"] == 6
    assert cfg["expert_dtype"] == "fp4"
