import numpy as np
from pathlib import Path
from tests.utils.mlx_synthetic import build_synthetic_mlx_moe, pack_u4
from tpu_inference.layers.common.quantization import mlx_dequantize
import jax.numpy as jnp
import json
from safetensors.numpy import load_file


def test_pack_roundtrip():
    vals = np.arange(64).reshape(1, 64) % 16
    packed = pack_u4(vals)                       # [1, 8]
    assert packed.shape == (1, 8) and packed.dtype == np.uint32


def test_build_writes_mlx_keys_and_negative_scales(tmp_path: Path):
    info = build_synthetic_mlx_moe(tmp_path, layers=1, experts=8, hidden=128, moe_inter=64)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["quantization"]["group_size"] == 64 and cfg["quantization"]["bits"] == 4
    assert cfg["architectures"] == ["Qwen3MoeForCausalLM"]
    st = load_file(tmp_path / "model.safetensors")
    # stacked experts, leading dim = 8
    gk = "model.layers.0.mlp.switch_mlp.gate_proj"
    assert st[gk + ".weight"].dtype == np.uint32
    assert st[gk + ".weight"].shape[0] == 8
    assert (st[gk + ".scales"] < 0).any()        # adversarial: some negative scales


def test_golden_matches_dequant(tmp_path: Path):
    info = build_synthetic_mlx_moe(tmp_path, layers=1, experts=8, hidden=128, moe_inter=64)
    st = load_file(tmp_path / "model.safetensors")
    gk = "model.layers.0.mlp.switch_mlp.gate_proj"
    w = mlx_dequantize(jnp.asarray(st[gk + ".weight"]),
                       jnp.asarray(st[gk + ".scales"]),
                       jnp.asarray(st[gk + ".biases"]),
                       group_size=64, bits=4)
    np.testing.assert_allclose(np.asarray(w, dtype=np.float32),
                               info["golden"][gk].astype(np.float32), atol=0.05)
