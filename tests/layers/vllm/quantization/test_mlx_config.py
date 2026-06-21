from tpu_inference.layers.vllm.quantization.mlx import VllmMLXConfig, is_mlx_quantized

class _HF:  # minimal stand-in for an HF config carrying an MLX quant block
    def __init__(self):
        self.quantization_config = {"group_size": 64, "bits": 4}

def test_is_mlx_quantized_true_for_groupsize_bits_without_quant_method():
    assert is_mlx_quantized(_HF()) is True

def test_is_mlx_quantized_false_when_quant_method_present():
    hf = _HF(); hf.quantization_config = {"group_size": 64, "bits": 4, "quant_method": "awq"}
    assert is_mlx_quantized(hf) is False

def test_from_config_parses_group_size_and_bits():
    cfg = VllmMLXConfig.from_config({"group_size": 64, "bits": 4})
    assert cfg.group_size == 64 and cfg.bits == 4
    assert VllmMLXConfig.get_name() == "mlx"
