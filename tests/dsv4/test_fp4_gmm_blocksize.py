# tests/dsv4/test_fp4_gmm_blocksize.py
import pytest

import tpu_inference.layers.vllm.quantization.mxfp4 as mxfp4


def test_requant_block_size_below_mxu_column_size():
    # v6e mxu_column_size == 256; block size must be < 256 to take the
    # dequant-in-VMEM branch in gmm_v2.should_dequantize_before_matmul.
    assert mxfp4.REQUANTIZED_BLOCK_SIZE < 256
    # Match the native MXFP4 group (and the NVFP4 precedent's small block).
    assert mxfp4.REQUANTIZED_BLOCK_SIZE == 32


def test_should_dequantize_true_for_new_block_size():
    # Mirror gmm_v2's decision on v6e: quant_block_size < mxu_column_size.
    import jax
    if jax.devices()[0].platform != "tpu":
        pytest.skip("gmm_v2 / get_tpu_info is TPU-only")
    from jax.experimental.pallas import tpu as pltpu
    mxu = pltpu.get_tpu_info().mxu_column_size
    assert mxu == 256, f"expected v6e mxu_column_size 256, got {mxu}"
    assert mxfp4.REQUANTIZED_BLOCK_SIZE < mxu


def test_dsv4_dispatches_to_mxfp4_moe_method():
    # Confirm the FP4 expert path routes to VllmMxfp4MoEMethod. Verified:
    # `if self.expert_dtype == "fp4":` (deepseek_v4_fp8.py:76) and
    # `return VllmMxfp4MoEMethod(...)` (deepseek_v4_fp8.py:83) both live in the
    # body of the regular instance method VllmDeepseekV4Fp8Config.get_quant_method.
    import inspect

    from tpu_inference.layers.vllm.quantization.deepseek_v4_fp8 import \
        VllmDeepseekV4Fp8Config
    body = inspect.getsource(VllmDeepseekV4Fp8Config.get_quant_method)
    assert "VllmMxfp4MoEMethod" in body
    assert 'self.expert_dtype == "fp4"' in body
