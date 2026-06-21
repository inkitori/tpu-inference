"""Pure-torch, CPU-runnable reference building blocks for DSV4 Phase-1 parity.

These helpers are the numerical ground truth for later parity tests (MoE FP4
dequant, ``_o_proj`` inverse RoPE, invariants). The full vLLM AMD model cannot
run on CPU, so these reproduce the exact reference numerics (spec 6.3 C).

Conventions verified against vLLM source (editable install at /home/enyouki/vllm):

* ``rotate_gptj`` even/odd interleave:
  ``model_executor/layers/rotary_embedding/common.py:25-29``
* cos_sin_cache ``cat((cos, sin), -1)`` and GPT-J ``repeat_interleave(2)``
  application (NOT ``cat((cos, cos))``, which is the NeoX path):
  ``model_executor/layers/rotary_embedding/deepseek_scaling_rope.py:250-297``
* inverse RoPE = forward RoPE with ``sin -> -sin``:
  ``tests/kernels/test_fused_inv_rope_fp8_quant.py:145``
* ``rmsnorm_no_weight`` returns fp32:
  ``tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py:109-119``
* ``break_fp4_bytes`` table ``[0,.5,1,1.5,2,3,4,6]``, low nibble first:
  ``model_executor/layers/quantization/utils/nvfp4_emulation_utils.py:328``
* ``_upcast_e8m0_to_fp32`` (value = ``2**(byte - 127)``):
  ``model_executor/layers/quantization/utils/fp8_utils.py:1049``
"""
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
)


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #
def rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with no learnable weight; returns fp32.

    Matches ``RMSNorm(head_dim, has_weight=False)``. Output is fp32 so callers
    can chain RoPE without an intermediate bf16 round (the kernel keeps the
    whole RMSNorm->RoPE pipeline in fp32 and rounds once at the final store).
    """
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(var + eps)


def rmsnorm_with_weight(x: torch.Tensor, w: torch.Tensor,
                        eps: float) -> torch.Tensor:
    """RMSNorm with a learnable weight; cast back to the input dtype."""
    return (rmsnorm_no_weight(x, eps) * w.float()).to(x.dtype)


# --------------------------------------------------------------------------- #
# GPT-J rotate / RoPE
# --------------------------------------------------------------------------- #
def build_cos_sin_cache(rope_dim: int, max_pos: int, theta: float,
                        mscale: float = 1.0) -> torch.Tensor:
    """``[max_pos, rope_dim]`` == ``cat((cos, sin), -1)``.

    Mirrors ``DeepseekV4ScalingRotaryEmbedding._compute_cos_sin_cache``: there
    are ``rope_dim // 2`` frequencies ``inv_freq = theta ** -(arange(0, dim, 2)
    / dim)``, and both cos and sin are scaled by ``mscale``. Returned in fp32.
    """
    half = rope_dim // 2
    # inv_freq[j] = theta ** -(2j / rope_dim), j in [0, half).
    inv_freq = 1.0 / (theta**(torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos() * mscale
    sin = freqs.sin() * mscale
    return torch.cat((cos, sin), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    """GPT-J rotation: interleaved pairs (x0,x1),(x2,x3),... -> (-x1,x0,-x3,x2)."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _gptj_cos_sin(positions, cos_sin_cache, rope_dim):
    """Gather cos/sin for ``positions`` and expand GPT-J style.

    GPT-J interleaves pairs, so each half-frequency cos/sin is
    ``repeat_interleave(2)``-d (NOT concatenated as in NeoX).
    """
    cs = cos_sin_cache[positions.long()].to(torch.float32)  # [..., rope_dim]
    half = rope_dim // 2
    cos = cs[..., :half].repeat_interleave(2, dim=-1)
    sin = cs[..., half:].repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope_gptj_last_k(x, positions, cos_sin_cache,
                           rope_dim) -> torch.Tensor:
    """Forward GPT-J interleaved RoPE on the last ``rope_dim`` dims of ``x``.

    ``x`` is ``[T, H, D]``; the leading ``D - rope_dim`` ("nope") dims pass
    through unrotated. cos/sin are ``[T, rope_dim]`` and broadcast over heads.
    """
    cos, sin = _gptj_cos_sin(positions, cos_sin_cache, rope_dim)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    out = x.clone().float()
    rot = out[..., -rope_dim:]
    out[..., -rope_dim:] = rot * cos + rotate_gptj(rot) * sin
    return out.to(x.dtype)


def apply_inverse_rope_gptj_last_k(x, positions, cos_sin_cache,
                                   rope_dim) -> torch.Tensor:
    """Inverse GPT-J RoPE: forward RoPE with ``sin -> -sin``.

    Used by Task 8 ``_o_proj`` to undo the RoPE rotation. Same last-``rope_dim``
    slicing as the forward pass.
    """
    cos, sin = _gptj_cos_sin(positions, cos_sin_cache, rope_dim)
    cos = cos.unsqueeze(-2)
    sin = -sin.unsqueeze(-2)  # inverse RoPE: negate sin
    out = x.clone().float()
    rot = out[..., -rope_dim:]
    out[..., -rope_dim:] = rot * cos + rotate_gptj(rot) * sin
    return out.to(x.dtype)


# --------------------------------------------------------------------------- #
# FP4 e2m1 / ue8m0 dequant (thin re-export wrappers around vLLM utils)
# --------------------------------------------------------------------------- #
def break_fp4_e2m1(packed_u8: torch.Tensor, out_dtype) -> torch.Tensor:
    """Unpack packed FP4 e2m1 nibbles to ``out_dtype`` (low nibble first)."""
    return break_fp4_bytes(packed_u8, out_dtype)


def upcast_e8m0_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor:
    """Convert ue8m0 (unsigned 8-bit exponent, bias 127) block scales to fp32."""
    return _upcast_e8m0_to_fp32(scale_u8)
