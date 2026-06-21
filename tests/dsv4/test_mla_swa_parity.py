"""Dense-MLA parity: the mla_swa kernel vs a pure-jax dense reference, using the
same ragged metadata our forward_mqa builds. <=window sequences => full causal.

This ports the proven cache/page-index build from
``tests/kernels/deepseek_v4/mla_swa_test.py`` (the passing kernel oracle) but with
mini-config-compatible dims: ``num_heads=8, head_dim=512, sliding_window=128,
page_size=16`` and all ``new_kv_lens <= 128`` so each sequence fits inside one
window starting from an empty cache. With ``kv_lens`` starting at 0 every attended
key is a freshly-written new token within one window, so the SWA mask never fires
and the kernel reduces to full causal attention over ``new_kv`` -- exactly what
``_dense_ref`` computes. The loose ``rtol=atol=0.1`` absorbs the kernel's internal
fp8 KV quantization (the reference uses unquantized bf16; that gap is expected).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.experimental.deepseek_v4 import mla_swa
from tpu_inference.kernels.experimental.deepseek_v4.mla_swa import (
    cdiv, get_dtype_packing)


def _fp8_quantize_dequantize_kv(kv):
    """Round-trip bf16 KV through the kernel's DSv4 fp8 format so the reference
    attends to the SAME quantized keys/values the kernel does. Mirrors
    ``mla_swa.quantize_kv_inputs`` / ``quantize_and_dequantize_ref_cache``:
    the first 448 dims (NoPE) are blocked into 7x64 and stored e4m3 with an
    e8m0 per-block scale; dims 448:512 (RoPE) stay bf16. Without this the
    reference would use unquantized bf16 KV and a handful of elements would
    drift just past the 0.1 tolerance -- the kernel oracle dequantizes its
    reference cache the same way (mla_swa_test.py:106-131)."""
    nope = kv[..., :448]
    rope = kv[..., 448:512]
    batch_dims = kv.shape[:-1]
    nope_blocked = nope.reshape(*batch_dims, 7, 64)
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    x_amax = jnp.clip(jnp.max(jnp.abs(nope_blocked), axis=-1, keepdims=True),
                      1e-4, None)
    sf = jnp.power(2.0, jnp.ceil(jnp.log2(x_amax / fp8_max)))
    fp8_quant = (nope_blocked * (1.0 / sf)).astype(jnp.float8_e4m3fn)
    scales = sf.reshape(*batch_dims, 7).astype(jnp.float8_e8m0fnu)
    dequant_nope = (fp8_quant.astype(jnp.bfloat16) *
                    scales[..., None].astype(jnp.bfloat16)).reshape(
                        *batch_dims, 448)
    return jnp.concatenate([dequant_nope, rope], axis=-1)


def _dense_ref(q, new_kv, kv_lens, cu_q_lens, sm_scale, sliding_window):
    # Per-seq dense MLA with causal + SWA mask (mla_swa_test.py:248-270).
    # Quantize/dequantize KV through the kernel's fp8 format so this is a
    # faithful parity check of the kernel math, not of fp8 rounding.
    new_kv = _fp8_quantize_dequantize_kv(new_kv)
    outs = []
    num_seqs = kv_lens.shape[0]
    for i in range(num_seqs):
        qs, qe = int(cu_q_lens[i]), int(cu_q_lens[i + 1])
        q_len = qe - qs
        if q_len == 0:
            continue
        kv_len = int(kv_lens[i])
        k = new_kv[qs:qe]                      # [kv segment] (<=window so all-new)
        q_i = q[qs:qe]                         # [q_len, heads, head_dim]
        attn = jnp.einsum("qnh,kh->nqk", q_i, k,
                          preferred_element_type=jnp.float32) * sm_scale
        q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32,
                                                             attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span
        mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        attn = jnp.where(mask, -1e30, attn)
        probs = jax.nn.softmax(attn, axis=-1).astype(k.dtype)
        out_i = jnp.einsum("nqk,kl->qnl", probs, k[:, :512]).astype(q_i.dtype)
        outs.append(out_i)
    return jnp.concatenate(outs, axis=0)


def test_mla_swa_dense_parity():
    """Run the real mla_swa ragged kernel on v6e-8 and assert it matches the
    independent pure-jax dense reference within rtol=atol=0.1."""
    rng = np.random.default_rng(1234)
    rng_key = jax.random.PRNGKey(1234)

    kv_dtype = jnp.bfloat16
    q_dtype = jnp.bfloat16
    kv_packing = get_dtype_packing(kv_dtype)

    # Mini-config (parity dims from the task brief). All new_kv_lens <= window.
    batch_size = 4
    num_heads = 8
    head_dim = 512
    sliding_window = 128
    page_size = 16

    # Cache/page-index build ported from mla_swa_test.setUp (lines 299-325).
    # Provision enough pages per seq to hold a full window's worth of tokens.
    pages_per_seq = cdiv(sliding_window * 10, page_size)
    page_indices = jnp.arange(batch_size * pages_per_seq, dtype=jnp.int32)

    # SWA cache lives in the DSv4 FP8 uint8 640-layout. The kernel writes the
    # quantized new KV into this cache itself; we start it at zeros.
    sw_physical_page_size = page_size + 4
    swc_cache = jnp.zeros(
        (
            batch_size * pages_per_seq,
            sw_physical_page_size // kv_packing,
            get_dtype_packing(jnp.uint8),
            640,
        ),
        dtype=jnp.uint8,
    )

    # Variable-length prefill, every length in (window//2, window] so each
    # sequence fits inside a single window from an empty cache.
    rng_key, subkey = jax.random.split(rng_key)
    new_kv_lens = jax.random.randint(
        subkey,
        shape=(batch_size,),
        minval=sliding_window // 2,
        maxval=sliding_window + 1,  # inclusive upper bound == sliding_window
        dtype=jnp.int32,
    )
    cu_q_lens = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32),
         jnp.cumulative_sum(new_kv_lens, dtype=jnp.int32)])
    # Fresh cache: kv_lens == new_kv_lens (all tokens are new this step).
    kv_lens = new_kv_lens
    total_tokens = int(jnp.sum(new_kv_lens))

    q = jnp.array(rng.random(size=(total_tokens, num_heads, head_dim),
                             dtype=np.float32)).astype(q_dtype)
    new_kv = jnp.array(rng.random(size=(total_tokens, head_dim),
                                  dtype=np.float32)).astype(kv_dtype)
    # All sequences are prefill (no decode): distribution = [decode, prefill_bnd, total].
    distribution = jnp.array([0, 0, batch_size], dtype=jnp.int32)

    assert int(jnp.max(new_kv_lens)) <= sliding_window

    out, _swc_cache, _lse, _m = (
        mla_swa.mla_sliding_window_ragged_paged_attention(
            q,
            new_kv,
            swc_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=1.0,
            sliding_window=sliding_window,
            num_queries_per_block=8,
            num_kv_pages_per_block=2,
            logical_page_size=page_size,
        ))
    out.block_until_ready()

    ref = _dense_ref(q, new_kv, kv_lens, cu_q_lens, sm_scale=1.0,
                     sliding_window=sliding_window)

    out_np = np.asarray(out)
    ref_np = np.asarray(ref)
    print(f"kv_lens: {np.asarray(kv_lens)}")
    print(f"cu_q_lens: {np.asarray(cu_q_lens)}")
    print(f"out shape: {out_np.shape}  ref shape: {ref_np.shape}")
    print(f"Max Diff Out: {np.max(np.abs(out_np - ref_np))}")
    assert out_np.shape == ref_np.shape
    np.testing.assert_allclose(ref_np, out_np, rtol=0.1, atol=0.1)
