# Copyright 2026 Google LLC
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
"""TPU interception for DeepSeek-V4 MLA attention (torchax path).

``DeepseekV4Attention`` is a plain ``nn.Module`` that vLLM instantiates directly
(the AMD decoder does ``self.attn = DeepseekV4ROCMAiterMLAAttention(...)``, a
``DeepseekV4Attention`` subclass, in ``deepseek_v4/amd/model.py``). Unlike the
MHC ops or the attention-impl bases, it is NOT a vLLM ``CustomOp`` and has no
``register_oot`` hook, so there is no registry-based way to swap it. Its
constructor is also CUDA-bound (allocates ``torch.cuda.Event``), so it cannot
run on TPU as-is.

Instead we substitute the class symbol before the model is built. Because
``amd/model.py`` does ``from ...amd.rocm import DeepseekV4ROCMAiterMLAAttention``,
the name is bound into the ``amd.model`` module namespace at import time; patching
it on ``amd.rocm`` alone would not take effect. ``patch_deepseek_v4_mla_cls``
rebinds it on ``amd.model`` directly. It is invoked from
``_maybe_patch_for_deepseek_v4`` in ``vllm_model_wrapper`` while ``is_rocm`` is
forced True and the package has been reloaded onto the AMD implementation.
"""
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view, torch_view
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention.attention import \
    get_attention_context
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from tpu_inference.kernels.experimental.deepseek_v4.mla_swa import \
    mla_sliding_window_ragged_paged_attention
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.backends.flash_attn_mla import \
    PallasMLAttentionBackend
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)


def build_swa_ragged_metadata(attn_metadata):
    """Map vLLM AttentionMetadata -> mla_swa kernel ragged args.

    Straight pass-through (verified): the TPU runner already produces all four
    fields in exactly the layout the kernel consumes, so no reshape and no
    per-rank slicing is needed here. ``block_tables`` is ALREADY flat 1D
    ``i32[max_seqs*pages_per_seq]`` on the TPU path (the runner does
    ``.reshape(-1)``), and ``request_distribution`` / ``query_start_loc`` are
    built GLOBAL (``i32[3*dp_size]`` / ``max_num_reqs + dp_size``) -- the
    DP->local localization is done by SPMD sharding inside the ``forward_mqa``
    shard_map, NOT by hand-slicing here.

    attention_metadata.py fields:
      seq_lens             -> kv_lens      i32[max_seqs]
      block_tables         -> page_indices i32[max_seqs*pages_per_seq] (flat)
      query_start_loc      -> cu_q_lens    i32[max_seqs+1]
      request_distribution -> distribution i32[3] = [decode, prefill_bnd, total]
    """
    return (attn_metadata.seq_lens, attn_metadata.block_tables,
            attn_metadata.query_start_loc, attn_metadata.request_distribution)


class VllmDeepseekV4MLAAttention(DeepseekV4Attention):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
    ) -> None:
        # Build the full DeepseekV4Attention parameter manifest (attn_sink,
        # fused_wqa_wkv, q_norm, wq_b, kv_norm, wo_a/wo_b, rotary_emb, plus the
        # per-layer indexer/compressor sub-modules) by running the real base
        # __init__, so the checkpoint loads with the correct FP8-block quant
        # wiring. The only GPU-bound line in the base __init__ is
        # ``self.ln_events = [torch.cuda.Event() ...]`` (CUDA stream ordering,
        # irrelevant on TPU). ``torch.cuda.Stream`` is already neutralized by
        # ``_maybe_patch_for_deepseek_v4`` in the wrapper; here we additionally
        # neutralize ``torch.cuda.Event`` for the duration of construction.
        _orig_event = torch.cuda.Event
        torch.cuda.Event = lambda *args, **kwargs: None
        try:
            super().__init__(
                vllm_config,
                prefix,
                topk_indices_buffer=topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            )
        finally:
            torch.cuda.Event = _orig_event
        # ``get_kv_cache_spec`` reports the cache dtype string straight from
        # cache_config (base stores the resolved ``kv_cache_dtype`` separately).
        self.cache_dtype = vllm_config.cache_config.cache_dtype
        # Logical KV page size = the vLLM cache block_size (a token count, not a
        # byte size, and NOT the physical cache tensor shape). Stored here for
        # the mla_swa kernel's required ``logical_page_size`` kwarg. Same proven
        # pattern as ``self.cache_dtype`` above. ``block_size`` is a concrete int
        # before layers are built.
        self._kv_block_size = vllm_config.cache_config.block_size

    # Abstract platform hooks required to instantiate the DeepseekV4Attention
    # ABC; unused on the TPU pass-through path.
    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def forward_mqa(self, q: torch.Tensor, kv: torch.Tensor,
                    positions: torch.Tensor, output: torch.Tensor) -> None:
        # Single ragged MLA+SWA kernel: it quantizes new_kv, writes the FP8 KV
        # cache, and applies BOTH causal and sliding-window masks internally.
        # There is NO Python decode/prefill branch -- the split is encoded in
        # `distribution`. No sink (Phase 1).
        #
        # CRITICAL: the kernel MUST be entered through an explicit jax.shard_map
        # with P(ATTN_DATA) on the four ragged-metadata args. On the production
        # DP-attention mesh, `distribution`/`query_start_loc` arrive GLOBAL
        # (i32[3*dp_size] / max_num_reqs+dp_size), but the kernel asserts the
        # local distribution shape is (3,). The surrounding plain jax.jit (no
        # in_shardings) does NOT localize a global array -- only entering a
        # shard_map with in_specs=P(ATTN_DATA) does. Mirrors the production MLA
        # path (attention_interface.mla_attention / sharded_ragged_paged_attention).
        attn_metadata, _, _, _ = get_attention_context(self.prefix)
        if attn_metadata is None:
            # Warmup dummy: no metadata -> produce zeros.
            output.zero_()
            return
        kv_lens, page_indices, cu_q_lens, distribution = \
            build_swa_ragged_metadata(attn_metadata)

        ctx = get_vllm_model_wrapper_context()
        mesh = ctx.mesh
        kv_cache_index = ctx.layer_name_to_kvcache_index[self.prefix]
        cache_kv = ctx.kv_caches[kv_cache_index]  # donated uint8 4D FP8 pool

        # q: [N, num_q_heads, head_dim] bf16; kv (new_kv): [N, head_dim] raw
        # bf16 (the kernel quantizes it internally); cache_kv stays uint8.
        q_j = jax_view(q.to(torch.bfloat16))
        kv_j = jax_view(kv.to(torch.bfloat16))
        cache_j = jax_view(cache_kv)

        # in_specs mirror attention_interface.mla_attention:526-539 /
        # sharded_ragged_paged_attention:366-379. q is token+head sharded; the
        # four ragged-metadata args each carry P(ATTN_DATA) so the global arrays
        # localize to the per-rank shard the kernel expects.
        q_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
        kv_spec = P(ShardingAxisName.ATTN_DATA, None)
        # Task 12: confirm the cache_kv pool sharding. DSv4 uses a SINGLE
        # physical page pool shared across attention types (page_indices select
        # into it), so a fully-replicated spec is the best-supported choice
        # here; the R1 MLA path's P(BATCH) cache layout is a different (per-rank)
        # cache and does not directly transfer. Confirm against the live KV
        # pool allocation on the DP-attention mesh.
        cache_spec = P()
        out_spec = q_spec
        # l/m are the lse/max aux returns, shape [N, num_q_heads]; mirror out's
        # token+head sharding minus the trailing head_dim axis.
        lm_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD)

        in_specs = (
            q_spec,  # q
            kv_spec,  # new_kv
            cache_spec,  # cache_kv (uint8 FP8 pool)
            P(ShardingAxisName.ATTN_DATA),  # kv_lens
            P(ShardingAxisName.ATTN_DATA),  # page_indices
            P(ShardingAxisName.ATTN_DATA),  # cu_q_lens
            P(ShardingAxisName.ATTN_DATA),  # distribution
        )
        out_specs = (out_spec, cache_spec, lm_spec, lm_spec)

        def _kernel(q, new_kv, cache, kv_lens, page_indices, cu_q_lens,
                    distribution):
            return mla_sliding_window_ragged_paged_attention(
                q,
                new_kv,
                cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                sm_scale=self.scale,
                sliding_window=self.window_size,
                logical_page_size=self._kv_block_size,
                num_kv_pages_per_block=2,
                num_queries_per_block=8,
            )

        out_j, updated_cache, _l, _m = jax.jit(
            jax.shard_map(
                _kernel,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            ))(q_j, kv_j, cache_j, jax_view(kv_lens),
               jax_view(page_indices), jax_view(cu_q_lens),
               jax_view(distribution))

        # Store the RAW JAX array back into the kv_caches list (NOT a
        # torch_view): step_fun_impl returns this list straight to JAX as
        # new_kv_caches without re-unwrapping, so a torchax.Tensor here leaks out
        # of the jit boundary ("not a valid JAX type"). Mirrors the production
        # MLA/flash paths (mla_attention.py:198, flash_attn.py:213), which store
        # the bare JAX array.
        ctx.kv_caches[kv_cache_index] = updated_cache
        output.copy_(torch_view(out_j))

    def _mla_swa_logical_page_size(self, cache_kv) -> int:
        # The logical page size == the vLLM KV-cache block_size (a logical token
        # count, NOT a byte size, and NOT the physical cache_kv tensor shape).
        # Stored on self in __init__ (self._kv_block_size). cache_kv is ignored.
        return self._kv_block_size

    def _o_proj(self, o: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        # o: [N, n_local_heads, head_dim]. Step 7 of the attention dataflow.
        # 1) inverse GPT-J RoPE on the last rope_head_dim dims, computed in fp32.
        o = self._inverse_rope_gptj(o, positions).to(o.dtype)
        # 2) group view -> grouped BMM via einsum("tgd,grd->tgr") over n_local_groups.
        #    wo_a_bf16 is the dequantized+reshaped weight built in
        #    process_weights_after_loading (Task 10): the raw self.wo_a.weight is
        #    2D float8_e4m3fn, NOT this layout — do NOT read it directly here.
        n = o.shape[0]
        o_g = o.reshape(n, self.n_local_groups, -1)   # [N, G, heads_per_g*head_dim]
        wo_a_w = self.wo_a_bf16  # bf16 [n_local_groups, o_lora_rank, heads_per_g*head_dim]
        z = torch.einsum("tgd,grd->tgr", o_g.float(),
                         wo_a_w.float()).to(o.dtype)  # [N, G, o_lora_rank]
        # 3) wo_b RowParallelLinear over the flattened [N, G*o_lora_rank].
        return self.wo_b(z.reshape(n, -1))

    def _inverse_rope_gptj(self, x: torch.Tensor,
                           positions: torch.Tensor) -> torch.Tensor:
        # GPT-J interleaved inverse RoPE on the last rope_head_dim dims (fp32).
        rope_dim = self.rope_head_dim
        cache = self.rotary_emb.cos_sin_cache  # [max_pos, rope_dim] = cat(cos, sin)
        cs = cache[positions.long()].float()
        half = rope_dim // 2
        cos = cs[..., :half].repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = -cs[..., half:].repeat_interleave(2, dim=-1).unsqueeze(-2)
        out = x.clone().float()
        rot = out[..., -rope_dim:]
        x1 = rot[..., ::2]
        x2 = rot[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        out[..., -rope_dim:] = rot * cos + rotated * sin
        return out

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.cache_dtype,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
            model_version="deepseek_v4",
        )

    def process_weights_after_loading(self, act_order: bool = False) -> None:
        # FP4 experts and FP8-block linears keep their quant; here we ONLY build
        # the bf16 grouped wo_a weight that _o_proj's einsum("tgd,grd->tgr")
        # consumes: [n_local_groups, o_lora_rank, heads_per_group*head_dim].
        # self.wo_a.weight is 2D float8_e4m3fn (n_local_groups*o_lora_rank,
        # heads_per_g*head_dim) with ue8m0 block scales in weight_scale_inv; the
        # 2D->3D reshape the GPU does at load (deepgemm_post_process_fp8_weight_
        # block) is CUDA-only, so do the dequant + reshape here. Mirrors
        # rocm_aiter_mla_sparse._get_cached_wo_a_bf16:967-1001.
        g = self.wo_a.bmm_batch_size          # == self.n_local_groups
        r = self.o_lora_rank
        # TPU linear-method PWAL transposes the weight to (in, out) = (d, g*r)
        # and re-stores it (unquantized.py:264 / fp8.py:155-156 jnp.transpose),
        # so the loaded weight is (d, g*r) -- NOT the GPU (g*r, d) layout. d is
        # therefore shape[0] (heads_per_group*head_dim), and we must .t() back to
        # (g*r, d) before the row-major (g outer, r inner) view that _o_proj's
        # einsum("tgd,grd->tgr") consumes as [g, r, d].
        d = self.wo_a.weight.shape[0]         # heads_per_group * head_dim
        if hasattr(self.wo_a, "weight_scale_inv"):
            w = self.wo_a.weight.t().contiguous().view(g, r, d).to(torch.float32)
            # weight_scale_inv is TPU-transposed to (d_blocks, g*r_blocks); .t()
            # restores the GPU (g*r_blocks, d_blocks) layout, so the per-group
            # view's trailing block dim is d_blocks == the TPU pre-transpose
            # shape[0]. (Mirrors the GPU ref view(g, -1, weight_scale_inv.
            # shape[-1]) which on the GPU (g*r_blocks, d_blocks) tensor is
            # d_blocks.) NOTE: this branch is DEAD on a real TPU FP8 build -- the
            # FP8 linear PWAL deletes weight_scale_inv and re-stores a
            # requantized scale as weight_scale (fp8.py:165-167), so the else
            # branch runs there. This branch survives only for hand-built parity
            # stubs in the GPU+ue8m0 (transposed) layout.
            scale = self._expand_wo_a_block_scales(
                self.wo_a.weight_scale_inv.t().contiguous().view(
                    g, -1, self.wo_a.weight_scale_inv.shape[0]), r, d)
            self.wo_a_bf16 = (w * scale).to(torch.bfloat16)
        else:
            # Non-quantized (synthetic) build: transpose back to (g*r, d), then
            # view+cast.
            self.wo_a_bf16 = self.wo_a.weight.t().contiguous().view(
                g, r, d).to(torch.bfloat16)

    def _expand_wo_a_block_scales(self, scale: torch.Tensor, r: int,
                                  d: int) -> torch.Tensor:
        # Decode ue8m0 -> fp32 then repeat_interleave each block up to (g, r, d).
        # scale: [g, r_blocks, d_blocks] ue8m0 (float8_e8m0fnu or raw uint8).
        # Mirrors rocm_aiter_mla_sparse._expand_2d_block_scales:863-874 +
        # _decode_e8m0_scales:853-860, with one widening: rocm decodes ONLY
        # float8_e8m0fnu via _upcast_e8m0_to_fp32 and value-casts everything
        # else. Real DSV4 checkpoints store weight_scale_inv as float8_e8m0fnu,
        # so that path is bit-identical to rocm. We additionally bit-reinterpret
        # raw uint8 as the same biased-exponent byte (2**(byte-127)) so a uint8
        # ue8m0 tensor (the synthetic-test format, and how some loaders surface
        # e8m0 bytes) decodes correctly instead of being treated as an integer.
        if scale.dtype == torch.float8_e8m0fnu or scale.dtype == torch.uint8:
            # ue8m0 byte -> fp32 power-of-two (== _upcast_e8m0_to_fp32:
            # exp_bits << 23, reinterpreted as float32).
            s = (scale.view(torch.uint8).to(torch.int32) << 23).view(
                torch.float32)
        else:
            s = scale.to(torch.float32)
        # rocm uses ceil(dim / num_blocks) as the repeat factor (matches our
        # r // r_blocks when blocks divide evenly, which they do for DSV4).
        r_blocks, d_blocks = s.shape[-2], s.shape[-1]
        block_r = -(-r // r_blocks)   # ceil
        block_d = -(-d // d_blocks)   # ceil
        s = torch.repeat_interleave(s, block_r, dim=-2)[..., :r, :]
        s = torch.repeat_interleave(s, block_d, dim=-1)[..., :, :d]
        return s

    def get_attn_backend(self) -> type[AttentionBackend]:
        return PallasMLAttentionBackend

    @staticmethod
    def _weighted_rmsnorm(x: torch.Tensor, w: torch.Tensor,
                          eps: float) -> torch.Tensor:
        # Pure-torch weighted RMSNorm, bit-algorithm-identical to the Triton
        # fused_q_kv_rmsnorm kernel AND tests/dsv4/torch_ref.rmsnorm_with_weight:
        # fp32 reduction over the last axis, weight multiplied in fp32 BEFORE the
        # single cast back, eps inside rsqrt(mean(x^2)+eps), plain weight (no
        # 1+w). We do NOT import fused_q_kv_rmsnorm: it is a @triton.jit kernel
        # with no torch/meta fallback, untraceable by torchax on TPU.
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (xf * w.float()).to(x.dtype)

    def _fused_qnorm_rope_kv_insert_q_only(self, q: torch.Tensor,
                                           kv: torch.Tensor,
                                           positions: torch.Tensor):
        # q: [N, n_local_heads, head_dim]; kv: [N, head_dim].
        # Replaces ONLY the q-side of the base CUDA _fused_qnorm_rope_kv_insert:
        #   (a) per-head weight-free RMSNorm over head_dim (fp32, then cast),
        #   (b) FORWARD GPT-J interleaved RoPE on the last rope_head_dim dims of
        #       q AND of kv (+sin; the mirror of Task-8 _o_proj's inverse -sin).
        # The kv fp8-quant + paged insert is left to the mla_swa kernel, so kv
        # is returned rope-applied but NOT quantized. Norm+rope stay in fp32 and
        # round once at the end (matches the base kernel's single store-round).
        rope_dim = self.rope_head_dim
        # weight-free per-head RMSNorm over head_dim (fp32 reduction).
        qf = q.float()
        q = (qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + self.eps)) \
            .to(q.dtype)
        cache = self.rotary_emb.cos_sin_cache  # [max_pos, rope_dim] = cat(cos,sin)
        cs = cache[positions.long()].float()
        half = rope_dim // 2
        cos = cs[..., :half].repeat_interleave(2, dim=-1)
        sin = cs[..., half:].repeat_interleave(2, dim=-1)

        def _rope(t, cos_, sin_):
            out = t.clone().float()
            rot = out[..., -rope_dim:]
            x1, x2 = rot[..., ::2], rot[..., 1::2]
            rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
            out[..., -rope_dim:] = rot * cos_ + rotated * sin_
            return out.to(t.dtype)

        q = _rope(q, cos.unsqueeze(-2), sin.unsqueeze(-2))  # broadcast over heads
        kv = _rope(kv, cos, sin)                            # [N, head_dim]
        return q, kv

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Steps 0-4,6-7 of the attention dataflow; step 5 (compressor/indexer)
        # skipped: every layer is forced dense through mla_swa (coherent <=128
        # tokens). We deliberately do NOT inherit the base forward/attention_impl
        # -- they pull in CUDA multi-stream machinery and the GPU fused
        # qnorm/rope/kv-insert custom op, none of which run on TPU.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype, device=hidden_states.device)

        # Steps 1-2: fused WQA/WKV GEMM -> split -> fused q/kv RMSNorm.
        # Call fused_wqa_wkv DIRECTLY (a MergedColumnParallelLinear, returns
        # (qr_kv, None)), NOT attn_gemm_parallel_execute -- the latter drives
        # CUDA multi-stream aux GEMMs (execute_in_parallel + ln_events) for the
        # indexer/compressor, which Phase 1 skips. (attention.py:194-201,409-412)
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        # Split sizes are [q_lora_rank, head_dim] = [1024, 512] (attention.py:339);
        # there is NO kv_lora_rank in DSV4 -- the KV latent dim IS head_dim (512).
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        # fused_q_kv_rmsnorm replaced by inline weighted RMSNorm (the Triton
        # kernel is untraceable on TPU -- see _weighted_rmsnorm). Same math.
        qr = self._weighted_rmsnorm(qr, self.q_norm.weight.data, self.eps)
        kv = self._weighted_rmsnorm(kv, self.kv_norm.weight.data, self.eps)

        # Step 3: wq_b -> [N, n_local_heads, head_dim].
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)

        # Step 4: q-side weight-free RMSNorm + FORWARD GPT-J RoPE on q and kv.
        # The kv fp8-quant + paged insert is done INSIDE the mla_swa kernel; we
        # hand raw (rope-applied) bf16 kv to it.
        q, kv = self._fused_qnorm_rope_kv_insert_q_only(q, kv, positions)

        # Step 6: attention. The mla_swa kernel output is [N, n_local_heads,
        # head_dim]; get_padded_num_q_heads returns num_heads unchanged so
        # padded_heads == n_local_heads and o_padded matches the kernel output
        # shape exactly (the slice below is then a no-op). If a future change
        # makes padded_heads > n_local_heads, the kernel must still write into
        # o_padded[:, :n_local_heads, :] (output.copy_ requires matching shapes).
        self.forward_mqa(q, kv, positions, o_padded)
        o = o_padded[:, :self.n_local_heads, :]

        # Step 7: inverse-rope o_proj.
        return self._o_proj(o, positions)


def patch_deepseek_v4_mla_cls() -> None:
    """Rebind ``DeepseekV4ROCMAiterMLAAttention`` to the TPU subclass.

    Must run after ``vllm.models.deepseek_v4.amd.model`` is imported (it holds
    its own ``from ...amd.rocm import DeepseekV4ROCMAiterMLAAttention``
    reference) and before the model is constructed.
    """
    import vllm.models.deepseek_v4.amd.model as ds_v4_amd_model
    ds_v4_amd_model.DeepseekV4ROCMAiterMLAAttention = VllmDeepseekV4MLAAttention
    logger.info(
        "Patched DeepseekV4ROCMAiterMLAAttention -> VllmDeepseekV4MLAAttention for TPU."
    )
