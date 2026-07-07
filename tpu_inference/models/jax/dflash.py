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
"""DFlash draft model (block-diffusion speculative decoding), JAX-native.

Faithful port of the z-lab DFlash reference (e.g. z-lab/gpt-oss-20b-DFlash):
a small Qwen3-style transformer that drafts a whole block of tokens in one
non-causal forward pass.

Reference semantics per drafting step::

    combined  = hidden_norm(fc(concat(target aux hidden states)))
    per layer: ctx K/V    = k/v_proj(combined)          # static, cacheable
               noise Q/KV = q/k/v_proj(hidden states)   # evolving noise block
               attn       = softmax(Q @ [ctx K | noise K]^T) @ [ctx V | ...]
                            (non-causal: full context + bidirectional block)
    logits    = target lm_head(norm(hidden[noise block][1:]))

Paged TPU integration: the flat token stream carries, per request,
``[a_i freshly-accepted context rows | block_size noise rows]``.  Both streams
run through the transformer, but the context rows' K/V are overridden with
projections of ``combined`` before each attention call — so the paged
attention kernel's end-aligned cache write lands real context K/V at those
rows' positions and noise K/V right behind them, while
``use_causal_mask=False`` gives every noise token full visibility.  Context
rows' attention outputs are garbage and never read; they cannot pollute the
noise rows because cross-token flow happens only through the (overridden)
K/V.

The checkpoint ships neither ``embed_tokens`` nor ``lm_head``: both are
shared from the target model (gpt-oss has untied embeddings, so these are two
DIFFERENT matrices — using the input embedding for logits drops draft
acceptance to 0%).  The proposer overwrites the zero-initialized placeholders
after load.
"""

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.rope import GptOssRotaryEmbedding
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.qwen2 import Qwen2MLP
from tpu_inference.models.jax.utils.weight_utils import (BaseWeightLoader,
                                                         get_default_maps,
                                                         load_hf_weights)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()
zeros_init = nnx.initializers.zeros_init()


class DFlashAttention(nnx.Module):
    """Qwen3-style attention with the DFlash context-K/V override.

    Rows where ``is_ctx`` is True contribute K/V computed from the shared
    projected context (``combined_ctx``) instead of from the evolving hidden
    states; everything else is standard Qwen3 attention (per-head q/k-norm,
    YaRN RoPE, attention biases), run non-causally over the paged KV cache.
    """

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh,
                 quant_config):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.num_heads)
        self.mesh = mesh
        use_bias = bool(getattr(config, "attention_bias", False))

        # Modern transformers folds rope_theta + yarn params into
        # rope_parameters; older configs keep rope_theta / rope_scaling.
        rope = dict(getattr(config, "rope_parameters", None) or {})
        if not rope:
            rope = dict(getattr(config, "rope_scaling", None) or {})
        rope_theta = rope.get("rope_theta") or getattr(
            config, "rope_theta", 10000.0)
        self.rotary_emb = GptOssRotaryEmbedding(
            head_dim=self.head_dim,
            rope_theta=rope_theta,
            dtype=dtype,
            initial_context_length=rope.get(
                "original_max_position_embeddings", 4096),
            rope_scaling_factor=rope.get("factor", 1.0),
            rope_ntk_alpha=rope.get("beta_slow", 1.0),
            rope_ntk_beta=rope.get("beta_fast", 32.0),
        )

        self.q_proj = JaxEinsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            bias_shape=(self.num_heads, self.head_dim) if use_bias else None,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(zeros_init, ("model", None)),
            rngs=rng,
            quant_config=quant_config,
        )
        self.k_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            bias_shape=(self.num_kv_heads,
                        self.head_dim) if use_bias else None,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(zeros_init, ("model", None)),
            rngs=rng,
            quant_config=quant_config,
        )
        self.v_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            bias_shape=(self.num_kv_heads,
                        self.head_dim) if use_bias else None,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(zeros_init, ("model", None)),
            rngs=rng,
            quant_config=quant_config,
        )
        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            bias_shape=(self.hidden_size, ) if use_bias else None,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            bias_init=nnx.with_partitioning(zeros_init, (None, )),
            rngs=rng,
            quant_config=quant_config,
        )
        self.q_norm = JaxRmsNorm(
            self.head_dim,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
        )
        self.k_norm = JaxRmsNorm(
            self.head_dim,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
        )

    def __call__(
        self,
        kv_cache: jax.Array,
        hidden_states: jax.Array,  # (T, D)
        combined_ctx: jax.Array,  # (T, D) — valid at ctx rows only
        is_ctx: jax.Array,  # (T,) bool
        md: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        positions = md.input_positions

        q = self.q_norm(self.q_proj(hidden_states))

        k_noise = self.k_norm(self.k_proj(hidden_states))
        v_noise = self.v_proj(hidden_states)

        # Context rows: K/V from the shared projected context, NOT from the
        # evolving hidden states. Same per-layer projections + k-norm + RoPE.
        k_ctx = self.k_norm(self.k_proj(combined_ctx))
        v_ctx = self.v_proj(combined_ctx)

        k = jnp.where(is_ctx[:, None, None], k_ctx, k_noise)
        v = jnp.where(is_ctx[:, None, None], v_ctx, v_noise)

        q, k = self.rotary_emb(q, k, positions)

        # Non-causal: the noise block attends to the whole context and
        # bidirectionally within itself. The kernel also writes k/v into the
        # paged cache end-aligned at [kv_len - q_len, kv_len).
        new_kv_cache, attn_out = attention(
            kv_cache,
            q,
            k,
            v,
            md,
            self.mesh,
            self.head_dim,
            use_causal_mask=False,
        )
        return new_kv_cache, self.o_proj(attn_out)


class DFlashDecoderLayer(nnx.Module):

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh,
                 quant_config):
        self.input_layernorm = JaxRmsNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
        )
        self.self_attn = DFlashAttention(config, dtype, rng, mesh,
                                         quant_config)
        self.post_attention_layernorm = JaxRmsNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
        )
        self.mlp = Qwen2MLP(config=config,
                            dtype=dtype,
                            rng=rng,
                            quant_config=quant_config)

    def __call__(
        self,
        kv_cache: jax.Array,
        hidden_states: jax.Array,
        combined_ctx: jax.Array,
        is_ctx: jax.Array,
        md: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        kv_cache, attn_out = self.self_attn(kv_cache, hidden_states,
                                            combined_ctx, is_ctx, md)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return kv_cache, residual + hidden_states


class DFlashWeightLoader(BaseWeightLoader):

    def __init__(self, vllm_config: VllmConfig, mesh: Mesh):
        super().__init__(vllm_config, framework="pt")
        self.vllm_config = vllm_config
        self.mesh = mesh

    def load_weights(self, model: "DFlashDraftModel", mappings: dict):
        # embed_tokens / lm_head are not in the checkpoint (the proposer
        # shares the target's weights in after load); materialize them so
        # check_all_loaded doesn't trip on abstract placeholders.
        from jax.sharding import NamedSharding, PartitionSpec
        for param in (model.embed_tokens, model.lm_head):
            if isinstance(param.value, jax.ShapeDtypeStruct):
                param.value = jax.device_put(
                    jnp.zeros(param.value.shape, param.value.dtype),
                    NamedSharding(self.mesh, PartitionSpec("model", None)))

        metadata_map = get_default_maps(
            self.vllm_config.speculative_config.draft_model_config, self.mesh,
            mappings)

        # Only what the checkpoint actually ships; embed_tokens / lm_head are
        # shared from the target by the proposer after load.
        filter_regex = (
            r"^(layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight|"
            r"layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)|"
            r"layers\.\d+\.self_attn\.(q_norm|k_norm)\.weight|"
            r"layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight|"
            r"(fc|hidden_norm|norm)\.weight)$")

        load_hf_weights(
            vllm_config=self.vllm_config,
            model=model,
            metadata_map=metadata_map,
            mesh=self.mesh,
            filter_regex=filter_regex,
            is_draft_model=True,
        )


class DFlashDraftModel(nnx.Module):
    """DFlash draft for speculative decoding; architectures=[DFlashDraftModel].

    Weight sources:
      - checkpoint: ``layers.*``, ``fc``, ``hidden_norm``, ``norm``
      - target model (shared post-load by the proposer): ``embed_tokens``,
        ``lm_head``
    """

    WeightLoader = DFlashWeightLoader

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh):
        self.vllm_config = vllm_config
        self.mesh = mesh
        rng = nnx.Rngs(rng_key)

        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        draft_config = spec_config.draft_model_config
        hf_config = draft_config.hf_config
        target_model_config = vllm_config.model_config
        dtype = target_model_config.dtype
        # DFlash draft checkpoints are unquantized bf16; never inherit the
        # target's (e.g. mxfp4) quant config.
        quant_config = None

        self.hidden_size = hf_config.hidden_size
        target_hidden_size = getattr(hf_config, "target_hidden_size",
                                     target_model_config.get_hidden_size())
        dflash_config = getattr(hf_config, "dflash_config", None) or {}
        target_layer_ids = dflash_config.get("target_layer_ids") or []
        num_target_layers = len(target_layer_ids) or getattr(
            hf_config, "num_target_layers", hf_config.num_hidden_layers)

        vocab_size = target_model_config.get_vocab_size()

        # Placeholders; the proposer shares the target's weights in (and
        # re-derives state leaves). Materialized as zeros so weight-loading
        # checks and jit tracing see concrete arrays.
        self.embed_tokens = nnx.Param(
            jnp.zeros((vocab_size, self.hidden_size), dtype=dtype),
            sharding=("model", None),
        )
        self.lm_head = nnx.Param(
            jnp.zeros((vocab_size, self.hidden_size), dtype=dtype),
            sharding=("model", None),
        )

        self.layers = nnx.List([
            DFlashDecoderLayer(hf_config, dtype, rng, mesh, quant_config)
            for _ in range(hf_config.num_hidden_layers)
        ])

        self.fc = JaxLinear(
            num_target_layers * target_hidden_size,
            self.hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
        )
        self.hidden_norm = JaxRmsNorm(
            self.hidden_size,
            epsilon=hf_config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
        )
        self.norm = JaxRmsNorm(
            self.hidden_size,
            epsilon=hf_config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
        )
        self.num_layers = hf_config.num_hidden_layers

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,  # (T,)
        target_hidden_states,  # (combined_ctx (T, D), is_ctx (T,) bool)
        attention_metadata: AttentionMetadata,
        _layer_name_to_kvcache_index=None,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array],
               Optional[jax.Array]]:
        combined_ctx, is_ctx = target_hidden_states

        x = jnp.take(self.embed_tokens.value, input_ids,
                     axis=0).astype(combined_ctx.dtype)

        # Resolve each draft layer's cache index from the runner's mapping;
        # fall back to the last num_layers entries.
        kv_index = dict(_layer_name_to_kvcache_index or ())
        draft_kv_start = len(kv_caches) - self.num_layers
        for i, layer in enumerate(self.layers):
            idx = kv_index.get(f"draft_layer.{i}", draft_kv_start + i)
            kv_caches[idx], x = layer(kv_caches[idx], x, combined_ctx, is_ctx,
                                      attention_metadata)

        hidden_states = self.norm(x)
        return kv_caches, hidden_states, [hidden_states], None

    def combine_hidden_states(self, hidden_states: jax.Array) -> jax.Array:
        """fc + hidden_norm over concatenated target aux hidden states."""
        return self.hidden_norm(self.fc(hidden_states))

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        """Logits through the (shared) target lm_head."""
        return hidden_states @ self.lm_head.value.T

    def load_weights(self, _rng_key: jax.Array):
        mappings = {
            # HF checkpoint key (".weight" stripped) -> model param path.
            "layers.*.input_layernorm": "layers.*.input_layernorm.weight",
            "layers.*.post_attention_layernorm":
            "layers.*.post_attention_layernorm.weight",
            "layers.*.self_attn.q_proj": "layers.*.self_attn.q_proj.weight",
            "layers.*.self_attn.k_proj": "layers.*.self_attn.k_proj.weight",
            "layers.*.self_attn.v_proj": "layers.*.self_attn.v_proj.weight",
            "layers.*.self_attn.o_proj": "layers.*.self_attn.o_proj.weight",
            "layers.*.self_attn.q_proj.bias":
            "layers.*.self_attn.q_proj.bias",
            "layers.*.self_attn.k_proj.bias":
            "layers.*.self_attn.k_proj.bias",
            "layers.*.self_attn.v_proj.bias":
            "layers.*.self_attn.v_proj.bias",
            "layers.*.self_attn.o_proj.bias":
            "layers.*.self_attn.o_proj.bias",
            "layers.*.self_attn.q_norm": "layers.*.self_attn.q_norm.weight",
            "layers.*.self_attn.k_norm": "layers.*.self_attn.k_norm.weight",
            "layers.*.mlp.gate_proj": "layers.*.mlp.gate_proj.weight",
            "layers.*.mlp.up_proj": "layers.*.mlp.up_proj.weight",
            "layers.*.mlp.down_proj": "layers.*.mlp.down_proj.weight",
            "fc": "fc.weight",
            "hidden_norm": "hidden_norm.weight",
            "norm": "norm.weight",
        }
        loader = self.WeightLoader(self.vllm_config, self.mesh)
        loader.load_weights(self, mappings)
