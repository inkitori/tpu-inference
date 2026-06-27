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
"""Torchax wrapper for the PyTorch DFlash draft model.

Loads the original HuggingFace DFlash model (pure PyTorch) and wraps it
so it can run on TPU via torchax + JAX JIT.  This avoids rewriting the
model in JAX and guarantees numerical equivalence with the GPU reference.
"""

import functools
from typing import Any

import jax
import torch
import torch.nn
import torchax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from transformers import AutoModel

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import \
    shard_model_to_tpu
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def _pin(t: torch.Tensor, spec: PartitionSpec, mesh: Mesh) -> torch.Tensor:
    """Pin a torchax tensor's layout to ``spec`` via with_sharding_constraint.

    Head-sharding the draft attention projection WEIGHTS is not enough on the
    torchax path: the per-step forward reshapes q/k/v with ``.view(n,b,nh,hd)``
    and ``repeat_kv`` expands KVH->Hq with ``.expand().reshape()``; those ops
    cross the sharded head axis, so XLA inserts all-gathers and runs the score
    matmul REPLICATED (an 8-chip HLO dump showed 72 all-gathers, incl. the
    bf16[N,64,B,C+B] SCORES tensor gathered back to all 64 heads -- no
    speedup). Round-tripping the activation through ``jax_view`` and pinning it
    with ``jax.lax.with_sharding_constraint`` forces XLA to KEEP the head axis
    sharded across those reshapes, so the score matmul / softmax / scores@v run
    head-parallel (an isolated JAX probe measured 0.44ms head-sharded vs 4.27ms
    replicated, 9.6x). Pure layout constraint -- same math, same dtypes.
    """
    return torch_view(
        jax.lax.with_sharding_constraint(jax_view(t),
                                         NamedSharding(mesh, spec)))


class _DFlashRunner(torch.nn.Module):
    """Wrapper that adapts the HF DFlash model for ``functional_call``."""

    def __init__(self, dflash_model: torch.nn.Module, mesh: Mesh | None = None):
        super().__init__()
        self.dflash = dflash_model
        # Mesh the forward runs under; needed by _draft_forward_cached to pin
        # the attention head axis (see _pin). MUST be the same mesh the jit
        # runs on, so it is threaded in from DFlashTorchaxWrapper. Optional only
        # for unit tests that exercise the non-cached routing (which never pins).
        self._mesh = mesh

    def forward(self, **kwargs) -> torch.Tensor:
        if "hidden_state" in kwargs:
            return self._compute_logits(kwargs["hidden_state"],
                                        kwargs["embed_weight"])
        elif "raw_hidden" in kwargs:
            return self._combine_hidden(kwargs["raw_hidden"])
        elif "kv_project_raw" in kwargs:
            return self._kv_project(kwargs["kv_project_raw"],
                                    kwargs["kv_position_ids"])
        elif "cached_k" in kwargs:
            return self._draft_forward_cached(
                kwargs["noise_embedding"],
                kwargs["cached_k"],
                kwargs["cached_v"],
                kwargs["noise_position_ids"],
                kwargs["attention_mask"],
            )
        else:
            return self._draft_forward(
                kwargs["noise_embedding"],
                kwargs["target_hidden"],
                kwargs["position_ids"],
                kwargs.get("attention_mask"),
            )

    def _draft_forward(
        self,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the DFlash model (no KV cache, no causal mask).

        Args:
            noise_embedding: (1, block_size, D) embedded noise block.
            target_hidden: (1, ctx_len, num_layers*D) raw concatenated
                           target hidden states (NOT yet projected).
            position_ids: (1, ctx_len + block_size) positions for RoPE
                          covering both context and noise.
            attention_mask: (1, 1, 1, ctx_len + block_size) additive float
                           bias (bf16): 0.0 for valid ctx/noise keys and
                           finfo(bf16).min for padding ctx keys. Added
                           directly onto attn_weights in eager attention.
        Returns:
            hidden_states: (1, block_size, D) – the draft model output
                           after the final norm (before lm_head).
        """
        return self.dflash(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )

    def _combine_hidden(self, raw_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target hidden states through fc + norm."""
        return self.dflash.hidden_norm(self.dflash.fc(raw_hidden))

    def _kv_project(
        self,
        raw_new: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-layer draft K (post-RoPE) and V (no norm, no RoPE) for a
        set of context rows, in isolation.

        This reproduces EXACTLY the K/V the draft attention computes for the
        context rows during its forward (Qwen3DFlashAttention.forward), but over
        ONLY the given rows rather than the full [ctx|noise] concat. Because:
          * fc + hidden_norm are applied per-row,
          * k_proj / v_proj are per-row linear maps,
          * k_norm is RMSNorm over the head_dim, applied per (row, head),
          * RoPE is applied per position,
        projecting just these rows (with their absolute positions for RoPE) is
        numerically identical to slicing them out of the in-forward K/V.

        Args:
            raw_new: (N, M, raw_hidden_dim) bf16 raw concatenated target hidden
                     states for M context rows.
            position_ids: (N, M) int absolute context position of each row
                          (drives RoPE).
        Returns:
            (k_all, v_all): each (num_layers, N, M, KVH, head_dim). k_all is
            post-RoPE (matching the forward's k just before attention); v_all is
            the raw projected V (no norm, no RoPE), in the same per-row layout.
        """
        # Same as transformers.models.qwen3.modeling_qwen3.rotate_half, imported
        # exactly as the remote DFlash modeling code does.
        from transformers.models.qwen3.modeling_qwen3 import rotate_half

        cfg = self.dflash.config
        kvh = cfg.num_key_value_heads
        hd = getattr(cfg, "head_dim",
                     cfg.hidden_size // cfg.num_attention_heads)

        bsz, m = raw_new.shape[:-1]

        # Shared across layers: fc -> hidden_norm (matches forward line 177).
        th = self.dflash.hidden_norm(self.dflash.fc(raw_new))  # (N, M, D)

        # RoPE tables from the absolute context positions. rotary_emb uses `th`
        # only for dtype/device; cos/sin are (N, M, hd).
        cos, sin = self.dflash.rotary_emb(th, position_ids)
        # apply_rotary_pos_emb does cos.unsqueeze(1) -> (N, 1, M, hd) so it
        # broadcasts over the head axis of k in (N, KVH, M, hd).
        c = cos.unsqueeze(1)
        s = sin.unsqueeze(1)

        k_layers = []
        v_layers = []
        for layer in self.dflash.layers:
            sa = layer.self_attn
            # k_proj/v_proj are per-row linears with bias (attention_bias=True).
            k = sa.k_proj(th).view(bsz, m, kvh, hd)  # (N, M, KVH, hd)
            v = sa.v_proj(th).view(bsz, m, kvh, hd)  # (N, M, KVH, hd)
            # Remote order (lines 79-82): k_norm in (N, *, KVH, hd) layout, then
            # transpose to (N, KVH, *, hd), then RoPE on k (all positions).
            k = sa.k_norm(k).transpose(1, 2)  # (N, KVH, M, hd)
            k = k * c + rotate_half(k) * s  # (N, KVH, M, hd), post-RoPE
            k = k.transpose(1, 2)  # back to (N, M, KVH, hd)
            # v: NO norm, NO RoPE; keep the per-row (N, M, KVH, hd) layout.
            k_layers.append(k)
            v_layers.append(v)

        k_all = torch.stack(k_layers, dim=0)  # (L, N, M, KVH, hd)
        v_all = torch.stack(v_layers, dim=0)  # (L, N, M, KVH, hd)
        return k_all, v_all

    def _draft_forward_cached(
        self,
        noise_embedding: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        noise_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cache-consuming draft forward (DFLASH_KV_CACHE=1 path).

        Re-drives the DFlash decoder torchax-side WITHOUT recomputing context
        K/V. The context K (post-RoPE) and V for every cached row were projected
        and written into the per-slot caches in earlier steps (via
        ``_kv_project`` / ``_batched_kv_write``); here each layer only projects
        the B NOISE rows and attends over ``[cached ctx K/V | fresh noise K/V]``.
        This makes the per-step draft cost O(B) instead of O(C) -- the whole
        point of the KV cache. Numerically equivalent to ``_draft_forward`` over
        the full [ctx|noise] concat (the cached ctx K may differ ~1 bf16 ULP from
        a same-graph recompute because RoPE FMA fuses differently across the
        kv_project jit and this jit; greedy argmax is robust to it).

        Args:
            noise_embedding: (N, B, D) embedded noise block.
            cached_k: (L, N, C, KVH, hd) cached context K (post-RoPE), already
                      sliced to the active (num_reqs, padded_ctx) sub-block.
            cached_v: (L, N, C, KVH, hd) cached context V (no norm, no RoPE).
            noise_position_ids: (N, B) absolute positions of the noise rows
                                (the last B columns of the full position_ids);
                                drive RoPE for the noise q/k only.
            attention_mask: (N, 1, 1, C+B) additive float bias (bf16): 0.0 for
                            valid ctx/noise keys, finfo(bf16).min for padding
                            ctx keys. Added onto attn_weights, masks KEYS only.
        Returns:
            hidden_states: (N, B, D) draft output after the final norm.
        """
        from transformers.models.qwen3.modeling_qwen3 import (
            eager_attention_forward, rotate_half)

        cfg = self.dflash.config
        kvh = cfg.num_key_value_heads
        nh = cfg.num_attention_heads
        hd = getattr(cfg, "head_dim",
                     cfg.hidden_size // cfg.num_attention_heads)

        hidden_states = noise_embedding  # (N, B, D)
        n, b = hidden_states.shape[:2]

        # RoPE tables at the NOISE positions only (the cached ctx K already has
        # its RoPE baked in at write time). cos/sin: (N, B, hd). Applied to both
        # noise q and noise k below (q_len == B, so cos[..., -B:, :] == cos).
        cos, sin = self.dflash.rotary_emb(hidden_states, noise_position_ids)
        c = cos.unsqueeze(1)  # (N, 1, B, hd) -> broadcasts over head axis
        s = sin.unsqueeze(1)

        for layer_idx, layer in enumerate(self.dflash.layers):
            sa = layer.self_attn
            residual = hidden_states
            hs = layer.input_layernorm(hidden_states)  # (N, B, D)

            # noise q/k/v ONLY (no context projection).
            q = sa.q_proj(hs).view(n, b, nh, hd)
            q = sa.q_norm(q).transpose(1, 2)  # (N, Hq, B, hd)
            k_noise = sa.k_proj(hs).view(n, b, kvh, hd)
            v_noise = sa.v_proj(hs).view(n, b, kvh, hd)
            k_noise = sa.k_norm(k_noise).transpose(1, 2)  # (N, KVH, B, hd)
            v_noise = v_noise.transpose(1, 2)  # (N, KVH, B, hd)

            # RoPE on noise q and noise k (noise positions).
            q = q * c + rotate_half(q) * s
            k_noise = k_noise * c + rotate_half(k_noise) * s

            # cached ctx K/V for this layer: (N, C, KVH, hd) -> (N, KVH, C, hd)
            # to match the noise k/v layout, then concat along the key axis.
            k_ctx = cached_k[layer_idx].transpose(1, 2)  # (N, KVH, C, hd)
            v_ctx = cached_v[layer_idx].transpose(1, 2)  # (N, KVH, C, hd)
            k = torch.cat([k_ctx, k_noise], dim=2)  # (N, KVH, C+B, hd)
            v = torch.cat([v_ctx, v_noise], dim=2)  # (N, KVH, C+B, hd)

            # PIN the head axes so XLA keeps the attention sharded instead of
            # all-gathering q/k/v back to replicated across the .view/repeat_kv
            # reshapes (see _pin). q head axis = axis 1 (Hq); k/v head axis =
            # axis 1 (KVH). With these pinned, repeat_kv's KVH->Hq expand and
            # the score matmul stay head-parallel.
            q = _pin(q, PartitionSpec(None, ShardingAxisName.ATTN_HEAD, None,
                                      None), self._mesh)
            k = _pin(k, PartitionSpec(None, ShardingAxisName.KV_CACHE_HEAD,
                                      None, None), self._mesh)
            v = _pin(v, PartitionSpec(None, ShardingAxisName.KV_CACHE_HEAD,
                                      None, None), self._mesh)

            # GQA expansion (KVH -> Hq) happens INSIDE eager_attention_forward;
            # pass KVH-head k/v. Mask (N,1,1,C+B) broadcasts over (N,Hq,B,C+B).
            attn_output, _ = eager_attention_forward(
                sa,
                q,
                k,
                v,
                attention_mask,
                scaling=sa.scaling,
                dropout=0.0,
                sliding_window=sa.sliding_window,
            )
            attn_output = attn_output.reshape(n, b, -1)
            attn_output = sa.o_proj(attn_output)
            hidden_states = residual + attn_output

            residual = hidden_states
            hs = layer.post_attention_layernorm(hidden_states)
            hs = layer.mlp(hs)
            hidden_states = residual + hs

        return self.dflash.norm(hidden_states)  # (N, B, D)

    @staticmethod
    def _compute_logits(
        hidden_state: torch.Tensor,
        logits_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Draft logits: hidden @ W^T, where W is the target's OUTPUT
        projection (lm_head). For an untied target (gpt-oss) this is a
        distinct matrix from the input embedding; the proposer passes
        ``lm_head_weight_jax`` here, falling back to the embedding only when
        the target ties them."""
        return torch.nn.functional.linear(hidden_state, logits_weight)


class DFlashTorchaxWrapper:
    """Load the HF DFlash model on CPU, shard to TPU, expose JIT-compiled
    pure functions that the DFlash proposer can call."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.model: _DFlashRunner | None = None
        self.params: dict | None = None
        self.embed_weight_jax: jax.Array | None = None
        # Output projection (lm_head) weight used to turn draft hidden states
        # into draft logits. For a target with tie_word_embeddings=False (e.g.
        # gpt-oss) this is a DISTINCT matrix from the input embedding, so it
        # must be captured separately. Falls back to the input embedding when
        # the target ties them.
        self.lm_head_weight_jax: jax.Array | None = None

    def load(
        self,
        draft_model_path: str,
        target_model_state: Any,
    ) -> None:
        """Load HF DFlash model, shard weights to TPU, share embeddings."""

        logger.info("Loading DFlash PyTorch model via AutoModel from %s",
                    draft_model_path)

        with jax.default_device(jax.devices("cpu")[0]):
            hf_model = AutoModel.from_pretrained(
                draft_model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",  # pure-PyTorch ops for torchax
            )
            hf_model.eval()

        self.model = _DFlashRunner(hf_model, self.mesh)
        self.params = shard_model_to_tpu(self.model, self.mesh)
        # shard_model_to_tpu returns torchax tensors; convert to JAX view
        # so JAX JIT can trace through them.
        self.params = jax_view(self.params)

        # Head-shard the draft attention projections. shard_model_to_tpu()
        # REPLICATES every draft weight (PartitionSpec()), so the per-step
        # draft attention computes all 64 query heads on every chip (~48ms on
        # v6e-8). The q/k/v/o projections are head-parallel: re-shard their
        # head (output for q/k/v, contraction for o) dimension onto the
        # ShardingAxisName.ATTN_HEAD ('model', size 8) axis so each chip owns
        # 8 q-heads / 1 kv-head (~1.5ms). This is a pure layout change on the
        # already-on-device jax.Arrays (a resharding device_put): the values
        # are byte-identical, only their device placement changes. XLA then
        # propagates the head-sharding through q@k^T / softmax / scores@v
        # (per-head independent) and auto-inserts the o_proj all-reduce
        # (bf16 reduction-order differences only, already accepted -- see the
        # _draft_forward_cached docstring). nn.Linear weight is [out, in], so
        # the head axis is axis 0 for q/k/v (output) and axis 1 for o
        # (contraction); o_proj.bias and q_norm/k_norm stay replicated.
        self._reshard_draft_attn_heads()

        # Share weights from the target model. The DFlash draft model owns NO
        # embedding / lm_head of its own; the custom modeling code consumes a
        # `noise_embedding` (built from the target INPUT embedding) and projects
        # its output through the target OUTPUT projection (lm_head). For an
        # untied target (gpt-oss: tie_word_embeddings=False) these are two
        # DISTINCT matrices, so we capture both: the input embedding for the
        # noise embedding, and lm_head for the draft logits. The HF DFlash
        # reference does exactly this: noise_embedding = target.embed_tokens(.),
        # draft_logits = target.lm_head(draft_hidden).
        #
        # Two backends pass different target_model_state shapes:
        #  - flax_nnx target: an nnx.State whose `.model.embed_tokens`/`.embed`
        #    exposes the embedding (`.embedding` or `.weight`).
        #  - vllm/torchax target (our gpt-oss path): a flat params dict of
        #    jax.Arrays keyed by torch-style fully-qualified names. The gpt-oss
        #    input embedding lives at `vllm_model.model.embedding.weight`
        #    (VocabParallelEmbedding, replicated global jax.Array [vocab, hidden]).
        if isinstance(target_model_state, dict):
            for key in (
                    "vllm_model.model.embedding.weight",
                    "vllm_model.model.embed_tokens.weight",
                    "model.embedding.weight",
                    "model.embed_tokens.weight",
            ):
                if key in target_model_state:
                    self.embed_weight_jax = target_model_state[key]
                    break
            # The draft logits use the target's OUTPUT projection (lm_head),
            # NOT the input embedding. These are the same weight only when the
            # target ties them; gpt-oss has tie_word_embeddings=False, so the
            # lm_head is a separate [vocab, hidden] matrix. Capture it here and
            # fall back to the input embedding for tied targets.
            for key in (
                    "vllm_model.lm_head.weight",
                    "lm_head.weight",
            ):
                if key in target_model_state:
                    self.lm_head_weight_jax = target_model_state[key]
                    break
        else:
            target_model = getattr(target_model_state, "model", None)
            embed = None
            if target_model is not None:
                embed = getattr(target_model, "embed_tokens", None)
                if embed is None:
                    embed = getattr(target_model, "embed", None)
            if embed is not None:
                if hasattr(embed, "embedding"):
                    w = embed.embedding
                    # Unwrap nnx.Param → raw jax.Array
                    self.embed_weight_jax = w.value if hasattr(w,
                                                               "value") else w
                elif hasattr(embed, "weight"):
                    w = embed.weight
                    if hasattr(w, "value"):
                        self.embed_weight_jax = w.value
                    elif isinstance(w, torch.Tensor):
                        self.embed_weight_jax = jax_view(w)
                    else:
                        self.embed_weight_jax = w
        if self.embed_weight_jax is None:
            raise RuntimeError(
                "Could not find target model embedding to share with DFlash")

        # Tied-embedding targets expose no separate lm_head weight; reuse the
        # input embedding for the logits projection in that case.
        if self.lm_head_weight_jax is None:
            logger.info(
                "No separate target lm_head weight found; assuming tied "
                "embeddings and using the input embedding for DFlash draft "
                "logits.")
            self.lm_head_weight_jax = self.embed_weight_jax

        logger.info("DFlash torchax wrapper loaded successfully.")

    def _reshard_draft_attn_heads(self) -> None:
        """Re-shard the draft attention q/k/v/o projections on the head axis.

        ``shard_model_to_tpu`` replicates every draft weight; this post-pass
        re-shards ONLY the per-layer self-attention projection weights/biases
        so their head dimension lands on ShardingAxisName.ATTN_HEAD ('model').
        Matches on the torch fully-qualified key suffix; biases may be absent
        (skipped gracefully). Everything else (q_norm/k_norm, mlp, norms) is
        left replicated by the catch-all. Same math, only the layout changes.
        """
        H = ShardingAxisName.ATTN_HEAD  # 'model' under the active 2D sharding

        # key-suffix -> PartitionSpec (nn.Linear weight is [out, in]).
        #   q/k/v weight: shard output(head) dim (axis 0).
        #   q/k/v bias:   shard the single (head) dim.
        #   o   weight:   shard input/contraction(head) dim (axis 1); output
        #                 (D) replicated => XLA auto-inserts the all-reduce.
        #   o   bias:     replicated.
        suffix_specs = {
            ".self_attn.q_proj.weight": PartitionSpec(H, None),
            ".self_attn.q_proj.bias": PartitionSpec(H),
            ".self_attn.k_proj.weight": PartitionSpec(H, None),
            ".self_attn.k_proj.bias": PartitionSpec(H),
            ".self_attn.v_proj.weight": PartitionSpec(H, None),
            ".self_attn.v_proj.bias": PartitionSpec(H),
            ".self_attn.o_proj.weight": PartitionSpec(None, H),
            ".self_attn.o_proj.bias": PartitionSpec(),
        }

        resharded = 0
        for key in list(self.params.keys()):
            for suffix, spec in suffix_specs.items():
                if key.endswith(suffix):
                    self.params[key] = jax.device_put(
                        self.params[key], NamedSharding(self.mesh, spec))
                    resharded += 1
                    break

        logger.info(
            "DFlash draft attention head-sharded: re-sharded %d projection "
            "tensors onto axis %r.", resharded, H)

    def get_draft_forward_fn(self):
        """Return a JIT-compiled draft forward function.

        Signature::

            draft_forward(params, noise_input_ids, target_hidden,
                          position_ids, embed_weight, attention_mask,
                          num_reqs, padded_ctx) -> hidden_states

        ``target_hidden`` is the FULL persistent context buffer
        ``(max_num_reqs, buf_len, D')``; the active sub-block
        ``[:num_reqs, :padded_ctx]`` is sliced INSIDE this jit so XLA fuses the
        slice straight into the ``fc`` matmul that consumes it. Doing the slice
        here (rather than eagerly in prepare_inputs) avoids materializing a
        multi-GiB standalone copy of the context buffer on every decode step —
        that transient, on top of the persistent buffer, was the c=32 OOM.
        ``num_reqs`` and ``padded_ctx`` are static so the trace shape is fixed
        (one trace per (N, C), exactly as before).
        """
        model = self.model

        # Output is 3-D (N, block_size, D). MLP_DATA (= mesh 'data') is size 1
        # at DP=1 so the leading request axis is replicated (no divisibility
        # constraint on N); block + hidden stay replicated.
        hidden_sharding = NamedSharding(
            self.mesh, PartitionSpec(ShardingAxisName.MLP_DATA, None, None))

        @functools.partial(jax.jit,
                           out_shardings=hidden_sharding,
                           static_argnums=(6, 7))
        def draft_forward(
            params: dict,
            noise_input_ids: jax.Array,
            target_hidden: jax.Array,
            position_ids: jax.Array,
            embed_weight: jax.Array,
            attention_mask: jax.Array,
            num_reqs: int,
            padded_ctx: int,
        ) -> jax.Array:
            with torchax.default_env():
                # noise_input_ids/position_ids/attention_mask already carry the
                # active leading request axis N (= num_reqs). target_hidden is
                # the FULL buffer (max_num_reqs, buf_len, D'); slice the active
                # (N, C, D') sub-block here so the slice fuses into fc below.
                #   noise_input_ids (N, B), position_ids (N, C+B),
                #   attention_mask (N, C+B), sliced target_hidden (N, C, D').
                target_hidden = target_hidden[:num_reqs, :padded_ctx]
                p = torch_view(params)
                noise_ids_t = torch_view(noise_input_ids)  # (N, B)
                embed_w_t = torch_view(embed_weight)
                noise_emb = torch.nn.functional.embedding(
                    noise_ids_t, embed_w_t)  # (N, B, D)

                target_h = torch_view(target_hidden)  # (N, C, D')
                pos = torch_view(position_ids)  # (N, C+B)
                # attention_mask is an additive float bias (bf16) of shape
                # (N, C+B). Reshape to (N, 1, 1, C+B) so it broadcasts over the
                # head + query axes of attn_weights (N, H, B, C+B) in the HF
                # eager_attention_forward add -- masks keys per request, never
                # queries.
                mask_t = torch_view(attention_mask)  # (N, C+B)
                mask = mask_t.reshape(mask_t.shape[0], 1, 1,
                                      mask_t.shape[1])  # (N,1,1,C+B)

                output = torch.func.functional_call(
                    model,
                    p,
                    kwargs={
                        "noise_embedding": noise_emb,
                        "target_hidden": target_h,
                        "position_ids": pos,
                        "attention_mask": mask,
                    },
                    tie_weights=False,
                )
                # output: (N, block_size, D)
                return jax_view(output)

        return draft_forward

    def get_draft_forward_cached_fn(self):
        """Return a JIT-compiled cache-consuming draft forward function.

        Signature::

            draft_forward_cached(params, noise_input_ids, k_cache, v_cache,
                                 position_ids, embed_weight, attention_mask,
                                 win_start, num_reqs, padded_ctx)
                -> hidden_states

        The DFLASH_KV_CACHE=1 analogue of ``get_draft_forward_fn``. Instead of
        the raw context buffer it consumes the FULL per-slot K/V caches
        ``(L, max_num_reqs, buf_len, KVH, hd)`` and GATHERS the active
        ``(L, num_reqs, padded_ctx, KVH, hd)`` window INSIDE this jit (so XLA
        fuses the gather and we never materialize a standalone copy of the
        caches). ``win_start`` (N,) gives the per-slot start row of the window:
        all-zeros == the plain ``[0:padded_ctx)`` prefix (Lever A off); else
        ``max(0, ctx_len_i - W)`` selects the LAST ``W == padded_ctx`` context
        positions (Lever A on), shrinking the O(C*B) attention-score matmul to
        O(W*B). ``num_reqs``/``padded_ctx`` are static so the trace shape is
        fixed (one trace per (N, C), matching ``draft_forward``). Per-step
        compute is O(B): only the B noise rows are projected; the context K/V
        come straight from the caches. Output sharding matches ``draft_forward``.
        """
        model = self.model

        hidden_sharding = NamedSharding(
            self.mesh, PartitionSpec(ShardingAxisName.MLP_DATA, None, None))

        @functools.partial(jax.jit,
                           out_shardings=hidden_sharding,
                           static_argnums=(8, 9))
        def draft_forward_cached(
            params: dict,
            noise_input_ids: jax.Array,
            k_cache: jax.Array,
            v_cache: jax.Array,
            position_ids: jax.Array,
            embed_weight: jax.Array,
            attention_mask: jax.Array,
            win_start: jax.Array,
            num_reqs: int,
            padded_ctx: int,
        ) -> jax.Array:
            with torchax.default_env():
                # Gather the active (L, N, C, KVH, hd) sub-block out of the full
                # caches. C == padded_ctx (static). For each request slot i the
                # window covers absolute cache rows
                # [win_start[i] : win_start[i] + padded_ctx). When win_start is
                # all-zeros (windowing off) this is exactly the [0:padded_ctx)
                # prefix slice, so the non-windowed path is unchanged. When
                # windowing is on, win_start[i] = max(0, ctx_len_i - W) so the
                # window is the LAST W context positions -- the O(C*B) attention-
                # score matmul shrinks to O(W*B). The gather fuses into the
                # consuming attention; the cached K already carries its absolute-
                # position RoPE (baked at write time), so a contiguous row gather
                # preserves RoPE alignment with no re-RoPE.
                k_cache = k_cache[:, :num_reqs]  # (L, N, buf_len, KVH, hd)
                v_cache = v_cache[:, :num_reqs]
                ws = win_start[:num_reqs]  # (N,)

                def _win_slot(cache_ln, start):
                    # cache_ln: (L, buf_len, KVH, hd); take rows [start:start+C)
                    # along the buf_len axis (axis 1).
                    return jax.lax.dynamic_slice_in_dim(
                        cache_ln, start, padded_ctx, axis=1)

                # vmap over the request axis (axis 1 of the (L, N, ...) cache).
                k_cache = jax.vmap(_win_slot, in_axes=(1, 0),
                                   out_axes=1)(k_cache, ws)
                v_cache = jax.vmap(_win_slot, in_axes=(1, 0),
                                   out_axes=1)(v_cache, ws)
                p = torch_view(params)
                noise_ids_t = torch_view(noise_input_ids)  # (N, B)
                embed_w_t = torch_view(embed_weight)
                noise_emb = torch.nn.functional.embedding(
                    noise_ids_t, embed_w_t)  # (N, B, D)

                k_t = torch_view(k_cache)  # (L, N, C, KVH, hd)
                v_t = torch_view(v_cache)
                pos = torch_view(position_ids)  # (N, C+B)
                # The last B columns are the noise positions (RoPE for noise
                # q/k); the cached ctx K already has its RoPE baked in.
                noise_pos = pos[:, padded_ctx:]  # (N, B)
                # attention_mask is an additive float bias (N, C+B); reshape to
                # (N, 1, 1, C+B) so it broadcasts over (N, Hq, B, C+B) keys.
                mask_t = torch_view(attention_mask)  # (N, C+B)
                mask = mask_t.reshape(mask_t.shape[0], 1, 1, mask_t.shape[1])

                output = torch.func.functional_call(
                    model,
                    p,
                    kwargs={
                        "noise_embedding": noise_emb,
                        "cached_k": k_t,
                        "cached_v": v_t,
                        "noise_position_ids": noise_pos,
                        "attention_mask": mask,
                    },
                    tie_weights=False,
                )
                return jax_view(output)

        return draft_forward_cached

    def get_combine_hidden_fn(self):
        """Return a JIT-compiled combine_hidden_states function.

        Signature::

            combine_fn(params, raw_hidden) -> projected_hidden
        """
        model = self.model

        hidden_sharding = NamedSharding(
            self.mesh, PartitionSpec(ShardingAxisName.MLP_DATA, None))

        @functools.partial(jax.jit, out_shardings=hidden_sharding)
        def combine_fn(
            params: dict,
            raw_hidden: jax.Array,
        ) -> jax.Array:
            with torchax.default_env():
                p = torch_view(params)
                h = torch_view(raw_hidden)
                out = torch.func.functional_call(
                    model,
                    p,
                    kwargs={"raw_hidden": h},
                    tie_weights=False,
                )
                return jax_view(out)

        return combine_fn

    def get_kv_project_fn(self):
        """Return a JIT-compiled standalone K/V projection function.

        Signature::

            kv_project(params, raw_new, position_ids) -> (k_all, v_all)

        Computes, for ``M`` context rows, the per-layer draft K (post-RoPE) and
        V (no norm, no RoPE) that the draft attention produces for those rows
        during its forward -- but in isolation, over only those rows. See
        ``_DFlashRunner._kv_project`` for why this is bit-identical to slicing
        the rows out of the in-forward K/V.

        ``raw_new`` is (N, M, raw_hidden_dim); ``position_ids`` is (N, M) (the
        absolute context positions for RoPE). M may vary, so it is NOT static --
        the trace specializes per leading shape, exactly like the other fns.
        Both outputs are (num_layers, N, M, KVH, head_dim), head-sharded on the
        KVH axis (axis 3) to match the head-sharded K/V cache they are written
        into (avoids a reshard on write).
        """
        model = self.model

        # K/V outputs are (L, N, M, KVH, hd); shard the KVH axis (axis 3) on
        # ShardingAxisName.KV_CACHE_HEAD so they are produced in the same layout
        # as the head-sharded _k_cache/_v_cache that _batched_kv_write stores
        # them into (no reshard on write).
        kv_sharding = NamedSharding(
            self.mesh,
            PartitionSpec(None, None, None, ShardingAxisName.KV_CACHE_HEAD,
                          None))

        @functools.partial(jax.jit, out_shardings=(kv_sharding, kv_sharding))
        def kv_project(
            params: dict,
            raw_new: jax.Array,
            position_ids: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            with torchax.default_env():
                p = torch_view(params)
                raw_t = torch_view(raw_new)
                pos_t = torch_view(position_ids)
                out = torch.func.functional_call(
                    model,
                    p,
                    kwargs={
                        "kv_project_raw": raw_t,
                        "kv_position_ids": pos_t,
                    },
                    tie_weights=False,
                )
                return jax_view(out[0]), jax_view(out[1])

        return kv_project

    def get_compute_logits_fn(self):
        """Return a JIT-compiled compute_logits function.

        Signature::

            logits_fn(params, hidden_states, logits_weight) -> logits

        ``logits_weight`` is the target's OUTPUT projection (lm_head), not the
        input embedding (they differ for an untied target like gpt-oss).
        """
        model = self.model

        # Batched: hidden in (N, num_spec, D) -> logits (N, num_spec, vocab).
        # Leading request axis on MLP_DATA (size 1 -> replicated at DP=1);
        # vocab stays tensor-parallel on MLP_TENSOR exactly as before.
        logits_sharding = NamedSharding(
            self.mesh,
            PartitionSpec(ShardingAxisName.MLP_DATA, None,
                          ShardingAxisName.MLP_TENSOR))

        @functools.partial(jax.jit, out_shardings=logits_sharding)
        def logits_fn(
            params: dict,
            hidden_states: jax.Array,
            logits_weight: jax.Array,
        ) -> jax.Array:
            with torchax.default_env():
                p = torch_view(params)
                h = torch_view(hidden_states)
                w = torch_view(logits_weight)
                out = torch.func.functional_call(
                    model,
                    p,
                    kwargs={
                        "hidden_state": h,
                        "embed_weight": w
                    },
                    tie_weights=False,
                )
                return jax_view(out)

        return logits_fn
