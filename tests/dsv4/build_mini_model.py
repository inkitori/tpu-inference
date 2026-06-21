"""Construct the DeepSeek-V4-Flash *mini* model on the torchax/vLLM path with
synthetic FP4-expert / FP8-linear weights, and run one eager forward.

This is the Task-12 INTEGRATION CRUX glue. It deliberately mirrors the ONE test
that drives the vllm/torchax build path -- ``tests/models/common/test_model_loader.py::
test_get_vllm_model`` -- and the runner's ``model_fn``/``compute_logits_fn`` call
sites (``tpu_inference/runner/tpu_runner.py``), but BYPASSES the runner so the
test owns the KV-cache pool and the AttentionMetadata (the production runner
KV-spec produces the wrong layout for the mla_swa kernel -- see DSV4 ledger; we
build the kernel-contract pool ourselves here).

No full-model load: ``load_format="dummy"`` -> vLLM ``DummyModelLoader``, then a
monkeypatch fills the uint8 FP4 expert tensors the dummy loader skips.
"""
from __future__ import annotations

import tempfile
from unittest.mock import patch

import jax
import jax.numpy as jnp
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tests.dsv4.mini_config import make_dsv4_mini_config


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# --------------------------------------------------------------------------- #
# Synthetic weight fill -- fills EVERY parameter in the real quant dtype.
# --------------------------------------------------------------------------- #
# This fully REPLACES vLLM's DummyModelLoader.initialize_dummy_weights because the
# TPU branch of initialize_single_dummy_weight (weight_utils.py:1306) crashes on
# fp8 params: it does `torch.rand(dtype=param.dtype)` which raises
# "check_uniform_bounds not implemented for Float8_e4m3fn". (The non-TPU branch
# handles fp8 via a float16 intermediate, but the TPU branch does not.) And it
# SKIPS the uint8 FP4/MXFP4 expert tensors entirely (`not torch.is_floating_point`
# -> return), leaving them torch.empty() garbage that dequants to NaN/Inf.
#
# So we fill all params ourselves, each in its stored quant dtype:
#   * uint8  experts:  *_weight = FP4-packed nibbles (any byte valid); *_weight_scale
#                      = packed e8m0 block scales kept in a sane magnitude band.
#   * float8_e4m3fn:   block-quantized FP8 linear weights (random via fp32 cast).
#   * float (bf16/f32): weight_scale_inv and any plain float weights -- small.
#   * other int:       zeroed (deterministic).
@torch.no_grad()
def fill_synthetic_weights(model: torch.nn.Module, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    n_uint8 = 0
    n_fp8 = 0
    for name, p in model.named_parameters():
        if p.device.type == "meta":
            continue  # deferred (online-quant) params handled elsewhere.
        if p.dtype == torch.uint8:
            if name.endswith("weight_scale"):
                # Packed e8m0 block scales: byte b -> 2**(b-127). Keep magnitudes
                # sane (124..131 -> 2^-3..2^+4); full 0..255 gives NaN/huge.
                p.copy_(torch.randint(124, 132, p.shape, generator=g,
                                      dtype=torch.uint8))
            else:
                # FP4-packed nibbles (e2m1): any byte is a valid pair of codes.
                p.copy_(torch.randint(0, 256, p.shape, generator=g,
                                     dtype=torch.uint8))
            n_uint8 += 1
        elif p.dtype == torch.float8_e4m3fn:
            # Random small fp8 weights via fp32 -> fp8 cast (no fp8 rand on TPU).
            tmp = (torch.rand(p.shape, generator=g, dtype=torch.float32) - 0.5) \
                * 0.1
            p.copy_(tmp.to(torch.float8_e4m3fn))
            n_fp8 += 1
        elif torch.is_floating_point(p):
            # weight_scale_inv (f32) and any plain float weight: keep small so the
            # block-scaled dequant stays well-conditioned.
            tmp = (torch.rand(p.shape, generator=g, dtype=torch.float32) - 0.5) \
                * 0.1 + 1.0
            p.copy_(tmp.to(p.dtype))
        else:
            p.zero_()  # other integer params: deterministic zero.
    if n_uint8 == 0:
        raise RuntimeError(
            "fill_synthetic_weights filled 0 uint8 expert tensors -- the model "
            "has no FP4/MXFP4 experts? Check the quant config plumbed through.")


# --------------------------------------------------------------------------- #
# VllmConfig construction (mirror test_get_vllm_model).
# --------------------------------------------------------------------------- #
def _make_vllm_config_from_mini(cfg: dict, block_size: int = 64):
    """Build the full VllmConfig offline from the mini cfg via EngineArgs.

    The mini cfg is injected as ``hf_overrides`` onto the real DeepSeek-V4-Flash
    repo id (resolved from the read-only HF cache -- config only, never weights).
    """
    from vllm.engine.arg_utils import EngineArgs

    # MLA models on TPU require DP-attention to be enabled in additional_config
    # (vLLM VllmConfig validation, and our 6-axis mesh selection).
    additional_config = {
        "sharding": {
            "sharding_strategy": {
                "enable_dp_attention": True
            }
        }
    }
    engine_args = EngineArgs(
        model="deepseek-ai/DeepSeek-V4-Flash",
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        load_format="dummy",          # -> vLLM DummyModelLoader (random weights)
        dtype="bfloat16",
        hf_overrides=cfg,
        max_model_len=cfg["max_position_embeddings"],
        trust_remote_code=True,
        additional_config=additional_config,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16
    # Explicitly pin the DSV4 quant method (hf_overrides plumbs it, but be sure):
    # get_tpu_quantization_config maps this -> VllmDeepseekV4Fp8Config.
    vllm_config.model_config.quantization = "deepseek_v4_fp8"
    # Small KV blocks: the runner default (1024) wastes the pool; we build the
    # pool ourselves below, sized off this.
    vllm_config.cache_config.block_size = block_size
    # DSV4 FlashMLA fp8 layout requires an fp8 kv-cache dtype (the attention
    # ctor asserts kv_cache_dtype.startswith("fp8") and rewrites it to the
    # canonical "fp8_ds_mla" = UE8M0 block-scaled fp8 packed as uint8). The
    # EngineArgs auto-detection logs fp8 but leaves cache_dtype="auto"; pin it.
    vllm_config.cache_config.cache_dtype = "fp8_ds_mla"
    return vllm_config


def build_mini_model(mesh, cfg=None):
    """Build the DSV4 mini-model under the torchax path + load synthetic weights.

    Returns ``(model, vllm_config)``. ``model`` is the ModelInterface from
    ``get_vllm_model`` (``.model_fn``, ``.compute_logits_fn``, ``.state_leaves``,
    ``.model`` = the VllmModelWrapper). ``_maybe_patch_for_deepseek_v4`` AUTO-fires
    inside ``VllmModelWrapper.load_weights`` (arch is DeepseekV4ForCausalLM) -- we
    do NOT call it manually.
    """
    cfg = cfg or make_dsv4_mini_config()
    vllm_config = _make_vllm_config_from_mini(cfg)

    # Replicate the production worker-init IR setup (WorkerBase.__init__ ->
    # worker_base.py:93-96). On TPU/eager, ir_enable_torch_wrap is False, which
    # makes vLLM IR ops (e.g. rms_norm) emit their pure-torch native composite
    # (aten ops torchax can trace) instead of the opaque `vllm_ir.*` custom op
    # (which torchax's __torch_dispatch__ does not intercept -> env assertion).
    # We bypass the runner/worker, so we must do this ourselves before tracing.
    import vllm.ir as _vllm_ir
    vllm_config.kernel_config.ir_op_priority.set_default()
    _vllm_ir.set_default_torch_wrap(
        vllm_config.compilation_config.ir_enable_torch_wrap)

    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized, init_distributed_environment)
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

    from tpu_inference.distributed.jax_parallel_state import \
        init_pp_distributed_environment
    from tpu_inference.models.common import model_loader

    rng = jax.random.PRNGKey(42)

    # vLLM's torch TP is collapsed to world_size=1 (TP=1/PP=1); the JAX MESH does
    # the real 8-way sharding. This is the repo-wide pattern (test_get_vllm_model,
    # tests/layers/vllm/*).
    with set_current_vllm_config(vllm_config):
        if not torch.distributed.is_initialized():
            temp_file = tempfile.mkstemp()[1]
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{temp_file}",
                backend="gloo",
            )
            ensure_model_parallel_initialized(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )
    # JAX single-rank pp group (the wrapper patches get_pp_group -> jax variant).
    init_pp_distributed_environment(
        ip="", rank=0, world_size=1, device=jax.devices()[0], need_pp=False)

    # Monkeypatch the dummy loader to fill ALL synthetic weights ourselves
    # (uint8 FP4 experts + fp8 linears + scales), BEFORE process_weights_after_
    # loading (which runs right after load_weights in BaseModelLoader.load_model;
    # no hook between the two). We do NOT call the original
    # DummyModelLoader.load_weights -- its TPU path crashes on fp8 dtypes
    # (torch.rand(dtype=float8_e4m3fn)) and skips the uint8 expert tensors.
    def _patched_load_weights(self, model, model_config):
        fill_synthetic_weights(model)
        return None

    with patch.object(DummyModelLoader, "load_weights", _patched_load_weights):
        model = model_loader.get_vllm_model(vllm_config, rng, mesh)

    # Build wo_a_bf16 on each MLA attention via the SAME production helper the
    # real loaders use (_process_dsv4_mla_weights in vllm_model_loader). vLLM's
    # post-load PWAL driver gates pass-2 on isinstance(module, (Attention,
    # MLAAttention, MMEncoderAttention)) -- VllmDeepseekV4MLAAttention is a
    # DeepseekV4Attention/AttentionLayerBase, matches NONE of those, so its
    # process_weights_after_loading (which sets self.wo_a_bf16, read by _o_proj)
    # never fires via that driver. The production IncrementalModelLoader /
    # RunaiIncrementalModelLoader now call _process_dsv4_mla_weights after their
    # process_weights_after_loading pass; the load_format="dummy" path used here
    # routes through vLLM's DummyModelLoader (not our loaders), so we invoke the
    # same helper explicitly to mirror production.
    import torchax

    from tpu_inference.models.vllm.vllm_model_loader import \
        _process_dsv4_mla_weights
    # model.model is the VllmModelWrapper; its .model is the _VllmRunner nn.Module.
    nn_module = model.model.model
    with torchax.default_env():
        _process_dsv4_mla_weights(nn_module, vllm_config.model_config)

    # Sanity: the helper must have found the patched MLA attention modules.
    from tpu_inference.layers.vllm.custom_ops.deepseek_v4_attention import \
        VllmDeepseekV4MLAAttention
    n_pwal = sum(
        1 for _, m in nn_module.named_modules()
        if isinstance(m, VllmDeepseekV4MLAAttention))
    if n_pwal == 0:
        raise RuntimeError(
            "no VllmDeepseekV4MLAAttention modules found -- the DSV4 MLA patch "
            "did not install (check _maybe_patch_for_deepseek_v4 / arch).")

    return model, vllm_config


# --------------------------------------------------------------------------- #
# Eager forward (build the KV pool + AttentionMetadata, call model_fn).
# --------------------------------------------------------------------------- #
def _attn_layers(vllm_config):
    """Enumerate attention modules -> {prefix: index}.

    NB: DeepseekV4Attention is NOT a subclass of vLLM's Attention/MLAAttention
    (it shares the AttentionLayerBase ancestor), so we enumerate via
    AttentionLayerBase. The dict keys equal each module's ``.prefix``.
    """
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import \
        AttentionLayerBase
    layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    return {name: i for i, name in enumerate(layers.keys())}


def build_mini_forward_args(model, vllm_config, input_ids, positions):
    """Build the positional args tuple for ``model.model_fn`` (Task-12 contract).

    Returns ``(model_fn, args, n)`` where ``model_fn`` is ``model.model_fn`` and
    ``args`` is the exact positional tuple passed to it (the same one
    ``run_mini_forward`` calls and the AOT gate lowers). Static positions of the
    underlying jitted ``run_model`` are (6, 9, 10): the
    ``layer_name_to_kvcache_index`` tuple, ``is_first_rank``, ``is_last_rank``.

    Builds the KV-cache pool to the mla_swa kernel contract (uint8, last-dim 640,
    REPLICATED ``P()``) and the GLOBAL-shaped AttentionMetadata (the shard_map
    localizes it per dp-rank). All values read off the live vllm_config -- not
    hardcoded. Single source of truth for both the eager forward and the AOT gate.
    """
    from tpu_inference.layers.common.attention_metadata import AttentionMetadata
    from tpu_inference.layers.common.sharding import ShardingAxisName

    # ModelInterface.model is the VllmModelWrapper, which carries the mesh.
    mesh = model.model.mesh
    n = int(input_ids.shape[0])

    # --- dims off the live config (mirror tpu_runner) --------------------------
    dp_size = vllm_config.sharding_config.total_dp_size
    block_size = vllm_config.cache_config.block_size
    sched = vllm_config.scheduler_config
    # R = max_num_reqs (runner: max(dp_size * max_num_seqs, MIN); use the simple
    # dp_size * max_num_seqs and pad up so R % dp_size == 0).
    R = dp_size * sched.max_num_seqs
    max_model_len = vllm_config.model_config.max_model_len
    P_blocks = _cdiv(max_model_len, block_size)        # max_num_blocks_per_req
    # T = padded token budget. One prefill of n tokens on dp-rank 0; give each
    # rank an equal token window large enough to hold n (kept tiny).
    per_rank_tokens = max(n, 1)
    T = per_rank_tokens * dp_size

    n_pages = _cdiv(n, block_size)

    # --- KV-cache pool (kernel contract: uint8 (nb, cdiv(bs,4), 4, 640)) -------
    n_attn_layers = len(_attn_layers(vllm_config))
    num_blocks = max(n_pages * dp_size + 2, dp_size + 2)
    kv_shape = (num_blocks, _cdiv(block_size, 4), 4, 640)
    kv_caches = [
        jax.device_put(jnp.zeros(kv_shape, jnp.uint8),
                       NamedSharding(mesh, P()))  # REPLICATED
        for _ in range(n_attn_layers)
    ]

    l2idx = _attn_layers(vllm_config)

    # --- AttentionMetadata (GLOBAL, dp-concatenated; shard_map splits axis 0) --
    attn_spec = NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA))

    def _put_attn(x):
        return jax.device_put(x, attn_spec)

    R_per = R // dp_size

    # seq_lens: i32[R], rank0 leads with [n, 0, ...], later ranks zeros.
    seq_lens = jnp.zeros((R,), jnp.int32).at[0].set(n)

    # block_tables: flat i32[R*P]. Row 0 = [0,1,...,n_pages-1, 0...].
    block_tables = jnp.zeros((R, P_blocks), jnp.int32)
    block_tables = block_tables.at[0, :n_pages].set(jnp.arange(n_pages,
                                                              dtype=jnp.int32))
    block_tables = block_tables.reshape(-1)

    # query_start_loc: i32[R + dp_size], dp-concatenated. Each rank gets a
    # [R_per + 1] block; rank0 block = [0, n, n, ...], other ranks all zeros.
    qsl = jnp.zeros((R + dp_size,), jnp.int32)
    # rank0 block occupies indices [0 : R_per+1]; set [1:] to n (cumsum of 1 seq).
    qsl = qsl.at[1:R_per + 1].set(n)

    # request_distribution: i32[3*dp_size]. rank0 triple [num_decode, bnd, n_reqs]
    # = [0, 0, 1] (one chunked-prefill seq); other ranks [0,0,0].
    request_distribution = jnp.zeros((3 * dp_size,), jnp.int32).at[2].set(1)

    # input_positions: i32[T], rank0 window [0..n-1, pad0].
    input_positions = jnp.zeros((T,), jnp.int32)
    input_positions = input_positions.at[:n].set(
        jnp.asarray(positions.numpy(), jnp.int32))

    # input_ids (separate model_fn arg): i32[T], rank0 holds the n ids then pad.
    input_ids_j = jnp.zeros((T,), jnp.int32)
    input_ids_j = input_ids_j.at[:n].set(
        jnp.asarray(input_ids.numpy(), jnp.int32))

    padded_num_reqs = int(R)

    attn_metadata = AttentionMetadata(
        input_positions=_put_attn(input_positions),
        block_tables=_put_attn(block_tables),
        seq_lens=_put_attn(seq_lens),
        query_start_loc=_put_attn(qsl),
        request_distribution=_put_attn(request_distribution),
        mamba_state_indices=None,
        padded_num_reqs=padded_num_reqs,
    )

    input_ids_sharded = _put_attn(input_ids_j)
    input_positions_sharded = _put_attn(input_positions)

    # --- model_fn positional args ---------------------------------------------
    # Signature mirrors tpu_runner.py: (params, kv_caches, input_ids, attn_md,
    # input_embeds, input_positions, layer_name_to_kvcache_index, lora_metadata,
    # intermediate_tensors, is_first_rank, is_last_rank). Positions 6/9/10 are the
    # static args of the underlying jitted run_model.
    args = (
        model.state_leaves,
        kv_caches,
        input_ids_sharded,
        attn_metadata,
        None,                          # input_embeds
        input_positions_sharded,
        tuple(l2idx.items()),          # static (pos 6)
        None,                          # lora_metadata
        None,                          # intermediate_tensors
        True,                          # is_first_rank   (static, pos 9)
        True,                          # is_last_rank    (static, pos 10)
    )
    return model.model_fn, args, n


def run_mini_forward(model, vllm_config, input_ids, positions):
    """One eager forward returning logits for a single N-token prefill request.

    Thin wrapper over ``build_mini_forward_args`` (the single source of truth for
    the model_fn args) + ``compute_logits_fn``.
    """
    model_fn, args, n = build_mini_forward_args(model, vllm_config, input_ids,
                                                positions)
    # --- call the jitted model_fn (sets wrapper + forward contexts for free) ---
    new_kv_caches, hidden_states, *_ = model_fn(*args)

    logits = model.compute_logits_fn(model.state_leaves, hidden_states, None)

    # Slice to the n real prefill tokens (rank0 window leads) and to torch.
    logits_np = jax.device_get(logits)
    return torch.from_numpy(logits_np[:n])
