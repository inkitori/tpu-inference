"""Parity: split-stream (ctx write + noise-only fwd) vs legacy combined
stream in DFlashDraftModel, same synthetic inputs, real weights, one chip.

Two steps (prefill-ish then decode-ish), mirroring parity_test.py.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")

import glob

import jax
import jax.numpy as jnp
import numpy as np

DRAFT_SNAP = glob.glob(
    "/tmp/gcs/bucket/hub/models--z-lab--gpt-oss-20b-DFlash/snapshots/*")[0]
TARGET_SNAP = glob.glob(
    "/tmp/gcs/bucket/hub/models--openai--gpt-oss-20b/snapshots/*")[0]

D_CTX = 14400
D = 2880
B = 8
P = int(os.environ.get("PARITY_P", "300"))
A2 = 3
MASK_ID = 200000
TOK0_STEP1 = 7
TOK0_STEP2 = 11

rng = np.random.default_rng(0)
raw_ctx_step1 = rng.standard_normal((P, D_CTX)).astype(np.float32) * 0.5
raw_ctx_step2 = rng.standard_normal((A2, D_CTX)).astype(np.float32) * 0.5
emb_tok0_s1 = rng.standard_normal((D, )).astype(np.float32) * 0.2
emb_tok0_s2 = rng.standard_normal((D, )).astype(np.float32) * 0.2
emb_mask = rng.standard_normal((D, )).astype(np.float32) * 0.2

from vllm.engine.arg_utils import EngineArgs

os.environ["VLLM_USE_V1"] = "1"

engine_args = EngineArgs(
    model=TARGET_SNAP,
    max_model_len=2048,
    tensor_parallel_size=1,
    speculative_config={
        "model": DRAFT_SNAP,
        "num_speculative_tokens": 7,
        "method": "dflash",
    },
)
vllm_config = engine_args.create_engine_config()

from tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64 import \
    get_kv_cache_shape
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.models.jax.dflash import DFlashDraftModel

vllm_config.model_config.dtype = jnp.bfloat16

devs = jax.devices()[:1]
mesh = jax.sharding.Mesh(
    np.array(devs).reshape((1, ) * len(MESH_AXIS_NAMES)), MESH_AXIS_NAMES)

with jax.set_mesh(mesh):
    model = DFlashDraftModel(vllm_config, jax.random.key(0), mesh)
    model.load_weights(jax.random.key(0))

    V = vllm_config.model_config.get_vocab_size()
    table = np.zeros((V, D), dtype=np.float32)
    table[TOK0_STEP1] = emb_tok0_s1
    table[TOK0_STEP2] = emb_tok0_s2
    table[MASK_ID] = emb_mask
    model.embed_tokens.value = jnp.asarray(table, dtype=jnp.bfloat16)

    PAGE = 16
    NPAGES = (P + A2 + B) // PAGE + 2
    cache_shape = get_kv_cache_shape(NPAGES, PAGE, 8, 64, jnp.bfloat16)
    layer_map = tuple((f"draft_layer.{i}", i) for i in range(8))

    def md_for(positions, kv_len, T):
        return AttentionMetadata(
            input_positions=jnp.asarray(positions, jnp.int32),
            block_tables=jnp.arange(NPAGES, dtype=jnp.int32),
            seq_lens=jnp.asarray([kv_len], jnp.int32),
            query_start_loc=jnp.asarray([0, T], jnp.int32),
            request_distribution=jnp.asarray([0, 0, 1], jnp.int32),
            padded_num_reqs=1,
        )

    def legacy_step(kv_caches, raw_ctx, tok0, pos_all, kv_len):
        a = raw_ctx.shape[0]
        T = a + B
        combined = model.combine_hidden_states(
            jnp.asarray(raw_ctx, dtype=jnp.bfloat16))
        combined_full = jnp.zeros((T, D), jnp.bfloat16).at[:a].set(combined)
        is_ctx = jnp.arange(T) < a
        ids = np.full((T, ), MASK_ID, np.int32)
        ids[:a] = 9
        ids[a] = tok0
        md = md_for(pos_all, kv_len, T)
        kv_caches, hidden, _, _ = model(kv_caches, jnp.asarray(ids),
                                        (combined_full, is_ctx), md,
                                        layer_map)
        return kv_caches, np.asarray(hidden.astype(jnp.float32))[a:]

    def split_step(kv_caches, raw_ctx, tok0, pos_all, kv_len):
        a = raw_ctx.shape[0]
        combined = model.combine_hidden_states(
            jnp.asarray(raw_ctx, dtype=jnp.bfloat16))
        ctx_md = md_for(pos_all[:a], kv_len - B, a)
        ids_noise = np.full((B, ), MASK_ID, np.int32)
        ids_noise[0] = tok0
        noise_md = md_for(pos_all[a:], kv_len, B)
        kv_caches, hidden, _, _ = model(kv_caches,
                                        jnp.asarray(ids_noise),
                                        (combined, ctx_md), noise_md,
                                        layer_map)
        return kv_caches, np.asarray(hidden.astype(jnp.float32))

    def fresh_caches():
        return [jnp.zeros(cache_shape, dtype=jnp.bfloat16) for _ in range(8)]

    # ---- legacy two steps
    kv_l = fresh_caches()
    pos1 = np.arange(P + B, dtype=np.int32)
    kv_l, leg1 = legacy_step(kv_l, raw_ctx_step1, TOK0_STEP1, pos1, P + B)
    pos2 = np.arange(P, P + A2 + B, dtype=np.int32)
    kv_l, leg2 = legacy_step(kv_l, raw_ctx_step2, TOK0_STEP2, pos2, P + A2 + B)

    # ---- split two steps
    kv_s = fresh_caches()
    kv_s, spl1 = split_step(kv_s, raw_ctx_step1, TOK0_STEP1, pos1, P + B)
    kv_s, spl2 = split_step(kv_s, raw_ctx_step2, TOK0_STEP2, pos2, P + A2 + B)

    # ---- compare caches too
    for i in (0, 7):
        dl = np.abs(np.asarray(kv_l[i], dtype=np.float32) -
                    np.asarray(kv_s[i], dtype=np.float32))
        print(f"cache layer {i}: max abs diff = {dl.max():.5f}")


def report(name, a, b):
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) *
                                   np.linalg.norm(b, axis=1) + 1e-9)
    print(f"{name}: max_abs={np.abs(a - b).max():.5f} per-row cos="
          f"{np.array2string(cos, precision=5, floatmode='fixed')}")


report("step1 noise hidden", leg1, spl1)
report("step2 noise hidden", leg2, spl2)
