"""Parity test: JAX DFlashDraftModel vs the HF reference (real weights).

Simulates two drafting steps for one request:
  step 1 (prefill): 13 ctx rows + 8 noise rows, fresh cache
  step 2 (decode):  3 accepted ctx rows + 8 noise rows, cached ctx

Compares the draft block hidden states (post final norm) between:
  - HF DFlashDraftModel (torch CPU, bf16, eager, DynamicCache + crop)
  - our DFlashDraftModel (JAX TPU, paged KV cache, non-causal hd64 kernel)
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")

import glob

import jax
import jax.numpy as jnp
import numpy as np
import torch

DRAFT_SNAP = glob.glob(
    "/tmp/gcs/bucket/hub/models--z-lab--gpt-oss-20b-DFlash/snapshots/*")[0]
TARGET_SNAP = glob.glob(
    "/tmp/gcs/bucket/hub/models--openai--gpt-oss-20b/snapshots/*")[0]

D_CTX = 14400  # 5 layers * 2880
D = 2880
B = 8  # block size
P = 13  # prompt/ctx rows step 1
A2 = 3  # accepted rows step 2
MASK_ID = 200000
TOK0_STEP1 = 7  # noise[0] id step 1
TOK0_STEP2 = 11  # noise[0] id step 2

rng = np.random.default_rng(0)
raw_ctx_step1 = rng.standard_normal((P, D_CTX)).astype(np.float32) * 0.5
raw_ctx_step2 = rng.standard_normal((A2, D_CTX)).astype(np.float32) * 0.5
emb_tok0_s1 = rng.standard_normal((D, )).astype(np.float32) * 0.2
emb_tok0_s2 = rng.standard_normal((D, )).astype(np.float32) * 0.2
emb_mask = rng.standard_normal((D, )).astype(np.float32) * 0.2

# ---------------------------------------------------------------- HF side
print("=== HF reference (torch CPU) ===", flush=True)
from transformers import AutoModel
from transformers.cache_utils import DynamicCache

hf = AutoModel.from_pretrained(DRAFT_SNAP,
                               trust_remote_code=True,
                               torch_dtype=torch.bfloat16,
                               attn_implementation="eager")
hf.eval()

cache = DynamicCache()


def hf_step(raw_ctx, noise_embs, pos_start_cache, pos_end):
    target_hidden = torch.from_numpy(raw_ctx).to(torch.bfloat16).unsqueeze(0)
    noise_embedding = torch.stack(
        [torch.from_numpy(e).to(torch.bfloat16) for e in noise_embs],
        dim=0).unsqueeze(0)
    position_ids = torch.arange(pos_start_cache, pos_end).unsqueeze(0)
    with torch.no_grad():
        out = hf(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            is_causal=False,
        )
    return out.squeeze(0).float().numpy()


noise1 = [emb_tok0_s1] + [emb_mask] * 7
hf_out1 = hf_step(raw_ctx_step1, noise1, 0, P + B)  # positions 0..20
cache.crop(P)  # drop noise K/V, keep ctx [0,13)
noise2 = [emb_tok0_s2] + [emb_mask] * 7
# step 2: ctx rows at positions 13..15, noise at 16..23; cache holds 13
hf_out2 = hf_step(raw_ctx_step2, noise2, P, P + A2 + B)
print("hf_out1", hf_out1.shape, "hf_out2", hf_out2.shape, flush=True)

del hf, cache

# ---------------------------------------------------------------- JAX side
print("=== JAX DFlashDraftModel (TPU) ===", flush=True)
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
from tpu_inference.models.jax.dflash import DFlashDraftModel

from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES

# get_flax_model normally converts the torch dtype; constructing the model
# directly requires doing it by hand.
vllm_config.model_config.dtype = jnp.bfloat16

devs = jax.devices()[:1]
mesh = jax.sharding.Mesh(
    np.array(devs).reshape((1, ) * len(MESH_AXIS_NAMES)), MESH_AXIS_NAMES)

with jax.set_mesh(mesh):
    model = DFlashDraftModel(vllm_config, jax.random.key(0), mesh)
    model.load_weights(jax.random.key(0))

    # Fake embedding table: only rows we use matter.
    V = vllm_config.model_config.get_vocab_size()
    table = np.zeros((V, D), dtype=np.float32)
    table[TOK0_STEP1] = emb_tok0_s1
    table[TOK0_STEP2] = emb_tok0_s2
    table[MASK_ID] = emb_mask
    model.embed_tokens.value = jnp.asarray(table, dtype=jnp.bfloat16)

    # Paged KV cache: 8 layers, page_size 16, enough pages.
    PAGE = 16
    NPAGES = 8
    cache_shape = get_kv_cache_shape(NPAGES, PAGE, 8, 64, jnp.bfloat16)
    kv_caches = [
        jnp.zeros(cache_shape, dtype=jnp.bfloat16) for _ in range(8)
    ]
    layer_map = tuple((f"draft_layer.{i}", i) for i in range(8))

    def jax_step(kv_caches, raw_ctx, tok0, positions, kv_len, num_rows):
        a = raw_ctx.shape[0]
        combined = model.combine_hidden_states(
            jnp.asarray(raw_ctx, dtype=jnp.bfloat16))
        T = num_rows  # a + B (no padding for simplicity)
        combined_full = jnp.zeros((T, D), jnp.bfloat16).at[:a].set(combined)
        is_ctx = jnp.arange(T) < a
        ids = np.full((T, ), MASK_ID, np.int32)
        ids[:a] = 9  # arbitrary; ctx row embeds are irrelevant
        ids[a] = tok0
        md = AttentionMetadata(
            input_positions=jnp.asarray(positions, jnp.int32),
            block_tables=jnp.arange(NPAGES, dtype=jnp.int32),
            seq_lens=jnp.asarray([kv_len], jnp.int32),
            query_start_loc=jnp.asarray([0, T], jnp.int32),
            request_distribution=jnp.asarray([0, 0, 1], jnp.int32),
            padded_num_reqs=1,
        )
        kv_caches, hidden, _, _ = model(kv_caches, jnp.asarray(ids),
                                        (combined_full, is_ctx), md,
                                        layer_map)
        return kv_caches, np.asarray(hidden.astype(jnp.float32))

    # step 1: rows = 13 ctx + 8 noise, positions 0..20, kv_len 21
    pos1 = np.arange(P + B, dtype=np.int32)
    kv_caches, jax_out1_full = jax_step(kv_caches, raw_ctx_step1, TOK0_STEP1,
                                        pos1, P + B, P + B)
    jax_out1 = jax_out1_full[P:]  # noise rows

    # step 2: rows = 3 ctx (pos 13..15) + 8 noise (pos 16..23), kv_len 24
    pos2 = np.arange(P, P + A2 + B, dtype=np.int32)
    kv_caches, jax_out2_full = jax_step(kv_caches, raw_ctx_step2, TOK0_STEP2,
                                        pos2, P + A2 + B, A2 + B)
    jax_out2 = jax_out2_full[A2:]

def report(name, hf_out, jax_out):
    # hf_out: (block rows) — HF returns only the noise-block rows? The
    # reference returns norm(hidden) for ALL forwarded rows (ctx + noise);
    # slice its noise tail.
    hf_noise = hf_out[-B:]
    diff = np.abs(hf_noise - jax_out)
    denom = np.abs(hf_noise) + 1e-3
    print(f"{name}: max_abs={diff.max():.4f} mean_abs={diff.mean():.5f} "
          f"max_rel={(diff/denom).max():.3f} "
          f"cos={np.sum(hf_noise*jax_out)/ (np.linalg.norm(hf_noise)*np.linalg.norm(jax_out)):.6f}")

report("step1", hf_out1, jax_out1)
report("step2", hf_out2, jax_out2)
