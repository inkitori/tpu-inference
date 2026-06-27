"""Dump the compiled HLO of the REAL _draft_forward_cached to see if the
attention runs head-sharded or replicated. Count all-gather / all-reduce and
check the per-head dim of the dot ops. Also report compiled flops/bytes.

Run:
  HF_HOME=/home/enyouki/local_hf \
  ~/tpu-tooling/tpu-env.sh python scratchpad/probe_hlo.py
"""
import os

os.environ.setdefault("HF_HOME", "/home/enyouki/local_hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

DRAFT = "z-lab/gpt-oss-20b-DFlash"
AX = ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp")


def main():
    from tpu_inference.layers.common.sharding import ShardingAxisName  # noqa
    from tpu_inference.models.vllm.dflash import DFlashTorchaxWrapper
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((1, 1, 1, 1, 8, 1)), axis_names=AX)
    with jax.set_mesh(mesh):
        import transformers
        cfg0 = transformers.AutoConfig.from_pretrained(DRAFT,
                                                       trust_remote_code=True)
        vocab = getattr(cfg0, "vocab_size", 201088)
        hidden = cfg0.hidden_size
        fake = jnp.zeros((vocab, hidden), dtype=jnp.bfloat16)
        w = DFlashTorchaxWrapper(mesh)
        w.load(DRAFT, {"vllm_model.model.embedding.weight": fake,
                       "vllm_model.lm_head.weight": fake})
        cfg = w.model.dflash.config
        L = cfg.num_hidden_layers
        KVH = cfg.num_key_value_heads
        HD = getattr(cfg, "head_dim", hidden // cfg.num_attention_heads)
        B = cfg.block_size
        params = w.params
        fwd = w.get_draft_forward_cached_fn()

        N, C = 32, 4096
        buf = C + 8
        kv_s = NamedSharding(mesh, PartitionSpec(None, None, None,
                                                 ShardingAxisName.KV_CACHE_HEAD,
                                                 None))
        kc = jax.device_put(jnp.zeros((L, N, buf, KVH, HD), jnp.bfloat16), kv_s)
        vc = jax.device_put(jnp.zeros((L, N, buf, KVH, HD), jnp.bfloat16), kv_s)
        nid = jnp.zeros((N, B), jnp.int32)
        pos = jnp.zeros((N, C + B), jnp.int32)
        att = jnp.zeros((N, C + B), jnp.bfloat16)
        ws = jnp.zeros((N,), jnp.int32)

        lowered = fwd.lower(params, nid, kc, vc, pos, fake, att, ws, N, C)
        comp = lowered.compile()
        hlo = comp.as_text()
        n_ag = hlo.count("all-gather")
        n_ar = hlo.count("all-reduce")
        n_rs = hlo.count("reduce-scatter")
        n_cp = hlo.count("collective-permute")
        print(f"HLO collectives: all-gather={n_ag} all-reduce={n_ar} "
              f"reduce-scatter={n_rs} collective-permute={n_cp}")
        # show the biggest all-gathers (C-sized => the replication smoking gun)
        import re
        ags = [l.strip() for l in hlo.splitlines()
               if "all-gather" in l and "f32" in l or
               ("all-gather" in l and "bf16" in l)]
        print(f"\nfirst 6 all-gather lines:")
        for l in ags[:6]:
            print("  " + l[:160])
        # dot dimensions involving the head/seq
        dots = [l.strip() for l in hlo.splitlines() if "dot(" in l]
        print(f"\ntotal dot ops: {len(dots)}; first 4 big dots:")
        big = [l for l in dots if "4104" in l or "4096" in l]
        for l in big[:4]:
            print("  " + l[:170])
        # cost analysis
        try:
            ca = comp.cost_analysis()
            if isinstance(ca, (list, tuple)):
                ca = ca[0]
            print(f"\nflops={ca.get('flops'):.3e}  "
                  f"bytes accessed={ca.get('bytes accessed'):.3e}")
        except Exception as e:
            print(f"cost_analysis n/a: {e}")


if __name__ == "__main__":
    main()
