"""L3 minimal torchax repro: isolate WHERE the head-shard all-gather happens
and test the inline-attention + pin-scores fix. NO model load -> compiles fast.

Mimics ONE dflash layer's noise-attention chain on dummy head-sharded weights:
  q_proj -> view(n,b,nh,hd) -> q_norm(rmsnorm) -> transpose -> RoPE -> pin
  k/v_proj -> view -> norm -> transpose -> RoPE -> concat(ctx) -> pin
  attention (3 variants):
    BROKEN : transformers.eager_attention_forward (view/repeat_kv inside)
    PIN_OUT: eager + pin attn_output (sibling's current edit)
    INLINE : inline repeat_kv+matmul+f32softmax+scores@v, pin the SCORES tensor

Reports per-variant: all-gather count, whether the score dot operand is
64-head (replicated) or 8-head (sharded), and the scores sharding.
"""
import os
os.environ.setdefault("HF_HOME", "/home/enyouki/local_hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view

AX = ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp")
H = "model"


def _real_mesh():
    return Mesh(np.array(jax.devices()).reshape((1, 1, 1, 1, 8, 1)),
                axis_names=AX)


def _pin(t, spec, mesh):
    return torch_view(
        jax.lax.with_sharding_constraint(jax_view(t), NamedSharding(mesh, spec)))


def main():
    mesh = _real_mesh()
    N, B, nh, kvh, hd, C, D = 4, 8, 64, 8, 64, 512, 2880
    S = C + B
    n_rep = nh // kvh
    scaling = hd ** -0.5
    from transformers.models.qwen3.modeling_qwen3 import (
        eager_attention_forward)

    with jax.set_mesh(mesh):
        rep = NamedSharding(mesh, PartitionSpec())
        # weight shardings matching dflash: q/k/v weight P('model',None);
        # o weight P(None,'model').
        wq_s = NamedSharding(mesh, PartitionSpec(H, None))
        wo_s = NamedSharding(mesh, PartitionSpec(None, H))

        key = jax.random.PRNGKey(0)

        def rn(k, shp, sc=0.02):
            return jax.random.normal(jax.random.PRNGKey(k), shp,
                                     jnp.bfloat16) * sc

        wq = jax.device_put(rn(1, (nh * hd, D)), wq_s)
        wk = jax.device_put(rn(2, (kvh * hd, D)), wq_s)
        wv = jax.device_put(rn(3, (kvh * hd, D)), wq_s)
        wo = jax.device_put(rn(4, (D, nh * hd)), wo_s)
        hs_in = jax.device_put(rn(5, (N, B, D), 0.1), rep)
        kc = jax.device_put(rn(6, (N, kvh, C, hd), 0.1),
                            NamedSharding(mesh, PartitionSpec(None, H, None,
                                                              None)))
        vc = jax.device_put(rn(7, (N, kvh, C, hd), 0.1),
                            NamedSharding(mesh, PartitionSpec(None, H, None,
                                                              None)))
        cos = jax.device_put(rn(8, (N, 1, B, hd), 1.0), rep)
        sin = jax.device_put(rn(9, (N, 1, B, hd), 1.0), rep)
        mask = jax.device_put(jnp.zeros((N, 1, 1, S), jnp.bfloat16), rep)

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        class FakeMod:
            num_key_value_groups = n_rep
            training = False

        def build_qkv(hs, wq, wk, wv, kc, vc, cos, sin):
            q = torch.nn.functional.linear(hs, wq).view(N, B, nh, hd)
            q = q.transpose(1, 2)  # (N, nh, B, hd)
            k = torch.nn.functional.linear(hs, wk).view(N, B, kvh, hd)
            v = torch.nn.functional.linear(hs, wv).view(N, B, kvh, hd)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            k = torch.cat([kc, k], dim=2)  # (N, kvh, S, hd)
            v = torch.cat([vc, v], dim=2)
            q = _pin(q, PartitionSpec(None, H, None, None), mesh)
            k = _pin(k, PartitionSpec(None, H, None, None), mesh)
            v = _pin(v, PartitionSpec(None, H, None, None), mesh)
            return q, k, v

        def fwd_broken(hs, wq, wk, wv, wo, kc, vc, cos, sin, mask):
            q, k, v = build_qkv(hs, wq, wk, wv, kc, vc, cos, sin)
            out, _ = eager_attention_forward(FakeMod(), q, k, v, mask,
                                             scaling=scaling, dropout=0.0)
            out = out.reshape(N, B, -1)
            return torch.nn.functional.linear(out, wo)

        def fwd_pinout(hs, wq, wk, wv, wo, kc, vc, cos, sin, mask):
            q, k, v = build_qkv(hs, wq, wk, wv, kc, vc, cos, sin)
            out, _ = eager_attention_forward(FakeMod(), q, k, v, mask,
                                             scaling=scaling, dropout=0.0)
            out = _pin(out, PartitionSpec(None, None, H, None), mesh)
            out = out.reshape(N, B, -1)
            return torch.nn.functional.linear(out, wo)

        def repeat_kv_inline(x):  # (N, kvh, S, hd) -> (N, nh, S, hd)
            xt = jax_view(x)
            xt = jnp.broadcast_to(xt[:, :, None, :, :],
                                  (N, kvh, n_rep, S, hd))
            xt = xt.reshape(N, kvh * n_rep, S, hd)
            return torch_view(xt)

        def fwd_inline(hs, wq, wk, wv, wo, kc, vc, cos, sin, mask):
            q, k, v = build_qkv(hs, wq, wk, wv, kc, vc, cos, sin)
            ks = repeat_kv_inline(k)
            vs = repeat_kv_inline(v)
            aw = torch.matmul(q, ks.transpose(2, 3)) * scaling  # (N,nh,B,S)
            aw = aw + mask
            # PIN scores head-sharded BEFORE softmax.
            aw = _pin(aw, PartitionSpec(None, H, None, None), mesh)
            aw = torch.nn.functional.softmax(aw, dim=-1,
                                             dtype=torch.float32).to(q.dtype)
            out = torch.matmul(aw, vs)  # (N, nh, B, hd)
            out = _pin(out, PartitionSpec(None, H, None, None), mesh)
            out = out.transpose(1, 2).reshape(N, B, nh * hd)
            return torch.nn.functional.linear(out, wo)

        # ---- SHARD_MAP variant: pure-jax attention mapped over 'model' ----
        # Inside shard_map each chip sees its LOCAL shard: q (N, nh/8=8, B, hd),
        # k/v (N, kvh/8=1, S, hd). n_rep=8 so local repeat 1->8. Compute local
        # attention + local o_proj contribution; psum over 'model' for o_proj.
        from jax.experimental.shard_map import shard_map

        def _attn_shard(q, k, v, mask, wo):
            # shapes per-shard: q (N, 8, B, hd) k/v (N, 1, S, hd)
            lnh = q.shape[1]
            lkvh = k.shape[1]
            rep = lnh // lkvh
            ks = jnp.broadcast_to(k[:, :, None], (N, lkvh, rep, S, hd)).reshape(
                N, lnh, S, hd)
            vs = jnp.broadcast_to(v[:, :, None], (N, lkvh, rep, S, hd)).reshape(
                N, lnh, S, hd)
            aw = jnp.matmul(q, jnp.swapaxes(ks, 2, 3)) * scaling  # (N,8,B,S)
            aw = aw + mask
            aw = jax.nn.softmax(aw.astype(jnp.float32), -1).astype(jnp.bfloat16)
            o = jnp.matmul(aw, vs)  # (N, 8, B, hd)
            o = jnp.swapaxes(o, 1, 2).reshape(N, B, lnh * hd)  # (N,B,512)
            # wo shard is (D, nh*hd/8 = 512); local o_proj then psum.
            o = jnp.matmul(o, wo.T)  # (N, B, D)
            return jax.lax.psum(o, axis_name=H)

        def fwd_shardmap(hs, wq, wk, wv, wo, kc, vc, cos, sin, mask):
            q, k, v = build_qkv(hs, wq, wk, wv, kc, vc, cos, sin)
            qj, kj, vj = jax_view(q), jax_view(k), jax_view(v)
            woj, mj = jax_view(wo), jax_view(mask)
            Pm = PartitionSpec
            sm = shard_map(
                _attn_shard, mesh=mesh,
                in_specs=(Pm(None, H, None, None), Pm(None, H, None, None),
                          Pm(None, H, None, None), Pm(),
                          Pm(None, H)),
                out_specs=Pm(),
                check_rep=False)
            out = sm(qj, kj, vj, mj, woj)
            return torch_view(out)

        def jitwrap(fn):
            @jax.jit
            def j(hs, wq, wk, wv, wo, kc, vc, cos, sin, mask):
                with torchax.default_env():
                    args = [torch_view(x) for x in
                            (hs, wq, wk, wv, wo, kc, vc, cos, sin, mask)]
                    return jax_view(fn(*args))
            return j

        jargs = (hs_in, wq, wk, wv, wo, kc, vc, cos, sin, mask)
        for name, fn in [("BROKEN", fwd_broken), ("PIN_OUT", fwd_pinout),
                         ("INLINE", fwd_inline), ("SHARDMAP", fwd_shardmap)]:
            jf = jitwrap(fn)
            comp = jf.lower(*jargs).compile()
            txt = comp.as_text()
            nag = txt.count("all-gather")
            nar = txt.count("all-reduce")
            # find score dot: look for the big [.,.,B,S] = [.,.,8,520] operand
            big = [l.strip() for l in txt.splitlines()
                   if "all-gather" in l and "bf16" in l]
            heads64 = sum(1 for l in big if "64,8,520" in l or ",64," in l
                          and "520" in l)
            # time it (this is just 1 layer; relative is what matters)
            import statistics
            import time
            jax.block_until_ready(comp(*jargs))
            ts = []
            for _ in range(50):
                t0 = time.perf_counter()
                jax.block_until_ready(comp(*jargs))
                ts.append((time.perf_counter() - t0) * 1e3)
            ms = statistics.median(ts)
            print(f"\n=== {name}: all-gather={nag} all-reduce={nar}  "
                  f"1-layer={ms:.4f} ms ===")
            for l in big[:3]:
                print("  AG:", l[:130])


if __name__ == "__main__":
    main()
