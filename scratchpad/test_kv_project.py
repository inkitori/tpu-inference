"""Bit-identical proof for DFlash kv_project (increment 1).

Loads the REAL draft model through DFlashTorchaxWrapper, then proves that the
standalone _kv_project (per-layer K post-RoPE, V no-norm/no-RoPE over a set of
context rows) is bit-identical to what the actual draft forward computes for
those same context rows.

Oracle: run the full draft forward once with the loaded module's
`eager_attention_forward` monkeypatched to record the (key, value) tensors that
the real attention passes into it -- i.e. post-RoPE k (N,KVH,C+B,hd) and the
transposed v (N,KVH,C+B,hd) the forward built at lines 80/82. We slice the
context rows [:, :, :C, :] from those, transpose to (N,C,KVH,hd), and compare to
kv_project's output. The oracle K/V therefore come from the ACTUAL forward path,
the projection K/V from the new code under test.

Run:
  HF_HOME=/home/enyouki/local_hf \
  ~/tpu-tooling/tpu-env.sh python scratchpad/test_kv_project.py
"""

import os
import sys

os.environ.setdefault("HF_HOME", "/home/enyouki/local_hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import jax
import jax.numpy as jnp
import numpy as np
import torch

DRAFT_MODEL_PATH = "z-lab/gpt-oss-20b-DFlash"


def _make_single_device_mesh():
    devices = np.array(jax.devices()[:1])
    return jax.sharding.Mesh(devices, axis_names=("model", ))


def main():
    from torchax.interop import jax_view, torch_view
    from tpu_inference.models.vllm.dflash import DFlashTorchaxWrapper

    mesh = _make_single_device_mesh()

    with jax.set_mesh(mesh):
        wrapper = DFlashTorchaxWrapper(mesh)

        # The wrapper.load() shares embeddings from a target model. For this
        # standalone numerics test we don't have a real target; provide a dict
        # with a synthetic embedding + lm_head so load() succeeds. kv_project
        # uses NONE of these (only fc/hidden_norm/k_proj/v_proj/k_norm/rotary),
        # so the synthetic embeddings are irrelevant to the numerics under test.
        hf_cfg_peek = __import__("transformers").AutoConfig.from_pretrained(
            DRAFT_MODEL_PATH, trust_remote_code=True)
        vocab = getattr(hf_cfg_peek, "vocab_size", 201088)
        hidden = hf_cfg_peek.hidden_size
        fake_embed = jnp.zeros((vocab, hidden), dtype=jnp.bfloat16)
        target_state = {
            "vllm_model.model.embedding.weight": fake_embed,
            "vllm_model.lm_head.weight": fake_embed,
        }
        wrapper.load(DRAFT_MODEL_PATH, target_state)

        cfg = wrapper.model.dflash.config
        L = cfg.num_hidden_layers
        D = cfg.hidden_size
        KVH = cfg.num_key_value_heads
        HD = getattr(cfg, "head_dim", D // cfg.num_attention_heads)
        B = cfg.block_size
        raw_dim = wrapper.model.dflash.fc.in_features

        N, C = 2, 24
        rng = np.random.default_rng(0)
        raw_ctx_np = rng.standard_normal((N, C, raw_dim)).astype(np.float32)
        raw_ctx = jnp.asarray(raw_ctx_np, dtype=jnp.bfloat16)

        # Absolute context positions: 0..C-1 per row.
        pos_ctx = jnp.broadcast_to(jnp.arange(C, dtype=jnp.int32),
                                   (N, C)).copy()

        # Build the full draft-forward inputs. noise block: zeros + a next-token
        # at slot 0; full positions cover ctx (0..C-1) then noise (C..C+B-1).
        embed_w = wrapper.embed_weight_jax  # synthetic zeros; only used to embed
        # noise ids -> noise embeddings (zeros). That's fine: the K/V we compare
        # are the CONTEXT rows, independent of the noise rows (k_norm + RoPE are
        # per-position). Use a real next token id to exercise the path.
        noise_ids = np.zeros((N, B), dtype=np.int32)
        noise_ids[:, 0] = 7  # arbitrary
        noise_ids_j = jnp.asarray(noise_ids)
        pos_full = jnp.broadcast_to(jnp.arange(C + B, dtype=jnp.int32),
                                    (N, C + B)).copy()
        attn_mask = jnp.zeros((N, C + B), dtype=jnp.bfloat16)  # no padding

        # ----------------------------------------------------------------
        # Single-graph bit-identical test.
        #
        # kv_project (code under test) and the real draft forward are run in the
        # SAME jit graph. WHY: bf16 RoPE (k*cos + rotate_half(k)*sin) is FMA-
        # rounding-sensitive to how XLA fuses it, and XLA makes DIFFERENT fusion
        # decisions in two SEPARATELY-jitted graphs (kv_project alone vs the full
        # forward), yielding ~1 bf16 ULP cross-graph drift on K (V, which has no
        # RoPE/FMA, is bit-exact even across separate graphs). That drift is an
        # XLA fusion artifact, NOT a numerics bug: when both paths share one
        # graph, XLA fuses them identically and K is bit-exact too. Co-locating
        # them is the faithful test of the OP SEQUENCE -- which is exactly what
        # must match when increment 2 calls kv_project inside the proposer graph.
        #
        # The oracle K/V come from the ACTUAL forward path: we patch the remote
        # module's eager_attention_forward to stash the (key, value) tensors the
        # real attention passes into it (post-RoPE k and transposed v), then
        # slice the context rows. The projection K/V come from the real
        # _DFlashRunner._kv_project method.
        # ----------------------------------------------------------------
        import torchax
        remote_mod = type(wrapper.model.dflash).__module__
        mod = sys.modules[remote_mod]
        orig_eager = mod.eager_attention_forward

        @jax.jit
        def combined(params, raw_ctx_j, pos_ctx_j, noise_ids_j_, pos_full_j,
                     mask_j, embed_w_j):
            with torchax.default_env():
                p = torch_view(params)
                # --- code under test: real _kv_project method ---
                k_all, v_all = torch.func.functional_call(
                    wrapper.model,
                    p,
                    kwargs={
                        "kv_project_raw": torch_view(raw_ctx_j),
                        "kv_position_ids": torch_view(pos_ctx_j),
                    },
                    tie_weights=False,
                )  # each (L, N, C, KVH, HD)

                # --- oracle: real forward, capture eager-attn (k, v) ---
                stash = []

                def spy_eager(module, query, key, value, *a, **kw):
                    stash.append((key, value))  # post-RoPE k, transposed v
                    return orig_eager(module, query, key, value, *a, **kw)

                mod.eager_attention_forward = spy_eager
                try:
                    noise_emb = torch.nn.functional.embedding(
                        torch_view(noise_ids_j_), torch_view(embed_w_j))
                    mask_t = torch_view(mask_j).reshape(N, 1, 1, C + B)
                    _ = torch.func.functional_call(
                        wrapper.model,
                        p,
                        kwargs={
                            "noise_embedding": noise_emb,
                            "target_hidden": torch_view(raw_ctx_j),
                            "position_ids": torch_view(pos_full_j),
                            "attention_mask": mask_t,
                        },
                        tie_weights=False,
                    )
                finally:
                    mod.eager_attention_forward = orig_eager
                ks = [jax_view(k) for (k, v) in stash]  # (N,KVH,C+B,HD) each
                vs = [jax_view(v) for (k, v) in stash]
                return jax_view(k_all), jax_view(v_all), ks, vs

        k_all, v_all, ks, vs = combined(wrapper.params, raw_ctx, pos_ctx,
                                        noise_ids_j, pos_full, attn_mask,
                                        embed_w)
        k_proj_all = np.asarray(jax.device_get(k_all))  # (L,N,C,KVH,HD)
        v_proj_all = np.asarray(jax.device_get(v_all))
        captured = list(zip(ks, vs))
        assert len(captured) == L, f"captured {len(captured)} layers, expected {L}"

    # ---------- Compare ----------
    all_pass = True
    print(f"\nDFlash kv_project bit-identical check "
          f"(N={N}, C={C}, B={B}, L={L}, KVH={KVH}, HD={HD})")
    print("-" * 64)
    for li in range(L):
        k_oracle = np.asarray(jax.device_get(captured[li][0]))  # (N,KVH,C+B,HD)
        v_oracle = np.asarray(jax.device_get(captured[li][1]))
        # context rows only, transpose to (N,C,KVH,HD)
        k_o = np.transpose(k_oracle[:, :, :C, :], (0, 2, 1, 3))
        v_o = np.transpose(v_oracle[:, :, :C, :], (0, 2, 1, 3))
        k_p = k_proj_all[li]  # (N,C,KVH,HD)
        v_p = v_proj_all[li]

        k_eq = np.array_equal(k_p, k_o)
        v_eq = np.array_equal(v_p, v_o)
        k_diff = float(np.max(np.abs(k_p.astype(np.float32) -
                                     k_o.astype(np.float32))))
        v_diff = float(np.max(np.abs(v_p.astype(np.float32) -
                                     v_o.astype(np.float32))))
        ok = k_eq and v_eq
        all_pass = all_pass and ok
        print(f"layer {li:2d}: K exact={k_eq} (max|d|={k_diff:.3e})  "
              f"V exact={v_eq} (max|d|={v_diff:.3e})  "
              f"{'PASS' if ok else 'FAIL'}")

    print("-" * 64)
    print("RESULT:", "PASS (bf16-exact)" if all_pass else "FAIL")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
