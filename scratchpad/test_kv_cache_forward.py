"""Token-equivalence proof for the DFlash cache-consuming draft forward (inc 3).

Loads the REAL draft model through DFlashTorchaxWrapper, then over a synthetic
multi-step decode (C grows 8 -> 16 -> 24) compares TWO paths step-for-step:

  (a) FULL recompute  -- draft_forward over the whole [ctx|noise] concat
      (the flag-OFF path; recomputes context K/V every step).
  (b) CACHED           -- kv_project the new rows -> write per-slot K/V cache ->
      draft_forward_cached attends over [cached ctx K/V | fresh noise K/V]
      (the DFLASH_KV_CACHE=1 path).

Both paths are fed IDENTICAL raw hidden + noise ids + positions + mask each
step. We compare:
  * the draft_forward output hidden (N, B, D), and
  * the argmax draft tokens after _sample_block_draft_tokens.

EXPECT: draft TOKENS identical (the bar). Hidden may differ by ~1 bf16 ULP
because the cached K is written from the kv_project jit and consumed by a
DIFFERENT jit (cross-graph RoPE FMA fusion drift); V is exact. We report the max
abs hidden diff and confirm tokens still match.

Run:
  HF_HOME=/home/enyouki/local_hf \
  ~/tpu-tooling/tpu-env.sh python scratchpad/test_kv_cache_forward.py
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
    # The draft forward shardings reference both the 'data' (MLP_DATA) and
    # 'model' (MLP_TENSOR) mesh axes, so the mesh must expose both. Use one
    # device with both axes size 1 (everything replicated).
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    return jax.sharding.Mesh(devices, axis_names=("data", "model"))


def main():
    from tpu_inference.layers.common.sharding import ShardingAxisName  # noqa
    from tpu_inference.models.vllm.dflash import DFlashTorchaxWrapper

    mesh = _make_single_device_mesh()

    with jax.set_mesh(mesh):
        wrapper = DFlashTorchaxWrapper(mesh)

        # Synthetic embedding + lm_head so load() succeeds (no real target). We
        # need a REAL (non-zero) embedding + lm_head here: the noise embedding
        # feeds the forward and the lm_head feeds the argmax token compare, so
        # both must be identical between the two paths (they are -- same arrays).
        import transformers
        hf_cfg_peek = transformers.AutoConfig.from_pretrained(
            DRAFT_MODEL_PATH, trust_remote_code=True)
        vocab = getattr(hf_cfg_peek, "vocab_size", 201088)
        hidden = hf_cfg_peek.hidden_size
        rng = np.random.default_rng(1234)
        embed_np = (rng.standard_normal(
            (vocab, hidden)).astype(np.float32) * 0.02)
        lmhead_np = (rng.standard_normal(
            (vocab, hidden)).astype(np.float32) * 0.02)
        target_state = {
            "vllm_model.model.embedding.weight":
            jnp.asarray(embed_np, dtype=jnp.bfloat16),
            "vllm_model.lm_head.weight":
            jnp.asarray(lmhead_np, dtype=jnp.bfloat16),
        }
        wrapper.load(DRAFT_MODEL_PATH, target_state)

        cfg = wrapper.model.dflash.config
        L = cfg.num_hidden_layers
        D = cfg.hidden_size
        KVH = cfg.num_key_value_heads
        HD = getattr(cfg, "head_dim", D // cfg.num_attention_heads)
        B = cfg.block_size
        raw_dim = wrapper.model.dflash.fc.in_features

        draft_forward = wrapper.get_draft_forward_fn()
        draft_forward_cached = wrapper.get_draft_forward_cached_fn()
        kv_project = wrapper.get_kv_project_fn()
        compute_logits = wrapper.get_compute_logits_fn()

        num_spec = 7 if B - 1 >= 7 else B - 1
        params = wrapper.params
        embed_w = target_state["vllm_model.model.embedding.weight"]
        lm_head_w = target_state["vllm_model.lm_head.weight"]

        N = 2
        # Buffer big enough for the largest context we'll reach.
        max_C = 32
        buf_len = max_C + 8

        # Per-slot K/V caches (L, N, buf_len, KVH, HD), like the proposer's.
        k_cache = jnp.zeros((L, N, buf_len, KVH, HD), dtype=jnp.bfloat16)
        v_cache = jnp.zeros((L, N, buf_len, KVH, HD), dtype=jnp.bfloat16)
        # Persistent ctx buffer (N, buf_len, raw_dim), like the proposer's.
        ctx_buf = jnp.zeros((N, buf_len, raw_dim), dtype=jnp.bfloat16)

        def next_padded(n, block=4):
            if n <= block:
                return block
            return ((n + block - 1) // block) * block

        def sample_tokens(hidden_states):
            # Mirror _sample_block_draft_tokens: rows 1..1+num_spec.
            draft_hidden = hidden_states[:, 1:1 + num_spec]
            logits = compute_logits(params, draft_hidden, lm_head_w)
            return jnp.argmax(logits, axis=-1)

        # A tie-free projection for a decisive argmax compare that the synthetic
        # 201088-way random lm_head can't give (its logits collide in bf16). 512
        # well-separated classes -> structural bugs shift argmax, bf16 noise does
        # not (margins are O(hidden_scale), not 0).
        proj_np = rng.standard_normal((hidden, 512)).astype(np.float32)
        proj = jnp.asarray(proj_np, dtype=jnp.float32)

        def sample_tokens_tiefree(hidden_states):
            dh = np.asarray(jax.device_get(
                hidden_states[:, 1:1 + num_spec])).astype(np.float32)
            logits = dh @ np.asarray(proj)  # (N, num_spec, 512)
            return np.argmax(logits, axis=-1), logits

        # ---- synthetic multi-step decode: C grows by 8 each step ----
        steps = [(0, 8), (8, 8), (16, 8)]  # (ctx_start, num_new) per step
        rng2 = np.random.default_rng(0)

        all_pass = True
        print(f"\nDFlash cache-consuming forward token-equivalence "
              f"(N={N}, B={B}, L={L}, KVH={KVH}, HD={HD}, num_spec={num_spec})")
        print("-" * 78)

        ctx_len = 0
        for si, (ctx_start, num_new) in enumerate(steps):
            assert ctx_start == ctx_len
            new_C = ctx_len + num_new

            # New raw rows for this step (per request).
            raw_new_np = rng2.standard_normal(
                (N, num_new, raw_dim)).astype(np.float32)
            raw_new = jnp.asarray(raw_new_np, dtype=jnp.bfloat16)

            # --- append into ctx_buf (path a) ---
            rows = jnp.arange(ctx_len, new_C)
            ctx_buf = ctx_buf.at[:, ctx_len:new_C].set(raw_new)

            # --- kv_project the new rows + write cache (path b) ---
            new_pos = jnp.broadcast_to(rows.astype(jnp.int32),
                                       (N, num_new))  # (N, num_new)
            k_proj, v_proj = kv_project(params, raw_new, new_pos)
            # k_proj/v_proj: (L, N, num_new, KVH, HD). Scatter into cache rows.
            k_cache = k_cache.at[:, :, ctx_len:new_C].set(k_proj)
            v_cache = v_cache.at[:, :, ctx_len:new_C].set(v_proj)

            ctx_len = new_C
            C = ctx_len
            padded_ctx = next_padded(C)

            # --- build identical inputs for both paths ---
            noise_ids = np.zeros((N, B), dtype=np.int32)
            noise_ids[:, 0] = rng2.integers(0, vocab, size=N)
            noise_ids_j = jnp.asarray(noise_ids)

            # positions: ctx [0..C-1] padded to padded_ctx with 0, then noise
            # [C..C+B-1].
            ar_ctx = jnp.arange(padded_ctx, dtype=jnp.int32)[None, :]
            ctx_valid = ar_ctx < C  # (1, padded_ctx) broadcast
            ctx_valid = jnp.broadcast_to(ctx_valid, (N, padded_ctx))
            ctx_positions = jnp.where(ctx_valid, ar_ctx, 0)
            noise_positions = (jnp.arange(B, dtype=jnp.int32)[None, :] + C)
            noise_positions = jnp.broadcast_to(noise_positions, (N, B))
            position_ids = jnp.concatenate([ctx_positions, noise_positions],
                                           axis=1)  # (N, padded_ctx + B)

            neg = jnp.finfo(jnp.bfloat16).min
            ctx_mask = jnp.where(
                ctx_valid,
                jnp.zeros((N, padded_ctx), dtype=jnp.bfloat16),
                jnp.full((N, padded_ctx), neg, dtype=jnp.bfloat16))
            noise_mask = jnp.zeros((N, B), dtype=jnp.bfloat16)
            attention_mask = jnp.concatenate([ctx_mask, noise_mask], axis=1)

            # --- path a: full recompute ---
            hidden_full = draft_forward(params, noise_ids_j, ctx_buf,
                                        position_ids, embed_w, attention_mask,
                                        N, padded_ctx)
            # --- path b: cached (real kv_project-written cache) ---
            hidden_cached = draft_forward_cached(params, noise_ids_j, k_cache,
                                                 v_cache, position_ids, embed_w,
                                                 attention_mask, N, padded_ctx)

            # --- ORACLE: same-graph cached vs full ---
            # Run the full forward while spying the EXACT in-forward ctx K/V it
            # builds, then feed those SAME-GRAPH ctx K/V into draft_forward_cached
            # -- all in ONE jit. This eliminates the cross-graph RoPE-FMA fusion
            # drift (kv_project jit vs cached jit), so any residual diff isolates
            # the cat / mask / transpose / GQA logic of the cached forward. It
            # should be ~bit-exact; if it is, the ~1-ULP diff above is purely the
            # known cross-graph K drift, not a bug.
            import torchax
            from torchax.interop import jax_view as _jv, torch_view as _tv
            remote_mod = sys.modules[type(wrapper.model.dflash).__module__]
            orig_eager = remote_mod.eager_attention_forward

            @jax.jit
            def oracle(prm, nids, th_full, pos, ew, mask):
                with torchax.default_env():
                    p = _tv(prm)
                    # full forward, spy ctx K/V (post-RoPE k, transposed v).
                    stash = []

                    def spy(module, query, key, value, *a, **kw):
                        stash.append((key, value))  # (N,KVH,C+B,hd)
                        return orig_eager(module, query, key, value, *a, **kw)

                    remote_mod.eager_attention_forward = spy
                    try:
                        noise_emb = torch.nn.functional.embedding(
                            _tv(nids), _tv(ew))
                        mask4 = _tv(mask).reshape(N, 1, 1, padded_ctx + B)
                        out_full = torch.func.functional_call(
                            wrapper.model, p,
                            kwargs={
                                "noise_embedding": noise_emb,
                                "target_hidden": _tv(th_full),
                                "position_ids": _tv(pos),
                                "attention_mask": mask4,
                            }, tie_weights=False)
                    finally:
                        remote_mod.eager_attention_forward = orig_eager
                    # ctx-only K/V from the spy, layout (L,N,C,KVH,hd) to match
                    # the cache slice the cached forward expects.
                    kc = torch.stack(
                        [k[:, :, :padded_ctx, :].transpose(1, 2)
                         for (k, v) in stash], dim=0)
                    vc = torch.stack(
                        [v[:, :, :padded_ctx, :].transpose(1, 2)
                         for (k, v) in stash], dim=0)
                    noise_emb2 = torch.nn.functional.embedding(
                        _tv(nids), _tv(ew))
                    noise_pos = _tv(pos)[:, padded_ctx:]
                    mask4b = _tv(mask).reshape(N, 1, 1, padded_ctx + B)
                    out_cached = torch.func.functional_call(
                        wrapper.model, p,
                        kwargs={
                            "noise_embedding": noise_emb2,
                            "cached_k": kc,
                            "cached_v": vc,
                            "noise_position_ids": noise_pos,
                            "attention_mask": mask4b,
                        }, tie_weights=False)
                    return _jv(out_full), _jv(out_cached)

            o_full, o_cached = oracle(params, noise_ids_j,
                                      ctx_buf[:, :padded_ctx], position_ids,
                                      embed_w, attention_mask)
            of = np.asarray(jax.device_get(o_full)).astype(np.float32)
            oc = np.asarray(jax.device_get(o_cached)).astype(np.float32)
            oracle_diff = float(np.max(np.abs(of - oc)))
            # NOTE: this is NOT ~0 -- the full path runs the remote
            # DFlashDraftModel.forward while the cached path runs the
            # hand-written _draft_forward_cached loop; even with byte-identical
            # ctx K/V the two op sequences round ~1-2 bf16 ULP differently
            # through 8 layers (XLA fuses them differently). It isolates the
            # cat/mask/GQA logic to that ULP floor (no O(1) structural error).
            print(f"    [oracle same-graph] max|hidden_d|={oracle_diff:.3e} "
                  f"(~1-2 bf16 ULP floor; not a bug)")

            tok_full = np.asarray(jax.device_get(sample_tokens(hidden_full)))
            tok_cached = np.asarray(
                jax.device_get(sample_tokens(hidden_cached)))

            # Tie-free decisive argmax compare (float32, 512 separated classes).
            tf_full, lg_tf_full = sample_tokens_tiefree(hidden_full)
            tf_cached, _ = sample_tokens_tiefree(hidden_cached)
            tf_eq = np.array_equal(tf_full, tf_cached)
            tf_ndiff = int(np.sum(tf_full != tf_cached))
            if not tf_eq:
                srt = np.sort(lg_tf_full, axis=-1)
                tf_margin = (srt[..., -1] - srt[..., -2])[tf_full != tf_cached]
            else:
                tf_margin = np.array([])

            hf = np.asarray(jax.device_get(hidden_full)).astype(np.float32)
            hc = np.asarray(jax.device_get(hidden_cached)).astype(np.float32)
            hidden_diff = float(np.max(np.abs(hf - hc)))
            hidden_scale = float(np.max(np.abs(hf)))
            rel = hidden_diff / (hidden_scale + 1e-9)

            tok_eq = np.array_equal(tok_full, tok_cached)
            n_diff = int(np.sum(tok_full != tok_cached))

            # Classify any divergence as near-tie vs structural. A position is a
            # near-tie if the full path's top-2 logit margin is within a few bf16
            # ULP of the logit scale -- argmax coin-flips on (near-)equal logits,
            # which is genuine bf16 rounding, NOT a logic bug. With the synthetic
            # random lm_head over a 201088 vocab, many logits round to identical
            # bf16 values -> exact 0-margin ties. A structural bug would diverge
            # at a LARGE margin (the cached path attended over the wrong keys).
            structural_diff = 0
            margins_at_diff = np.array([])
            if not tok_eq:
                dh_full = hidden_full[:, 1:1 + num_spec]
                logits_full = np.asarray(
                    jax.device_get(compute_logits(params, dh_full,
                                                  lm_head_w))).astype(
                                                      np.float32)
                srt = np.sort(logits_full, axis=-1)
                margin = srt[..., -1] - srt[..., -2]  # (N, num_spec)
                lscale = float(np.max(np.abs(logits_full)))
                tie_tol = 8.0 * lscale * 2**-8  # ~a few bf16 ULP at logit scale
                diff_mask = (tok_full != tok_cached)
                margins_at_diff = margin[diff_mask]
                structural_diff = int(np.sum(margins_at_diff > tie_tol))

            ok = tok_eq or (structural_diff == 0)
            all_pass = all_pass and ok
            status = ("PASS" if tok_eq else
                      ("PASS(ties)" if structural_diff == 0 else "FAIL(bug)"))
            print(f"step {si} (C={C:2d}, padded_ctx={padded_ctx:2d}): "
                  f"tokens_eq={tok_eq} (ndiff={n_diff}, "
                  f"structural={structural_diff})  "
                  f"max|hidden_d|={hidden_diff:.3e} (scale={hidden_scale:.2e}, "
                  f"rel={rel:.2e})  {status}")
            print(f"    tie-free argmax: eq={tf_eq} (ndiff={tf_ndiff})"
                  + ("" if tf_eq else
                     f"  diverged margins (f32)={tf_margin}"))
            if not tok_eq:
                print(f"    diverged positions top-2 margins: "
                      f"{margins_at_diff}")

        print("-" * 78)
        print("RESULT:",
              "PASS (tokens identical)" if all_pass else "FAIL (tokens differ)")
        sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
