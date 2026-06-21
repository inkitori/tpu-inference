# DeepSeek-V4-Flash bring-up — HANDOFF (read this first)

Goal: serve `deepseek-ai/DeepSeek-V4-Flash` **text-only** on TPU v6e-8 via the torchax path
(`MODEL_IMPL_TYPE=vllm`), accurate/coherent output, experts kept FP4 + linears FP8.
Companion doc with full detail: **`DSV4_BRINGUP_NOTES.md`** (architecture map, kernel inventory, PR inventory,
numeric references). Read that second.

## TL;DR state (as of this handoff)
The model **constructs and loads** on v6e-8 through the vllm/torchax path. The attention **forward is still a
pass-through stub** (returns hidden_states unchanged) — so output is NOT yet correct. Next big task = implement the
real attention forward using the #2903 Pallas kernels (now in-tree). Everything else (weight load, FP4 experts,
DP+EP sharding, mHC, MoE plumbing) is wired.

## Environment (all confirmed working)
- TPU v6e-8 (`ct6e-standard-8t-tpu`), 8 chips, 31.25 GiB/chip free. 1.4 TB host RAM. venv: `source /home/enyouki/.venv/bin/activate` (jax 0.10.1).
- vLLM editable `/home/enyouki/vllm` @ `ecf9d8352` (has `DeepseekV4ForCausalLM`; compatible with skeleton — verified).
- tpu-inference `/home/enyouki/tpu-inference`, branch `ds-v4-flash-torchax` **fast-forwarded to upstream/main `2b09274e`**
  (added remote `upstream` = vllm-project/tpu-inference). PR branches fetched as refs: `pr-2903 pr-2858 pr-2905 pr-2950 pr-1906`.
- Weights: 2nd gcsfuse mount as enyouki (read-only) at **`/home/enyouki/dsv4-weights`** (cache `/dev/shm/dsv4cache`).
  Snapshot: `/home/enyouki/dsv4-weights/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/553034d7dd9e06c2eeaee68cf85a17d6d4754cf0`.
  If the mount is gone after reboot, remount:
  `gcsfuse -o ro --implicit-dirs --only-dir vllm --cache-dir /dev/shm/dsv4cache --client-protocol grpc --file-cache-max-size-mb 160000 --file-cache-cache-file-for-range-read --file-cache-enable-parallel-downloads personal-mark-eu /home/enyouki/dsv4-weights`
  (Do NOT touch the systemd root mount at /tmp/gcs/bucket — it's root-only.)

## Launch
`/home/enyouki/dsv4_run/boot.sh` → `examples/offline_inference.py`, logs to `/home/enyouki/dsv4_run/boot.log`.
Critical env/flags (all required, discovered the hard way):
- `MODEL_IMPL_TYPE=vllm`, `NEW_MODEL_DESIGN=1` (MLA models require it), `MOE_REQUANTIZE_WEIGHT_DTYPE=fp4` (keep experts FP4).
- `--tensor-parallel-size 8 --enable-expert-parallel --kv-cache-dtype fp8`
- `--additional-config '{"sharding": {"sharding_strategy": {"enable_dp_attention": true, "expert_parallelism": 8, "tensor_parallelism": 1}}, "replicate_attn_weights": "True", "sparse_matmul": "True"}'`
- First load is slow (~12 min, cold gcsfuse read of 148 GiB). Re-runs should be faster (file cache warms in /dev/shm).

## Changes made this session (UNCOMMITTED — in working tree)
1. `tpu_inference/layers/vllm/custom_ops/mhc.py` — applied #2950 (real `VllmHCHeadOp.forward_tpu` gated collapse)
   AND implemented `VllmMHCFusedPostPreOp.forward_tpu` (was NotImplementedError) by composing
   `mhc_post_torch` + `mhc_pre_torch` (mirrors vLLM `MHCFusedPostPreOp.forward_native`). mHC pre/post already used torch impls.
2. `tpu_inference/layers/vllm/custom_ops/deepseek_v4_attention.py` — `VllmDeepseekV4MLAAttention.__init__` now calls the
   real `super().__init__()` (base `DeepseekV4Attention.__init__`) with `torch.cuda.Event` mocked, so it registers the
   FULL param manifest (attn_sink, fused_wqa_wkv, q_norm, wq_b, kv_norm, wo_a/wo_b, rotary_emb, + per-layer
   indexer[ratio4]/compressor[ratio>1]/swa_cache). This fixed `KeyError: attn_sink`. **forward/forward_mqa/_o_proj are
   still stubs** (forward returns hidden_states; the other two raise NotImplementedError — not reached while forward is pass-through).
3. Brought kernel files into tree (git-staged adds) from PRs:
   - #2903: `tpu_inference/kernels/experimental/deepseek_v4/{__init__,mla,mla_swa}.py` + tests
   - #2858: `tpu_inference/kernels/experimental/deepseek_v4/{compress_norm_rope,compress_store,compressor}.py` + test
   - (took #2903's `__init__.py`; the only add/add conflict)
   NOT yet brought in: #2905 indexer files (`streamindex_topk.py`, `deepseek_v4_indexer.py`) — defer (dead/broken, see notes).
   #1906 = irrelevant for V4 (experts are native mxfp4 via VllmMxfp4MoEMethod). Skip.

## What's CONFIRMED working in the boot
config validation · tokenizer_mode deepseek_v4 · mesh `attn_dp_expert=8` (DP attention + EP) · expert_dtype resolved
**fp4** · MoE GMM EP kernel · kv cache remapped to **fp8_ds_mla** (uint8, 576B slots) · all 43 layers construct incl.
indexer + compressor · checkpoint shards loading (no mapping/KeyErrors). HBM-after-load number: **CHECK boot.log** (was
mid-load at handoff — grep for "weights took"/"GiB"/"KV cache"). Risk-1 (FP4/HBM fit) looked on track (19 GiB/chip
expected sharded; 31.25 free).

## NEXT STEPS (in order)
1. **Confirm boot outcome** in `/home/enyouki/dsv4_run/boot.log`: did weight load finish + HBM fit? The profile/dummy
   forward runs after load with pass-through attention — it may surface the next tracing error (mHC, MoE mxfp4, o_proj,
   or lm_head). Catalog it. (Output will be garbage until attention forward is real — expected.)
2. **Implement the attention forward** (Task #3 — the big one). In `VllmDeepseekV4MLAAttention`, implement
   `forward` / `forward_mqa` / `_o_proj`. Reference: vLLM base `vllm/models/deepseek_v4/attention.py:318` forward and
   the per-step dataflow. Use in-tree kernels:
   - `kernels/experimental/deepseek_v4/mla_swa.py` → `mla_sliding_window_ragged_paged_attention(...)` (mla_swa.py:932):
     dense sliding-window, WRITES KV to cache, no sink. Use for **compress_ratio==1 layers** (layers 0,1, and last).
   - `kernels/experimental/deepseek_v4/mla.py` → `mla_ragged_paged_attention(...)` (mla.py:704): compressed regime
     (ratio 4/128), reads compressed cache, consumes `attention_sinks` + `topk_indices`(CSA)/`kv_lens_to_attend`(HCA) +
     SWA fold-in state (swa_acc/l/m from the SWA kernel run with unnormalized_output=True).
   - You must WRITE the per-layer `compress_ratio` dispatcher (no PR provides it).
   - Dataflow per forward: fused_wqa_wkv → split qr/kv → q_norm/kv_norm → wq_b → per-head weight-free Q RMSNorm + GPT-J
     RoPE → kv RoPE+quant+paged insert → forward_mqa(kernel) → _o_proj(inverse-RoPE + wo_a BMM + wo_b).
   - **START with ratio==1 dense SWA only**, route ALL layers through it first, smoke-test coherence at SHORT context
     (≤128 tokens) where every regime ≈ dense attention over the full sequence (the spec's "dense-equivalent" insight).
   - Parity refs (vLLM torch, in /home/enyouki/vllm): sparse-MLA softmax+sink `tests/kernels/attention/test_rocm_triton_attn_dsv4.py:77-87`;
     inverse-RoPE o_proj `tests/kernels/test_fused_inv_rope_fp8_quant.py:177`. RoPE = GPT-J interleaved, YaRN, base=compress_rope_theta if ratio>1 else rope_theta.
3. **Compressor** (#2858, files in tree): wire into forward step-1 + compressed-cache writes for ratio 4/128. Parity:
   `/home/enyouki/vllm/tests/kernels/test_compressor_kv_cache.py:460`.
4. **Indexer** (#2905): bring in + rework (dead code, broken `streamindex_topk` call missing `distribution`, wrong
   kwargs, cache-format mismatch vs compressor uint8). Wire top-k → mla.py `topk_indices`. Decode re-selection is the landmine.
5. Per-component parity → short-context full-forward logits parity vs vLLM → coherence smoke. Use adversarial agents per kernel.

## Gotchas / things the next agent must know
- The attention runs vLLM's **AMD** model variant (`vllm/models/deepseek_v4/amd/model.py`), forced by the wrapper's
  `_maybe_patch_for_deepseek_v4` (vllm_model_wrapper.py:95) which also mocks cuda.Stream + device_type=cpu. The nvidia
  variant is cute-dsl/DeepGEMM-bound and unusable on TPU.
- Attention class is patched in by replacing the symbol `DeepseekV4ROCMAiterMLAAttention` (deepseek_v4_attention.py:111).
- mHC fused_post_pre / pre / post all run as **pure torch traced by torchax** (no kernel) — keep it that way unless perf demands.
- 41 of 43 layers are ratio 4/128 (compressor/indexer) — so true long-context coherence needs compressor+indexer correct;
  but SHORT-context (≤128 tok) coherence is reachable with just the dense path because the sliding window covers the whole seq.
- Don't touch `glm5.2-dsa*` branches (user's unrelated/broken work). Don't reinstall vLLM (deliberately at ecf9d8352).
- Tasks tracked via Task tool (#1 done, #2 in progress, #3-8 pending). DSV4_BRINGUP_NOTES.md has the kernel signatures
  + numeric references + the full PR integration analysis.
