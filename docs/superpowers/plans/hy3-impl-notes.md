# Hy3-4bit bring-up — impl notes (working)

Spec: docs/superpowers/specs/2026-06-22-hy3-preview-4bit-bringup-design.md

## Key paths
- Gating: `tpu_inference/layers/common/fused_moe_gmm.py` (`fused_moe_func` sig 467-499; gating block 531-541; `apply_scoring_fn` 82-90)
- moe_apply: `tpu_inference/layers/common/moe.py` (GMM case 133-153; reads layer attrs)
- interface: `tpu_inference/layers/vllm/interface/moe.py` (`vllm_moe_apply` 60-89, passes whole `layer`)
- MLX config+MoE method: `tpu_inference/layers/vllm/quantization/mlx.py` (`from_config` 81-85; `get_quant_method` 104-107; `_process` jit 442-456; w13 fold 446-452; w2 currently bf16 ~456)
- weight transform: `tpu_inference/models/vllm/mlx_weight_transform.py` (`_dequant_to_bf16` 51-75; triplet 106-117; `_SWITCH` 45-47)
- moe_weights: `tpu_inference/layers/common/process_weights/moe_weights.py` (FusedMoEWeights 40-57; FUSED_MOE groupbias assert 314)
- gmm_v2: `tpu_inference/kernels/megablox/gmm_v2.py`

## Hy3 facts
num_experts=192, top8, 1 shared, moe_inter=1536, hidden=4096, gs=64, 80 layers, layer0 dense.
sigmoid + expert_bias(sel-only) + route_norm + router_scaling_factor=2.826. NO expert groups. MTP absent.
Quant: bits4 gs64 affine; 79 overrides `model.layers.{1..79}.mlp.router.gate:{gs64,bits8}`.
expert_bias=F32[192] plain. router.gate=8bit triplet. shared_mlp/embed/lm_head=4bit triplet.
Checkpoint: /tmp/gcs/bucket/hub/models--mlx-community--Hy3-preview-4bit/snapshots/8e4d56f18efd912b8c7581a8ccfa8b2a79ba3469/

## vLLM hy_v3.py wiring (confirmed)
FusedMoE gets e_score_correction_bias=self.expert_bias, routed_scaling_factor=router_scaling_factor(2.826),
scoring_func="sigmoid", renormalize=route_norm(True), use_grouped_topk=True num_expert_group=1 topk_group=1
(==plain topk). So layer.* attrs ARE populated. Router gate = GateLinear(quant_config=None FORCED unquant, F32),
prefix mlp.gate; vLLM strips `router.` from `router.gate.` only.
BLOCKERS (vLLM doesn't handle; fix in transform):
 - mlp.router.expert_bias NOT remapped -> KeyError. Transform rename -> mlp.expert_bias.
 - mlp.shared_mlp.{g,u,d}_proj: vLLM HYV3MoEFused registers self.shared_mlp (HYV3FeedForward) -> real param
   `...mlp.shared_mlp.{gate,up,down}_proj.*`. Transform PRESERVES the shared_mlp. infix (keep 4bit; vLLM stacks
   gate+up into gate_up_proj). [VERIFIED correct; earlier strip-to-bare-mlp was WRONG (bare mlp. = dense layer0 only).]
 - routed_scaling_factor could be None -> guard to 1.0.
switch_mlp->experts ALREADY done by transform. embed/lm_head dequant ALREADY done. Dense layer0 mlp passthrough OK.

## Plan / status
1. [DONE-code] Gating (fused_moe_gmm + moe.py + interface/moe.py): e_score_correction_bias(traced,None),
   routed_scaling_factor(static,1.0). math: s=score; sel=s+bias; idx=topk(sel,k); w=take_along(s,idx);
   if renorm w/=(sum+1e-20); w*=rsf. interface reads layer attrs, jax_view bias. Qwen3 no-op.
2. 8bit router gate: transform dequant mlp.router.gate triplet(bits8 gs64)->bf16 single .weight; from_config
   parse per-module overrides, accept only *.mlp.router.gate bits8 gs64 reject else (gate already unquant in vLLM). [ ]
   + transform: rename expert_bias, strip shared_mlp (the 2 blockers above). [ ]
3. [DONE-code] w2 4bit in-kernel: mlx.py _process mirror w13 fold for w2, GATED on w2_num_blocks%mlp_tp==0
   (Hy3 24%8=0; Qwen3 12%8!=0 falls back bf16). process_moe_weights/shard handle w2 symmetric to w13 (verified).
   apply_monolithic branches on self._w2_int4.
4. [DONE-verified] generic expert count/shapes: loader reads hf_config.num_experts (192) generically;
   gate=is_mlx_quantized (no arch allowlist); no hardcoded-128/all-MoE assumptions; dense layer0 -> LinearBase.
5. run: tp=8 greedy coherence. Ensure GMM backend (not FUSED_MOE; groupbias asserts None there). [ ]
   ENV: ~/vllm_env/bin/python (NOT docker). vLLM editable /home/enyouki/vllm. HF_HOME=/tmp/gcs/bucket
   (model at /tmp/gcs/bucket/hub/models--mlx-community--Hy3-preview-4bit). gcsfuse root-owned ro; maybe sudo+HF_HUB_OFFLINE=1.

## RUN OUTCOME: loads + decodes at tp=8 but GARBAGE — every prompt = token 0 (`!`) x128, input-independent
## (= NaN/degenerate logits). routed_scaling_factor=2.826 + e_score_correction_bias CONFIRMED reach kernel.
## Two source fixes needed to get crash->running (both VERIFIED correct by review):
##   A) mlx_weight_transform.py: PRESERVE shared_mlp. infix (was strip; strip KeyError'd on down_proj.biases).
##   B) interface/moe.py: e_score_correction_bias arrives as raw un-moved nn.Parameter (FusedMoE plain alias,
##      not register_parameter'd) -> .to(device="jax") before jax_view (added import torchax).
## 4-agent adversarial review CLEARED: transform (every name lands right), routing math (byte-exact vs HF+vLLM),
## int4 fold/unpack/groupbias/w2-shard, shared-expert exec. lm_head-fp32 theory = DEAD (precision can't -> constant).
## PINPOINTED (instrumented tp=8 forward, jit even w/ enforce_eager; via jax.debug.print on jax_view):
##   First NaN = L1.moe.final_out (routed 4-bit expert gmm_v2 w13 combine, FIRST MoE layer). SPARSE NaN,
##   finite parts normal-magnitude. CLEAN before it: embed, L0 dense, L1 attn (qkv/qknorm/rope/core/oproj),
##   moe.input, router_logits (8bit gate, healthy 2.74/-5.17), shared_out (0.19/-0.35). NaN -> residual ->
##   logits NaN -> argmax token0 '!'. CLEARS all reviewer suspects (attn/rope/qknorm/oproj/embed/gate/shared).
##   => bug is in the ROUTED quantized GMM, tp=8-specific (tp=2 e2e passes exact). Leading: int4 scale/groupbias
##   shard misalign @tp=8, OR empty/uneven expert groups (12-tok prompt over 192 experts -> most experts 0 tokens).
## ROOT CAUSE (static-confirmed): gmm_v2 OOB per-group scale/groupbias read in w2 (down) at tp=8 ONLY.
##   w2 in-dim 1536/8=192/shard -> 3 quant blocks (gs64). calculate_tiling: tile_k=align_to(192,128)=256 (rhs
##   tiny -> shrink loops never fire) -> num_quant_blocks_per_tile_k=cdiv(256,64)=4 > 3 real blocks. BlockSpec
##   DMAs block[0:4] from 3-long axis -> OOB read (disable_bounds_checks=True) -> NaN scale/groupbias. k-tail
##   mask zeros matmul VALUE but NaN_scale*0=NaN & NaN_groupbias*block_sum=NaN -> SPARSE NaN. tp<=4 safe:
##   1536/tp in {1536,768,384} all %128==0 -> tile_k==size_k. w13 always safe (4096 contraction replicated).
##   gmm_v2.py: 210-211, 309/316, 407/419/432, 944, 965-967, 1372.
## FIX: gmm_v2 keep tile_k lane-aligned(256) but BOUND quant-block DMA/index to real blocks (no over-read);
##   tail block -> 0 via k-mask. Escape hatch: w2_keep_int4=False (mlx.py:476-479) -> w2 bf16 (w13 int4 stays).
## FIX DONE + VERIFIED: gmm_v2.py (+75/-9) new num_quant_blocks_per_tile_k_read clamps scale/groupbias DMA+index
##   to real blocks when tile_k over-aligns past size_k (single-tile case); no-op for 128-aligned size_k so
##   tp<=4 + W4A8 are byte-identical. tail block contributes 0 via existing k-mask. +assert in make_gmm_configs.
##   REGRESSIONS PASS: gmm_v2_groupbias 5, gmm_test wq 10/3skip, W4A8 4, mlx e2e tp=1&2 STILL exact-match.
##   CAVEAT: unit test can't deterministically repro the NaN (OOB read is HBM-content dependent) -> the invariant
##   assert is the deterministic guard; the tp=8 model run is the real e2e proof.
## DONE — ACCEPTANCE PASSED (2026-06-23): tp=8 greedy run yields COHERENT English on all 4 prompts
##   ("capital of France is **Paris**", correct sky-is-blue physics, etc). Constant-`!` garbage GONE. Clean
##   exit. The gmm_v2 OOB fix resolved it end-to-end. (One earlier launch failed only on a TPU collision with
##   a duplicate run from the kernel-fix agent — not a code issue; relaunched single-owner -> success.)
## REMAINING: commit (branch hy3, identity inkitori) once user approves; full diff = gating + w2-int4 +
##   transform (shared_mlp/expert_bias/8bit-gate) + interface bias-move + gmm_v2 OOB fix + tests.

Tests: [A DONE 20 pass] CPU routing-parity + compute_moe_routing helper extracted from fused_moe_func.
[B DONE] w2 int4: oracle CPU 3 pass; gmm_v2 TPU 3 pass; e2e TPU tp=1&2 exact MLX==REF (w2 int4 proven).
  (agent B also fixed test fixture _expert_map=None — vLLM weight_loader change, baseline-broken.)

## Pre-run blockers cleared (verified):
- MTP absent: HYV3ForCausalLM never builds MTP head; no missing-param check; spec-decode not auto-enabled. OK no-op.
- Merged quant linear (dense layer0 gate_up/qkv, shared expert gate_up): qkv proven on real Qwen3; gate_up
  unit-tested to golden (test_mlx_linear_method). Structurally identical. No code change.
- gcsfuse /tmp/gcs/bucket is root-owned -> run needs sudo. HF_HUB_OFFLINE=1, HF_HOME=/tmp/gcs/bucket.
## Run recipe: sudo HF_HOME=/tmp/gcs/bucket HF_HUB_OFFLINE=1 MODEL_IMPL_TYPE=vllm SKIP_JAX_PRECOMPILE=1
##   ~/vllm_env/bin/python -u scripts/run_hy3.py. Monitor: tail -n0 -f log | grep (NOT pgrep waiters); end sentinel.
