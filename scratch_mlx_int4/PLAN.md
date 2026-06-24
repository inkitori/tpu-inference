# GOAL: 150+ per-user tok/s (vllm bench serve), Hy3-preview-4bit, 4096in/1024out
# MODEL_IMPL_TYPE=vllm, fp8 KV cache. Also >=100 tok/s/user @ 8 concurrent. Keep correctness+coherence.

## Model (MoE): 80 layers, hidden 4096, dense MLP only @layer0 (interm 13312).
Layers 1-79 MoE: 192 experts, 8 active + 1 shared, moe_interm 1536. 64 q / 8 kv heads, hd 128. int4 g64.

## Current state (committed a99cee7c)
- Dense linears, MoE experts (w13+w2), shared expert, attn QKV/O all int4 IN-KERNEL via gmm_v2. fp8 KV plumbed. router gate bf16.
- Microbench: decode matmuls 2-7x faster than old dequant+einsum; PREFILL 0.58x (SLOWER at M=2048).
- e2e NOT yet measured with this code.

## Commands
SERVE: HF_HOME=/tmp/gcs/bucket/vllm HF_HUB_OFFLINE=1 MODEL_IMPL_TYPE=vllm /home/enyouki/vllm_env/bin/vllm serve mlx-community/Hy3-preview-4bit --trust-remote-code --tensor-parallel-size 8 --kv-cache-dtype fp8 --max-model-len 10240 --max-num-seqs 512 --max-num-batched-tokens 8192 --no-enable-prefix-caching --gpu-memory-utilization 0.95 --seed 42
BENCH: /home/enyouki/vllm_env/bin/vllm bench serve --model mlx-community/Hy3-preview-4bit --backend vllm --host 127.0.0.1 --port 8000 --dataset-name random --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1.0 --ignore-eos --request-rate inf  [+ --num-prompts N --max-concurrency C]
per-user tok/s = 1000/MeanTPOT_ms. Measure @ concurrency 1 (target 150) and @ 8 (target 100).

## Ops
- TPU free check: HF_HUB_OFFLINE=1 /home/enyouki/vllm_env/bin/python -c "import jax;print(len(jax.devices()))" == 8
- Kill server: pkill -f "vllm serve"; then kill -9 the EngineCore PID DIRECTLY (proctitle truncates to "VLLM::EngineCor" w/o final 'e' so `pkill -f EngineCore` MISSES it). After kill: rm -f /tmp/libtpu_lockfile. Reload ~15min, batch e2e.
- microbench dir: scratch_mlx_int4/ (test_correctness.py, benchmark.py). Use -u.

## BASELINE (committed a99cee7c, W4A16 in-kernel gmm) — 2026-06-24
- c=1: TPOT 26.74ms => 37.4 tok/s/user (target 150, need 4x). TTFT 644ms.
- c=8: TPOT 39.36ms => 25.4 tok/s/user (target 100, need 4x). TTFT 3.3s.
- Coherence: GOOD ("capital of France is Paris...", sky-blue answer correct).
- DIAGNOSIS: HBM floor ~0.8ms/token but actual 26.7ms = 34x off => NOT memory-bound.
  At batch=1 the int4 gmm is VPU-DEQUANT bound (unpack int4->bf16 for every weight, O(K*N) VPU work before tiny matmul).

## HYPOTHESIS / lever #1: W4A8 (int8 activations + quantized-lhs gmm path)
- int8 act -> gmm does int8xint4->int32 on MXU; affine scale/groupbias applied POST-matmul O(N/group) not O(K*N). Kills VPU dequant.
- gmm_v2 HAS a quantized path (gmm_v2.py ~494-566) but wrapper forces maybe_quantize_lhs=False for affine (fused_moe_gmm.py:156, and mlx _mlx_int4_matmul). Need to enable + verify affine math + correctness/coherence + speedup.
- Plan: microbench ONE MoE layer gmm (M=8, real int4, TP=8) bf16-path vs int8-path: prove VPU-bound, prove correctness, prove speedup. THEN apply to attn/dense/shared too. THEN one e2e reload.

## W4A8 int8 activations (int8act_bench.py) — RE-OPENED (was wrongly dismissed)
- Quantized-lhs path in gmm_v2 BROKEN for affine: lhs quant_block 512 vs rhs group 64 -> mis-applies scale (40-50% err). Gated off (fused_moe_gmm.py:138-156, mlx.py:170). Fix: lhs block->64 gives correct 1.5e-2.
- "no speedup" verdict was from PER-CALL timing = INVALID: dispatch floor is 125us/call, so 150us per-call gmm was ~125 dispatch + ~25 kernel; int8 vs bf16 delta lost in noise.
- Need CLEAN device-time (chained-in-jit) bf16 vs int8 ratio to settle. (in flight)

## KEY: dispatch floor = 125us/call. In real model all 80 layers = ONE graph (dispatch paid once). So e2e (26.74ms) is REAL device compute, ~334us/layer. Isolated per-call microbench USELESS; must use chained-in-jit device-time OR real-decode profile.

## METHODOLOGY FIX (important)
- Per-call microbench timing is DISPATCH-CONTAMINATED: single gmm shows ~150us flat M=2..64, but e2e is 334us/LAYER with 4-6 gmms/layer => true in-graph per-gmm must be <=55us. Must measure DEVICE-TIME via in-jit loop or jax.profiler, NOT per-call wall clock.
- a08d4490 measuring true device-time (dispatch floor vs in-jit gmm) now.

## KEY REFRAME: decode is FIXED-COST-bound (weight streaming), not per-token compute.
- c=8 TPOT 39.4ms is only 1.47x c=1 26.7ms despite 8x tokens => per-step cost ~constant in batch. Batching gives 5.4x/token efficiency. Target = cut the fixed per-step weight-stream+dequant cost.
- dense int4 gmm device-time ~45us (3.5x over HBM floor 13us) due to VPU dequant. MoE grouped gmm (8 experts, M=1 each) is prime suspect for being even worse.

## PROFILING (in flight, bdf1u71ff, ~20min): examples/tpu_profiling.py, batch=1, input64/output96, --profiler set programmatically. Trace -> scratch_mlx_int4/prof. Parser ready: parse_trace.py (per-op device time, categorized, /95 steps).
## Profiler facts: VLLM_TORCH_PROFILER_DIR is IGNORED on this build. Use examples/tpu_profiling.py (sets profiler_config.profiler='torch' programmatically) OR serve with CLI flags --profiler torch --torch-profiler-dir. Readout: parse <dir>/**/*.trace.json.gz (chrome trace) by device plane. fp8 kv => fp8_e5m2 on v6e.

## PROFILE RESULT (decisive) — batch=1 decode, parse_trace.py on scratch_mlx_int4/prof
- decode step = jit_step_fun 26.3ms device (== TPOT 26.7ms) => fully DEVICE-BOUND.
- Leaf-op breakdown per step: MoE expert gmm ~77% (~22ms!), collective 8%(0.4ms), moe-routing 5%(0.25ms), attention 5%(0.24ms), fusion 3.6%. Dense attn gmms (qkv/o) NOT in top40 => minor.
- MoE gmm op: gmm_v2-g_192-m_128-k_4096-act_silu-n_512-tm_64... => M PADDED TO 128 (batch=1: 1 tok pads to 16, x8 topk =128 => 16x the rows needed!). ~139us/call, ~20x over HBM floor (5.9us). 79 layers x2 gmm.
- TARGET: the MoE expert grouped gmm. Fix it -> 4x. Other 23% (~4ms) is collective+route+attn.

## NEXT: MoE-gmm microbench (no e2e reloads). Reproduce decode MoE gmm (TP=8, g192, 8 active, M=128, hidden4096, moe_interm1536, int4 g64). Test: (a) M 128->16->8 padding, (b) EP vs TP sharding, (c) tile tuning. device-time(chained-in-jit)+correctness. Then 1 e2e reload to confirm winner.

## MoE-GMM MICROBENCH RESULT (moe_gmm_bench.py) — KEY
- gmm cost ∝ #DISTINCT ACTIVE EXPERTS (~3.7us each in TP; every chip runs every active expert). Profile 139us = ~32 active (NOT 8): batch=1 pads 1 tok->16 toks, padding toks route to garbage experts => 8 active inflates to ~32. Token-padding ~4x's MoE work!
- EP beats TP everywhere: 1.5x @8 active, 2x @64; halves down (no psum). Wired for MLX, correct (rl2 1e-3).
- M-padding reduction helps (TP 128->8: 7.7->4.7ms/step).
- int8-lhs helps MoE gmm (gate_up 45->27us!) but needs affine fix (lhs block 64). [dense gmm it didn't help]
- Projections (8 active): TP128=85, TP8=115, EP128=96, EP16=116 tok/s. 150 needs EP + low-pad + int8-lhs + trim collectives.

## LEVERS (stack, on 77% MoE hotspot): (1) reduce decode token-padding 16->min (cuts active experts 32->8, ~3x, mainly c=1); (2) EP --enable-expert-parallel (~1.5-2x both, removes psum=>helps collective 2.3ms); (3) int8-lhs MoE+affine fix (~1.4x gate_up).

## PADDING-MASK IMPLEMENTED (uncommitted, syntax OK) — diff in 3 files:
- fused_moe_gmm.py compute_moe_routing: mask rows >= num_actual_tokens to expert0/zero-weight (shared routing, helps TP+EP).
- moe.py moe_apply + vllm/interface/moe.py vllm_moe_apply: num_actual_tokens = jnp.max(query_start_loc) from fwd ctx, guard ATTN_DATA==1, None fallback.
- CPU-verified: active experts 95->8, real row byte-identical. Correctness safe (max(qsl) >= N real tokens always => never masks real rows).
- int8-lhs affine fix located (gmm_v2.py:1204 lhs quant_block=rhs 64) — NOT yet applied (secondary).

## EP OOM LEARNING: EP at gpu-util 0.95 OOMs compiling 4096-prefill (needs 351M, 189M free). EP whole-expert dispatch uses more peak HBM than TP at big prefill. FIX: gpu-util 0.88. (Confirmed "Using GMM EP kernel" engages before OOM.)

## Progress
- [x] baseline (c1 37, c8 25); W4A8 dense dead; profile (MoE gmm 77%); microbench (EP>TP, padding inflates active experts); padding-mask coded+CPU-verified
- [ ] EP+mask server loading @0.88 (b7iymhe8x, watcher bp9493dpi). Bench c1/c8 + coherence when ready. Projected EP+mask ~105 tok/s c1.
- [ ] if works: commit. consider M-padding reduction + int8-lhs for more.
