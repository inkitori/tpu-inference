# MLX int4 dense linear — status

Goal: make `VllmMLXLinearMethod` do int4 matmul IN-KERNEL (via `gmm_v2`) instead
of dequant→bf16→einsum every step. Done + correctness-validated on v6e-8.

## Code changes (production)
`tpu_inference/layers/vllm/quantization/mlx.py`:
- `VllmMLXLinearMethod.process_weights_after_loading`: unpacks ONCE to gmm_v2
  layout — int4 codes `[1,in,out]`, scale/groupbias `[1,in//gs,1,out]` (signed
  `q-8`, `groupbias=bias+8*scale`). Sharding specs mirror `tensor_parallel_gmm`.
- `apply`: calls new module helper `_mlx_int4_matmul` (single-group gmm_v2 in a
  `shard_map` + psum for RowParallel) instead of einsum.
- Trimmed the MoE `w2` bf16 fallback — w2 is now always int4 (valid for hy3).

## Validated (ALL PASS — `test_correctness.py`)
Fold algebra bit-exact (0.0); gmm vs old dequant+einsum ~2.5e-3 (bf16 tol) for
qkv/o/gate_up/down at tp=1 col+row; real sharding at tp=8 (row psum + col).

## vmem caveat
gmm_v2 sets BOTH tiling target and hard cap from `vmem_limit_bytes`. The
unsharded down_proj K=13312 at **tp=1** needs the FULL 128MB (`128<<20`); the
default `0.9*cap` is ~9MB short. So `_mlx_int4_matmul` takes an optional
`vmem_limit_bytes` (default None). Production runs tp>=2 → K shards small →
default is fine. Bench/test pass `128<<20`.

## Running (interpreter: /home/enyouki/vllm_env/bin/python — has jax 0.9.2)
- `python -u scratch_mlx_int4/test_correctness.py`   (correctness)
- `python -u scratch_mlx_int4/benchmark.py`          (decode latency old vs new)
- Always use `-u` (Pallas compiles ~15-30s each, output block-buffers otherwise).
- Persistent compile cache at `scratch_mlx_int4/.jax_cache` → 2nd run is fast.
- Eager does NOT speed it up: gmm_v2 is a Pallas kernel, always compiled.

## TODO / not done
- Benchmark numbers: see `bench.out` (was running at handoff).
- Full method-object e2e (construct VllmMLXLinearMethod + load real hy3) not run;
  only the kernel path + transform math are unit-tested. The method wiring is
  mechanical glue over the validated helper.
