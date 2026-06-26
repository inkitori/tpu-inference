# MLX int4 MoE gmm_v2 microbench — baseline & opt log

Bench: `bench/mlx_moe_gmm_bench.py`. Run: `~/tpu-tooling/tpu-env.sh python bench/mlx_moe_gmm_bench.py`
TPU v6e-8. E=192, topk=8, gs=64, TP=8 per-shard isolated kernels.
GMM1 = w13 rhs[192,4096,384] (64 blocks). GMM2 = w2 rhs[192,192,4096] (3 blocks).
Gate: rel_L2 < 1e-2 (baseline ~2.4e-3). Device-time ms, imbalanced routing (headline).

## Baseline (per-call ms)
| M_seqs | gmm1 | gmm2 |
|--:|--:|--:|
| 8   | 0.2050 | 0.0747 |
| 16  | 0.3408 | 0.1211 |
| 32  | 0.5332 | 0.1864 |
| 64  | 0.6573 | 0.2303 |
| 128 | 0.7200 | 0.2570 |

## Optimizations
- F1: width-64 K-slice matmuls -> full-tile-k dequant (fold scale into rhs cols) + single matmul; groupbias via hoisted per-block lhs sums @ gbias. Touches only the `lhs_cfgs.quant_dtype is None` (W4A16) branch in `_matmul` (gmm_v2.py ~424-474). Absorbs F2 (hoist block_lhs_sum out of start_n loop). Exact in fp (scale is per-(block,n), constant across k-in-block).
- F8: argsort -> counting sort in fused_moe_gmm routing (separate bench needed).

## Results
- **F1: KEPT.** GMM1 -36% across M (0.72->0.46ms @M128); GMM2 +8% (cheap 3-block kernel, full-tile dequant overhead > MXU gain). Net per-MoE-layer (gmm1+gmm2) -24.8% @M64 (0.888->0.668ms). rel_L2 2.9e-3 < 1e-2 PASS. F2 absorbed (lhs_block_sums hoisted out of start_n loop). Pallas note: clamped block idx built via Python-int list + concatenate, not jnp.asarray gather (kernel can't capture array constants).
- **F8: REJECTED by bench.** Routing two-argsort block = 0.7-2.2% of GMM pair, dominated by ~3.4us fixed dispatch floor (doesn't scale with N). Scatter-invert (#2) and counting-sort (#1) both verified EXACT integer-perm match but neither faster. No production edit. Bench kept at bench/moe_routing_bench.py.
- Optional follow-up (not done): gate F1 on num_quant_blocks_per_tile_k to recover GMM2's 8% — skipped as a fragile magic-number heuristic in a shared general-purpose kernel.
