# gmm_v2: how the kernel works, and why only `tile_m` mattered for decode

Notes from tuning `tpu_inference/kernels/megablox/gmm_v2.py` for Hy3-preview-4bit
decode (W4A16 MLX affine int4, TP=8/EP on v6e-8). All timings below are from a
single v6e chip running the real per-chip decode workload: 24 local experts,
`m=256` rows (32 tokens x top-8), w13 `k=4096 n=3072` (fused silu), w2
`k=1536 n=4096`, group sizes drawn from real routing (~18 active groups,
1-4 rows each).

## 1. What the kernel computes

`gmm_v2(lhs, rhs, group_sizes, ...)` is a **grouped matmul**: `lhs [M, K]` holds
token rows sorted by expert, `group_sizes [G]` says how many consecutive rows
belong to each expert, and `rhs [G, K, N]` holds each expert's weight. Row block
`g` is multiplied only by `rhs[g]`:

```
out[rows_of_g, :] = lhs[rows_of_g, :] @ (scale_g * q_g + groupbias_g)
```

For the MLX affine path the weight stays packed int4 in HBM
(`q` int4 + per-64-group fp32 `scale`/`groupbias`) and is dequantized **inside**
the kernel, so HBM traffic is ~4.5 bits/weight instead of 16.

## 2. Kernel structure

`kernel_main` (gmm_v2.py) runs once per shard and has three parts:

1. **Metadata pass** (`fill_metadata`): walks `group_sizes` on-core and builds a
   compact list of *gm steps* — one entry per `(group, m-tile)` pair, skipping
   empty groups. A group with `rows <= tile_m` costs exactly one gm step; a
   bigger group costs `ceil(rows / tile_m)` steps. This is what makes the kernel
   "ragged-aware": experts that received no tokens cost nothing.

2. **Pipelined grid** (`pltpu.emit_pipeline`, `grid = (num_n, num_gm, num_k)`):
   - `num_n = ceil(N / tile_n)` — outermost,
   - `num_gm` — the gm-step list from the metadata pass,
   - `num_k = ceil(K / tile_k)` — innermost.
   Per grid step the pipeline DMAs the blocks named by the BlockSpecs and calls
   `inner_kernel`. Buffers: lhs tile `[tile_m, tile_k]` double-buffered, rhs
   tile `[tile_k, tile_n]` (+ scale/groupbias rows) **triple**-buffered — so the
   next group's weight load overlaps the current group's compute — output
   double-buffered. Consecutive grid steps that name the same block index skip
   the re-fetch.

3. **`inner_kernel`** (per grid step), W4A16 path:
   - unpack the int4 tile (`pltpu.bitcast` on the packed uint32) — VPU work,
   - dequantize: cast to bf16 and multiply by the per-quant-block scale
     (a reshape-broadcast per 128-column MXU chunk) — VPU work **proportional
     to `tile_k x tile_n`, independent of `tile_m`**,
   - one MXU matmul `[tile_m, tile_k] @ [tile_k, 128]` per column chunk —
     MXU work **proportional to `tile_m`**,
   - the affine `+groupbias` term is folded in as a second small matmul
     `lhs_block_sums [tile_m, num_blocks] @ groupbias [num_blocks, 128]`,
   - mask rows outside `[m_start, m_end)` of the current group and write out.
     (Masking happens at the *output*; the MXU still processes all `tile_m`
     rows.) `partial_out_ref` handles adjacent groups sharing an 8-row sublane.

## 3. The tiling parameters and their defaults

`calculate_tiling` picks `TileSizes(tile_m, tile_k, tile_n)`:

| param | meaning | default at our shapes | constraint |
|---|---|---|---|
| `tile_m` | lhs rows per MXU pass; granularity of gm steps | `128 * lhs_mod / rhs_mod` = **64** for bf16 x int4 | multiple of the 8-row bf16 sublane |
| `tile_k` | contraction chunk; k-chunks accumulate into `acc_ref` | full K (**4096** / **1536**) — only shrinks if VMEM forces it | must stay compatible with the 64-wide quant block |
| `tile_n` | output-column chunk per rhs DMA | full per-projection N (**1536** / **4096**) — shrinks first when VMEM is tight | >= 2x MXU column size |

At our shapes everything fits VMEM, so the default grid degenerates to
`(1, num_gm, 1)`: **one grid step per active expert**, each DMA-ing that
expert's full weight tile once. That is already optimal DMA-wise — every
active expert's bytes are read exactly once — which is the key to understanding
the sweep results.

## 4. What the sweep showed

Decode w13 (`m=256`, ~18 active groups, fp32 meta), measured 2026-07-02:

| config | time | vs default |
|---|---|---|
| default (`tm64, tk4096, tn1536`) | 179 us | — |
| `tm32` | 167 us | -7% |
| **`tm16`** | **161 us** | **-10%** |
| `tm8` | 182 us | +2% |
| `tk2048` (split k) | 161-178 us | ~0 to -10%, noisy, never beat tm16 |
| `tn768` (split n) | 189-210 us | +6..18% worse |
| `tm128` / `tm256` | 181-367 us | much worse |
| bf16 rhs (no dequant, 4x bytes) | 361 us at **77-90% of HBM peak** | — |
| int4 default | 179 us at **~50% of HBM peak** | — |

(w2 behaves the same; `tm16` -9%.)

## 5. Why capping `tile_m` at 16 helped

At decode, routing spreads 256 rows over 24 local experts: the typical active
group has **1-4 real rows**. A gm step always runs the MXU over all `tile_m`
rows — rows past the group's end are computed and masked at the output. So per
active expert:

- `tile_m=64`: the MXU chews 64 rows x 4096 k x 1536 n, of which 1-4 rows are
  real — **~94-98% of the MXU work is wasted**.
- `tile_m=16`: same DMA, same dequant, but 4x less MXU work per group.

This only matters because the kernel at these shapes is **compute-bound, not
DMA-bound** (~50% of HBM peak while the bf16 variant proves 77-90% is
reachable): shrinking wasted MXU rows directly shortens the critical path. The
DMA volume is *identical* at every `tile_m` — an active expert's weight tile is
fetched once regardless — which is why the win is ~10% and not 4x.

Why not `tile_m=8`? Two costs turn around below 16:
- groups with 9-16 rows now need **two** gm steps instead of one — more grid
  steps, each with fixed pipeline/masking overhead, and the second step re-uses
  the same rhs block (no new DMA, but the whole inner_kernel prologue reruns);
- 8 rows equals the bf16 sublane granularity exactly, so misaligned group
  boundaries force more `partial_out_ref` revisit handling per step.

Measured: `tm8` is worse than `tm64`. 16 is the empirical sweet spot: big
enough to cover nearly all decode groups in one step, small enough to waste
little MXU.

`small_m_tiling` (gmm_v2.py) implements exactly this: default tiling, then
`tile_m = min(16, tile_m)` when `size_m <= 256`. It is bit-exact (row tiling
does not change any dot product's accumulation order) and a no-op at prefill.
It is wired in only for the MLX affine path (`rhs_groupbias is not None` in
`fused_moe_gmm.gmm_wrapper`, plus the dense `_mlx_int4_matmul`), so every other
quant path keeps byte-identical tiling.

## 6. Why `tile_k` and `tile_n` didn't help

Both parameters only *redistribute* work — the bytes read from HBM and the
dequant/MXU element counts are unchanged — so at best they break even, and each
split adds real overhead:

- **Splitting `tile_k`** (4096 -> 2048/1024): `num_k > 1` turns the k loop into
  multi-step accumulation — each (n, gm) pair now reads and rewrites the
  `[tile_m, tile_n]` accumulator per k step (extra VPU/VMEM traffic), and the
  grid has more steps. Nothing is saved: the same weight bytes arrive, just in
  two DMAs. `tile_k` also must stay a multiple (or divisor) of the 64-wide
  quant block. It exists as a *pressure valve* for when `[tile_k, tile_n]`
  doesn't fit VMEM — which is not our case.

- **Splitting `tile_n`** (1536 -> 768): `n` is the **outermost** grid dim, so
  `num_n=2` runs the entire per-expert gm sweep twice — twice the pipeline
  fixed costs and per-step prologues, plus smaller (less efficient) DMA
  transfers — again with zero byte savings. Its floor (`2x MXU column size`)
  exists to keep the MXU fed; there is no upside to going below the
  VMEM-fitting maximum.

- **Growing them** was not an option: both were already at their maximum
  (full K, full per-projection N).

The general rule this sweep illustrates: in a memory-bound-by-design kernel
whose DMA schedule is already minimal, tiling changes only pay when they cut
**wasted compute** (here: `tile_m` vs 1-4-row groups) or enable **overlap**;
re-chunking the reduction (`tile_k`) or the output (`tile_n`) just adds seams.

## 7. What actually limits the kernel now

After `tm16` + the fused-broadcast dequant change (reshape-broadcast scale
multiply instead of `concat`+`jnp.repeat`, and skipping the clamped-tail gather
when `tile_k` doesn't over-align), w13 is at 147.6 us ≈ **~60% of HBM peak**.
The remaining gap to the bf16 variant's 77-90% is the per-element VPU dequant
(int4 unpack + bf16 convert + scale multiply), which no tiling parameter can
remove — it scales with the weight bytes themselves. Options beyond tiling:
push more of the dequant onto the MXU (the W4A8 quantized-lhs path — measured
roughly break-even here), or reduce metadata traffic (bf16 scales measured
*slower* due to extra VPU casts).
