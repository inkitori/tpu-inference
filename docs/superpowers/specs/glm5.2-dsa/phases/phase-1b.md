# Phase 1b — multi-device S1 gate

The gate that catches the S1 / uninit-HBM / reshard class. De-risks ~80% of the model and brings up the
multi-device gate. **Run on the single-host v6e-8 8-chip mesh via the real sharded weight-loader** (core §H6
is the gate definition; read it).

**Precondition:** **1a hard-gates 1b** — never debug math on the mesh. 1a's dense backbone must be green.
**Core anchors:** §F8 (loader), §G3, §H6 (THE multi-device gate def + localization), §H11, §H12, §H13, §I1–§I6, §J1, §J2.

## Deliverables
- Run the dense backbone on the **8-chip mesh via the real `JaxAutoWeightsLoader`** (core §F8).
- **NaN-poison HBM** before load; assert no NaN/garbage leaks into outputs.
- Assert **N-device fp32 == 1-device fp32** (sharding is value-invariant); compare 1/2/4/8-device.
- Exercise `expert_axis_name` sharding.
- **When the gate trips, localize with the per-weight/per-stage checksum probe** (core §H6), not deductive
  bisection — NaN-poison misses the coherent-but-wrong S1 variant.
- **Sharding-geometry stress fixture on the MEDIUM config — TP mode** (core §I5): experts **not** divisible
  by the `MLP_TENSOR` axis (e.g. 10 or 14 experts on the 8-chip axis; idle data-shard rows) so the GMM_TP
  reduction tiles non-trivially. It need not pass full HF math parity; its job is to arm the S1 gate.
- **(FIX) EP-mode geometry gate (production's real sharding mode).** Production routed-expert sharding is
  **pure even EP** (`use_ep=True`). EP and TP run the **same fused backend** (`GMM_EP`/`GMM_TP` →
  `fused_moe_func`) but reshard over **different axes** — EP over `ShardingAxisName.EXPERT` (`ragged_gather` +
  `psum`, `fused_moe_gmm.py:231-232,:478,:704-709`), TP over `MLP_TENSOR` (core §I4). The TP-mode fixture
  above does **not** exercise the EXPERT reshard. Add a gate that runs the **medium config in pure-EP mode** —
  8 devices on the `expert`/`attn_dp_expert` axis, experts **even-divisible** (e.g. 16 experts → 2/shard),
  `total_TP==1` — through the same **N-dev==1-dev fp32 + NaN-poison + per-weight checksum** pipeline. **Assert
  `ep_size = product(mesh, ShardingAxisName.EXPERT)` divides `n_routed_experts` evenly** (the production
  invariant; the fused path *silently masks* uneven shards rather than faulting — core §I5). Without this, the
  EXPERT-axis reshard the real model actually uses is first-contacted only at Phase R.

## Acceptance gates (numeric)
- **N-device fp32 == 1-device fp32** across 1/2/4/8-device.
- NaN-poison clean (no NaN/garbage in outputs).
- **Per-weight 1-dev-vs-N-dev checksum identical** (the localizing probe, core §H6).
- Medium-config **TP-mode** sharding-geometry stress fixture clean (no NaN/garbage, value-invariant).
- **(FIX)** Medium-config **EP-mode** gate green: N-dev==1-dev fp32 + NaN-poison clean in pure-EP (the
  `fused_moe_func` EXPERT-axis `ragged_gather`+`psum` reshard exercised; `ep_size | n_routed_experts` asserted).

## Phase-specific risks & fixtures
- **S1 / uninit-HBM / reshard** — silent, coherent-but-wrong, CPU-invisible (core §H6; the fix is
  host-stack-into-sharded-layout, `gmm_v2(zero_initialize=True)` disproven).
- **Tiny config doesn't reproduce S1** → the medium-config stress fixture arms a triggering geometry (core §I5).
- **64-head divisibility** — verify 64 is divisible by the model-axis size before running (core §I3).
- Mesh compile fixtures: no size-1 token-axis `with_sharding_constraint` gather; no-fallback ragged-gather
  compile check.
