# RPA v3 fp8_e5m2 TPU v6e tuner-mapping verification

## 1. Executive summary

Final conclusion label: **VALIDATED_SLOWDOWN**

The earlier `~1.2x` number was real in the tuner checkpoint, but it was not a production-wrapper speedup. For the four overlapping shapes, the checkpoint reports a `1.220632x` geomean versus the tuner sweep's own successful baseline. That baseline was not the current runtime wrapper's selected block tuple.

I tested the suspected fix: reconstructing the four-value RPA v3 block tuple exactly as the historical tuner did, with `bq_csz <= 32` and `bkv_csz = page_size`. On real TPU v6e with `q=bfloat16` and `kv=float8_e5m2`, that exact tuner tuple was slower than the current old wrapper mapping:

| workload | comparison | geomean speedup | conclusion |
|---|---|---:|---|
| decode-only wrapper | old mapping `63800815` / exact tuner tuple `fd812530` | `0.558417x` | exact tuner tuple slowed down |
| tuner-like mixed wrapper | old mapping `63800815` / exact tuner tuple `fd812530` | `0.698204x` | exact tuner tuple slowed down |
| historical tuner checkpoint only | tuner winner / tuner baseline | `1.220632x` | real tuner-internal win, not production-wrapper proof |

The production kernel mapping was restored after the experiment in `cae90ede`, so final HEAD does not leave the known-slower exact tuner tuple active. The only retained code change is the verification harness support for `--workload tuner_mixed`.

Biggest caveat: this still uses a small four-shape representative matrix, not a full production trace. It is strong enough to reject "just use the exact tuner tuple and claim 1.2x" for these shapes.

## 2. Git state

Baseline for the new A/B:

```text
638008157153cce7d414c4ec3b58a3dd80297ca2 Add RPA v3 tuned wiring benchmark evidence
```

Experimental candidate for the new A/B:

```text
fd81253017662792d3a3d9595880f9d23461572a Mirror RPA v3 tuner block-size mapping
```

Final restored HEAD while writing this report:

```text
cae90ede4e3af0471fc14853b114329655775408 Restore RPA v3 default compute chunk mapping
```

Relevant commits:

```text
691fcc89 Add v6e RPA v3 e5m2 tuned block sizes
514dedf3 Wire RPA v3 tuned block sizes into non-HD64 path
63800815 Add RPA v3 tuned wiring benchmark evidence
fd812530 Mirror RPA v3 tuner block-size mapping
cae90ede Restore RPA v3 default compute chunk mapping
```

Net source-code diff versus `origin/rpa-tuning` after restoring the kernel mapping is verification-only:

```text
benchmarks/rpa_v3_fp8_e5m2_verify.py | 103 ++++++++++++++++++++++++++++-------
```

The original table-only candidate `691fcc89` did not change benchmark, test, CI, logging, dtype mapping, or lookup code. Later verification commits did change benchmark/test code and must be treated as verification artifacts, not primary proof by themselves.

## 3. Adversarial review findings

Independent agent checks converged on these points:

- The original table commit really added `TPU v6e` `q_bfloat16_kv_float8_e5m2` entries.
- Before `514dedf3`, `v3/tuned_block_sizes.py` was effectively dead for the non-HD64 runtime wrapper.
- After `514dedf3`, the wrapper consumed the table, but reconstructed only `bq_sz` and `bkv_sz` from the tuner output while deriving compute chunks from the old default heuristic.
- The historical tuner reconstructed `(bq_sz, bkv_sz, bq_csz, bkv_csz)` as `(bq, bkv_p * page_size, divisor_at_most_32, page_size)`.
- The historical tuner used `distribution = [0, 0, actual_num_seqs]`, so its evidence is mixed-path evidence. It did not separately prove decode and prefill tuning.
- The current wrapper benchmark that found only `~1.01x` was valid for its own narrow decode-only workload, but it was not measuring the same tuple or workload that produced the tuner checkpoint's `~1.2x`.

Reward-hacking surfaces:

| Surface | Status |
|---|---|
| e5m2 silently mapped to e4m3fn | Ruled out in these logs; rows print `kv_dtype=float8_e5m2`. |
| Benchmark ran off TPU | Ruled out; logs show `BACKEND=tpu`, `TPU_INFO=V6E:8`. |
| Async dispatch or compile timing | Mostly ruled out; correctness and first post-correctness call run before timed iterations, warmups run, and timed calls synchronize. |
| Winning cases only | Ruled out for this matrix; all four planned cases are reported for both workloads, including slowdowns. |
| Wrong baseline for the `1.2x` claim | Confirmed. The `1.2x` denominator was tuner-internal, not the production wrapper's old mapping. |

## 4. Benchmark methodology

Harness:

```text
benchmarks/rpa_v3_fp8_e5m2_verify.py
```

The harness is verification-only. It imports the actual RPA v3 wrapper, allocates real `float8_e5m2` KV cache inputs, checks correctness against `ref_ragged_paged_attention`, excludes compile/warmup from timed iterations, and synchronizes each timed call with `jax.block_until_ready()`.

New mode added for this check:

```text
--workload tuner_mixed
```

That mode mirrors the historical tuner workload shape:

```text
actual_num_seqs=35
max_num_tokens=128
max_num_seqs=128
total_num_pages=128
distribution=[0, 0, 35]
```

Commands:

```bash
PYTHONPATH=/tmp/rpa_verify_stubs /tmp/rpa_v3_verify_venv/bin/python benchmarks/rpa_v3_fp8_e5m2_verify.py \
  --repo-path /tmp/tpu-inference-rpa-baseline-63800815 \
  --output-csv benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_tuner_mapping_verification/baseline_63800815_decode_results.csv \
  --iterations 21 --warmups 5 --workload decode

PYTHONPATH=/tmp/rpa_verify_stubs /tmp/rpa_v3_verify_venv/bin/python benchmarks/rpa_v3_fp8_e5m2_verify.py \
  --repo-path /home/enyouki/tpu-inference \
  --output-csv benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_tuner_mapping_verification/candidate_fd812530_decode_results.csv \
  --iterations 21 --warmups 5 --workload decode

PYTHONPATH=/tmp/rpa_verify_stubs /tmp/rpa_v3_verify_venv/bin/python benchmarks/rpa_v3_fp8_e5m2_verify.py \
  --repo-path /tmp/tpu-inference-rpa-baseline-63800815 \
  --output-csv benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_tuner_mapping_verification/baseline_63800815_tuner_mixed_results.csv \
  --iterations 21 --warmups 5 --workload tuner_mixed

PYTHONPATH=/tmp/rpa_verify_stubs /tmp/rpa_v3_verify_venv/bin/python benchmarks/rpa_v3_fp8_e5m2_verify.py \
  --repo-path /home/enyouki/tpu-inference \
  --output-csv benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_tuner_mapping_verification/candidate_fd812530_tuner_mixed_results.csv \
  --iterations 21 --warmups 5 --workload tuner_mixed
```

Environment:

```text
backend=tpu
TPU_INFO=V6E:8
device_kinds=['TPU v6 lite', ... x8]
jax=0.9.2
jaxlib=0.9.2
libtpu=0.0.39
tpu-info=0.7.1
tpu-inference=NOT_INSTALLED
```

Correctness:

All baseline and candidate rows reported `correctness=pass` and `updated_kv_cache_equal_ref=true`.

## 5. Benchmark matrix

Speedup is `baseline_median / candidate_median`. Values below `1.0x` mean the exact tuner tuple was slower than the old wrapper mapping.

| workload | case id | q dtype | KV dtype | page | q heads | KV heads | head dim | max_model_len | distribution | baseline block sizes | exact tuner tuple block sizes | baseline median ms | candidate median ms | speedup | percent change | correctness |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---|
| decode | 32q8kv ctx2048 | bfloat16 | float8_e5m2 | 128 | 32 | 8 | 128 | 2048 | `[8, 8, 8]` | `(32,1024,1,1024)` | `(32,1024,32,128)` | 0.248350 | 0.320820 | 0.774110 | -29.18% | pass/pass |
| decode | 32q8kv ctx8192 | bfloat16 | float8_e5m2 | 128 | 32 | 8 | 128 | 8192 | `[8, 8, 8]` | `(32,4096,1,4096)` | `(32,4096,32,128)` | 0.300510 | 0.723940 | 0.415103 | -140.90% | pass/pass |
| decode | 64q8kv ctx2048 | bfloat16 | float8_e5m2 | 128 | 64 | 8 | 128 | 2048 | `[8, 8, 8]` | `(16,2048,1,2048)` | `(16,2048,16,128)` | 0.241150 | 0.327930 | 0.735371 | -35.99% | pass/pass |
| decode | 64q8kv ctx8192 | bfloat16 | float8_e5m2 | 128 | 64 | 8 | 128 | 8192 | `[8, 8, 8]` | `(16,4096,1,4096)` | `(16,4096,16,128)` | 0.304230 | 0.739320 | 0.411500 | -143.01% | pass/pass |
| tuner_mixed | 32q8kv ctx2048 | bfloat16 | float8_e5m2 | 128 | 32 | 8 | 128 | 2048 | `[0, 0, 35]` | `(32,1024,32,512)` | `(32,1024,32,128)` | 0.453560 | 0.587500 | 0.772017 | -29.53% | pass/pass |
| tuner_mixed | 32q8kv ctx8192 | bfloat16 | float8_e5m2 | 128 | 32 | 8 | 128 | 8192 | `[0, 0, 35]` | `(32,4096,32,512)` | `(32,4096,32,128)` | 1.071390 | 1.675250 | 0.639540 | -56.36% | pass/pass |
| tuner_mixed | 64q8kv ctx2048 | bfloat16 | float8_e5m2 | 128 | 64 | 8 | 128 | 2048 | `[0, 0, 35]` | `(16,2048,16,512)` | `(16,2048,16,128)` | 0.443880 | 0.589230 | 0.753322 | -32.75% | pass/pass |
| tuner_mixed | 64q8kv ctx8192 | bfloat16 | float8_e5m2 | 128 | 64 | 8 | 128 | 8192 | `[0, 0, 35]` | `(16,4096,16,512)` | `(16,4096,16,128)` | 1.074760 | 1.682120 | 0.638932 | -56.51% | pass/pass |

Geomeans:

```text
decode exact-tuner-tuple geomean vs old mapping:      0.558417x
tuner_mixed exact-tuner-tuple geomean vs old mapping: 0.698204x
```

Historical tuner checkpoint overlap:

| shape | tuner baseline | tuner winner | latency baseline us | latency winner us | speedup |
|---|---|---|---:|---:|---:|
| 32q8kv ctx2048 | `(16,32)` | `(8,32)` | 625 | 617 | 1.012966 |
| 32q8kv ctx8192 | `(16,32)` | `(32,32)` | 1695 | 1687 | 1.004742 |
| 64q8kv ctx2048 | `(16,32)` | `(16,16)` | 869 | 622 | 1.397106 |
| 64q8kv ctx8192 | `(16,32)` | `(32,16)` | 2640 | 1691 | 1.561206 |

Tuner checkpoint overlap geomean: `1.220632x`.

## 6. Raw evidence

Raw files:

- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_tuner_mapping_verification/env.log`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_tuner_mapping_verification/comparison_summary.csv`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_tuner_mapping_verification/tuner_checkpoint_overlap.csv`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_tuner_mapping_verification/baseline_63800815_decode.log`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_tuner_mapping_verification/candidate_fd812530_decode.log`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_tuner_mapping_verification/baseline_63800815_tuner_mixed.log`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_tuner_mapping_verification/candidate_fd812530_tuner_mixed.log`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_tuner_mapping_verification/*_results.csv`

Evidence snippets:

```text
BACKEND=tpu
TPU_INFO=V6E:8
KV_DTYPE=float8_e5m2
Q_DTYPE=bfloat16
```

Old mapping example:

```text
actual_mixed_block_sizes={'bq_sz': 32, 'bkv_sz': 4096, 'bq_csz': 32, 'bkv_csz': 512}
```

Exact tuner tuple example:

```text
actual_mixed_block_sizes={'bq_sz': 32, 'bkv_sz': 4096, 'bq_csz': 32, 'bkv_csz': 128}
```

## 7. Independent audit

Agent D verdict: **PASS, with caveats**.

The auditor verified:

- Baseline SHA `638008157153cce7d414c4ec3b58a3dd80297ca2`.
- Experimental SHA `fd81253017662792d3a3d9595880f9d23461572a`.
- Current HEAD `cae90ede4e3af0471fc14853b114329655775408` restored the production/default compute-chunk mapping after the experimental tuner-tuple commit.
- Logs show `backend=tpu`, `TPU_INFO=V6E:8`, and `tpu_type=v6e-8`.
- Raw CSV rows use `q_dtype=bfloat16` and `kv_dtype=float8_e5m2`.
- Correctness passed for every baseline and candidate case, with `updated_kv_cache_equal_ref=True`.
- Old/current mapping used larger/default chunks such as mixed `bkv_csz=512`; experimental `fd812530` used the tuner-style `bkv_csz=128`.
- Recomputed geomeans match the report: `0.558417x` for decode and `0.698204x` for tuner-mixed.

Remaining caveat: this is synthetic wrapper evidence, not a full production serving benchmark.

## 8. Final conclusion

The likely explanation for the initial `1.2x` is not that the user imagined it and not that the table entries are obviously fake. It is that the `1.2x` was a tuner-internal result against a tuner-internal baseline. The tuner did not compare against the current production wrapper's old compute-chunk mapping.

The exact tuner tuple should not be shipped as the production mapping based on this evidence. For the representative decode-only and tuner-like mixed wrapper workloads measured here, it slowed down every case.

What needs to be done next:

1. Retune against the actual production wrapper and production distribution, not just the external tuner harness.
2. Sweep or store the full four-value tuple `(bq_sz, bkv_sz, bq_csz, bkv_csz)`, not only `(bkv_p, bq)`.
3. Tune decode, prefill, and mixed paths separately, because the existing checkpoint is mixed-path evidence.
4. Compare against the current production-selected block tuple, not the tuner sweep's first successful baseline.
5. Only claim speedup after correctness passes and a predeclared production matrix shows median speedup and geomean speedup against that real baseline.
