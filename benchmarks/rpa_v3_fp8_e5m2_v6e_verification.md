# RPA v3 fp8_e5m2 TPU v6e verification

## 1. Executive summary

Final conclusion label: **INCONCLUSIVE_INVALID_BENCHMARK**

The latest commit **does add TPU v6e RPA v3 `float8_e5m2` tuning-table data** in `tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py`. The new table keys match the lookup helper format for `TPU v6e`, page sizes 128 and 256, `q_bfloat16_kv_float8_e5m2`, head shape, rounded `max_model_len`, and `sliding_window`.

Valid benchmarks did **not** validate a speedup from that tuning commit. The verification harness ran real TPU v6e RPA v3 fp8_e5m2 work, but the measured non-HD64 RPA v3 wrapper did not consume `v3/tuned_block_sizes.py`. In both baseline and candidate logs, the actual measured kernel block source is `fallback_default_get_default_block_sizes`.

No valid overall tuned-table geomean speedup is reported. The representative fallback-path timing geomean was `0.990330x`, but that number is not valid evidence about the new tuned table because the measured actual block sizes were identical fallback defaults in both revisions.

Biggest caveats:

- Production shape was not inferred from config/history, so the benchmark matrix is representative decode-heavy coverage, not production-proven coverage.
- The fresh harness uses lightweight import stubs for vLLM/Torch/TorchAX to import the kernel in this repository environment. That is acceptable for kernel-only smoke timing, but it is not a full serving-stack benchmark.
- Existing upstream RPA v3 correctness tests exist, but existing upstream benchmark infrastructure in this repo did not provide a trusted fp8_e5m2 TPU v6e performance benchmark for this claim.
- The most important caveat is stronger than normal benchmark noise: static and runtime evidence show the modified table is not wired into the measured non-HD64 RPA v3 kernel path.

## 2. Git state

Baseline SHA: `4958977a929d4ee639db269056b1299aaf37984f`

Candidate SHA: `691fcc89aa52818598b30d76c6de969427c29a8e`

Latest commit:

```text
commit 691fcc89aa52818598b30d76c6de969427c29a8e
Author: Codex <codex@openai.com>
Date:   Sun Jun 21 14:06:01 2026 +0000

    Add v6e RPA v3 e5m2 tuned block sizes

 .../ragged_paged_attention/v3/tuned_block_sizes.py | 2120 +++++++++++++-------
 1 file changed, 1385 insertions(+), 735 deletions(-)
```

Files changed by candidate:

```text
M tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py
```

Candidate benchmark/test/CI/logging/dtype-mapping changes: none found in `HEAD~1..HEAD`. The candidate commit only changed tuning data. This means no candidate-added benchmark/test code was used as primary evidence, and there was no direct benchmark-script reward-hacking surface in the latest commit.

Raw git evidence:

- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/git_state_and_diff.log`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/tuning_table_entry_summary.txt`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/static_callsite_and_dtype_search.log`

Tuning-table summary from the raw artifact:

```text
HEAD~1 TPU v6e page=128 q_bfloat16_kv_float8_e4m3fn head_keys=38 entries=266
HEAD~1 TPU v6e page=256 q_bfloat16_kv_float8_e4m3fn head_keys=38 entries=228
HEAD   TPU v6e page=128 q_bfloat16_kv_float8_e5m2   head_keys=38 entries=266
HEAD   TPU v6e page=256 q_bfloat16_kv_float8_e5m2   head_keys=38 entries=228
```

## 3. Adversarial review findings

Agent A verdict: **runtime intent not validated**.

Evidence:

- The latest commit adds `q_bfloat16_kv_float8_e5m2` entries for `TPU v6e`.
- The key format is consistent with `get_lookup_keys`: device name from `get_device_name()`, power-of-two page size, `jnp.dtype(...).name` dtype names, q/kv/head dimensions, rounded `max_model_len`, and `sliding_window`.
- No benchmark, test, CI, logging, dtype mapping, or lookup helper was changed by the candidate.
- `tpu_inference/utils.py` maps `"fp8_e5m2"` to `jnp.float8_e5m2`; no evidence was found that e5m2 is silently mapped to e4m3fn.
- The non-HD64 RPA v3 kernel wrapper does not import or call `v3/tuned_block_sizes.py`. When block sizes are omitted, it calls `get_default_block_sizes()` in `kernel.py`.
- The HD64 path imports `tuned_block_sizes_hd64.py`, not the modified file.

Reward-hacking surfaces:

| Surface | Status |
|---|---|
| Benchmark code modified by candidate | Ruled out. Candidate changed only `v3/tuned_block_sizes.py`. |
| Benchmark uses e4m3fn instead of e5m2 | Ruled out for fresh harness logs; they print `KV_DTYPE=float8_e5m2` and CSV rows have `kv_dtype=float8_e5m2`. |
| Dtype alias maps e5m2 to e4m3fn | Ruled out in `tpu_inference/utils.py` for `fp8_e5m2`; generic `fp8` remains e4m3fn in one utility, but the harness and target key use explicit e5m2. |
| Tests skipped or xfailed | Existing correctness subset passed: `8 passed, 53 deselected`. |
| Fallback heuristic used instead of tuned table | **Confirmed unresolved for speed claim.** Runtime logs show `actual_kernel_block_source=fallback_default_get_default_block_sizes` for every measured case. |
| Benchmark measures JIT compile or async dispatch | Mostly ruled out for the fresh harness: correctness/first call happen before timed loops, warmups run, and each timed call uses `jax.block_until_ready`. |
| Benchmark filters to winning cases | Ruled out for the fresh harness: all four predeclared cases are reported, including slowdowns. |
| Wrong baseline | Ruled out: logs show baseline `4958977a929d4ee639db269056b1299aaf37984f` and candidate `691fcc89aa52818598b30d76c6de969427c29a8e`. |

## 4. Benchmark methodology

### Existing infrastructure review

Agent B found existing RPA v3 correctness tests in `HEAD~1`:

- `tests/kernels/ragged_paged_attention_kernel_v3_test.py`
- `tests/kernels/ragged_paged_attention_kernel_v3_hd64_test.py`

The non-HD64 test includes `jnp.float8_e5m2` cases and skips only when the device is older than TPU v5. It is a correctness test, not a performance benchmark.

Benchmark-like Buildkite files existed in `HEAD~1`, but the sampled generic RPA v3 microbenchmark YAML records correctness/performance as `unverified` metadata instead of running a measurement. The `scripts/rpa_v3_e5m2_resume_sweep.sh` helper also existed in `HEAD~1` and forces `RPA_KV_DTYPE=float8_e5m2`, but it depends on external tuner checkpoints and tools under `/tmp`/GCS and is not a self-contained benchmark path in this repo.

Because existing upstream infrastructure was insufficient for a trusted fp8_e5m2 TPU v6e RPA v3 performance result, I wrote a narrow verification-only harness:

- `benchmarks/rpa_v3_fp8_e5m2_verify.py`

Agent B reviewed the harness and rejected it as a valid tuned-table benchmark because the actual measured non-HD64 RPA v3 wrapper does not call `v3/tuned_block_sizes.py`. Agent B accepted it only as a transparent smoke/verification harness for the actual wrapper/fallback path.

### Harness behavior

The harness:

- Imports the actual `tpu_inference.kernels.ragged_paged_attention.v3.kernel` path.
- Allocates real `jnp.float8_e5m2` K/V and KV cache inputs.
- Requires `jax.default_backend() == "tpu"` and `tpu_info` beginning with `V6E:`.
- Prints device/backend/dtypes, selected lookup key, helper table result, and actual measured block-size source.
- Runs correctness for the same shape against `ref_ragged_paged_attention` before timing.
- Checks updated KV cache equality against the reference.
- Separates first-call/correctness from timed loops.
- Runs warmups before timed iterations.
- Calls `jax.block_until_ready()` for correctness, warmups, and timed calls.
- Reports all predeclared cases and raw timing arrays.

The harness also calls `get_tuned_block_sizes()` for reporting. That call proves whether the helper table has a candidate entry, but it does not prove the measured wrapper used that entry. The runtime column `actual_kernel_block_source` is therefore the controlling evidence for the speedup claim.

### Commands

Baseline environment:

```bash
/tmp/rpa_v3_verify_venv/bin/python - <<'PY' > benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/baseline_4958977_env.log
import importlib.metadata as md
import subprocess
import jax
print('git_rev=', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
print('backend=', jax.default_backend())
print('devices=', jax.devices())
print('device_kinds=', [getattr(d, 'device_kind', None) for d in jax.devices()])
for pkg in ['jax', 'jaxlib', 'libtpu', 'tpu-info', 'tpu-inference']:
    try:
        version = md.version(pkg)
    except md.PackageNotFoundError:
        version = 'NOT_INSTALLED'
    print(f'{pkg}={version}')
from tpu_info import device as tpu_device
print('tpu_info_chips=', tpu_device.get_local_chips())
PY
```

Baseline benchmark:

```bash
/tmp/rpa_v3_verify_venv/bin/python benchmarks/rpa_v3_fp8_e5m2_verify.py \
  --repo-path /home/enyouki/tpu-inference \
  --output-csv benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/baseline_4958977_results.csv \
  --iterations 9 \
  --warmups 3
```

Candidate benchmark:

```bash
/tmp/rpa_v3_verify_venv/bin/python benchmarks/rpa_v3_fp8_e5m2_verify.py \
  --repo-path /home/enyouki/tpu-inference \
  --output-csv benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/candidate_691fcc8_results.csv \
  --iterations 9 \
  --warmups 3
```

Existing correctness subset:

```bash
PYTHONPATH=/tmp/rpa_verify_stubs:$PYTHONPATH \
  /tmp/rpa_v3_verify_venv/bin/python -m pytest \
  tests/kernels/ragged_paged_attention_kernel_v3_test.py \
  -k 'quantized_kv_cache or quantized_attention' -q -s
```

### Device and versions

From the environment and benchmark logs:

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

The repo was imported from `/home/enyouki/tpu-inference` via `PYTHONPATH`; it was not installed as a package.

### Correctness methodology

Correctness was checked two ways:

- Existing upstream RPA v3 fp8 correctness subset: `8 passed, 53 deselected, 8 warnings in 80.88s`.
- Per benchmark case and per revision, the harness compared `ragged_paged_attention()` output and updated KV cache against `ref_ragged_paged_attention()` for the same dtype/path/shape. All four baseline and four candidate cases reported `correctness=pass` and `updated_kv_cache_equal_ref=true`.

## 5. Benchmark matrix

These cases are representative decode-heavy shapes for TPU v6e, not production-proven shapes. The main claim dtype is `KV dtype=float8_e5m2`; query dtype is `bfloat16`; page size is 128; sliding window is `None`.

| case id | device | q dtype | KV dtype | page size | q heads | KV heads | head dim | max_model_len/context | sliding_window | baseline selected block size | candidate selected block size | baseline median ms | candidate median ms | speedup | percent change | correctness status | notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---|---|
| decode_gqa_32q8kv_ctx2048_p128 | V6E:8 | bfloat16 | float8_e5m2 | 128 | 32 | 8 | 128 | 2048 | None | actual fallback `bq=1,bkv=2048,bq_c=1,bkv_c=2048`; table miss `(16,32)` | actual fallback `bq=1,bkv=2048,bq_c=1,bkv_c=2048`; table hit `(8,32)` | 0.229689991 | 0.241479953 | 0.951176 | -5.133% | baseline pass; candidate pass | candidate table entry exists, but measured kernel used identical fallback defaults |
| decode_gqa_32q8kv_ctx8192_p128 | V6E:8 | bfloat16 | float8_e5m2 | 128 | 32 | 8 | 128 | 8192 | None | actual fallback `bq=1,bkv=8192,bq_c=1,bkv_c=8192`; table miss `(16,32)` | actual fallback `bq=1,bkv=8192,bq_c=1,bkv_c=8192`; table hit `(32,32)` | 0.317390077 | 0.307430048 | 1.032398 | 3.138% | baseline pass; candidate pass | candidate table entry exists, but measured kernel used identical fallback defaults |
| decode_gqa_64q8kv_ctx2048_p128 | V6E:8 | bfloat16 | float8_e5m2 | 128 | 64 | 8 | 128 | 2048 | None | actual fallback `bq=1,bkv=2048,bq_c=1,bkv_c=2048`; table miss `(16,32)` | actual fallback `bq=1,bkv=2048,bq_c=1,bkv_c=2048`; table hit `(16,16)` | 0.226849923 | 0.234709936 | 0.966512 | -3.465% | baseline pass; candidate pass | candidate table entry exists, but measured kernel used identical fallback defaults |
| decode_gqa_64q8kv_ctx8192_p128 | V6E:8 | bfloat16 | float8_e5m2 | 128 | 64 | 8 | 128 | 8192 | None | actual fallback `bq=1,bkv=8192,bq_c=1,bkv_c=8192`; table miss `(16,32)` | actual fallback `bq=1,bkv=8192,bq_c=1,bkv_c=8192`; table hit `(32,16)` | 0.313339988 | 0.309180003 | 1.013455 | 1.328% | baseline pass; candidate pass | candidate table entry exists, but measured kernel used identical fallback defaults |

Fallback-path geomean across the four correctness-passing representative cases: `0.990330x`.

This geomean is deliberately not reported as a valid tuned-table speedup because the actual measured block source was fallback in both revisions.

## 6. Raw evidence

Raw artifacts:

- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/baseline_4958977_env.log`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/baseline_4958977_benchmark.log`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/baseline_4958977_results.csv`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/candidate_691fcc8_env.log`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/candidate_691fcc8_benchmark.log`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/candidate_691fcc8_results.csv`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/combined_results.csv`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/combined_summary.txt`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/candidate_existing_correctness_pytest.log`
- `benchmarks/artifacts/rpa_v3_fp8_e5m2_v6e_verification/baseline_existing_infra.log`

Representative candidate stdout snippets:

```text
GIT_SHA=691fcc89aa52818598b30d76c6de969427c29a8e
BACKEND=tpu
TPU_INFO=V6E:8
KV_DTYPE=float8_e5m2
Q_DTYPE=bfloat16
```

```text
lookup_key=('TPU v6e', 128, 'q_bfloat16_kv_float8_e5m2',
            'q_head-32_kv_head-8_head-128',
            'max_model_len-2048-sw-None')
tuned_table_has_entry=true
tuned_table_block_bkv_pages_bq=(8, 32)
actual_kernel_block_source=fallback_default_get_default_block_sizes
actual_decode_block_sizes={'bq_sz': 1, 'bkv_sz': 2048, 'bq_csz': 1, 'bkv_csz': 2048}
```

Representative baseline stdout snippets:

```text
GIT_SHA=4958977a929d4ee639db269056b1299aaf37984f
BACKEND=tpu
TPU_INFO=V6E:8
KV_DTYPE=float8_e5m2
Q_DTYPE=bfloat16
tuned_table_has_entry=false
actual_kernel_block_source=fallback_default_get_default_block_sizes
```

Static call-site evidence:

```text
tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py imports tuned_block_sizes_hd64.py
tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes.py defines get_tuned_block_sizes
No production RPA v3 non-HD64 caller of v3/tuned_block_sizes.py was found.
```

The non-HD64 wrapper in `kernel.py` fills missing block sizes through `_prepare_block_sizes()`, which calls `get_default_block_sizes()` when `d_block_sizes`, `p_block_sizes`, or `m_block_sizes` are `None`.

## 7. Independent audit

Agent D verdict: **Fail for validating the latest tuning commit.**

Agent D confirmed:

- The benchmark ran on TPU v6e: logs show TPU backend, eight `TPU v6 lite` devices, `TPU_INFO=V6E:8`, and `tpu_info` reporting V6E.
- The benchmark used `float8_e5m2` KV inputs/cache, not e4m3fn or bf16. The harness hard-codes `jnp.float8_e5m2`, and logs/CSVs report `KV_DTYPE=float8_e5m2`.
- The baseline and candidate SHAs are correct.
- Timing methodology is reasonable for the smoke benchmark: correctness/first call precede timing, three warmups run, and each timed iteration synchronizes with `jax.block_until_ready`.
- The matrix coverage is complete: four planned cases and four result rows in each CSV, including slowdowns.
- A recomputed sample matches the combined CSV: `0.229689991 / 0.241479953 = 0.951176x`, a `5.13%` slowdown for `decode_gqa_32q8kv_ctx2048_p128`.

Agent D's invalidating finding:

> The harness exercised the RPA v3 kernel and separately exercised the tuning lookup, but the measured kernel calls did not pass tuned block sizes. The non-HD64 kernel falls back to `get_default_block_sizes` when block sizes are `None`, and every row logged `actual_kernel_block_source=fallback_default_get_default_block_sizes` despite candidate lookup hits.

Missing evidence needed for validation:

- A benchmark where the actual RPA v3 kernel uses the new e5m2 tuned block sizes, either by wiring non-HD64 `kernel.py` to `tuned_block_sizes.py` or by passing explicit tuned `d_block_sizes`/`m_block_sizes`.
- Logs showing the actual measured block source is tuned and actual block sizes differ from fallback defaults.

## 8. Final conclusion

**INCONCLUSIVE_INVALID_BENCHMARK**

The latest commit did add RPA v3 TPU v6e fp8_e5m2 tuning-table entries, but the available benchmark evidence cannot validate that those entries speed up production RPA v3 execution. The measured non-HD64 kernel path used fallback default block sizes in both revisions. The correct conclusion is not "optimized" or "speedup"; it is that the candidate table data exists but was not shown to affect the active measured path.
