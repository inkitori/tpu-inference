#!/usr/bin/env python3
"""Verification-only RPA v3 fp8_e5m2 TPU benchmark harness."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import subprocess
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path


def _install_lightweight_stubs() -> None:
    """Provide the tiny vLLM/Torch surface needed by kernel-only imports."""

    if "vllm" not in sys.modules:
        vllm = types.ModuleType("vllm")
        sys.modules["vllm"] = vllm

    logger_mod = types.ModuleType("vllm.logger")

    class _VllmLogger(logging.Logger):
        _once_messages: set[tuple[object, ...]] = set()

        def info_once(self, msg, *args, **kwargs):
            key = ("info", msg, args)
            if key not in self._once_messages:
                self._once_messages.add(key)
                self.info(msg, *args, **kwargs)

        def warning_once(self, msg, *args, **kwargs):
            key = ("warning", msg, args)
            if key not in self._once_messages:
                self._once_messages.add(key)
                self.warning(msg, *args, **kwargs)

    def init_logger(name):
        logging.setLoggerClass(_VllmLogger)
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(name)
        if not isinstance(logger, _VllmLogger):
            logger.__class__ = _VllmLogger
        return logger

    logger_mod._VllmLogger = _VllmLogger
    logger_mod.init_logger = init_logger
    sys.modules["vllm.logger"] = logger_mod
    sys.modules["vllm"].logger = logger_mod

    envs_mod = types.ModuleType("vllm.envs")
    envs_mod.VLLM_TPU_USING_PATHWAYS = False
    envs_mod.VLLM_TPU_BUCKET_PADDING_GAP = 128
    sys.modules["vllm.envs"] = envs_mod
    sys.modules["vllm"].envs = envs_mod

    utils_mod = types.ModuleType("vllm.utils")
    utils_mod.cdiv = lambda a, b: (a + b - 1) // b
    sys.modules["vllm.utils"] = utils_mod
    sys.modules["vllm"].utils = utils_mod

    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")

        class _TorchDType:
            pass

        class _Tensor:
            pass

        torch_mod.dtype = _TorchDType
        torch_mod.Tensor = _Tensor
        torch_mod.bfloat16 = _TorchDType()
        torch_mod.uint8 = _TorchDType()
        torch_mod.float8_e4m3fn = _TorchDType()
        torch_mod.float8_e4m3fnuz = _TorchDType()
        torch_mod.float8_e5m2 = _TorchDType()
        torch_mod.float8_e5m2fnuz = _TorchDType()
        sys.modules["torch"] = torch_mod

    torchax_mod = types.ModuleType("torchax")
    torchax_ops_mod = types.ModuleType("torchax.ops")
    mappings_mod = types.ModuleType("torchax.ops.mappings")
    mappings_mod.j2t_dtype = lambda dtype: dtype
    mappings_mod.t2j_dtype = lambda dtype: dtype

    def _unsupported_t2j(*_args, **_kwargs):
        raise RuntimeError("torchax stub cannot convert tensors")

    mappings_mod.t2j = _unsupported_t2j
    sys.modules.setdefault("torchax", torchax_mod)
    sys.modules.setdefault("torchax.ops", torchax_ops_mod)
    sys.modules["torchax.ops.mappings"] = mappings_mod


@dataclass(frozen=True)
class Case:
    case_id: str
    page_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    max_model_len: int
    sliding_window: int | None
    num_seqs: int = 8


def _matrix() -> list[Case]:
    return [
        Case("decode_gqa_32q8kv_ctx2048_p128", 128, 32, 8, 128, 2048, None),
        Case("decode_gqa_32q8kv_ctx8192_p128", 128, 32, 8, 128, 8192, None),
        Case("decode_gqa_64q8kv_ctx2048_p128", 128, 64, 8, 128, 2048, None),
        Case("decode_gqa_64q8kv_ctx8192_p128", 128, 64, 8, 128, 8192, None),
    ]


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_to(x: int, a: int) -> int:
    return _cdiv(x, a) * a


def _decode_heavy_lens(max_num_tokens: int, max_model_len: int,
                       actual_num_seqs: int) -> tuple[list[int], list[int]]:
    assert max_num_tokens >= actual_num_seqs
    if actual_num_seqs == 1:
        cu_q_lens = [0, max_num_tokens]
    else:
        cu_q_lens = list(range(actual_num_seqs))
        prefill_q_len = max_num_tokens - (actual_num_seqs - 1)
        cu_q_lens.append(cu_q_lens[-1] + prefill_q_len)

    kv_lens = []
    for seq_idx in range(actual_num_seqs):
        q_len = cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx]
        kv_lens.append(max_model_len if q_len == 1 else q_len)
    return cu_q_lens, kv_lens


def _block_until_ready(tree):
    import jax

    jax.block_until_ready(tree)
    return tree


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("jax", "jaxlib", "libtpu", "tpu-info", "tpu-inference"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def _git_sha(repo_path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"], text=True).strip()


def _make_inputs(case: Case, *, copies: int, seed: int, args):
    import jax.numpy as jnp
    import numpy as np
    from tpu_inference.kernels.ragged_paged_attention.v3.util import (
        align_to, cdiv, get_dtype_packing)

    rng = np.random.default_rng(seed)
    q_dtype = jnp.bfloat16
    kv_dtype = jnp.float8_e5m2
    pages_per_seq = cdiv(case.max_model_len, case.page_size)
    if args.workload == "decode":
        actual_num_seqs = case.num_seqs
        max_num_tokens = align_to(actual_num_seqs, 128)
        max_num_seqs = align_to(actual_num_seqs, 8)
        total_pages = actual_num_seqs * pages_per_seq
        cu_q_lens_np = np.arange(actual_num_seqs + 1, dtype=np.int32)
        kv_lens_np = np.full((actual_num_seqs, ),
                             case.max_model_len,
                             dtype=np.int32)
        page_indices_np = np.arange(total_pages, dtype=np.int32).reshape(
            actual_num_seqs, pages_per_seq)
        page_indices_np = np.pad(
            page_indices_np,
            ((0, max_num_seqs - actual_num_seqs), (0, 0)),
        ).reshape(-1)
        distribution_values = [actual_num_seqs, actual_num_seqs, actual_num_seqs]
    elif args.workload == "tuner_mixed":
        actual_num_seqs = args.tuner_actual_num_seqs
        max_num_tokens = args.tuner_max_num_tokens
        max_num_seqs = args.tuner_max_num_seqs
        total_pages = args.tuner_total_num_pages
        cu_q_lens, kv_lens = _decode_heavy_lens(max_num_tokens,
                                                case.max_model_len,
                                                actual_num_seqs)
        cu_q_lens_np = np.asarray(cu_q_lens, dtype=np.int32)
        kv_lens_np = np.asarray(kv_lens, dtype=np.int32)
        page_indices_np = (np.arange(max_num_seqs * pages_per_seq,
                                     dtype=np.int32) % total_pages)
        distribution_values = [0, 0, actual_num_seqs]
    else:
        raise ValueError(f"Unsupported workload {args.workload!r}")

    if max_num_tokens < int(cu_q_lens_np[-1]):
        raise ValueError(f"{max_num_tokens=} is too small for {cu_q_lens_np[-1]=}")
    if max_num_seqs < actual_num_seqs:
        raise ValueError(f"{max_num_seqs=} is too small for {actual_num_seqs=}")
    if total_pages < 1:
        raise ValueError(f"{total_pages=} must be positive")

    padded_head_dim = align_to(case.head_dim, 128)
    kv_packing = get_dtype_packing(kv_dtype)
    num_kv_heads_x2 = align_to(case.kv_heads * 2, kv_packing)
    kv_cache_shape = (
        total_pages,
        case.page_size,
        num_kv_heads_x2 // kv_packing,
        kv_packing,
        padded_head_dim,
    )

    def random_array(shape, dtype):
        arr = rng.random(shape, dtype=np.float32)
        return jnp.asarray(arr).astype(dtype)

    q_list = [
        random_array((max_num_tokens, case.q_heads, case.head_dim), q_dtype)
        for _ in range(copies)
    ]
    k_list = [
        random_array((max_num_tokens, case.kv_heads, case.head_dim), kv_dtype)
        for _ in range(copies)
    ]
    v_list = [
        random_array((max_num_tokens, case.kv_heads, case.head_dim), kv_dtype)
        for _ in range(copies)
    ]
    correctness_kv_cache = random_array(kv_cache_shape, kv_dtype)
    timing_kv_cache = random_array(kv_cache_shape, kv_dtype)
    kv_lens = jnp.pad(
        jnp.asarray(kv_lens_np, dtype=jnp.int32),
        (0, max_num_seqs - actual_num_seqs),
    )
    cu_q_lens = jnp.pad(
        jnp.asarray(cu_q_lens_np, dtype=jnp.int32),
        (0, max_num_seqs + 1 - (actual_num_seqs + 1)),
    )
    page_indices = jnp.asarray(page_indices_np, dtype=jnp.int32)
    distribution = jnp.array(distribution_values, dtype=jnp.int32)
    static = {
        "workload": args.workload,
        "actual_num_seqs": actual_num_seqs,
        "actual_num_tokens": int(cu_q_lens_np[actual_num_seqs]),
        "max_num_tokens": max_num_tokens,
        "max_num_seqs": max_num_seqs,
        "pages_per_seq": pages_per_seq,
        "total_pages": total_pages,
        "kv_cache_shape": kv_cache_shape,
        "distribution": repr(distribution_values),
        "q_dtype": str(jnp.dtype(q_dtype)),
        "kv_dtype": str(jnp.dtype(kv_dtype)),
    }
    return (q_list, k_list, v_list, correctness_kv_cache, timing_kv_cache,
            kv_lens, page_indices, cu_q_lens, distribution, static)


def _table_hit(table, key) -> bool:
    device, page_size, dtypes, head_dims, extra = key
    try:
        table[device][page_size][dtypes][head_dims][extra]
    except KeyError:
        return False
    return True


def _run_case(case: Case, args, modules) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax._src import test_util as jtu

    rpa_kernel, tuned = modules
    copies = 1 + args.warmups + args.iterations + 1
    (q_list, k_list, v_list, correctness_kv_cache, timing_kv_cache, kv_lens,
     page_indices, cu_q_lens, distribution, static) = _make_inputs(
         case, copies=copies, seed=args.seed, args=args)

    kwargs = {
        "use_causal_mask": True,
        "sliding_window": case.sliding_window,
        "sm_scale": case.head_dim**-0.5,
    }

    lookup_key = tuned.get_lookup_keys(
        case.page_size,
        jnp.bfloat16,
        jnp.float8_e5m2,
        case.q_heads,
        case.kv_heads,
        case.head_dim,
        case.max_model_len,
        case.sliding_window,
    )
    has_table_entry = _table_hit(tuned.TUNED_BLOCK_SIZES, lookup_key)
    tuned_block = tuned.get_tuned_block_sizes(
        jnp.bfloat16,
        jnp.float8_e5m2,
        case.q_heads,
        case.kv_heads,
        case.head_dim,
        case.page_size,
        static["max_num_tokens"],
        static["pages_per_seq"],
        case.sliding_window,
    )
    decode_default = rpa_kernel.get_default_block_sizes(
        jnp.bfloat16,
        jnp.float8_e5m2,
        case.q_heads,
        case.kv_heads,
        case.head_dim,
        case.page_size,
        static["max_num_tokens"],
        static["max_num_seqs"],
        static["pages_per_seq"],
        case=rpa_kernel.RpaCase.DECODE,
    )
    mixed_default = rpa_kernel.get_default_block_sizes(
        jnp.bfloat16,
        jnp.float8_e5m2,
        case.q_heads,
        case.kv_heads,
        case.head_dim,
        case.page_size,
        static["max_num_tokens"],
        static["max_num_seqs"],
        static["pages_per_seq"],
        case=rpa_kernel.RpaCase.MIXED,
    )
    decode_selected = rpa_kernel.get_selected_block_sizes(
        jnp.bfloat16,
        jnp.float8_e5m2,
        case.q_heads,
        case.kv_heads,
        case.head_dim,
        case.page_size,
        static["max_num_tokens"],
        static["max_num_seqs"],
        static["pages_per_seq"],
        case=rpa_kernel.RpaCase.DECODE,
        sliding_window=case.sliding_window,
    )
    mixed_selected = rpa_kernel.get_selected_block_sizes(
        jnp.bfloat16,
        jnp.float8_e5m2,
        case.q_heads,
        case.kv_heads,
        case.head_dim,
        case.page_size,
        static["max_num_tokens"],
        static["max_num_seqs"],
        static["pages_per_seq"],
        case=rpa_kernel.RpaCase.MIXED,
        sliding_window=case.sliding_window,
    )
    selected_source = ("tuned_table"
                       if has_table_entry else
                       "fallback_default_get_default_block_sizes")

    correctness = "not_run"
    max_abs_err = None
    max_rel_err = None
    kv_cache_equal = None
    correctness_wall_ms = None
    try:
        correctness_start = time.perf_counter()
        expected, expected_kv_cache = rpa_kernel.ref_ragged_paged_attention(
            q_list[0], k_list[0], v_list[0], correctness_kv_cache, kv_lens,
            page_indices, cu_q_lens, distribution, **kwargs)
        output, updated_kv_cache = rpa_kernel.ragged_paged_attention(
            q_list[0], k_list[0], v_list[0], correctness_kv_cache, kv_lens,
            page_indices, cu_q_lens, distribution, **kwargs)
        expected = _block_until_ready(expected[:static["actual_num_tokens"]])
        output = _block_until_ready(output[:static["actual_num_tokens"]])
        kv_cache_equal = bool(
            np.asarray(
                _block_until_ready(jnp.all(updated_kv_cache == expected_kv_cache))))
        expected_np = np.asarray(expected, dtype=np.float32)
        output_np = np.asarray(output, dtype=np.float32)
        abs_err = np.abs(output_np - expected_np)
        rel_err = abs_err / np.maximum(np.abs(expected_np), 1e-6)
        max_abs_err = float(np.max(abs_err))
        max_rel_err = float(np.max(rel_err))
        np.testing.assert_allclose(output_np, expected_np, atol=args.atol,
                                   rtol=args.rtol)
        if not kv_cache_equal:
            raise AssertionError("updated KV cache did not match reference")
        correctness_wall_ms = (time.perf_counter() - correctness_start) * 1000.0
        correctness = "pass"
    except Exception as exc:  # pylint: disable=broad-except
        correctness_wall_ms = (time.perf_counter() - correctness_start) * 1000.0
        correctness = f"fail:{type(exc).__name__}:{exc}"

    times_ms: list[float] = []
    first_post_correctness_start = time.perf_counter()
    output, kv_state = rpa_kernel.ragged_paged_attention(
        q_list[1], k_list[1], v_list[1], timing_kv_cache, kv_lens, page_indices,
        cu_q_lens, distribution, **kwargs)
    _block_until_ready((output, kv_state))
    first_post_correctness_ms = (
        time.perf_counter() - first_post_correctness_start) * 1000.0

    cursor = 2
    for _ in range(args.warmups):
        output, kv_state = rpa_kernel.ragged_paged_attention(
            q_list[cursor], k_list[cursor], v_list[cursor], kv_state, kv_lens,
            page_indices, cu_q_lens, distribution, **kwargs)
        _block_until_ready((output, kv_state))
        cursor += 1

    for _ in range(args.iterations):
        start = time.perf_counter()
        output, kv_state = rpa_kernel.ragged_paged_attention(
            q_list[cursor], k_list[cursor], v_list[cursor], kv_state, kv_lens,
            page_indices, cu_q_lens, distribution, **kwargs)
        _block_until_ready((output, kv_state))
        times_ms.append((time.perf_counter() - start) * 1000.0)
        cursor += 1

    times_np = np.asarray(times_ms, dtype=np.float64)
    return {
        **asdict(case),
        **static,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "device_kinds": [getattr(d, "device_kind", "") for d in jax.devices()],
        "is_tpu_v5_or_newer": bool(jtu.is_device_tpu_at_least(version=5)),
        "lookup_key": repr(lookup_key),
        "tuned_table_has_entry": has_table_entry,
        "tuned_table_block_bkv_pages_bq": repr(tuned_block),
        "actual_kernel_block_source": selected_source,
        "actual_decode_block_sizes": repr(decode_selected),
        "actual_mixed_block_sizes": repr(mixed_selected),
        "default_decode_block_sizes": repr(decode_default),
        "default_mixed_block_sizes": repr(mixed_default),
        "correctness": correctness,
        "correctness_wall_ms": correctness_wall_ms,
        "updated_kv_cache_equal_ref": kv_cache_equal,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "first_post_correctness_ms": first_post_correctness_ms,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "raw_times_ms": json.dumps(times_ms),
        "median_ms": float(np.median(times_np)),
        "mean_ms": float(np.mean(times_np)),
        "min_ms": float(np.min(times_np)),
        "max_ms": float(np.max(times_np)),
        "std_ms": float(np.std(times_np, ddof=1)) if len(times_np) > 1 else 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", type=Path, default=Path.cwd())
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=0.2)
    parser.add_argument("--rtol", type=float, default=0.2)
    parser.add_argument("--workload",
                        choices=("decode", "tuner_mixed"),
                        default="decode")
    parser.add_argument("--tuner-actual-num-seqs", type=int, default=35)
    parser.add_argument("--tuner-max-num-tokens", type=int, default=128)
    parser.add_argument("--tuner-max-num-seqs", type=int, default=128)
    parser.add_argument("--tuner-total-num-pages", type=int, default=128)
    args = parser.parse_args()

    repo_path = args.repo_path.resolve()
    sys.path.insert(0, str(repo_path))
    _install_lightweight_stubs()

    import jax
    from tpu_info import device
    from tpu_inference.kernels.ragged_paged_attention.v3 import kernel
    from tpu_inference.kernels.ragged_paged_attention.v3 import tuned_block_sizes

    sha = _git_sha(repo_path)
    versions = _package_versions()
    try:
        chip_type, chip_count = device.get_local_chips()
        tpu_info = f"{chip_type.name}:{chip_count}"
    except Exception as exc:  # pylint: disable=broad-except
        tpu_info = f"unavailable:{type(exc).__name__}:{exc}"

    print("HARNESS_SOURCE=benchmarks/rpa_v3_fp8_e5m2_verify.py")
    print(f"REPO_PATH={repo_path}")
    print(f"GIT_SHA={sha}")
    print(f"BACKEND={jax.default_backend()}")
    print(f"DEVICES={jax.devices()}")
    print(f"DEVICE_KINDS={[getattr(d, 'device_kind', '') for d in jax.devices()]}")
    print(f"TPU_INFO={tpu_info}")
    print(f"VERSIONS={versions}")
    print("KV_DTYPE=float8_e5m2")
    print("Q_DTYPE=bfloat16")
    print(f"WORKLOAD={args.workload}")
    if args.workload == "tuner_mixed":
        print(f"TUNER_ACTUAL_NUM_SEQS={args.tuner_actual_num_seqs}")
        print(f"TUNER_MAX_NUM_TOKENS={args.tuner_max_num_tokens}")
        print(f"TUNER_MAX_NUM_SEQS={args.tuner_max_num_seqs}")
        print(f"TUNER_TOTAL_NUM_PAGES={args.tuner_total_num_pages}")
    print("MATRIX=" + json.dumps([asdict(case) for case in _matrix()]))

    if jax.default_backend() != "tpu":
        raise RuntimeError(f"Expected TPU backend, got {jax.default_backend()}")
    if not tpu_info.startswith("V6E:"):
        raise RuntimeError(f"Expected TPU v6e, got {tpu_info}")

    rows: list[dict[str, object]] = []
    for case in _matrix():
        print(f"CASE_START={case.case_id}", flush=True)
        row = _run_case(case, args, (kernel, tuned_block_sizes))
        row = {
            "sha": sha,
            "repo_path": str(repo_path),
            "package_versions": json.dumps(versions, sort_keys=True),
            "tpu_info": tpu_info,
            **row,
        }
        print("CASE_RESULT=" + json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)

    _write_csv(args.output_csv, rows)
    print(f"WROTE_CSV={args.output_csv}")
    failed = [row for row in rows if row["correctness"] != "pass"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
