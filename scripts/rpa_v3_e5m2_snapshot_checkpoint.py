#!/usr/bin/env python3
"""Snapshot the live RPA v3 e5m2 tuner DB to local disk and GCS."""

from __future__ import annotations

import datetime as _datetime
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path


DB_SRC = Path(os.environ.get("RPA_E5M2_DB_SRC",
                             "/tmp/rpa_v3_e5m2_v6e_full_db"))
TARGETS_SRC = Path(os.environ.get("RPA_E5M2_TARGETS",
                                  "/tmp/rpa_v3_e5m2_targets.json"))
TOOLS_SRC = Path(os.environ.get("RPA_E5M2_TOOLS", "/tmp/rpa-tuner-tools"))
STUBS_SRC = Path(os.environ.get("RPA_E5M2_STUBS", "/tmp/rpa-stubs"))
CHECKPOINT_ROOT = Path(
    os.environ.get("RPA_E5M2_CHECKPOINT_ROOT",
                   "/home/enyouki/rpa_v3_e5m2_checkpoint"))
GCS_URI = os.environ.get(
    "RPA_E5M2_CHECKPOINT_GCS",
    "gs://personal-mark-eu/tpu-inference-checkpoints/rpa_v3_e5m2_v6e")
REPO = Path(__file__).resolve().parents[1]


def _load_json_with_retry(path: Path, attempts: int = 10):
    last_err = None
    for _ in range(attempts):
        try:
            with path.open() as f:
                return json.load(f)
        except json.JSONDecodeError as err:
            last_err = err
            time.sleep(0.5)
    raise RuntimeError(f"could not read stable JSON from {path}: {last_err}")


def _atomic_json_dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp_name, path)


def _copy_json_table(src: Path, dst: Path) -> None:
    _atomic_json_dump(dst, _load_json_with_retry(src))


def _newest_mtime(src: Path) -> float:
    newest = src.stat().st_mtime
    for path in src.rglob("*"):
        if path.exists():
            newest = max(newest, path.stat().st_mtime)
    return newest


def _tar_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists() and dst.stat().st_mtime >= _newest_mtime(src):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tarfile.open(tmp, "w:gz") as tf:
        tf.add(src, arcname=src.name)
    os.replace(tmp, dst)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def main() -> int:
    if not DB_SRC.exists():
        print(f"missing source DB: {DB_SRC}", file=sys.stderr)
        return 1

    db_dst = CHECKPOINT_ROOT / "db"
    artifacts_dst = CHECKPOINT_ROOT / "artifacts"
    for src in sorted(DB_SRC.glob("*.json")):
        _copy_json_table(src, db_dst / src.name)

    _copy_if_exists(TARGETS_SRC, artifacts_dst / TARGETS_SRC.name)
    _tar_dir(TOOLS_SRC, artifacts_dst / "rpa-tuner-tools.tar.gz")
    _tar_dir(STUBS_SRC, artifacts_dst / "rpa-stubs.tar.gz")

    _copy_if_exists(REPO / "scripts/rpa_v3_e5m2_resume_sweep.sh",
                    CHECKPOINT_ROOT / "scripts/rpa_v3_e5m2_resume_sweep.sh")
    _copy_if_exists(REPO / "scripts/rpa_v3_e5m2_snapshot_checkpoint.py",
                    CHECKPOINT_ROOT /
                    "scripts/rpa_v3_e5m2_snapshot_checkpoint.py")
    _copy_if_exists(REPO / "docs/rpa_v3_e5m2_tuning_recovery.md",
                    CHECKPOINT_ROOT /
                    "docs/rpa_v3_e5m2_tuning_recovery.md")

    results_path = db_dst / "CaseResults.json"
    results = _load_json_with_retry(results_path) if results_path.exists() else []
    statuses = Counter(row.get("ProcessedStatus") for row in results)
    last_case = max([row["CaseId"] for row in results], default=-1)
    metadata = {
        "updated_at_utc": _datetime.datetime.now(
            _datetime.timezone.utc).isoformat(),
        "source_db": str(DB_SRC),
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "gcs_uri": GCS_URI,
        "case_set_id": "rpa_v3_e5m2_v6e_full",
        "run_id": "001",
        "results": len(results),
        "last_case": last_case,
        "status_counts": dict(statuses),
        "total_cases": 8930,
        "measure_iters": 21,
        "candidate_bkv_p": [1, 2, 4, 8, 16, 32],
        "candidate_bq_sz": [8, 16, 32, 64, 128],
    }
    _atomic_json_dump(CHECKPOINT_ROOT / "metadata.json", metadata)

    gsutil = shutil.which("gsutil")
    if gsutil and GCS_URI:
        subprocess.run(
            [gsutil, "-m", "rsync", "-r", str(CHECKPOINT_ROOT), GCS_URI],
            check=True,
        )

    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

