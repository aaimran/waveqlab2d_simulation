#!/usr/bin/env python3
"""Run a generated FNO/PINO manifest on one GPU, restart safely, and archive HDF5 results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_simulation.py"
STAGING = ROOT / "output"


def read_manifest(dataset_root: Path) -> list[dict[str, str]]:
    path = dataset_root / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def valid_hdf5(path: Path, row: dict[str, str]) -> tuple[bool, str]:
    if not path.is_file() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        import h5py
        import numpy as np

        with h5py.File(path, "r") as archive:
            required = {"DomainOutput_l", "DomainOutput_r", "snap_times"}
            if not required.issubset(archive.keys()):
                return False, f"missing keys {sorted(required - set(archive.keys()))}"
            if "metadata" not in archive.attrs:
                return False, "missing finalized metadata"
            expected_frames = int(row["expected_frames"])
            left = archive["DomainOutput_l"]
            right = archive["DomainOutput_r"]
            times = archive["snap_times"][:]
            if left.ndim != 4 or right.ndim != 4:
                return False, "domain output is not rank four"
            if left.shape[2:] != (expected_frames, 5):
                return False, f"unexpected left shape {left.shape}"
            if right.shape[2:] != (expected_frames, 5):
                return False, f"unexpected right shape {right.shape}"
            if times.shape != (expected_frames,):
                return False, f"unexpected snap_times shape {times.shape}"
            if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
                return False, "invalid snapshot times"
            # WaveQLab snapshots at zero-based iterations 0, iplot, 2*iplot, ...
            # after advancing the state, so times are dt, (iplot+1)*dt, ... .
            dt = float(row["dt_s"])
            iplot = int(row["iplot"])
            expected_start = dt
            expected_end = ((expected_frames - 1) * iplot + 1) * dt
            expected_stride = iplot * dt
            if not np.isclose(float(times[0]), expected_start, atol=1.0e-6):
                return False, f"first snapshot is {times[0]}, expected {expected_start}"
            if not np.allclose(np.diff(times), expected_stride, atol=1.0e-6, rtol=0.0):
                return False, f"snapshot stride is not {expected_stride}"
            if not np.isclose(float(times[-1]), expected_end, atol=1.0e-5):
                return False, f"last snapshot is {times[-1]}, expected {expected_end}"
            # Sample first/final velocity frames without loading the full archive.
            for dataset in (left, right):
                sample = dataset[:, :, (0, expected_frames - 1), :2]
                if not np.all(np.isfinite(sample)):
                    return False, "nonfinite velocity sample"
        return True, "ok"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def find_valid(directory: Path, prefix: str, row: dict[str, str]) -> Path | None:
    for path in sorted(directory.glob(f"{prefix}_*.h5"), reverse=True):
        valid, _ = valid_hdf5(path, row)
        if valid:
            return path
    return None


def record(status_file: Path, **values: object) -> None:
    status_file.parent.mkdir(parents=True, exist_ok=True)
    values["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with status_file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(values, sort_keys=True, default=str) + "\n")


def solver_time(console: str) -> float | None:
    matches = re.findall(r"CUDA simulation time\s*=\s*([0-9.eE+-]+)\s*s", console)
    return float(matches[-1]) if matches else None


def move_timing(prefix: str, raw_dir: Path) -> Path | None:
    candidates = sorted((STAGING / "timing").glob(f"{prefix}_*_timing.npz"))
    if not candidates:
        return None
    timing_dir = raw_dir / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    source = candidates[-1]
    target = timing_dir / source.name
    if target.exists():
        target.unlink()
    shutil.move(str(source), str(target))
    return target


def run_case(
    dataset_root: Path,
    row: dict[str, str],
    status_file: Path,
    rerun: bool,
) -> tuple[str, str, float | None]:
    case_id = row["case_id"]
    prefix = row["output_prefix"]
    input_path = dataset_root / row["input_file"]
    raw_dir = dataset_root / "raw" / row["split"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir = dataset_root / "status" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    existing = find_valid(raw_dir, prefix, row)
    if existing and not rerun:
        return "skipped", str(existing), None

    staged = find_valid(STAGING, prefix, row)
    if staged and not rerun:
        target = raw_dir / staged.name
        if target.exists():
            target.unlink()
        shutil.move(str(staged), str(target))
        move_timing(prefix, raw_dir)
        record(status_file, case_id=case_id, split=row["split"],
               evaluation_group=row["evaluation_group"], status="recovered",
               output=str(target), bytes=target.stat().st_size)
        return "recovered", str(target), None

    command = [
        sys.executable,
        str(RUNNER),
        str(input_path),
        "--backend", "cuda",
        "--precision", "float64",
        "--device", "0",
        "-np", "1",
    ]
    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.monotonic() - started
    console_path = log_dir / f"{case_id}.log"
    console_path.write_text(process.stdout, encoding="utf-8")
    staged = find_valid(STAGING, prefix, row) if process.returncode == 0 else None
    if staged is None:
        record(status_file, case_id=case_id, split=row["split"],
               evaluation_group=row["evaluation_group"], status="failed",
               returncode=process.returncode, elapsed_s=elapsed,
               console_log=str(console_path), tail=process.stdout[-4000:])
        return "failed", str(console_path), None

    target = raw_dir / staged.name
    if target.exists():
        target.unlink()
    shutil.move(str(staged), str(target))
    timing = move_timing(prefix, raw_dir)
    gpu_time = solver_time(process.stdout)
    record(status_file, case_id=case_id, split=row["split"],
           evaluation_group=row["evaluation_group"], status="complete",
           output=str(target), timing=str(timing) if timing else None,
           bytes=target.stat().st_size, elapsed_s=elapsed, gpu_time_s=gpu_time,
           console_log=str(console_path))
    return "completed", str(target), gpu_time


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate restart-safe WaveQLab2D HDF5 results on one CUDA GPU.")
    parser.add_argument("dataset_root", type=Path, help="Generated Stage 1 dataset root")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--group", choices=("id", "near_ood", "far_ood"))
    parser.add_argument("--limit", type=int, help="Run at most N selected cases")
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun cases even when validated results already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the selected queue without running simulations")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    args = parser.parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    rows = read_manifest(dataset_root)
    if args.split:
        rows = [row for row in rows if row["split"] == args.split]
    if args.group:
        rows = [row for row in rows if row["evaluation_group"] == args.group]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[:args.limit]
    if not rows:
        parser.error("selection contains no cases")

    print(f"Dataset: {dataset_root}")
    print(f"Selected cases: {len(rows)}")
    print("Backend: one CUDA GPU, sequential execution")
    if args.dry_run:
        for row in rows[:10]:
            print(f"  {row['case_id']} {row['split']} {row['evaluation_group']}")
        if len(rows) > 10:
            print(f"  ... {len(rows) - 10} more")
        return 0

    status_file = dataset_root / "status" / "generation-status.jsonl"
    completed = skipped = recovered = failed = 0
    gpu_times = []
    batch_started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        free_gb = shutil.disk_usage(dataset_root).free / 1024**3
        if free_gb < args.min_free_gb:
            record(status_file, status="stopped-low-disk", free_gb=free_gb)
            print(f"STOP: free disk {free_gb:.1f} GB < {args.min_free_gb:.1f} GB")
            return 2
        status, detail, gpu_time = run_case(dataset_root, row, status_file, args.rerun)
        if status == "completed":
            completed += 1
            label = "DONE"
        elif status == "recovered":
            recovered += 1
            label = "RECOVER"
        elif status == "skipped":
            skipped += 1
            label = "SKIP"
        else:
            failed += 1
            label = "FAIL"
        if gpu_time is not None:
            gpu_times.append(gpu_time)
        eta = None
        if gpu_times:
            eta = (len(rows) - index) * sum(gpu_times) / len(gpu_times)
        eta_text = f" ETA={eta/60:.1f}m" if eta is not None else ""
        print(f"[{index:04d}/{len(rows):04d}] {label} {row['case_id']}{eta_text}", flush=True)
        if status == "failed":
            print(f"  inspect: {detail}")

    elapsed = time.monotonic() - batch_started
    record(status_file, status="batch-complete", selected=len(rows),
           completed=completed, recovered=recovered, skipped=skipped,
           failed=failed, elapsed_s=elapsed)
    print(
        f"Batch complete: completed={completed} recovered={recovered} "
        f"skipped={skipped} failed={failed} wall={elapsed/3600:.2f}h")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
