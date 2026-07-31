#!/usr/bin/env python3
"""Compare matching WaveQLab2D CPU and GPU NPZ result archives."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


DEFAULT_RTOL = 1.0e-6
DEFAULT_ATOL = 1.0e-9
CHUNK_SIZE = 1_000_000


def compare_numeric(
    cpu: np.ndarray,
    gpu: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, float | int | bool]:
    """Compare numeric arrays in chunks to limit temporary memory usage."""
    cpu_flat = cpu.reshape(-1)
    gpu_flat = gpu.reshape(-1)
    max_abs = 0.0
    max_rel = 0.0
    diff_sq = 0.0
    cpu_sq = 0.0
    mismatch_count = 0
    nonfinite_mismatch_count = 0

    for start in range(0, cpu_flat.size, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, cpu_flat.size)
        a = cpu_flat[start:stop].astype(np.float64, copy=False)
        b = gpu_flat[start:stop].astype(np.float64, copy=False)

        finite = np.isfinite(a) & np.isfinite(b)
        same_nonfinite = ((np.isnan(a) & np.isnan(b)) |
                          (np.isposinf(a) & np.isposinf(b)) |
                          (np.isneginf(a) & np.isneginf(b)))
        nonfinite_mismatch_count += int(np.count_nonzero(~finite & ~same_nonfinite))

        if np.any(finite):
            af = a[finite]
            bf = b[finite]
            diff = np.abs(af - bf)
            tolerance = atol + rtol * np.abs(bf)
            mismatch_count += int(np.count_nonzero(diff > tolerance))
            max_abs = max(max_abs, float(np.max(diff)))
            denominator = np.maximum(np.maximum(np.abs(af), np.abs(bf)), atol)
            max_rel = max(max_rel, float(np.max(diff / denominator)))
            diff_sq += float(np.dot(diff, diff))
            cpu_sq += float(np.dot(af, af))

    mismatch_count += nonfinite_mismatch_count
    rmse = np.sqrt(diff_sq / max(cpu_flat.size, 1))
    relative_l2 = np.sqrt(diff_sq / cpu_sq) if cpu_sq > 0.0 else np.sqrt(diff_sq)
    return {
        "passed": mismatch_count == 0,
        "mismatches": mismatch_count,
        "nonfinite_mismatches": nonfinite_mismatch_count,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "rmse": float(rmse),
        "relative_l2": float(relative_l2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare matching WaveQLab2D CPU and GPU NPZ outputs.")
    parser.add_argument("cpu", type=Path, help="CPU result .npz file")
    parser.add_argument("gpu", type=Path, help="GPU result .npz file")
    parser.add_argument(
        "--rtol", type=float, default=DEFAULT_RTOL,
        help=f"Relative tolerance (default: {DEFAULT_RTOL:g}).")
    parser.add_argument(
        "--atol", type=float, default=DEFAULT_ATOL,
        help=f"Absolute tolerance (default: {DEFAULT_ATOL:g}).")
    args = parser.parse_args()

    if args.rtol < 0.0 or args.atol < 0.0:
        parser.error("--rtol and --atol must be nonnegative")
    for label, path in (("CPU", args.cpu), ("GPU", args.gpu)):
        if not path.is_file():
            parser.error(f"{label} result not found: {path}")

    print(f"CPU:  {args.cpu.resolve()}")
    print(f"GPU:  {args.gpu.resolve()}")
    print(f"Tolerance: atol={args.atol:g}, rtol={args.rtol:g}\n")

    overall_pass = True
    with np.load(args.cpu, allow_pickle=False) as cpu_archive, \
            np.load(args.gpu, allow_pickle=False) as gpu_archive:
        cpu_keys = set(cpu_archive.files)
        gpu_keys = set(gpu_archive.files)
        cpu_only = sorted(cpu_keys - gpu_keys)
        gpu_only = sorted(gpu_keys - cpu_keys)
        if cpu_only:
            overall_pass = False
            print("CPU-only keys:", ", ".join(cpu_only))
        if gpu_only:
            overall_pass = False
            print("GPU-only keys:", ", ".join(gpu_only))

        for key in sorted(cpu_keys & gpu_keys):
            if key == "metadata":
                print(f"SKIP  {key:<28} backend/timing metadata intentionally differs")
                continue

            cpu = cpu_archive[key]
            gpu = gpu_archive[key]
            if cpu.shape != gpu.shape:
                overall_pass = False
                print(f"FAIL  {key:<28} shape {cpu.shape} != {gpu.shape}")
                continue

            if np.issubdtype(cpu.dtype, np.number) and np.issubdtype(gpu.dtype, np.number):
                stats = compare_numeric(cpu, gpu, rtol=args.rtol, atol=args.atol)
                overall_pass &= bool(stats["passed"])
                status = "PASS" if stats["passed"] else "FAIL"
                print(
                    f"{status}  {key:<28} shape={str(cpu.shape):<20} "
                    f"max_abs={stats['max_abs']:.6e}  "
                    f"rmse={stats['rmse']:.6e}  "
                    f"rel_l2={stats['relative_l2']:.6e}  "
                    f"bad={stats['mismatches']}"
                )
            else:
                passed = np.array_equal(cpu, gpu)
                overall_pass &= passed
                status = "PASS" if passed else "FAIL"
                print(f"{status}  {key:<28} shape={cpu.shape} exact comparison")

    print("\nRESULT:", "CPU and GPU outputs MATCH" if overall_pass else
          "CPU and GPU outputs DO NOT MATCH")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
