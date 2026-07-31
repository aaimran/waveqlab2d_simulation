#!/usr/bin/env python3
"""Benchmark one WaveQLab2D simulation repeatedly on a single CUDA GPU."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics

from benchmark_utils import (
    BENCHMARK_ROOT,
    execute,
    prepare_input,
    write_csv,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a matched WaveQLab2D simulation on one CUDA GPU.")
    parser.add_argument("input", type=Path, help="Same base input used by CPU benchmark")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_ROOT / "gpu")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input not found: {args.input}")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    root = args.output_dir.resolve()
    rows: list[dict[str, object]] = []
    solver_times = []
    for repetition in range(1, args.repetitions + 1):
        prefix = f"bench_gpu_single_r{repetition:02d}"
        input_path = prepare_input(
            args.input.resolve(),
            root / "inputs" / f"{prefix}.in",
            prefix,
            "cuda",
        )
        print(f"[gpu] repetition={repetition}/{args.repetitions}", flush=True)
        result = execute(
            input_path,
            1,
            root / "logs" / f"{prefix}.stdout.log",
        )
        if result["returncode"] != 0 or result["solver_time_s"] is None:
            raise RuntimeError(f"GPU benchmark failed; inspect {result['log']}")
        solver_time = float(result["solver_time_s"])
        solver_times.append(solver_time)
        rows.append({
            "backend": "cuda",
            "gpu_count": 1,
            "repetition": repetition,
            "solver_time_s": solver_time,
            "process_elapsed_s": result["elapsed_s"],
            "mean_step_ms": result["mean_step_ms"],
            "log": result["log"],
        })

    summary = {
        "repetitions": args.repetitions,
        "median_solver_time_s": statistics.median(solver_times),
        "mean_solver_time_s": statistics.mean(solver_times),
        "min_solver_time_s": min(solver_times),
        "max_solver_time_s": max(solver_times),
        "rows": rows,
    }
    write_csv(root / "single_gpu.csv", rows)
    write_json(root / "benchmark_results.json", summary)
    print(f"GPU median simulation time: {summary['median_solver_time_s']:.6f} s")
    print(f"GPU benchmark results: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
