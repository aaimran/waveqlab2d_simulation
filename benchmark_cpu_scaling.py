#!/usr/bin/env python3
"""Benchmark WaveQLab2D CPU scaling and fixed-core concurrent throughput."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import statistics
import time

from benchmark_utils import (
    BENCHMARK_ROOT,
    execute,
    prepare_input,
    write_csv,
    write_json,
)


SINGLE_THREADS = (32, 16, 8, 4, 2, 1)
CONCURRENT_MATRIX = ((8, 4), (4, 8), (2, 16), (1, 32))


def run_single(base: Path, repetitions: int, root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline: float | None = None
    for threads in SINGLE_THREADS:
        times = []
        for repetition in range(1, repetitions + 1):
            prefix = f"bench_cpu_single_np{threads:02d}_r{repetition:02d}"
            input_path = prepare_input(
                base,
                root / "inputs" / "single" / f"{prefix}.in",
                prefix,
                "cpu",
            )
            print(f"[single] threads={threads} repetition={repetition}/{repetitions}", flush=True)
            result = execute(
                input_path,
                threads,
                root / "logs" / "single" / f"{prefix}.stdout.log",
            )
            if result["returncode"] != 0 or result["solver_time_s"] is None:
                raise RuntimeError(f"CPU benchmark failed; inspect {result['log']}")
            solver_time = float(result["solver_time_s"])
            times.append(solver_time)
            rows.append({
                "phase": "single",
                "threads_per_sim": threads,
                "concurrency": 1,
                "total_threads": threads,
                "repetition": repetition,
                "solver_time_s": solver_time,
                "process_elapsed_s": result["elapsed_s"],
                "mean_step_ms": result["mean_step_ms"],
                "log": result["log"],
            })
        median = statistics.median(times)
        if threads == 1:
            baseline = median
        for row in rows:
            if row["phase"] == "single" and row["threads_per_sim"] == threads:
                row["median_solver_time_s"] = median

    # The np=1 baseline runs last, so calculate speedups after all timings exist.
    baseline_rows = [
        row for row in rows
        if row["phase"] == "single" and row["threads_per_sim"] == 1
    ]
    if not baseline_rows:
        raise RuntimeError("missing np=1 CPU baseline")
    baseline = float(baseline_rows[0]["median_solver_time_s"])
    for row in rows:
        threads = int(row["threads_per_sim"])
        median = float(row["median_solver_time_s"])
        speedup = baseline / median
        row["speedup_vs_np1"] = speedup
        row["parallel_efficiency"] = speedup / threads
    return rows


def run_concurrent(base: Path, repetitions: int, root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threads, concurrency in CONCURRENT_MATRIX:
        for repetition in range(1, repetitions + 1):
            jobs = []
            for index in range(1, concurrency + 1):
                prefix = (
                    f"bench_cpu_concurrent_np{threads:02d}_c{concurrency:02d}_"
                    f"r{repetition:02d}_s{index:02d}"
                )
                input_path = prepare_input(
                    base,
                    root / "inputs" / "concurrent" / f"{prefix}.in",
                    prefix,
                    "cpu",
                )
                jobs.append((index, prefix, input_path))

            print(
                f"[concurrent] threads/sim={threads} simulations={concurrency} "
                f"total_threads={threads * concurrency} repetition={repetition}/{repetitions}",
                flush=True,
            )
            batch_started = time.monotonic()
            results = []
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(
                        execute,
                        input_path,
                        threads,
                        root / "logs" / "concurrent" / f"{prefix}.stdout.log",
                    ): (index, prefix)
                    for index, prefix, input_path in jobs
                }
                for future in as_completed(futures):
                    index, prefix = futures[future]
                    result = future.result()
                    if result["returncode"] != 0 or result["solver_time_s"] is None:
                        raise RuntimeError(f"concurrent benchmark failed; inspect {result['log']}")
                    results.append((index, prefix, result))
            batch_elapsed = time.monotonic() - batch_started
            throughput = concurrency / batch_elapsed
            solver_times = [float(result["solver_time_s"]) for _, _, result in results]
            rows.append({
                "phase": "concurrent-summary",
                "threads_per_sim": threads,
                "concurrency": concurrency,
                "total_threads": threads * concurrency,
                "repetition": repetition,
                "batch_elapsed_s": batch_elapsed,
                "simulations_per_second": throughput,
                "simulations_per_hour": throughput * 3600.0,
                "median_solver_time_s": statistics.median(solver_times),
                "min_solver_time_s": min(solver_times),
                "max_solver_time_s": max(solver_times),
            })
            for index, prefix, result in sorted(results):
                rows.append({
                    "phase": "concurrent-process",
                    "threads_per_sim": threads,
                    "concurrency": concurrency,
                    "total_threads": threads * concurrency,
                    "repetition": repetition,
                    "simulation_index": index,
                    "solver_time_s": result["solver_time_s"],
                    "process_elapsed_s": result["elapsed_s"],
                    "mean_step_ms": result["mean_step_ms"],
                    "log": result["log"],
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CPU thread-scaling and fixed-32-thread throughput benchmarks.")
    parser.add_argument("input", type=Path, help="Base WaveQLab2D simulation input")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--phase", choices=("all", "single", "concurrent"), default="all")
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_ROOT / "cpu")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input not found: {args.input}")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if (os.cpu_count() or 1) < 32:
        print("WARNING: fewer than 32 logical CPUs are visible; results may oversubscribe")

    root = args.output_dir.resolve()
    rows: list[dict[str, object]] = []
    if args.phase in ("all", "single"):
        rows.extend(run_single(args.input.resolve(), args.repetitions, root))
        write_csv(root / "single_scaling.csv", rows)
    if args.phase in ("all", "concurrent"):
        concurrent_rows = run_concurrent(args.input.resolve(), args.repetitions, root)
        rows.extend(concurrent_rows)
        write_csv(root / "concurrent_throughput.csv", concurrent_rows)

    write_json(root / "benchmark_results.json", rows)
    print(f"CPU benchmark results: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
