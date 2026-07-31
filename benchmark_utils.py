"""Shared helpers for WaveQLab2D CPU and GPU performance benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import time


SIMULATION_ROOT = Path(__file__).resolve().parent
RUNNER = SIMULATION_ROOT / "run_simulation.py"
BENCHMARK_ROOT = SIMULATION_ROOT / "benchmark"


def parse_input(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            values[key] = value
    return values


def replace_settings(template: str, settings: dict[str, object]) -> str:
    remaining = {key: str(value) for key, value in settings.items()}
    lines = []
    for line in template.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                comment = ""
                if "#" in line:
                    comment = "  #" + line.split("#", 1)[1]
                line = f"{key} = {remaining.pop(key)}{comment}"
        lines.append(line)
    if remaining:
        lines.extend(("", "# Performance benchmark overrides"))
        lines.extend(f"{key} = {value}" for key, value in remaining.items())
    return "\n".join(lines) + "\n"


def benchmark_overrides(base_values: dict[str, str], prefix: str, backend: str) -> dict[str, object]:
    """Return settings that preserve compute work while minimizing result I/O."""
    y0 = float(base_values.get("y0", 0.5 * float(base_values["y_length"])))
    return {
        "compute_backend": backend,
        "precision": "float64",
        "output_prefix": prefix,
        "output_compression": "uncompressed",
        "output_streaming": "false",
        "output_timing": "true",
        "output_mode": "subdomain",
        "output_xlim": "-0.05,0.05",
        "output_ylim": f"{y0 - 0.05:.6f},{y0 + 0.05:.6f}",
        "station_file": "none",
        "iplot": 100000000,
    }


def prepare_input(base: Path, destination: Path, prefix: str, backend: str) -> Path:
    values = parse_input(base)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        replace_settings(
            base.read_text(encoding="utf-8"),
            benchmark_overrides(values, prefix, backend),
        ),
        encoding="utf-8",
    )
    return destination


def execute(input_path: Path, threads: int, output_log: Path) -> dict[str, object]:
    if not RUNNER.is_file():
        raise FileNotFoundError(f"simulation launcher not found: {RUNNER}")
    output_log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(RUNNER), str(input_path), "-np", str(threads)]
    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=SIMULATION_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.monotonic() - started
    output_log.write_text(process.stdout, encoding="utf-8")

    solver_time = _last_float(
        process.stdout,
        (r"total simulation time\s*=\s*([0-9.eE+-]+)\s*s",
         r"CUDA simulation time\s*=\s*([0-9.eE+-]+)\s*s"),
    )
    mean_step_ms = _last_float(
        process.stdout,
        (r"mean step \(total\)\s*=\s*([0-9.eE+-]+)\s*ms",),
    )
    return {
        "returncode": process.returncode,
        "elapsed_s": elapsed,
        "solver_time_s": solver_time,
        "mean_step_ms": mean_step_ms,
        "command": command,
        "log": str(output_log),
    }


def _last_float(text: str, patterns: tuple[str, ...]) -> float | None:
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])
    return None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
