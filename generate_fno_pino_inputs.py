#!/usr/bin/env python3
"""Generate the Stage 1 FNO/PINO source-position dataset inputs.

Dry-run mode performs the complete 1,500-case design and validation without
creating directories or files. Normal mode writes input decks and manifests
under ``fno-pino-benchmark/stage1-y0`` by default.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "fno-pino-benchmark" / "stage1-y0"
DEFAULT_SEED = 20260731

PHYSICAL_SIZE_KM = 30.0
PML_KM = 3.0
COMPUTATIONAL_SIZE_KM = PHYSICAL_SIZE_KM + 2.0 * PML_KM
OBJECTIVE_SIZE_KM = 20.0
DX_KM = 0.1
DT_S = 0.002
TEND_S = 10.0
IPLOT = 10
X0_KM = 0.0
CENTER_Y_KM = COMPUTATIONAL_SIZE_KM / 2.0
T0_S = 1.0
PERIOD_S = 0.2
M0 = 0.02824
PML_POINTS = round(PML_KM / DX_KM)

EXPECTED_COUNTS = {
    "train": 1050,
    "validation_id": 100,
    "validation_ood": 50,
    "test_id": 150,
    "test_ood": 150,
}


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    evaluation_group: str
    design: str
    design_index: int
    seed: int
    x0_km: float
    y0_km: float
    y_offset_km: float
    M0: float
    T_s: float
    t0_s: float
    tend_s: float
    dt_s: float
    iplot: int
    input_file: str
    output_prefix: str


def stratified_values(count: int, lower: float, upper: float, seed: int) -> list[float]:
    """Return a deterministic shuffled stratified design on [lower, upper]."""
    rng = random.Random(seed)
    width = (upper - lower) / count
    values = [lower + (index + rng.random()) * width for index in range(count)]
    rng.shuffle(values)
    return values


def symmetric_ood_offsets(
    count: int,
    lower_magnitude: float,
    upper_magnitude: float,
    seed: int,
) -> list[float]:
    if count % 2:
        raise ValueError("symmetric OOD count must be even")
    half = count // 2
    magnitudes = stratified_values(half, lower_magnitude, upper_magnitude, seed)
    offsets = [-value for value in magnitudes] + magnitudes
    random.Random(seed + 1).shuffle(offsets)
    return offsets


def make_group(
    name: str,
    split: str,
    evaluation_group: str,
    offsets: Iterable[float],
    seed: int,
) -> list[Case]:
    cases = []
    for index, offset in enumerate(offsets):
        case_id = f"{name}_{index:04d}"
        relative_file = f"input/{split}/{case_id}.in"
        cases.append(Case(
            case_id=case_id,
            split=split,
            evaluation_group=evaluation_group,
            design="deterministic-stratified-y0",
            design_index=index,
            seed=seed,
            x0_km=X0_KM,
            y0_km=CENTER_Y_KM + float(offset),
            y_offset_km=float(offset),
            M0=M0,
            T_s=PERIOD_S,
            t0_s=T0_S,
            tend_s=TEND_S,
            dt_s=DT_S,
            iplot=IPLOT,
            input_file=relative_file,
            output_prefix=f"fpop_y0_{case_id}",
        ))
    return cases


def build_cases(seed: int = DEFAULT_SEED) -> list[Case]:
    train = stratified_values(1050, -5.0, 5.0, seed + 10)
    validation_id = stratified_values(100, -5.0, 5.0, seed + 20)
    validation_ood = symmetric_ood_offsets(50, 5.0, 7.0, seed + 30)
    test_id = stratified_values(150, -5.0, 5.0, seed + 40)
    test_ood = symmetric_ood_offsets(150, 7.0, 8.0, seed + 50)

    # Required reference and extreme extrapolation cases.
    test_id[0] = 0.0
    test_ood[0] = -8.0
    test_ood[1] = 8.0

    cases = []
    cases.extend(make_group("train", "train", "id", train, seed + 10))
    cases.extend(make_group(
        "validation_id", "validation", "id", validation_id, seed + 20))
    cases.extend(make_group(
        "validation_ood", "validation", "near_ood", validation_ood, seed + 30))
    cases.extend(make_group("test_id", "test", "id", test_id, seed + 40))
    cases.extend(make_group("test_ood", "test", "far_ood", test_ood, seed + 50))
    validate_cases(cases)
    return cases


def validate_cases(cases: list[Case]) -> None:
    if len(cases) != 1500:
        raise ValueError(f"expected 1500 cases, got {len(cases)}")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs")
    locations = [round(case.y0_km, 9) for case in cases]
    if len(locations) != len(set(locations)):
        raise ValueError("duplicate y0 values across dataset splits")

    counts = {name: 0 for name in EXPECTED_COUNTS}
    for case in cases:
        group = case.case_id.rsplit("_", 1)[0]
        counts[group] += 1
        offset = case.y_offset_km
        if case.evaluation_group == "id" and not -5.0 <= offset <= 5.0:
            raise ValueError(f"ID offset out of range: {case}")
        if case.evaluation_group == "near_ood" and not 5.0 < abs(offset) <= 7.0:
            raise ValueError(f"near-OOD offset out of range: {case}")
        if case.evaluation_group == "far_ood" and not 7.0 < abs(offset) <= 8.0:
            raise ValueError(f"far-OOD offset out of range: {case}")
        if not 8.0 <= case.y0_km <= 28.0:
            raise ValueError(f"source lies outside objective window: {case}")
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"incorrect split counts: {counts}")

    if not any(case.split == "test" and case.y_offset_km == 0.0 for case in cases):
        raise ValueError("center reference case is missing")
    extremes = {case.y_offset_km for case in cases if case.evaluation_group == "far_ood"}
    if not {-8.0, 8.0}.issubset(extremes):
        raise ValueError("+/-8 km OOD reference cases are missing")


def input_settings(case: Case, output_format: str) -> dict[str, object]:
    streaming = output_format == "hdf5"
    return {
        "compute_backend": "cuda",
        "precision": "float64",
        "gpu_device": 0,
        "output_prefix": case.output_prefix,
        "output_format": output_format,
        "output_compression": "compressed",
        "output_streaming": str(streaming).lower(),
        "output_timing": "true",
        "output_mode": "subdomain",
        "output_xlim": "-10,10",
        "output_ylim": "8,28",
        "station_file": "none",
        "station_stride": 1,
        "station_interpolation": "nearest",
        "station_include_fault": "false",
        "iplot": IPLOT,
        "absorbing_method": "pml",
        "pml_width": PML_POINTS,
        "pml_profile_order": 3,
        "pml_reflection_target": "1e-6",
        "pml_max_damping": "auto",
        "boundary_x_left": "absorbing",
        "boundary_x_right": "absorbing",
        "boundary_y_surface": "absorbing",
        "boundary_y_depth": "absorbing",
        "pml_x_left": "true",
        "pml_x_right": "true",
        "pml_y_surface": "true",
        "pml_y_depth": "true",
        "response": "elastic",
        "q_storage": "conventional",
        "coarse_grain": "false",
        "Qp": 1_000_000_000.0,
        "Qs": 1_000_000_000.0,
        "dt": DT_S,
        "tend": TEND_S,
        "cfl": 0.5,
        "fd_type": "central",
        "sbp_family": "diagonal",
        "fd_order": 6,
        "order": 6,
        "mode": "II",
        "interface": "none",
        "x_left_length": COMPUTATIONAL_SIZE_KM / 2.0,
        "x_right_length": COMPUTATIONAL_SIZE_KM / 2.0,
        "y_length": COMPUTATIONAL_SIZE_KM,
        "x_left_resolution": DX_KM,
        "x_right_resolution": DX_KM,
        "y_resolution": DX_KM,
        "cp": 6.0,
        "cs": 3.464,
        "rho": 2.6702,
        "simulation_type": "PointSource",
        "source_type": "Gaussian",
        "M0": case.M0,
        "x0": case.x0_km,
        "y0": case.y0_km,
        "t0": case.t0_s,
        "T": case.T_s,
    }


SECTIONS = (
    ("Compute", ("compute_backend", "precision", "gpu_device")),
    ("Output", ("output_prefix", "output_format", "output_compression",
                "output_streaming", "output_timing", "output_mode", "output_xlim",
                "output_ylim", "station_file", "station_stride",
                "station_interpolation", "station_include_fault", "iplot")),
    ("Full-space boundary conditions", ("absorbing_method", "pml_width",
                "pml_profile_order", "pml_reflection_target", "pml_max_damping",
                "boundary_x_left", "boundary_x_right", "boundary_y_surface",
                "boundary_y_depth", "pml_x_left", "pml_x_right", "pml_y_surface",
                "pml_y_depth")),
    ("Material", ("response", "q_storage", "coarse_grain", "Qp", "Qs", "cp", "cs",
                  "rho")),
    ("Time and discretization", ("dt", "tend", "cfl", "fd_type", "sbp_family",
                                 "fd_order", "order", "mode", "interface")),
    ("36 km computational domain: 30 km physical plus external PML",
     ("x_left_length", "x_right_length", "y_length", "x_left_resolution",
      "x_right_resolution", "y_resolution")),
    ("Gaussian point source", ("simulation_type", "source_type", "M0", "x0", "y0",
                               "t0", "T")),
)


def render_input(case: Case, output_format: str) -> str:
    settings = input_settings(case, output_format)
    lines = [
        "# WaveQLab2D FNO/PINO Stage 1 y0 sweep",
        f"# case_id={case.case_id} split={case.split} group={case.evaluation_group}",
        "# Physical domain: 30x30 km; computational domain: 36x36 km",
        "# Objective window: 20x20 km; PML: 3 km external on every side",
    ]
    emitted = set()
    for title, keys in SECTIONS:
        lines.extend(("", f"# {title}"))
        for key in keys:
            lines.append(f"{key} = {settings[key]}")
            emitted.add(key)
    if emitted != set(settings):
        raise RuntimeError(f"unrendered settings: {sorted(set(settings) - emitted)}")
    return "\n".join(lines) + "\n"


def validate_with_solver(cases: list[Case], output_format: str) -> None:
    try:
        import config_2d
    except ImportError as error:
        raise RuntimeError(
            "cannot import WaveQLab2D config_2d; activate the simulation venv or "
            "use --skip-solver-validation") from error

    for case in cases:
        params = config_2d.build_params(input_settings(case, output_format), 1)
        config_2d.validate(params)
        if (params["nx_left"], params["nx_right"], params["ny"]) != (181, 181, 361):
            raise ValueError(f"unexpected grid for {case.case_id}")
        if params["nt"] != 5000 or params["iplot"] != 10:
            raise ValueError(f"unexpected time sampling for {case.case_id}")


MANIFEST_FIELDS = tuple(Case.__dataclass_fields__) + (
    "physical_domain_km", "computational_domain_km", "objective_domain_km",
    "pml_km", "pml_points", "dx_km", "nx_left", "nx_right", "ny",
    "snap_dt_s", "expected_frames", "field_order",
)


def manifest_row(case: Case) -> dict[str, object]:
    row = asdict(case)
    row.update({
        "physical_domain_km": PHYSICAL_SIZE_KM,
        "computational_domain_km": COMPUTATIONAL_SIZE_KM,
        "objective_domain_km": OBJECTIVE_SIZE_KM,
        "pml_km": PML_KM,
        "pml_points": PML_POINTS,
        "dx_km": DX_KM,
        "nx_left": 181,
        "nx_right": 181,
        "ny": 361,
        "snap_dt_s": DT_S * IPLOT,
        "expected_frames": round(TEND_S / (DT_S * IPLOT)),
        "field_order": "vx,vy",
    })
    return row


def ensure_writable_targets(root: Path, cases: list[Case], overwrite: bool) -> None:
    targets = [root / case.input_file for case in cases]
    targets.extend((root / "manifest.csv", root / "dataset.json"))
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        preview = "\n  ".join(str(path) for path in existing[:5])
        raise FileExistsError(
            f"{len(existing)} generated targets already exist; use --overwrite to "
            f"replace them. First targets:\n  {preview}")


def write_dataset(root: Path, cases: list[Case], output_format: str, overwrite: bool) -> None:
    ensure_writable_targets(root, cases, overwrite)
    for case in cases:
        path = root / case.input_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_input(case, output_format), encoding="utf-8")

    rows = [manifest_row(case) for case in cases]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "stage": "stage1-y0",
        "seed": cases[0].seed - 10,
        "total_cases": len(cases),
        "counts": EXPECTED_COUNTS,
        "output_format": output_format,
        "physical_domain_km": [30.0, 30.0],
        "computational_domain_km": [36.0, 36.0],
        "objective_window": {"x_km": [-10.0, 10.0], "y_km": [8.0, 28.0]},
        "pml_km": 3.0,
        "spatial_resolution_km": DX_KM,
        "temporal_resolution_s": DT_S,
        "snapshot_interval_s": DT_S * IPLOT,
        "tend_s": TEND_S,
        "fields": ["vx", "vy"],
    }
    (root / "dataset.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def print_summary(cases: list[Case], output_root: Path, dry_run: bool) -> None:
    counts = {key: 0 for key in EXPECTED_COUNTS}
    for case in cases:
        counts[case.case_id.rsplit("_", 1)[0]] += 1
    mode = "DRY RUN (no files written)" if dry_run else "GENERATED"
    print(mode)
    print(f"Stage:             y0")
    print(f"Total cases:       {len(cases)}")
    print(f"Counts:            {counts}")
    print(f"Train y0:          13 to 23 km (center +/-5 km)")
    print(f"Far-OOD y0:        10 to 11 and 25 to 26 km")
    print(f"Grid:              181 + 181 by 361")
    print(f"Steps/frames:      5000 / 500")
    print(f"Output root:       {output_root}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Stage 1 y0-sweep FNO/PINO input decks and manifests.")
    parser.add_argument("--stage", choices=("y0",), default="y0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--format", choices=("hdf5", "npz"), default="hdf5")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build, render, and validate all cases without writing any files.")
    parser.add_argument(
        "--skip-solver-validation", action="store_true",
        help="Skip validation through the installed WaveQLab2D config parser.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace generated targets that already exist.")
    args = parser.parse_args()

    cases = build_cases(args.seed)
    # Render every deck even in dry-run mode to catch missing settings.
    for case in cases:
        rendered = render_input(case, args.format)
        if not rendered.endswith("\n") or "y0 = " not in rendered:
            raise RuntimeError(f"failed to render {case.case_id}")
    if not args.skip_solver_validation:
        validate_with_solver(cases, args.format)

    output_root = args.output_root.expanduser().resolve()
    if not args.dry_run:
        write_dataset(output_root, cases, args.format, args.overwrite)
    print_summary(cases, output_root, args.dry_run)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
