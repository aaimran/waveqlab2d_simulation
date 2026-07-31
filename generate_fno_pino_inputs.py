#!/usr/bin/env python3
"""Generate the Stage 1 FNO/PINO source-position dataset inputs.

Dry-run mode performs the complete dataset design and validation without
creating directories or files. The generator supports a compact four-sided
PML design and a reflection-safe extended no-PML design.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = 20260731

BOUNDARY_DESIGNS = {
    "pml": {
        "directory": "stage1-y0",
        "computational_size_km": 15.0,
        "pml_km": 2.0,
        "absorbing_method": "pml",
        "pml_enabled": True,
    },
    "extended-nopml": {
        "directory": "stage1-y0-extended-nopml",
        "computational_size_km": 42.0,
        "pml_km": 0.0,
        "absorbing_method": "sat",
        "pml_enabled": False,
    },
}

CURRENT_BOUNDARY_DESIGN = "pml"
COMPUTATIONAL_SIZE_KM = 15.0
PML_KM = 2.0
PHYSICAL_SIZE_KM = 11.0
OBJECTIVE_SIZE_KM = 10.0
DX_KM = 0.1
DT_S = 0.002
TEND_S = 5.0
IPLOT = 10
X0_KM = 0.0
CENTER_Y_KM = COMPUTATIONAL_SIZE_KM / 2.0
T0_S = 1.0
PERIOD_S = 0.2
M0 = 0.02824
PML_POINTS = round(PML_KM / DX_KM)


def configure_boundary_design(name: str) -> dict[str, object]:
    """Select a domain/boundary design and update derived module constants."""
    global CURRENT_BOUNDARY_DESIGN
    global COMPUTATIONAL_SIZE_KM, PML_KM, PHYSICAL_SIZE_KM, CENTER_Y_KM, PML_POINTS
    design = BOUNDARY_DESIGNS[name]
    CURRENT_BOUNDARY_DESIGN = name
    COMPUTATIONAL_SIZE_KM = float(design["computational_size_km"])
    PML_KM = float(design["pml_km"])
    PHYSICAL_SIZE_KM = COMPUTATIONAL_SIZE_KM - 2.0 * PML_KM
    CENTER_Y_KM = COMPUTATIONAL_SIZE_KM / 2.0
    PML_POINTS = round(PML_KM / DX_KM)
    return design

DEFAULT_TOTAL_CASES = 500
EXPECTED_COUNTS: dict[str, int] = {}


def split_counts(total_cases: int) -> dict[str, int]:
    """Return stable 70/10/10/10 train/validation/ID-test/OOD-test counts."""
    if total_cases < 500 or total_cases % 500:
        raise ValueError("total cases must be 500 or a larger multiple of 500")
    test_id = total_cases // 10
    test_ood = total_cases // 10
    validation_total = total_cases // 10
    validation_ood = validation_total // 3
    validation_id = validation_total - validation_ood
    train = total_cases - test_id - test_ood - validation_id - validation_ood
    return {
        "train": train,
        "validation_id": validation_id,
        "validation_ood": validation_ood,
        "test_id": test_id,
        "test_ood": test_ood,
    }


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    evaluation_group: str
    boundary_design: str
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


def radical_inverse(index: int, base: int = 2) -> float:
    """Return one element of a deterministic low-discrepancy sequence."""
    value = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


def nested_values(
    count: int,
    lower: float,
    upper: float,
    sequence_start: int,
) -> list[float]:
    """Return a prefix-stable low-discrepancy design on [lower, upper]."""
    width = upper - lower
    return [
        lower + width * radical_inverse(sequence_start + index)
        for index in range(count)
    ]


def symmetric_ood_offsets(
    count: int,
    lower_magnitude: float,
    upper_magnitude: float,
    seed: int,
) -> list[float]:
    magnitude_count = (count + 1) // 2
    magnitudes = nested_values(
        magnitude_count, lower_magnitude, upper_magnitude, seed)
    offsets = []
    for magnitude in magnitudes:
        offsets.extend((-magnitude, magnitude))
    return offsets[:count]


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
            boundary_design=CURRENT_BOUNDARY_DESIGN,
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
            output_prefix=(
                f"fpop_y0_{case_id}" if CURRENT_BOUNDARY_DESIGN == "pml"
                else f"fpop_y0_nopml_{case_id}"
            ),
        ))
    return cases


def build_cases(
    seed: int = DEFAULT_SEED,
    boundary_design: str = "pml",
    total_cases: int = DEFAULT_TOTAL_CASES,
) -> list[Case]:
    global EXPECTED_COUNTS
    configure_boundary_design(boundary_design)
    EXPECTED_COUNTS = split_counts(total_cases)
    # Disjoint sequence ranges make all ID locations unique across splits while
    # preserving exact prefixes as total_cases increases.
    sequence_base = seed * 10
    train = nested_values(EXPECTED_COUNTS["train"], -3.0, 3.0, sequence_base + 10_000)
    validation_id = nested_values(
        EXPECTED_COUNTS["validation_id"], -3.0, 3.0, sequence_base + 1_000_000)
    validation_ood = symmetric_ood_offsets(
        EXPECTED_COUNTS["validation_ood"], 3.0, 4.0, sequence_base + 2_000_000)
    test_id = nested_values(
        EXPECTED_COUNTS["test_id"], -3.0, 3.0, sequence_base + 3_000_000)
    test_ood = symmetric_ood_offsets(
        EXPECTED_COUNTS["test_ood"], 4.0, 5.0, sequence_base + 4_000_000)

    # Required reference and extreme extrapolation cases.
    test_id[0] = 0.0
    test_ood[0] = -5.0
    test_ood[1] = 5.0

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
    expected_total = sum(EXPECTED_COUNTS.values())
    if len(cases) != expected_total:
        raise ValueError(f"expected {expected_total} cases, got {len(cases)}")
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
        if case.evaluation_group == "id" and not -3.0 <= offset <= 3.0:
            raise ValueError(f"ID offset out of range: {case}")
        if case.evaluation_group == "near_ood" and not 3.0 < abs(offset) <= 4.0:
            raise ValueError(f"near-OOD offset out of range: {case}")
        if case.evaluation_group == "far_ood" and not 4.0 < abs(offset) <= 5.0:
            raise ValueError(f"far-OOD offset out of range: {case}")
        objective_y_min = CENTER_Y_KM - OBJECTIVE_SIZE_KM / 2.0
        objective_y_max = CENTER_Y_KM + OBJECTIVE_SIZE_KM / 2.0
        if not objective_y_min <= case.y0_km <= objective_y_max:
            raise ValueError(f"source lies outside objective window: {case}")
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"incorrect split counts: {counts}")

    if not any(case.split == "test" and case.y_offset_km == 0.0 for case in cases):
        raise ValueError("center reference case is missing")
    extremes = {case.y_offset_km for case in cases if case.evaluation_group == "far_ood"}
    if not {-5.0, 5.0}.issubset(extremes):
        raise ValueError("+/-5 km OOD reference cases are missing")

    if CURRENT_BOUNDARY_DESIGN == "extended-nopml":
        # A reflected wave can affect the target as soon as it returns to its
        # source point, which is itself inside the objective window. Check the
        # fastest P wave for every source, including the +/-5 km OOD extremes.
        for case in cases:
            nearest_boundary_km = min(
                COMPUTATIONAL_SIZE_KM / 2.0,  # x direction; x0 is centered
                case.y0_km,
                COMPUTATIONAL_SIZE_KM - case.y0_km,
            )
            reflected_return_s = 2.0 * nearest_boundary_km / 6.0
            if reflected_return_s <= TEND_S:
                raise ValueError(
                    f"domain is not reflection-safe for {case.case_id}: "
                    f"P-wave return={reflected_return_s:.6f} s")


def input_settings(case: Case, output_format: str) -> dict[str, object]:
    streaming = output_format == "hdf5"
    design = BOUNDARY_DESIGNS[case.boundary_design]
    pml_enabled = str(bool(design["pml_enabled"])).lower()
    objective_y_min = CENTER_Y_KM - OBJECTIVE_SIZE_KM / 2.0
    objective_y_max = CENTER_Y_KM + OBJECTIVE_SIZE_KM / 2.0
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
        "output_xlim": "-5,5",
        "output_ylim": f"{objective_y_min:g},{objective_y_max:g}",
        "station_file": "none",
        "station_stride": 1,
        "station_interpolation": "nearest",
        "station_include_fault": "false",
        "iplot": IPLOT,
        "absorbing_method": design["absorbing_method"],
        "pml_width": PML_POINTS,
        "pml_profile_order": 3,
        "pml_reflection_target": "1e-6",
        "pml_max_damping": "auto",
        "boundary_x_left": "absorbing",
        "boundary_x_right": "absorbing",
        "boundary_y_surface": "absorbing",
        "boundary_y_depth": "absorbing",
        "pml_x_left": pml_enabled,
        "pml_x_right": pml_enabled,
        "pml_y_surface": pml_enabled,
        "pml_y_depth": pml_enabled,
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
    ("Computational domain",
     ("x_left_length", "x_right_length", "y_length", "x_left_resolution",
      "x_right_resolution", "y_resolution")),
    ("Gaussian point source", ("simulation_type", "source_type", "M0", "x0", "y0",
                               "t0", "T")),
)


def render_input(case: Case, output_format: str) -> str:
    settings = input_settings(case, output_format)
    if case.boundary_design == "pml":
        domain_comment = (
            "# Total domain: 15x15 km including PML; PML-free interior: 11x11 km")
        boundary_comment = (
            "# Objective window: 10x10 km; PML: 2 km on every outer side")
    else:
        domain_comment = (
            "# Extended no-PML domain: 42x42 km; SAT absorbing outer boundaries")
        boundary_comment = (
            "# Objective window: 10x10 km; worst P reflection return: 5.33 s")
    lines = [
        "# WaveQLab2D FNO/PINO Stage 1 y0 sweep",
        (f"# case_id={case.case_id} split={case.split} "
         f"group={case.evaluation_group} boundary={case.boundary_design}"),
        domain_comment,
        boundary_comment,
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
        expected_nx = round((COMPUTATIONAL_SIZE_KM / 2.0) / DX_KM) + 1
        expected_ny = round(COMPUTATIONAL_SIZE_KM / DX_KM) + 1
        if (params["nx_left"], params["nx_right"], params["ny"]) != (
                expected_nx, expected_nx, expected_ny):
            raise ValueError(f"unexpected grid for {case.case_id}")
        if params["nt"] != 2500 or params["iplot"] != 10:
            raise ValueError(f"unexpected time sampling for {case.case_id}")


MANIFEST_FIELDS = tuple(Case.__dataclass_fields__) + (
    "physical_domain_km", "computational_domain_km", "objective_domain_km",
    "pml_km", "pml_points", "dx_km", "nx_left", "nx_right", "ny",
    "snap_dt_s", "expected_frames", "field_order",
)


def manifest_row(case: Case) -> dict[str, object]:
    nx = round((COMPUTATIONAL_SIZE_KM / 2.0) / DX_KM) + 1
    ny = round(COMPUTATIONAL_SIZE_KM / DX_KM) + 1
    row = asdict(case)
    row.update({
        "physical_domain_km": PHYSICAL_SIZE_KM,
        "computational_domain_km": COMPUTATIONAL_SIZE_KM,
        "objective_domain_km": OBJECTIVE_SIZE_KM,
        "pml_km": PML_KM,
        "pml_points": PML_POINTS,
        "dx_km": DX_KM,
        "nx_left": nx,
        "nx_right": nx,
        "ny": ny,
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
        "boundary_design": CURRENT_BOUNDARY_DESIGN,
        "seed": cases[0].seed - 10,
        "total_cases": len(cases),
        "counts": EXPECTED_COUNTS,
        "output_format": output_format,
        "physical_domain_km": [PHYSICAL_SIZE_KM, PHYSICAL_SIZE_KM],
        "computational_domain_km": [COMPUTATIONAL_SIZE_KM, COMPUTATIONAL_SIZE_KM],
        "objective_window": {
            "x_km": [-5.0, 5.0],
            "y_km": [CENTER_Y_KM - 5.0, CENTER_Y_KM + 5.0],
        },
        "pml_km": PML_KM,
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
    print(f"Boundary design:   {CURRENT_BOUNDARY_DESIGN}")
    print(f"Total cases:       {len(cases)}")
    print(f"Counts:            {counts}")
    print(f"Train y0:          {CENTER_Y_KM-3:g} to {CENTER_Y_KM+3:g} km (center +/-3 km)")
    print(
        f"Far-OOD y0:        {CENTER_Y_KM-5:g} to {CENTER_Y_KM-4:g} and "
        f"{CENTER_Y_KM+4:g} to {CENTER_Y_KM+5:g} km")
    nx = round((COMPUTATIONAL_SIZE_KM / 2.0) / DX_KM) + 1
    ny = round(COMPUTATIONAL_SIZE_KM / DX_KM) + 1
    print(f"Grid:              {nx} + {nx} by {ny}")
    print(f"Steps/frames:      2500 / 250")
    print(f"Output root:       {output_root}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Stage 1 y0-sweep FNO/PINO input decks and manifests.")
    parser.add_argument("--stage", choices=("y0",), default="y0")
    parser.add_argument(
        "--boundary-design", choices=tuple(BOUNDARY_DESIGNS), default="pml",
        help="Use compact four-sided PML or a reflection-safe extended no-PML domain.")
    parser.add_argument(
        "--output-root", type=Path,
        help="Dataset root (default depends on --boundary-design).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--total-cases", type=int, default=DEFAULT_TOTAL_CASES,
        help="Dataset size: 500 or a larger multiple of 500 (default: 500).")
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

    cases = build_cases(args.seed, args.boundary_design, args.total_cases)
    # Render every deck even in dry-run mode to catch missing settings.
    for case in cases:
        rendered = render_input(case, args.format)
        if not rendered.endswith("\n") or "y0 = " not in rendered:
            raise RuntimeError(f"failed to render {case.case_id}")
    if not args.skip_solver_validation:
        validate_with_solver(cases, args.format)

    default_root = (
        ROOT / "fno-pino-benchmark" /
        str(BOUNDARY_DESIGNS[args.boundary_design]["directory"])
    )
    output_root = (args.output_root or default_root).expanduser().resolve()
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
