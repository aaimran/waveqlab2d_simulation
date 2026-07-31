#!/usr/bin/env python3
"""Create the WaveQLab2D CPU/CUDA environment for Punakha simulations.

Expected DGX layout::

    /scratch/aimran/waveqlab2d_0/          # solver checkout
    /scratch/aimran/waveqlab2d_simulation/ # this script and benchmark data

The script is idempotent. It reuses an existing virtual environment and
synchronizes the editable solver installation without deleting user data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


SIMULATION_ROOT = Path(__file__).resolve().parent
DEFAULT_SOLVER = SIMULATION_ROOT.parent / "waveqlab2d_0"


def environment_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def run(command: list[str], *, check: bool = True) -> int:
    print("+", subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, check=check).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a local CPU/CUDA environment for WaveQLab2D.")
    parser.add_argument(
        "--venv",
        type=Path,
        default=SIMULATION_ROOT / "venv",
        help="Environment directory (default: waveqlab2d_simulation/venv).",
    )
    parser.add_argument(
        "--solver",
        type=Path,
        default=DEFAULT_SOLVER,
        help="WaveQLab2D source checkout (default: sibling waveqlab2d_0).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to create the environment.",
    )
    parser.add_argument(
        "--no-upgrade-tools",
        action="store_true",
        help="Do not upgrade pip, setuptools, and wheel.",
    )
    args = parser.parse_args()

    venv = args.venv.expanduser().resolve()
    solver = args.solver.expanduser().resolve()
    pyproject = solver / "pyproject.toml"

    if not pyproject.is_file():
        parser.error(
            f"WaveQLab2D pyproject.toml not found at {pyproject}; "
            "pass the checkout with --solver")
    if sys.version_info < (3, 10):
        parser.error("WaveQLab2D requires Python 3.10 or newer")

    venv_python = environment_python(venv)
    if not venv_python.is_file():
        if venv.exists() and any(venv.iterdir()):
            parser.error(
                f"directory exists but is not a usable virtual environment: {venv}")
        venv.parent.mkdir(parents=True, exist_ok=True)
        print(f"Creating virtual environment: {venv}")
        run([args.python, "-m", "venv", str(venv)])
    else:
        print(f"Reusing virtual environment: {venv}")

    if not args.no_upgrade_tools:
        run([
            str(venv_python), "-m", "pip", "install", "--upgrade",
            "pip", "setuptools", "wheel",
        ])

    # The cuda extra installs core NumPy/Numba dependencies plus numba-cuda.
    run([str(venv_python), "-m", "pip", "install", "-e", f"{solver}[cuda]"])

    print("\nChecking CPU imports...")
    run([
        str(venv_python), "-c",
        "import numpy, numba; "
        "print('NumPy', numpy.__version__); "
        "print('Numba', numba.__version__)",
    ])

    print("\nChecking CUDA visibility (non-fatal during environment creation)...")
    cuda_status = run([
        str(venv_python), "-c",
        "from numba import cuda; "
        "print('CUDA available:', cuda.is_available()); "
        "print('GPU:', cuda.get_current_device().name if cuda.is_available() else 'not visible')",
    ], check=False)
    if cuda_status:
        print(
            "WARNING: CUDA probe failed. The CPU environment is usable, but run "
            "inside a SLURM GPU allocation before benchmarking the H100.",
            file=sys.stderr,
        )

    print("\nEnvironment ready.")
    print(f"Python:   {venv_python}")
    print(f"Solver:   {solver}")
    print(f"Activate: source {venv / 'bin' / 'activate'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(f"ERROR: command failed with exit code {error.returncode}", file=sys.stderr)
        raise SystemExit(error.returncode)
