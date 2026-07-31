#!/usr/bin/env python3
"""Run the installed WaveQLab2D solver using this directory as runtime root.

Place this file in /scratch/aimran/waveqlab2d_simulation. Input paths are
resolved from the current shell directory, while log and output directories
are always created beside this script.
"""

from pathlib import Path
import sys

import cli_2d


SIMULATION_ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    raise SystemExit(
        cli_2d.main(argv=sys.argv[1:], root_dir=str(SIMULATION_ROOT))
    )
