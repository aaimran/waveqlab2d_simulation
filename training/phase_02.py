#!/usr/bin/env python3
"""Phase 02: compute leakage-free train-only normalization and metadata."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from phase_common import available_rows, prepared_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    train = available_rows(root, "train")
    if not train:
        raise RuntimeError("no prepared training cases; run phase_01.py first")
    maximum = np.zeros(2, dtype=np.float64)
    energy = np.zeros(2, dtype=np.float64)
    count = 0
    for number, row in enumerate(train, 1):
        with h5py.File(prepared_path(root, row), "r") as archive:
            values = archive["velocity"]
            for start in range(0, values.shape[0], 10):
                block = values[start:start+10].astype(np.float64)
                maximum = np.maximum(maximum, np.max(np.abs(block), axis=(0,1,2)))
                energy += np.sum(block**2, axis=(0,1,2)); count += np.prod(block.shape[:3])
            if number == 1:
                x, y, time = archive["x"][:], archive["y"][:], archive["time"][:]
    rms = np.sqrt(energy / count)
    # A quiescent component must not cause division by zero downstream.
    scale = np.maximum(np.maximum(maximum, 5*rms), 1e-12).astype(np.float32)
    metadata = {
        "velocity_scale": scale.tolist(), "velocity_max_abs": maximum.tolist(),
        "velocity_rms": rms.tolist(), "x": x.tolist(), "y": y.tolist(),
        "time": time.tolist(), "dx_km": float(x[1]-x[0]),
        "dy_km": float(y[1]-y[0]), "saved_dt_s": float(time[1]-time[0]),
        "center_y_km": float((y[0]+y[-1])/2), "train_cases": len(train),
        "normalization_source": "training split only", "field_order": ["vx","vy"],
    }
    path = root / "prepared" / "normalization.json"
    path.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
