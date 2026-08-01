#!/usr/bin/env python3
"""Phase 01: validate raw HDF5 and extract compact vx/vy prepared cases."""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from phase_common import rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    selected = rows(root)[:args.limit] if args.limit else rows(root)
    if not selected:
        raise RuntimeError("manifest contains no selected cases")
    index_rows = []
    reference = None
    for number, row in enumerate(selected, 1):
        matches = sorted((root / "raw" / row["split"]).glob(f"{row['output_prefix']}_*.h5"))
        if len(matches) != 1:
            raise RuntimeError(f"{row['case_id']}: expected one raw HDF5, found {len(matches)}")
        destination = root / "prepared" / row["split"] / f"{row['case_id']}.h5"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not args.overwrite:
            index_rows.append(dict(row, prepared_file=str(destination.relative_to(root))))
            continue
        with h5py.File(matches[0], "r") as source:
            left = source["DomainOutput_l"][:-1, :, :, :2]
            right = source["DomainOutput_r"][:, :, :, :2]
            velocity = np.concatenate((left, right), axis=0).transpose(2, 0, 1, 3)
            x = np.concatenate((source["x_l"][:-1], source["x_r"][:])).astype(np.float32)
            y = source["y_fault"][:].astype(np.float32)
            time = source["snap_times"][:].astype(np.float32)
        if velocity.shape != (250, 101, 101, 2) or not np.all(np.isfinite(velocity)):
            raise RuntimeError(f"{row['case_id']}: invalid velocity shape/data {velocity.shape}")
        coordinates = (x.tolist(), y.tolist(), time.tolist())
        if reference is None:
            reference = coordinates
        elif coordinates != reference:
            raise RuntimeError(f"{row['case_id']}: coordinate mismatch")
        with h5py.File(destination, "w") as target:
            target.create_dataset("velocity", data=velocity, chunks=(1,101,101,2),
                                  compression="gzip", compression_opts=4, shuffle=True)
            target.create_dataset("x", data=x)
            target.create_dataset("y", data=y)
            target.create_dataset("time", data=time)
            target.attrs["case"] = json.dumps(row)
        index_rows.append(dict(row, prepared_file=str(destination.relative_to(root))))
        print(f"[{number:04d}/{len(selected):04d}] {row['case_id']}", flush=True)
    index_path = root / "prepared" / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=index_rows[0].keys())
        writer.writeheader(); writer.writerows(index_rows)
    print(f"Prepared {len(index_rows)} cases: {root/'prepared'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
