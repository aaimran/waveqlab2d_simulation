#!/usr/bin/env python3
"""Phase 04: run a bounded FNO hyperparameter search and select by validation error."""

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("dataset_root",type=Path)
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda")
    p.add_argument("--epochs",type=int,default=8); p.add_argument("--steps-per-epoch",type=int,default=250)
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    grid=list(itertools.product((24,32,48),(12,16,20),(5e-4,1e-3)))
    results=[]
    for index,(width,modes,lr) in enumerate(grid,1):
        run=args.output_dir/f"trial_{index:02d}_w{width}_m{modes}_lr{lr:g}"
        command=[sys.executable,str(Path(__file__).with_name("phase_03.py")),str(args.dataset_root),
                 "--output-dir",str(run),"--device",args.device,"--epochs",str(args.epochs),
                 "--steps-per-epoch",str(args.steps_per_epoch),"--width",str(width),
                 "--modes",str(modes),"--learning-rate",str(lr)]
        print("+"," ".join(command),flush=True); subprocess.run(command,check=True)
        score=json.loads((run/"summary.json").read_text())["best_validation_relative_l2"]
        results.append(dict(trial=index,width=width,modes=modes,learning_rate=lr,score=score,run=str(run)))
    results.sort(key=lambda row:row["score"])
    (args.output_dir/"search_results.json").write_text(json.dumps(results,indent=2)+"\n")
    (args.output_dir/"best_trial.json").write_text(json.dumps(results[0],indent=2)+"\n")
    print(json.dumps(results[0],indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
