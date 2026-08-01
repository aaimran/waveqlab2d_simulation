#!/usr/bin/env python3
"""Phase 06: matched FNO/PINO ID/OOD accuracy and inference benchmark."""

import argparse
import csv
import json
import math
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from phase_common import (available_rows, load_metadata, load_model, model_input,
                          prepared_path)


@torch.no_grad()
def evaluate(name, checkpoint, root, selected, meta, device, batch_size):
    model,_=load_model(checkpoint,device); model.eval()
    scale=torch.tensor(meta["velocity_scale"],device=device).view(1,2,1,1)
    records=[]; total_error=total_truth=0.0; inference=0.0
    for number,row in enumerate(selected,1):
        with h5py.File(prepared_path(root,row),"r") as archive:
            times=archive["time"][:]; truth=archive["velocity"][:].astype(np.float32)
        error_sq=truth_sq=0.0; peak_error=0.0
        for start in range(0,len(times),batch_size):
            stop=min(start+batch_size,len(times))
            inputs=torch.from_numpy(np.asarray([
                model_input(row,float(value),meta) for value in times[start:stop]])).to(device)
            if device.type=="cuda": torch.cuda.synchronize()
            began=time.perf_counter(); prediction=model(inputs)*scale
            if device.type=="cuda": torch.cuda.synchronize()
            inference+=time.perf_counter()-began
            predicted=prediction.cpu().numpy().transpose(0,2,3,1)
            target=truth[start:stop]; error_sq+=float(np.sum((predicted-target)**2,dtype=np.float64))
            truth_sq+=float(np.sum(target**2,dtype=np.float64))
            peak_error=max(peak_error,float(np.max(np.abs(predicted-target))))
        relative=math.sqrt(error_sq/max(truth_sq,1e-30)); total_error+=error_sq; total_truth+=truth_sq
        records.append(dict(model=name,case_id=row["case_id"],group=row["evaluation_group"],
                            y0_km=row["y0_km"],relative_l2=relative,max_abs_error=peak_error))
        print(f"[{name} {number:03d}/{len(selected):03d}] {row['case_id']} relL2={relative:.4e}",flush=True)
    return records,dict(model=name,cases=len(selected),global_relative_l2=math.sqrt(total_error/total_truth),
                        inference_seconds=inference,frames=sum(1 for _ in selected)*len(meta["time"]),
                        milliseconds_per_frame=1000*inference/(len(selected)*len(meta["time"])))


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("dataset_root",type=Path)
    p.add_argument("--fno",type=Path,required=True); p.add_argument("--pino",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda")
    p.add_argument("--batch-size",type=int,default=16); args=p.parse_args()
    root=args.dataset_root.resolve(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    selected=available_rows(root,"test"); meta=load_metadata(root); device=torch.device(args.device)
    all_records=[]; summaries=[]
    for name,path in (("FNO",args.fno),("PINO",args.pino)):
        records,summary=evaluate(name,path,root,selected,meta,device,args.batch_size)
        all_records+=records; summaries.append(summary)
    with (out/"per_case.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=all_records[0].keys()); writer.writeheader(); writer.writerows(all_records)
    grouped={}
    for model in ("FNO","PINO"):
        for group in ("id","far_ood"):
            values=[row["relative_l2"] for row in all_records if row["model"]==model and row["group"]==group]
            grouped[f"{model}_{group}"]={"cases":len(values),"median_relative_l2":float(np.median(values)),
                                         "p90_relative_l2":float(np.percentile(values,90))}
    report={"global":summaries,"groups":grouped}
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
