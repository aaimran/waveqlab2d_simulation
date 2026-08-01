#!/usr/bin/env python3
"""Phase 05: warm-start from FNO and fine-tune a velocity PINO."""

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from phase_common import (append_jsonl, available_rows, load_metadata, load_model,
                          now, random_batch, save_checkpoint, supervised_loss,
                          validation_relative_l2)


def navier(pred, rows, meta, scale):
    batch=len(rows); fields=pred.reshape(batch,3,2,*pred.shape[-2:])*scale[:,None]
    previous,current,following=fields[:,0],fields[:,1],fields[:,2]
    dt=float(meta["saved_dt_s"]); dx=float(meta["dx_km"]); dy=float(meta["dy_km"])
    acceleration=(following-2*current+previous)/dt**2
    dxx=(current[...,2:,1:-1]-2*current[...,1:-1,1:-1]+current[...,:-2,1:-1])/dx**2
    dyy=(current[...,1:-1,2:]-2*current[...,1:-1,1:-1]+current[...,1:-1,:-2])/dy**2
    dxy=(current[...,2:,2:]-current[...,2:,:-2]-current[...,:-2,2:]+current[...,:-2,:-2])/(4*dx*dy)
    cp,cs=6.0,3.464; coupling=cp**2-cs**2
    rhs=torch.stack((cp**2*dxx[:,0]+cs**2*dyy[:,0]+coupling*dxy[:,1],
                     cs**2*dxx[:,1]+cp**2*dyy[:,1]+coupling*dxy[:,0]),dim=1)
    residual=acceleration[...,1:-1,1:-1]-rhs
    x=np.asarray(meta["x"])[1:-1]; y=np.asarray(meta["y"])[1:-1]
    xx,yy=np.meshgrid(x,y,indexing="ij"); masks=[]
    for row in rows:
        radius=np.sqrt((xx-float(row["x0_km"]))**2+(yy-float(row["y0_km"]))**2)
        masks.append(radius>=0.5)
    mask=torch.from_numpy(np.asarray(masks)).to(residual.device)[:,None].expand_as(residual)
    return torch.mean((residual[mask]/(scale.mean()*dt**-2))**2)


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("dataset_root",type=Path); p.add_argument("fno_checkpoint",type=Path)
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda")
    p.add_argument("--epochs",type=int,default=15); p.add_argument("--steps-per-epoch",type=int,default=300)
    p.add_argument("--batch-size",type=int,default=2); p.add_argument("--learning-rate",type=float,default=1e-4)
    p.add_argument("--physics-weight",type=float,default=0.01); p.add_argument("--seed",type=int,default=20260731)
    args=p.parse_args(); root=args.dataset_root.resolve(); run=args.output_dir.resolve(); run.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device); model,state=load_model(args.fno_checkpoint,device)
    meta=load_metadata(root); train=available_rows(root,"train"); val=available_rows(root,"validation")
    scale=torch.tensor(meta["velocity_scale"],device=device).view(1,2,1,1); rng=random.Random(args.seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-5); best=math.inf
    for epoch in range(args.epochs):
        model.train(); dl=[]; pl=[]; started=now()
        for _ in range(args.steps_per_epoch):
            chosen,inputs,target,_=random_batch(root,train,meta,args.batch_size,rng,triplets=True)
            inputs,target=inputs.to(device),target.to(device); optimizer.zero_grad(set_to_none=True)
            predicted=model(inputs); current=predicted.reshape(args.batch_size,3,2,*predicted.shape[-2:])[:,1]
            data=supervised_loss(current,target); physics=navier(predicted,chosen,meta,scale)
            loss=data+args.physics_weight*physics; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step(); dl.append(float(data.detach().cpu())); pl.append(float(physics.detach().cpu()))
        score=validation_relative_l2(model,root,val,meta,device)
        record=dict(epoch=epoch+1,data_loss=float(np.mean(dl)),physics_loss=float(np.mean(pl)),
                    validation_relative_l2=score,elapsed_s=now()-started)
        append_jsonl(run/"history.jsonl",record); print(json.dumps(record),flush=True)
        extra=dict(epoch=epoch,metadata=meta,args=vars(args),validation=score,warm_start=str(args.fno_checkpoint))
        save_checkpoint(run/"last.pt",model,**extra)
        if score<best: best=score; save_checkpoint(run/"best.pt",model,**extra)
    (run/"summary.json").write_text(json.dumps({"best_validation_relative_l2":best},indent=2)+"\n"); return 0


if __name__ == "__main__": raise SystemExit(main())
