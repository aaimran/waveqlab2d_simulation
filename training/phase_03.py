#!/usr/bin/env python3
"""Phase 03: train the supervised time-conditioned FNO baseline."""

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from phase_common import (FNO2d, append_jsonl, available_rows, load_metadata,
                          now, random_batch, save_checkpoint, supervised_loss,
                          validation_relative_l2)


def train(args: argparse.Namespace) -> dict[str, float]:
    root, run = args.dataset_root.resolve(), args.output_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); rng = random.Random(args.seed)
    device = torch.device(args.device)
    metadata = load_metadata(root)
    train_rows = available_rows(root, "train")
    validation = available_rows(root, "validation")
    model = FNO2d(args.width, args.modes, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best = math.inf
    for epoch in range(args.epochs):
        model.train(); losses=[]; started=now()
        for _ in range(args.steps_per_epoch):
            _, inputs, targets, _ = random_batch(
                root, train_rows, metadata, args.batch_size, rng)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_loss(model(inputs), targets)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        score = validation_relative_l2(model, root, validation, metadata, device)
        scheduler.step()
        record = dict(epoch=epoch+1, train_loss=float(np.mean(losses)),
                      validation_relative_l2=score, learning_rate=optimizer.param_groups[0]["lr"],
                      elapsed_s=now()-started)
        append_jsonl(run/"history.jsonl", record); print(json.dumps(record), flush=True)
        state = dict(epoch=epoch, metadata=metadata, args=vars(args), validation=score)
        save_checkpoint(run/"last.pt", model, **state)
        if score < best:
            best=score; save_checkpoint(run/"best.pt", model, **state)
    summary={"best_validation_relative_l2":best}
    (run/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(); p.add_argument("dataset_root",type=Path)
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda")
    p.add_argument("--epochs",type=int,default=30); p.add_argument("--steps-per-epoch",type=int,default=500)
    p.add_argument("--batch-size",type=int,default=4); p.add_argument("--width",type=int,default=32)
    p.add_argument("--modes",type=int,default=16); p.add_argument("--layers",type=int,default=4)
    p.add_argument("--learning-rate",type=float,default=1e-3)
    p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--seed",type=int,default=20260731)
    return p


if __name__ == "__main__":
    raise SystemExit(0 if train(parser().parse_args()) else 1)
