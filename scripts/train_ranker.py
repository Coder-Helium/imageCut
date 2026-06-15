#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from dacc.data import CropRankerDataset
from dacc.losses import ranker_loss
from dacc.models import CropRanker
from dacc.utils import AverageMeter, get_device, load_config, save_checkpoint, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ranker_small.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CropRankerDataset(**cfg["train_dataset"])
    val_ds = CropRankerDataset(**cfg["val_dataset"]) if cfg.get("val_dataset", {}).get("jsonl_path") else None
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 0)) if val_ds else None

    model = CropRanker(**cfg.get("model", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    best_val = float("inf")
    history = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        metrics = {"epoch": epoch, "train_loss": train_loss}
        if val_loader:
            val_loss = run_eval(model, val_loader, device)
            metrics["val_loss"] = val_loss
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, metrics)
        else:
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, metrics)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, metrics)
        history.append(metrics)
        print(metrics)
    write_json(out_dir / "history.json", history)


def run_epoch(model, loader, optimizer, device) -> float:
    model.train()
    meter = AverageMeter()
    for batch in loader:
        image = batch["image"].to(device)
        crop = batch["crop"].to(device)
        box_feat = batch["box_feat"].to(device)
        target = batch["score"].to(device)
        pred = model(image, crop, box_feat)
        loss = ranker_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        meter.update(float(loss.detach().cpu()), image.size(0))
    return meter.avg


@torch.no_grad()
def run_eval(model, loader, device) -> float:
    model.eval()
    meter = AverageMeter()
    for batch in loader:
        pred = model(batch["image"].to(device), batch["crop"].to(device), batch["box_feat"].to(device))
        loss = ranker_loss(pred, batch["score"].to(device))
        meter.update(float(loss.detach().cpu()), pred.size(0))
    return meter.avg


if __name__ == "__main__":
    main()
