#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from dacc.data import DACCGeneratorDataset
from dacc.losses import dacc_loss
from dacc.metrics import acc_at_iou, top1_iou
from dacc.models import DACCNet
from dacc.utils import AverageMeter, get_device, load_config, save_checkpoint, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/daccnet_small.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DACCGeneratorDataset(**cfg["train_dataset"])
    val_ds = DACCGeneratorDataset(**cfg["val_dataset"]) if cfg.get("val_dataset", {}).get("jsonl_path") else None
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 0)) if val_ds else None

    model = DACCNet(**cfg.get("model", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    best_metric = -1.0
    history = []
    weights = cfg.get("loss_weights", {})

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_logs = train_one_epoch(model, train_loader, optimizer, device, weights)
        metrics = {"epoch": epoch, **{f"train_{k}": v for k, v in train_logs.items()}}
        if val_loader:
            val_logs = evaluate(model, val_loader, device, weights)
            metrics.update({f"val_{k}": v for k, v in val_logs.items()})
            if val_logs["top1_iou"] > best_metric:
                best_metric = val_logs["top1_iou"]
                save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, metrics)
        else:
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, metrics)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, metrics)
        history.append(metrics)
        print(metrics)
    write_json(out_dir / "history.json", history)


def train_one_epoch(model, loader, optimizer, device, weights):
    model.train()
    meters = {}
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["image"], batch["aspect"])
        loss, logs = dacc_loss(outputs, batch, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for k, v in logs.items():
            meters.setdefault(k, AverageMeter()).update(v, batch["image"].size(0))
    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def evaluate(model, loader, device, weights):
    model.eval()
    meters = {}
    ious = AverageMeter()
    acc75 = AverageMeter()
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["image"], batch["aspect"])
        loss, logs = dacc_loss(outputs, batch, weights)
        for k, v in logs.items():
            meters.setdefault(k, AverageMeter()).update(v, batch["image"].size(0))
        ious.update(top1_iou(outputs["boxes"], batch["target_boxes"], batch["target_mask"]), batch["image"].size(0))
        acc75.update(acc_at_iou(outputs["boxes"], batch["target_boxes"], batch["target_mask"], 0.75), batch["image"].size(0))
    out = {k: m.avg for k, m in meters.items()}
    out["top1_iou"] = ious.avg
    out["acc_iou75"] = acc75.avg
    return out


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


if __name__ == "__main__":
    main()
