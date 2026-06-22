#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dacc.data import PairwiseCropRankerDataset
from dacc.models import CropRanker
from dacc.utils import AverageMeter, get_device, load_config, save_checkpoint, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crop ranker with pairwise preferences such as CPC.")
    parser.add_argument("--config", default="configs/ranker_cpc_pairwise.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PairwiseCropRankerDataset(**cfg["train_dataset"])
    val_ds = PairwiseCropRankerDataset(**cfg["val_dataset"]) if cfg.get("val_dataset", {}).get("jsonl_path") else None
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 0)) if val_ds else None

    model = CropRanker(**cfg.get("model", {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    best_acc = -1.0
    history = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_logs = run_epoch(model, train_loader, optimizer, device)
        metrics = {"epoch": epoch, **{f"train_{k}": v for k, v in train_logs.items()}}
        if val_loader:
            val_logs = run_eval(model, val_loader, device)
            metrics.update({f"val_{k}": v for k, v in val_logs.items()})
            if val_logs["pairwise_acc"] > best_acc:
                best_acc = val_logs["pairwise_acc"]
                save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, metrics)
        else:
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, metrics)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, metrics)
        history.append(metrics)
        print(metrics)
    write_json(out_dir / "history.json", history)


def pairwise_loss(winner_scores: torch.Tensor, loser_scores: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight / weight.mean().clamp(min=1e-6)
    return (F.softplus(-(winner_scores - loser_scores)) * weight).mean()


def run_epoch(model, loader, optimizer, device) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    margin_meter = AverageMeter()
    for batch in loader:
        image = batch["image"].to(device)
        winner = batch["winner_crop"].to(device)
        loser = batch["loser_crop"].to(device)
        winner_feat = batch["winner_box_feat"].to(device)
        loser_feat = batch["loser_box_feat"].to(device)
        weight = batch["weight"].to(device)
        winner_score = model(image, winner, winner_feat)
        loser_score = model(image, loser, loser_feat)
        loss = pairwise_loss(winner_score, loser_score, weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        margin = winner_score - loser_score
        loss_meter.update(float(loss.detach().cpu()), image.size(0))
        acc_meter.update(float((margin > 0).float().mean().detach().cpu()), image.size(0))
        margin_meter.update(float(margin.mean().detach().cpu()), image.size(0))
    return {"loss": loss_meter.avg, "pairwise_acc": acc_meter.avg, "score_margin": margin_meter.avg}


@torch.no_grad()
def run_eval(model, loader, device) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    margin_meter = AverageMeter()
    for batch in loader:
        image = batch["image"].to(device)
        winner = batch["winner_crop"].to(device)
        loser = batch["loser_crop"].to(device)
        winner_feat = batch["winner_box_feat"].to(device)
        loser_feat = batch["loser_box_feat"].to(device)
        weight = batch["weight"].to(device)
        winner_score = model(image, winner, winner_feat)
        loser_score = model(image, loser, loser_feat)
        loss = pairwise_loss(winner_score, loser_score, weight)
        margin = winner_score - loser_score
        loss_meter.update(float(loss.detach().cpu()), image.size(0))
        acc_meter.update(float((margin > 0).float().mean().detach().cpu()), image.size(0))
        margin_meter.update(float(margin.mean().detach().cpu()), image.size(0))
    return {"loss": loss_meter.avg, "pairwise_acc": acc_meter.avg, "score_margin": margin_meter.avg}


if __name__ == "__main__":
    main()
