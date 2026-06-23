#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from rigcrop.data import RIGPairwiseDataset  # noqa: E402
from rigcrop.losses import graph_supervision_loss, pairwise_crop_loss, utility_distillation_loss  # noqa: E402
from rigcrop.model import RIGCropModel  # noqa: E402
from rigcrop.runtime import AverageMeter, get_device, load_config, save_checkpoint, set_seed, write_json  # noqa: E402


def _build_train_dataset(cfg: Dict[str, Any]):
    if cfg.get("train_datasets"):
        return ConcatDataset([RIGPairwiseDataset(**item) for item in cfg["train_datasets"]])
    return RIGPairwiseDataset(**cfg["train_dataset"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RIG-Crop with crop preference and middle-state supervision.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    distributed = _init_distributed()
    rank = _rank()
    is_main = rank == 0
    device = _distributed_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = _build_train_dataset(cfg)
    val_ds = RIGPairwiseDataset(**cfg["val_dataset"]) if cfg.get("val_dataset", {}).get("jsonl_path") else None
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 16)), shuffle=False, num_workers=int(cfg.get("num_workers", 0))) if val_ds else None

    model = RIGCropModel(**cfg.get("model", {})).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[_local_rank()] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    loss_weights = cfg.get("loss", {})
    best_acc = -1.0
    history = []
    if is_main:
        print(
            f"[rig-train] train_pairs={len(train_ds)} val_pairs={len(val_ds) if val_ds else 0} "
            f"device={device} world_size={_world_size()}",
            flush=True,
        )
    for epoch in range(1, int(cfg.get("epochs", 10)) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_logs = run_epoch(model, train_loader, optimizer, device, loss_weights)
        train_logs = _reduce_logs(train_logs) if distributed else train_logs
        metrics = {"epoch": epoch, **{f"train_{k}": v for k, v in train_logs.items()}}
        if val_loader and is_main:
            val_logs = run_eval(_unwrap(model), val_loader, device, loss_weights)
            metrics.update({f"val_{k}": v for k, v in val_logs.items()})
            if val_logs["pairwise_acc"] > best_acc:
                best_acc = val_logs["pairwise_acc"]
                save_checkpoint(out_dir / "best.pt", _unwrap(model), optimizer, epoch, metrics, cfg)
        elif not val_loader and is_main:
            save_checkpoint(out_dir / "best.pt", _unwrap(model), optimizer, epoch, metrics, cfg)
        if is_main:
            save_checkpoint(out_dir / "last.pt", _unwrap(model), optimizer, epoch, metrics, cfg)
            history.append(metrics)
            write_json(out_dir / "history.json", history)
            print(metrics, flush=True)
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


def run_epoch(model: RIGCropModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, loss_weights: Dict[str, Any]) -> Dict[str, float]:
    model.train()
    meters = _meters()
    for batch in loader:
        batch = _to_device(batch, device)
        winner = model(batch["image"], batch["winner_crop"], batch["winner_box_feat"])
        loser = model(batch["image"], batch["loser_crop"], batch["loser_box_feat"])
        losses = _losses(winner, loser, batch, loss_weights)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        _update_meters(meters, losses, winner["score"] - loser["score"], batch["image"].size(0))
    return {key: meter.avg for key, meter in meters.items()}


@torch.no_grad()
def run_eval(model: RIGCropModel, loader: DataLoader, device: torch.device, loss_weights: Dict[str, Any]) -> Dict[str, float]:
    model.eval()
    meters = _meters()
    for batch in loader:
        batch = _to_device(batch, device)
        winner = model(batch["image"], batch["winner_crop"], batch["winner_box_feat"])
        loser = model(batch["image"], batch["loser_crop"], batch["loser_box_feat"])
        losses = _losses(winner, loser, batch, loss_weights)
        _update_meters(meters, losses, winner["score"] - loser["score"], batch["image"].size(0))
    return {key: meter.avg for key, meter in meters.items()}


def _losses(winner: Dict[str, torch.Tensor], loser: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], weights: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    crop = pairwise_crop_loss(winner["score"], loser["score"], batch["weight"])
    graph_items = graph_supervision_loss(winner, batch)
    utility = utility_distillation_loss(winner["utility"], loser["utility"], batch["winner_utility"], batch["loser_utility"])
    node = graph_items["node_bbox"] + graph_items["node_role"] + graph_items["node_importance"] + graph_items["node_valid"]
    relation = graph_items["relation_policy"] + graph_items["relation_weight"]
    action = graph_items["action"]
    node_weight = float(weights.get("node", 0.3))
    relation_weight = float(weights.get("relation", 0.2))
    node_weighted = (
        float(weights.get("node_bbox", node_weight)) * graph_items["node_bbox"]
        + float(weights.get("node_role", node_weight)) * graph_items["node_role"]
        + float(weights.get("node_importance", node_weight)) * graph_items["node_importance"]
        + float(weights.get("node_valid", node_weight)) * graph_items["node_valid"]
    )
    relation_weighted = (
        float(weights.get("relation_policy", relation_weight)) * graph_items["relation_policy"]
        + float(weights.get("relation_weight", relation_weight)) * graph_items["relation_weight"]
    )
    total = (
        float(weights.get("crop_pair", 1.0)) * crop
        + node_weighted
        + relation_weighted
        + float(weights.get("utility", 0.3)) * utility
        + float(weights.get("action", 0.05)) * action
    )
    return {"total": total, "crop": crop, "node": node, "relation": relation, "utility": utility, "action": action}


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _meters() -> Dict[str, AverageMeter]:
    return {name: AverageMeter() for name in ["loss", "crop_loss", "node_loss", "relation_loss", "utility_loss", "action_loss", "pairwise_acc", "score_margin"]}


def _update_meters(meters: Dict[str, AverageMeter], losses: Dict[str, torch.Tensor], margin: torch.Tensor, n: int) -> None:
    meters["loss"].update(float(losses["total"].detach().cpu()), n)
    meters["crop_loss"].update(float(losses["crop"].detach().cpu()), n)
    meters["node_loss"].update(float(losses["node"].detach().cpu()), n)
    meters["relation_loss"].update(float(losses["relation"].detach().cpu()), n)
    meters["utility_loss"].update(float(losses["utility"].detach().cpu()), n)
    meters["action_loss"].update(float(losses["action"].detach().cpu()), n)
    meters["pairwise_acc"].update(float((margin > 0).float().mean().detach().cpu()), n)
    meters["score_margin"].update(float(margin.mean().detach().cpu()), n)


def _init_distributed() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(_local_rank())
    return True


def _distributed_device(device_name: str) -> torch.device:
    if _world_size() > 1 and torch.cuda.is_available():
        return torch.device("cuda", _local_rank())
    return get_device(device_name)


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _reduce_logs(logs: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(logs)
    tensor = torch.tensor([float(logs[k]) for k in keys], dtype=torch.float32, device=_distributed_device("auto"))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= max(_world_size(), 1)
    return {key: float(value) for key, value in zip(keys, tensor.detach().cpu().tolist())}


if __name__ == "__main__":
    main()
