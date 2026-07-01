#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Subset

from rigcrop.data import RIGPairwiseDataset  # noqa: E402
from rigcrop.losses import graph_supervision_loss, pairwise_crop_loss, query_proposal_loss, utility_distillation_loss  # noqa: E402
from rigcrop.model import RIGCropModel  # noqa: E402
from rigcrop.runtime import AverageMeter, get_device, load_checkpoint, load_config, save_checkpoint, set_seed, write_json  # noqa: E402


def _build_train_dataset(cfg: Dict[str, Any]):
    if cfg.get("train_datasets"):
        return ConcatDataset([RIGPairwiseDataset(**item) for item in cfg["train_datasets"]])
    return RIGPairwiseDataset(**cfg["train_dataset"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RIG-Crop with crop preference and middle-state supervision.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="", help="Optional checkpoint path to resume from.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    distributed = _init_distributed(timeout_seconds=int(cfg.get("ddp_timeout_seconds", 7200)))
    rank = _rank()
    is_main = rank == 0
    device = _distributed_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["output_dir"])
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = _build_train_dataset(cfg)
    val_ds = RIGPairwiseDataset(**cfg["val_dataset"]) if cfg.get("val_dataset", {}).get("jsonl_path") else None
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_eval_ds = Subset(val_ds, _rank_indices(len(val_ds))) if distributed and val_ds is not None else val_ds
    num_workers = int(cfg.get("num_workers", 0))
    persistent_workers = bool(cfg.get("persistent_workers", num_workers > 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=int(cfg.get("prefetch_factor", 2)) if num_workers > 0 else None,
    )
    val_loader = (
        DataLoader(
            val_eval_ds,
            batch_size=int(cfg.get("val_batch_size", cfg.get("batch_size", 16))),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=int(cfg.get("prefetch_factor", 2)) if num_workers > 0 else None,
        )
        if val_ds
        else None
    )

    model = RIGCropModel(**cfg.get("model", {})).to(device)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[_local_rank()] if device.type == "cuda" else None,
            find_unused_parameters=bool(cfg.get("find_unused_parameters", False)),
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    loss_weights = cfg.get("loss", {})
    best_acc = -1.0
    history = []
    start_epoch = 1
    resume_path = _resolve_resume_path(args.resume, cfg, out_dir)
    if resume_path:
        ckpt = load_checkpoint(resume_path, _unwrap(model), optimizer)
        _move_optimizer_state(optimizer, device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        history = _load_history(out_dir)
        best_acc = _best_acc_from_history(history, ckpt)
        if is_main:
            print(f"[rig-train] resumed checkpoint={resume_path} start_epoch={start_epoch} best_acc={best_acc:.6f}", flush=True)
    if is_main:
        print(
            f"[rig-train] train_pairs={len(train_ds)} val_pairs={len(val_ds) if val_ds else 0} "
            f"device={device} world_size={_world_size()}",
            flush=True,
        )
    for epoch in range(start_epoch, int(cfg.get("epochs", 10)) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_logs, train_count = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_weights,
            epoch=epoch,
            log_interval=int(cfg.get("log_interval", 1000)),
            is_main=is_main,
        )
        train_logs = _reduce_logs(train_logs, train_count) if distributed else train_logs
        metrics = {"epoch": epoch, **{f"train_{k}": v for k, v in train_logs.items()}}
        if val_loader:
            val_logs, val_count = run_eval(model, val_loader, device, loss_weights)
            val_logs = _reduce_logs(val_logs, val_count) if distributed else val_logs
            if is_main:
                metrics.update({f"val_{k}": v for k, v in val_logs.items()})
                if val_logs["pairwise_acc"] > best_acc:
                    best_acc = val_logs["pairwise_acc"]
                    save_checkpoint(out_dir / "best.pt", _unwrap(model), optimizer, epoch, metrics, cfg)
        elif is_main:
            save_checkpoint(out_dir / "best.pt", _unwrap(model), optimizer, epoch, metrics, cfg)
        if is_main:
            save_checkpoint(out_dir / "last.pt", _unwrap(model), optimizer, epoch, metrics, cfg)
            history.append(metrics)
            write_json(out_dir / "history.json", history)
            _plot_history(history, out_dir)
            print(metrics, flush=True)
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


def run_epoch(
    model: RIGCropModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: Dict[str, Any],
    epoch: int = 1,
    log_interval: int = 1000,
    is_main: bool = True,
) -> tuple[Dict[str, float], int]:
    model.train()
    meters = _meters()
    total_steps = len(loader)
    start_time = time.time()
    for step, batch in enumerate(loader, start=1):
        batch = _to_device(batch, device)
        graph = _encode_graph(model, batch["image"])
        winner, loser = _score_pair(
            model,
            batch["image"],
            batch.get("winner_crop"),
            batch.get("loser_crop"),
            batch["winner_box_feat"],
            batch["loser_box_feat"],
            graph,
        )
        losses = _losses(winner, loser, batch, loss_weights)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        _update_meters(meters, losses, _score_margin(winner, loser), batch["image"].size(0))
        if is_main and log_interval > 0 and (step == 1 or step % log_interval == 0 or step == total_steps):
            _print_train_progress(epoch, step, total_steps, meters, optimizer, start_time)
    return {key: meter.avg for key, meter in meters.items()}, _meter_count(meters)


@torch.no_grad()
def run_eval(model: RIGCropModel, loader: DataLoader, device: torch.device, loss_weights: Dict[str, Any]) -> tuple[Dict[str, float], int]:
    model.eval()
    meters = _meters()
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            graph = _encode_graph(model, batch["image"])
            winner, loser = _score_pair(
                model,
                batch["image"],
                batch.get("winner_crop"),
                batch.get("loser_crop"),
                batch["winner_box_feat"],
                batch["loser_box_feat"],
                graph,
            )
            losses = _losses(winner, loser, batch, loss_weights)
            _update_meters(meters, losses, _score_margin(winner, loser), batch["image"].size(0))
    return {key: meter.avg for key, meter in meters.items()}, _meter_count(meters)


def _losses(winner: Dict[str, torch.Tensor], loser: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], weights: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    crop = pairwise_crop_loss(_ranking_score(winner), _ranking_score(loser), batch["weight"])
    graph_items = graph_supervision_loss(winner, batch)
    utility = utility_distillation_loss(winner["utility"], loser["utility"], batch["winner_utility"], batch["loser_utility"])
    query = query_proposal_loss(winner, batch)
    node = graph_items["node_bbox"] + graph_items["node_role"] + graph_items["node_importance"] + graph_items["node_valid"]
    relation = graph_items["relation_policy"] + graph_items["relation_weight"]
    action = graph_items["action"]
    node_weight = float(weights.get("node", 0.1))
    relation_weight = float(weights.get("relation", 0.06))
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
        + float(weights.get("utility", 0.15)) * utility
        + float(weights.get("query", 0.05)) * query
        + float(weights.get("action", 0.0)) * action
    )
    return {"total": total, "crop": crop, "node": node, "relation": relation, "utility": utility, "query": query, "action": action}


def _score_pair(
    model: RIGCropModel,
    image: torch.Tensor,
    winner_crop: torch.Tensor | None,
    loser_crop: torch.Tensor | None,
    winner_box_feat: torch.Tensor,
    loser_box_feat: torch.Tensor,
    graph: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    batch_size = image.size(0)
    if _model_uses_crop_image(model):
        if winner_crop is None or loser_crop is None:
            raise ValueError("winner_crop and loser_crop are required when model uses crop_backbone mode")
        pair_image = torch.cat([image, image], dim=0)
        pair_crop = torch.cat([winner_crop, loser_crop], dim=0)
    else:
        pair_image = None
        pair_crop = None
    pair_box_feat = torch.cat([winner_box_feat, loser_box_feat], dim=0)
    pair_graph = _repeat_graph_for_pair(graph, batch_size)
    out = model(pair_image, pair_crop, pair_box_feat, graph=pair_graph)
    winner: Dict[str, torch.Tensor] = {}
    loser: Dict[str, torch.Tensor] = {}
    for key, value in out.items():
        if torch.is_tensor(value) and value.size(0) == batch_size * 2:
            winner[key] = value[:batch_size]
            loser[key] = value[batch_size:]
        else:
            winner[key] = value
            loser[key] = value
    return winner, loser


def _repeat_graph_for_pair(graph: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in graph.items():
        if torch.is_tensor(value) and value.size(0) == batch_size:
            out[key] = torch.cat([value, value], dim=0)
        else:
            out[key] = value
    return out


def _ranking_score(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    return out.get("score_logit", out["score"])


def _score_margin(winner: Dict[str, torch.Tensor], loser: Dict[str, torch.Tensor]) -> torch.Tensor:
    return _ranking_score(winner) - _ranking_score(loser)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _meters() -> Dict[str, AverageMeter]:
    return {name: AverageMeter() for name in ["loss", "crop_loss", "node_loss", "relation_loss", "utility_loss", "query_loss", "action_loss", "pairwise_acc", "score_margin"]}


def _update_meters(meters: Dict[str, AverageMeter], losses: Dict[str, torch.Tensor], margin: torch.Tensor, n: int) -> None:
    meters["loss"].update(float(losses["total"].detach().cpu()), n)
    meters["crop_loss"].update(float(losses["crop"].detach().cpu()), n)
    meters["node_loss"].update(float(losses["node"].detach().cpu()), n)
    meters["relation_loss"].update(float(losses["relation"].detach().cpu()), n)
    meters["utility_loss"].update(float(losses["utility"].detach().cpu()), n)
    meters["query_loss"].update(float(losses["query"].detach().cpu()), n)
    meters["action_loss"].update(float(losses["action"].detach().cpu()), n)
    meters["pairwise_acc"].update(float((margin > 0).float().mean().detach().cpu()), n)
    meters["score_margin"].update(float(margin.mean().detach().cpu()), n)


def _meter_count(meters: Dict[str, AverageMeter]) -> int:
    return int(meters["loss"].count) if "loss" in meters else 0


def _print_train_progress(
    epoch: int,
    step: int,
    total_steps: int,
    meters: Dict[str, AverageMeter],
    optimizer: torch.optim.Optimizer,
    start_time: float,
) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    sec_per_step = elapsed / max(step, 1)
    eta = sec_per_step * max(total_steps - step, 0)
    lr = optimizer.param_groups[0].get("lr", 0.0)
    print(
        "[rig-train-step] "
        f"epoch={epoch} step={step}/{total_steps} "
        f"loss={meters['loss'].avg:.4f} crop={meters['crop_loss'].avg:.4f} "
        f"node={meters['node_loss'].avg:.4f} rel={meters['relation_loss'].avg:.4f} "
        f"utility={meters['utility_loss'].avg:.4f} query={meters['query_loss'].avg:.4f} "
        f"acc={meters['pairwise_acc'].avg:.4f} margin={meters['score_margin'].avg:.4f} "
        f"lr={float(lr):.3g} elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}",
        flush=True,
    )


def _format_seconds(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _init_distributed(timeout_seconds: int = 7200) -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=max(int(timeout_seconds), 600)))
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


def _encode_graph(model, image: torch.Tensor) -> Dict[str, torch.Tensor]:
    if hasattr(model, "module"):
        return model(image, encode_only=True)
    return model.encode_graph(image)


def _model_uses_crop_image(model) -> bool:
    module = _unwrap(model)
    return bool(getattr(module, "uses_crop_image", lambda: True)())


def _reduce_logs(logs: Dict[str, float], count: int) -> Dict[str, float]:
    keys = sorted(logs)
    weighted_values = [float(logs[k]) * max(int(count), 0) for k in keys]
    tensor = torch.tensor(weighted_values + [float(max(int(count), 0))], dtype=torch.float64, device=_distributed_device("auto"))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_count = max(float(tensor[-1].detach().cpu()), 1.0)
    values = (tensor[:-1] / total_count).detach().cpu().tolist()
    return {key: float(value) for key, value in zip(keys, values)}


def _rank_indices(length: int) -> list[int]:
    world = max(_world_size(), 1)
    rank = _rank()
    return list(range(rank, int(length), world))


def _resolve_resume_path(cli_resume: str, cfg: Dict[str, Any], out_dir: Path) -> str:
    explicit = str(cli_resume or cfg.get("resume", "") or "").strip()
    if explicit:
        return explicit
    if bool(cfg.get("auto_resume", False)):
        candidate = out_dir / "last.pt"
        if candidate.exists():
            return str(candidate)
    return ""


def _load_history(out_dir: Path) -> list[Dict[str, Any]]:
    path = out_dir / "history.json"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _best_acc_from_history(history: list[Dict[str, Any]], ckpt: Dict[str, Any]) -> float:
    values = [float(item["val_pairwise_acc"]) for item in history if "val_pairwise_acc" in item]
    metrics = ckpt.get("metrics", {}) if isinstance(ckpt.get("metrics"), dict) else {}
    if "val_pairwise_acc" in metrics:
        values.append(float(metrics["val_pairwise_acc"]))
    return max(values) if values else -1.0


def _plot_history(history: list[Dict[str, float]], out_dir: Path) -> None:
    if not history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        if not getattr(_plot_history, "_warned", False):
            print(f"[rig-plot] skip plotting because matplotlib is unavailable: {exc}", flush=True)
            setattr(_plot_history, "_warned", True)
        return

    epochs = [int(item["epoch"]) for item in history]
    curves = [
        ("loss", "train_loss", "val_loss"),
        ("pairwise_acc", "train_pairwise_acc", "val_pairwise_acc"),
        ("score_margin", "train_score_margin", "val_score_margin"),
        ("node_loss", "train_node_loss", "val_node_loss"),
        ("relation_loss", "train_relation_loss", "val_relation_loss"),
        ("utility_loss", "train_utility_loss", "val_utility_loss"),
        ("query_loss", "train_query_loss", "val_query_loss"),
        ("action_loss", "train_action_loss", "val_action_loss"),
    ]
    available = [
        (name, train_key, val_key)
        for name, train_key, val_key in curves
        if train_key in history[-1] or val_key in history[-1]
    ]
    if not available:
        return

    rows = (len(available) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, max(4, rows * 3.2)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, train_key, val_key) in zip(axes.ravel(), available):
        ax.axis("on")
        if train_key in history[-1]:
            ax.plot(epochs, [float(item.get(train_key, float("nan"))) for item in history], label="train")
        if val_key in history[-1]:
            ax.plot(epochs, [float(item.get(val_key, float("nan"))) for item in history], label="val")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=180)
    plt.close(fig)

    for name, train_key, val_key in available:
        fig, ax = plt.subplots(figsize=(7, 4))
        if train_key in history[-1]:
            ax.plot(epochs, [float(item.get(train_key, float("nan"))) for item in history], label="train")
        if val_key in history[-1]:
            ax.plot(epochs, [float(item.get(val_key, float("nan"))) for item in history], label="val")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
