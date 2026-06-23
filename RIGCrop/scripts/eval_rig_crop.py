#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from rigcrop.data import RIGPairwiseDataset  # noqa: E402
from rigcrop.model import RIGCropModel  # noqa: E402
from rigcrop.runtime import get_device, load_checkpoint, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RIG-Crop on pairwise crop preferences.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-pairs-per-record", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    device = get_device(cfg.get("device", "auto"))
    model = RIGCropModel(**cfg.get("model", {})).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    ds = RIGPairwiseDataset(
        args.jsonl,
        image_size=args.image_size,
        crop_size=args.crop_size,
        max_records=args.max_records or None,
        max_pairs_per_record=args.max_pairs_per_record or None,
        max_nodes=int(cfg.get("model", {}).get("max_nodes", 8)),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    total = correct = 0
    weighted_correct = 0.0
    weight_sum = 0.0
    margins = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            graph = model.encode_graph(image)
            winner = model(image, batch["winner_crop"].to(device), batch["winner_box_feat"].to(device), graph=graph)
            loser = model(image, batch["loser_crop"].to(device), batch["loser_box_feat"].to(device), graph=graph)
            weight = batch["weight"].to(device)
            margin = winner["score"] - loser["score"]
            ok = margin > 0
            total += int(ok.numel())
            correct += int(ok.sum().cpu())
            weighted_correct += float((ok.float() * weight).sum().cpu())
            weight_sum += float(weight.sum().cpu())
            margins.extend(float(v) for v in margin.cpu().tolist())
    print(
        json.dumps(
            {
                "jsonl": str(Path(args.jsonl).resolve()),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "num_pairs": total,
                "pairwise_acc": correct / max(total, 1),
                "weighted_pairwise_acc": weighted_correct / max(weight_sum, 1e-6),
                "mean_score_margin": sum(margins) / max(len(margins), 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
