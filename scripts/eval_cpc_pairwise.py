#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from dacc.data import PairwiseCropRankerDataset
from dacc.models import CropRanker
from dacc.utils import get_device, load_checkpoint, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate crop ranker on CPC-style pairwise preferences.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-pairs-per-record", type=int, default=0)
    args = parser.parse_args()

    device = get_device("auto")
    ds = PairwiseCropRankerDataset(
        args.jsonl,
        image_size=args.image_size,
        crop_size=args.crop_size,
        max_records=args.max_records or None,
        max_pairs_per_record=args.max_pairs_per_record or None,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    model_cfg = load_config(args.config).get("model", {}) if args.config else {}
    model = CropRanker(**model_cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    total = 0
    correct = 0
    weighted_correct = 0.0
    weight_sum = 0.0
    margins = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            winner = batch["winner_crop"].to(device)
            loser = batch["loser_crop"].to(device)
            winner_feat = batch["winner_box_feat"].to(device)
            loser_feat = batch["loser_box_feat"].to(device)
            weights = batch["weight"].to(device)
            winner_score = model(image, winner, winner_feat)
            loser_score = model(image, loser, loser_feat)
            margin = winner_score - loser_score
            ok = margin > 0
            correct += int(ok.sum().detach().cpu())
            total += int(ok.numel())
            weighted_correct += float((ok.float() * weights).sum().detach().cpu())
            weight_sum += float(weights.sum().detach().cpu())
            margins.extend(float(v) for v in margin.detach().cpu().tolist())

    print(
        json.dumps(
            {
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
