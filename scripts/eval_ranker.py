#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from dacc.data import CropRankerDataset, unit_to_score
from dacc.metrics import spearmanr_np
from dacc.models import CropRanker
from dacc.utils import get_device, load_checkpoint, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-size", type=int, default=224)
    args = parser.parse_args()
    device = get_device("auto")
    ds = CropRankerDataset(args.jsonl, image_size=args.image_size, crop_size=args.crop_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    model_cfg = load_config(args.config).get("model", {}) if args.config else {}
    model = CropRanker(**model_cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["image"].to(device), batch["crop"].to(device), batch["box_feat"].to(device)).cpu().tolist()
            preds.extend([unit_to_score(x) for x in pred])
            targets.extend(batch["score_raw"].tolist())
    mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / max(len(preds), 1)
    print(json.dumps({"num_candidates": len(preds), "mse_raw_score": mse, "spearman": spearmanr_np(preds, targets)}, indent=2))


if __name__ == "__main__":
    main()
