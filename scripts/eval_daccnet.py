#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from dacc.data import DACCGeneratorDataset
from dacc.metrics import acc_at_iou, top1_iou
from dacc.models import DACCNet
from dacc.utils import get_device, load_checkpoint, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args()
    device = get_device("auto")
    ds = DACCGeneratorDataset(args.jsonl, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    model_cfg = load_config(args.config).get("model", {}) if args.config else {}
    model = DACCNet(**model_cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    ious, acc75 = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device), batch["aspect"].to(device))
            ious.append(top1_iou(out["boxes"].cpu(), batch["target_boxes"], batch["target_mask"]))
            acc75.append(acc_at_iou(out["boxes"].cpu(), batch["target_boxes"], batch["target_mask"], 0.75))
    print(json.dumps({"num_samples": len(ds), "top1_iou": sum(ious) / max(len(ious), 1), "acc_iou75": sum(acc75) / max(len(acc75), 1)}, indent=2))


if __name__ == "__main__":
    main()
