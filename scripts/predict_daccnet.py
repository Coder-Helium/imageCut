#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import torch

from dacc.box_ops import denormalize_xyxy
from dacc.data import read_image_rgb, resize_to_tensor
from dacc.models import DACCNet
from dacc.utils import get_device, load_checkpoint, load_config
from dacc.vocab import ACTION_VOCAB, ISSUE_VOCAB, aspect_to_float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-vis", default="")
    parser.add_argument("--aspect", default="original")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device("auto")
    model = DACCNet(**cfg.get("model", {})).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    img_rgb = read_image_rgb(args.image)
    h, w = img_rgb.shape[:2]
    tensor = resize_to_tensor(img_rgb, args.image_size).unsqueeze(0).to(device)
    image_aspect = w / max(float(h), 1.0)
    aspect = torch.tensor([[aspect_to_float(args.aspect, w, h) / max(image_aspect, 1e-6)]], dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(tensor, aspect)
    boxes = denormalize_xyxy(out["boxes"][0].cpu(), w, h)
    scores = out["scores"][0].cpu()
    actions = out["action_logits"][0].argmax(-1).cpu()
    issues = out["issue_logits"][0].argmax(-1).cpu()
    order = torch.argsort(scores, descending=True)[: args.topk]

    preds = []
    for rank, idx in enumerate(order.tolist(), start=1):
        preds.append(
            {
                "rank": rank,
                "box": [int(round(v)) for v in boxes[idx].tolist()],
                "score": float(scores[idx]),
                "action": ACTION_VOCAB[int(actions[idx])],
                "issue": ISSUE_VOCAB[int(issues[idx])],
            }
        )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"image": args.image, "aspect": args.aspect, "predictions": preds}, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_vis:
        vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        for p in preds:
            x1, y1, x2, y2 = p["box"]
            color = (0, 0, 255) if p["rank"] == 1 else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"#{p['rank']} {p['score']:.2f}", (x1, max(24, y1 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        Path(args.out_vis).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out_vis, vis)
    print(json.dumps({"out_json": args.out_json, "out_vis": args.out_vis, "num_predictions": len(preds)}, indent=2))


if __name__ == "__main__":
    main()
