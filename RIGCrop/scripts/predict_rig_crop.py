#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import torch

from rigcrop.anchors import generate_anchors  # noqa: E402
from rigcrop.box_ops import candidate_box_features, normalize_xyxy  # noqa: E402
from rigcrop.image_io import crop_rgb, draw_boxes, read_image_rgb, resize_to_tensor  # noqa: E402
from rigcrop.model import RIGCropModel  # noqa: E402
from rigcrop.runtime import get_device, load_checkpoint, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Image-only RIG-Crop inference.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-vis", default="")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    model = RIGCropModel(**cfg.get("model", {})).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    img = read_image_rgb(args.image)
    h, w = img.shape[:2]
    anchors = generate_anchors(w, h)
    image_tensor = resize_to_tensor(img, args.image_size).unsqueeze(0).to(device)
    results = []
    with torch.no_grad():
        graph = model.encode_graph(image_tensor)
        for start in range(0, len(anchors), args.batch_size):
            batch_boxes = anchors[start : start + args.batch_size]
            crops = torch.stack([resize_to_tensor(crop_rgb(img, box), args.crop_size) for box in batch_boxes]).to(device)
            box_feat = torch.tensor([candidate_box_features(normalize_xyxy(box, w, h)) for box in batch_boxes], dtype=torch.float32, device=device)
            image_batch = image_tensor.expand(len(batch_boxes), -1, -1, -1)
            graph_batch = {key: value.expand(len(batch_boxes), *value.shape[1:]) if torch.is_tensor(value) and value.size(0) == 1 else value for key, value in graph.items()}
            out = model(image_batch, crops, box_feat, graph=graph_batch)
            for box, score, util in zip(batch_boxes, out["score"].cpu().tolist(), out["utility"].cpu().tolist()):
                results.append({"box": box, "score": float(score), "utility": float(util)})
    results = sorted(results, key=lambda item: item["score"], reverse=True)[: args.topk]
    payload = {"image": str(Path(args.image).resolve()), "topk": results}
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out_vis:
        labels = [f"#{idx+1} {item['score']:.3f}" for idx, item in enumerate(results)]
        vis = draw_boxes(img, [item["box"] for item in results], labels)
        Path(args.out_vis).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out_vis), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
