#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from rigcrop.anchors import generate_anchors  # noqa: E402
from rigcrop.box_ops import candidate_box_features, denormalize_xyxy, normalize_xyxy  # noqa: E402
from rigcrop.image_io import crop_rgb, read_image_rgb, resize_to_tensor  # noqa: E402
from rigcrop.model import RIGCropModel  # noqa: E402
from rigcrop.runtime import get_device, load_checkpoint, load_config  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch predict RIG-Crop and export original/crop comparison images.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default="RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/best.pt")
    parser.add_argument("--config", default="RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--panel-height", type=int, default=720)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    json_dir = out_dir / "json"
    boxed_dir = out_dir / "boxed"
    crop_dir = out_dir / "crop"
    compare_dir = out_dir / "compare"
    for path in [json_dir, boxed_dir, crop_dir, compare_dir]:
        path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    device = get_device(args.device)
    model = RIGCropModel(**cfg.get("model", {})).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    images = _list_images(args.image_dir)
    summary = []
    with torch.inference_mode():
        for idx, image_path in enumerate(images, start=1):
            print(f"[{idx}/{len(images)}] {image_path}", flush=True)
            item = _predict_one(model, image_path, args, device)
            stem = _safe_stem(image_path)

            boxed_rgb = item.pop("_boxed_rgb")
            crop_rgb_img = item.pop("_crop_rgb")
            compare_rgb = item.pop("_compare_rgb")
            item["boxed"] = str(boxed_dir / f"{stem}.jpg")
            item["crop"] = str(crop_dir / f"{stem}.jpg")
            item["compare"] = str(compare_dir / f"{stem}.jpg")

            (json_dir / f"{stem}.json").write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            cv2.imwrite(str(boxed_dir / f"{stem}.jpg"), cv2.cvtColor(boxed_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(crop_dir / f"{stem}.jpg"), cv2.cvtColor(crop_rgb_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(compare_dir / f"{stem}.jpg"), cv2.cvtColor(compare_rgb, cv2.COLOR_RGB2BGR))
            summary.append(item)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] images={len(images)} out={out_dir}", flush=True)


@torch.no_grad()
def _predict_one(model: RIGCropModel, image_path: Path, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    img = read_image_rgb(image_path)
    h, w = img.shape[:2]
    image_tensor = resize_to_tensor(img, args.image_size).unsqueeze(0).to(device, non_blocking=True)
    graph = model.encode_graph(image_tensor)

    boxes = generate_anchors(w, h)
    if "query_boxes" in graph:
        query_boxes = [denormalize_xyxy(box, w, h) for box in graph["query_boxes"][0].detach().cpu().tolist()]
        boxes = _merge_boxes(boxes, query_boxes, w, h)

    ranked = _score_boxes(model, img, image_tensor, graph, boxes, args, device)
    top = ranked[: max(args.topk, 1)]
    best = top[0]
    best_box = best["box"]
    crop = crop_rgb(img, best_box)
    boxed = _draw_prediction(img, best_box, best["score"])
    compare = _comparison_panel(img, boxed, crop, args.panel_height)
    return {
        "image": str(image_path.resolve()),
        "width": w,
        "height": h,
        "topk": top,
        "best_box": best_box,
        "best_score": best["score"],
        "best_utility": best["utility"],
        "_boxed_rgb": boxed,
        "_crop_rgb": crop,
        "_compare_rgb": compare,
    }


def _score_boxes(
    model: RIGCropModel,
    img: np.ndarray,
    image_tensor: torch.Tensor,
    graph: Dict[str, torch.Tensor],
    boxes: List[List[int]],
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, Any]]:
    h, w = img.shape[:2]
    needs_crop = bool(getattr(model, "uses_crop_image", lambda: True)())
    results: List[Dict[str, Any]] = []
    for start in range(0, len(boxes), max(args.batch_size, 1)):
        batch_boxes = boxes[start : start + max(args.batch_size, 1)]
        crops = (
            torch.stack([resize_to_tensor(crop_rgb(img, box), args.crop_size) for box in batch_boxes]).to(device, non_blocking=True)
            if needs_crop
            else None
        )
        image_batch = image_tensor.expand(len(batch_boxes), -1, -1, -1) if needs_crop else None
        box_feat = torch.tensor(
            [candidate_box_features(normalize_xyxy(box, w, h)) for box in batch_boxes],
            dtype=torch.float32,
            device=device,
        )
        out = model(image_batch, crops, box_feat, graph=_expand_graph(graph, len(batch_boxes)))
        scores = out.get("score_logit", out["score"]).detach().cpu().tolist()
        utilities = out["utility"].detach().cpu().tolist()
        for box, score, utility in zip(batch_boxes, scores, utilities):
            results.append({"box": box, "score": float(score), "utility": float(utility)})
    return sorted(results, key=lambda item: item["score"], reverse=True)


def _expand_graph(graph: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in graph.items():
        out[key] = value.expand(batch_size, *value.shape[1:]) if torch.is_tensor(value) and value.size(0) == 1 else value
    return out


def _draw_prediction(img_rgb: np.ndarray, box: Sequence[int], score: float) -> np.ndarray:
    out = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(out, f"PRED {score:.2f}", (x1, max(28, y1 + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _comparison_panel(original_rgb: np.ndarray, boxed_rgb: np.ndarray, crop_rgb_img: np.ndarray, panel_height: int) -> np.ndarray:
    left = _fit_height(boxed_rgb, panel_height)
    right = _fit_height(crop_rgb_img, panel_height)
    label_h = 44
    gap = 12
    canvas_h = panel_height + label_h
    canvas_w = left.shape[1] + gap + right.shape[1]
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)
    canvas[label_h:, : left.shape[1]] = left
    canvas[label_h:, left.shape[1] + gap :] = right
    out = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.putText(out, "Original with predicted crop", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, "Final crop", (left.shape[1] + gap + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _fit_height(img_rgb: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    scale = target_h / max(h, 1)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img_rgb, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _merge_boxes(base: Iterable[Sequence[int]], extra: Iterable[Sequence[int]], image_w: int, image_h: int) -> List[List[int]]:
    out: List[List[int]] = []
    seen = set()
    for box in list(base) + list(extra):
        clean = _sanitize_box(box, image_w, image_h)
        key = tuple(clean)
        if key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def _sanitize_box(box: Sequence[int | float], image_w: int, image_h: int) -> List[int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    x1, x2 = sorted((max(0, min(image_w - 1, x1)), max(0, min(image_w, x2))))
    y1, y2 = sorted((max(0, min(image_h - 1, y1)), max(0, min(image_h, y2))))
    if x2 <= x1 + 4:
        x2 = min(image_w, x1 + 5)
    if y2 <= y1 + 4:
        y2 = min(image_h, y1 + 5)
    return [x1, y1, x2, y2]


def _list_images(image_dir: str) -> List[Path]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"image-dir not found: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)[:140]


if __name__ == "__main__":
    main()
