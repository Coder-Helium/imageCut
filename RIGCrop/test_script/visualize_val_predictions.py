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
from rigcrop.io import iter_jsonl, write_json  # noqa: E402
from rigcrop.model import RIGCropModel  # noqa: E402
from rigcrop.runtime import get_device, load_checkpoint, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize model top crop against validation ground-truth crop.")
    parser.add_argument("--jsonl", required=True, help="Validation JSONL with candidates and image_path.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-records", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--score-candidates-only", action="store_true", help="Only score JSONL candidates, not dense anchors.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    model = RIGCropModel(**cfg.get("model", {})).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    processed = 0
    for rec_idx, rec in enumerate(iter_jsonl(args.jsonl), start=1):
        if rec_idx < args.start_index:
            continue
        if args.max_records > 0 and processed >= args.max_records:
            break
        try:
            item = _visualize_record(model, rec, device, args, out_dir, rec_idx)
        except Exception as exc:  # noqa: BLE001
            item = {"record_index": rec_idx, "sample_id": rec.get("sample_id", ""), "error": str(exc)}
        rows.append(item)
        processed += 1
        print(json.dumps(item, ensure_ascii=False), flush=True)

    write_json(out_dir / "summary.json", {"jsonl": str(Path(args.jsonl).resolve()), "checkpoint": str(Path(args.checkpoint).resolve()), "records": rows})


@torch.no_grad()
def _visualize_record(
    model: RIGCropModel,
    rec: Dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path,
    rec_idx: int,
) -> Dict[str, Any]:
    img = read_image_rgb(rec["image_path"])
    h, w = img.shape[:2]
    gt = _ground_truth_crop(rec, w, h)
    candidates = _candidate_boxes(rec, w, h)
    if not args.score_candidates_only:
        candidates = _merge_boxes(generate_anchors(w, h), candidates, w, h)

    image_tensor = resize_to_tensor(img, args.image_size).unsqueeze(0).to(device)
    graph = model.encode_graph(image_tensor)
    if not args.score_candidates_only and "query_boxes" in graph:
        query_boxes = [denormalize_xyxy(box, w, h) for box in graph["query_boxes"][0].detach().cpu().tolist()]
        candidates = _merge_boxes(candidates, query_boxes, w, h)
    ranked = _score_boxes(model, img, image_tensor, graph, candidates, device, args.crop_size, args.batch_size)
    top = ranked[: max(1, args.topk)]
    pred = top[0]["box"]
    sample_id = str(rec.get("sample_id", f"record_{rec_idx:06d}"))
    stem = _safe_name(f"{rec_idx:06d}_{sample_id}")
    vis_path = out_dir / f"{stem}.jpg"
    crop_path = out_dir / f"{stem}_crops.jpg"
    vis = _draw_comparison(img, gt, pred, top)
    cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    crop_panel = _crop_panel(img, gt, pred)
    cv2.imwrite(str(crop_path), cv2.cvtColor(crop_panel, cv2.COLOR_RGB2BGR))
    return {
        "record_index": rec_idx,
        "sample_id": sample_id,
        "image_path": rec.get("image_path"),
        "gt_box": gt,
        "pred_box": pred,
        "pred_score": top[0]["score"],
        "iou": _iou(gt, pred),
        "topk": top,
        "vis": str(vis_path),
        "crops": str(crop_path),
    }


def _ground_truth_crop(rec: Dict[str, Any], image_w: int, image_h: int) -> List[int]:
    box = rec.get("best_crop")
    if isinstance(box, list) and len(box) >= 4:
        return _sanitize_abs_box(box, image_w, image_h)
    candidates = list(rec.get("candidates", []) or [])
    scored = [(cand, _teacher_score(cand)) for cand in candidates]
    scored = [(cand, score) for cand, score in scored if score is not None and isinstance(cand.get("box"), list)]
    if scored:
        best = max(scored, key=lambda item: float(item[1]))[0]
        return _sanitize_abs_box(best["box"], image_w, image_h)
    pairs = list(rec.get("pairwise_preferences", []) or [])
    if pairs and candidates:
        wins: Dict[str, float] = {}
        cand_map = {str(c.get("candidate_id")): c for c in candidates}
        for pair in pairs:
            wins[str(pair.get("winner"))] = wins.get(str(pair.get("winner")), 0.0) + float(pair.get("weight", 1.0) or 1.0)
            wins.setdefault(str(pair.get("loser")), 0.0)
        if wins:
            cid = max(wins, key=wins.get)
            if cid in cand_map and isinstance(cand_map[cid].get("box"), list):
                return _sanitize_abs_box(cand_map[cid]["box"], image_w, image_h)
    raise ValueError("No ground-truth crop found from best_crop, candidate scores, or pairwise preferences")


def _candidate_boxes(rec: Dict[str, Any], image_w: int, image_h: int) -> List[List[int]]:
    out: List[List[int]] = []
    for cand in rec.get("candidates", []) or []:
        box = cand.get("box")
        if isinstance(box, list) and len(box) >= 4:
            out.append(_sanitize_abs_box(box, image_w, image_h))
    return out


def _teacher_score(cand: Dict[str, Any]) -> float | None:
    scores = cand.get("scores", {}) if isinstance(cand.get("scores"), dict) else {}
    value = scores.get("mos", scores.get("final_score", cand.get("score", None)))
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_abs_box(box: Sequence[float], image_w: int, image_h: int) -> List[int]:
    values = [float(v) for v in box[:4]]
    if max(values) <= 1.5:
        values = [values[0] * image_w, values[1] * image_h, values[2] * image_w, values[3] * image_h]
    x1, y1, x2, y2 = [int(round(v)) for v in values]
    x1, x2 = sorted((max(0, min(image_w - 1, x1)), max(0, min(image_w, x2))))
    y1, y2 = sorted((max(0, min(image_h - 1, y1)), max(0, min(image_h, y2))))
    if x2 <= x1 + 1:
        x2 = min(image_w, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(image_h, y1 + 2)
    return [x1, y1, x2, y2]


def _score_boxes(
    model: RIGCropModel,
    img: np.ndarray,
    image_tensor: torch.Tensor,
    graph: Dict[str, torch.Tensor],
    boxes: List[List[int]],
    device: torch.device,
    crop_size: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    needs_crop = bool(getattr(model, "uses_crop_image", lambda: True)())
    h, w = img.shape[:2]
    results: List[Dict[str, Any]] = []
    for start in range(0, len(boxes), batch_size):
        batch_boxes = boxes[start : start + batch_size]
        if not batch_boxes:
            continue
        crops = torch.stack([resize_to_tensor(crop_rgb(img, box), crop_size) for box in batch_boxes]).to(device) if needs_crop else None
        image_batch = image_tensor.expand(len(batch_boxes), -1, -1, -1) if needs_crop else None
        box_feat = torch.tensor([candidate_box_features(normalize_xyxy(box, w, h)) for box in batch_boxes], dtype=torch.float32, device=device)
        out = model(image_batch, crops, box_feat, graph=_expand_graph(graph, len(batch_boxes)))
        scores = out.get("score_logit", out["score"]).detach().cpu().tolist()
        utils = out["utility"].detach().cpu().tolist()
        for box, score, util in zip(batch_boxes, scores, utils):
            results.append({"box": box, "score": float(score), "utility": float(util)})
    return sorted(results, key=lambda item: item["score"], reverse=True)


def _expand_graph(graph: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in graph.items():
        out[key] = value.expand(batch_size, *value.shape[1:]) if torch.is_tensor(value) and value.size(0) == 1 else value
    return out


def _draw_comparison(img_rgb: np.ndarray, gt: List[int], pred: List[int], top: List[Dict[str, Any]]) -> np.ndarray:
    out = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
    _draw_box_bgr(out, gt, (0, 210, 0), "GT")
    _draw_box_bgr(out, pred, (0, 0, 255), f"PRED {top[0]['score']:.2f}")
    for idx, item in enumerate(top[1:5], start=2):
        _draw_box_bgr(out, item["box"], (255, 150, 0), f"#{idx} {item['score']:.2f}", thickness=1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _draw_box_bgr(img_bgr: np.ndarray, box: Sequence[int], color: tuple[int, int, int], label: str, thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img_bgr, label, (x1, max(20, y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def _crop_panel(img_rgb: np.ndarray, gt: List[int], pred: List[int]) -> np.ndarray:
    gt_crop = _fit_panel_crop(crop_rgb(img_rgb, gt))
    pred_crop = _fit_panel_crop(crop_rgb(img_rgb, pred))
    panel = np.concatenate([gt_crop, pred_crop], axis=1)
    out = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.putText(out, "GT", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 0), 2, cv2.LINE_AA)
    cv2.putText(out, "PRED", (gt_crop.shape[1] + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _fit_panel_crop(img_rgb: np.ndarray, canvas_size: tuple[int, int] = (420, 360)) -> np.ndarray:
    canvas_w, canvas_h = canvas_size
    h, w = img_rgb.shape[:2]
    scale = min(canvas_w / max(w, 1), canvas_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)
    x0 = (canvas_w - new_w) // 2
    y0 = (canvas_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    cv2.rectangle(canvas, (x0, y0), (x0 + new_w - 1, y0 + new_h - 1), (35, 35, 35), 1)
    return canvas


def _merge_boxes(base: Iterable[Sequence[int]], extra: Iterable[Sequence[int]], image_w: int, image_h: int) -> List[List[int]]:
    out: List[List[int]] = []
    seen = set()
    for box in list(base) + list(extra):
        clean = _sanitize_abs_box(box, image_w, image_h)
        key = tuple(clean)
        if key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(area_a + area_b - inter, 1e-6))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:160]


if __name__ == "__main__":
    main()
