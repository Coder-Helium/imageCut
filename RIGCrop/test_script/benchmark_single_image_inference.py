#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rigcrop.anchors import generate_anchors  # noqa: E402
from rigcrop.box_ops import candidate_box_features, denormalize_xyxy, normalize_xyxy  # noqa: E402
from rigcrop.image_io import crop_rgb, read_image_rgb, resize_to_tensor  # noqa: E402
from rigcrop.io import iter_jsonl  # noqa: E402
from rigcrop.model import RIGCropModel  # noqa: E402
from rigcrop.runtime import get_device, load_checkpoint, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark single-image RIGFormer inference latency.")
    parser.add_argument("--image", default="", help="Image path. If omitted, use --jsonl and --record-index.")
    parser.add_argument("--jsonl", default="", help="Optional JSONL used to pick an image_path.")
    parser.add_argument("--record-index", type=int, default=1, help="1-based JSONL record index when --image is omitted.")
    parser.add_argument("--checkpoint", default="RIGCrop/runs/rig_crop_cpc_gaic_dinov3_pth/best.pt")
    parser.add_argument("--config", default="RIGCrop/configs/rig_crop_cpc_gaic_dinov3_pth.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=512, help="Candidate scoring batch size.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--score-anchors-only", action="store_true", help="Disable learned query boxes during benchmark.")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    image_path = _resolve_image_path(args.image, args.jsonl, args.record_index)
    device = get_device(args.device)
    if device.type != "cuda":
        print(f"[warn] benchmarking on {device}; use CUDA_VISIBLE_DEVICES=0 for single-4090 latency.", file=sys.stderr)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    cfg = load_config(args.config)
    model = RIGCropModel(**cfg.get("model", {})).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    img = read_image_rgb(image_path)
    with torch.inference_mode():
        for _ in range(max(args.warmup, 0)):
            _predict_once(model, img, args, device)
        _sync(device)

        times: List[float] = []
        last: Dict[str, Any] | None = None
        for _ in range(max(args.repeat, 1)):
            _sync(device)
            start = time.perf_counter()
            last = _predict_once(model, img, args, device)
            _sync(device)
            times.append(time.perf_counter() - start)

    assert last is not None
    metrics = _summarize(times, last, args, image_path, device)
    payload = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


def _predict_once(model: RIGCropModel, img, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    height, width = img.shape[:2]
    image_tensor = resize_to_tensor(img, args.image_size).unsqueeze(0).to(device, non_blocking=True)
    graph = model.encode_graph(image_tensor)

    boxes = generate_anchors(width, height)
    if not args.score_anchors_only and "query_boxes" in graph:
        query_boxes = [denormalize_xyxy(box, width, height) for box in graph["query_boxes"][0].detach().cpu().tolist()]
        boxes = _merge_boxes(boxes, query_boxes, width, height)

    ranked = _score_boxes(model, img, image_tensor, graph, boxes, args.crop_size, args.batch_size, device)
    topk = ranked[: max(args.topk, 1)]
    return {"num_candidates": len(boxes), "topk": topk}


def _score_boxes(
    model: RIGCropModel,
    img,
    image_tensor: torch.Tensor,
    graph: Dict[str, torch.Tensor],
    boxes: List[List[int]],
    crop_size: int,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    height, width = img.shape[:2]
    needs_crop = bool(getattr(model, "uses_crop_image", lambda: True)())
    results: List[Dict[str, Any]] = []
    for start in range(0, len(boxes), max(batch_size, 1)):
        batch_boxes = boxes[start : start + max(batch_size, 1)]
        crops = (
            torch.stack([resize_to_tensor(crop_rgb(img, box), crop_size) for box in batch_boxes]).to(device, non_blocking=True)
            if needs_crop
            else None
        )
        image_batch = image_tensor.expand(len(batch_boxes), -1, -1, -1) if needs_crop else None
        box_feat = torch.tensor(
            [candidate_box_features(normalize_xyxy(box, width, height)) for box in batch_boxes],
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


def _resolve_image_path(image: str, jsonl: str, record_index: int) -> str:
    if image:
        return image
    if not jsonl:
        raise SystemExit("Either --image or --jsonl is required.")
    for idx, rec in enumerate(iter_jsonl(jsonl), start=1):
        if idx == max(record_index, 1):
            path = rec.get("image_path")
            if not path:
                raise SystemExit(f"Record {idx} has no image_path: {jsonl}")
            return str(path)
    raise SystemExit(f"record-index {record_index} out of range: {jsonl}")


def _summarize(times: List[float], last: Dict[str, Any], args: argparse.Namespace, image_path: str, device: torch.device) -> Dict[str, Any]:
    times_ms = [t * 1000.0 for t in times]
    avg_ms = sum(times_ms) / max(len(times_ms), 1)
    metrics: Dict[str, Any] = {
        "image": str(Path(image_path).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "image_size": args.image_size,
        "crop_size": args.crop_size,
        "candidate_batch_size": args.batch_size,
        "num_candidates": last["num_candidates"],
        "warmup": args.warmup,
        "repeat": args.repeat,
        "avg_ms": avg_ms,
        "p50_ms": statistics.median(times_ms),
        "p90_ms": _percentile(times_ms, 0.90),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "fps": 1000.0 / max(avg_ms, 1e-6),
        "topk": last["topk"],
    }
    if device.type == "cuda":
        metrics["max_memory_GB"] = torch.cuda.max_memory_allocated(device) / 1024**3
    return metrics


def _percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
