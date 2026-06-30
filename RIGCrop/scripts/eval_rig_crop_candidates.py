#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rigcrop.box_ops import candidate_box_features, normalize_xyxy  # noqa: E402
from rigcrop.image_io import crop_rgb, read_image_rgb, resize_to_tensor  # noqa: E402
from rigcrop.io import iter_jsonl, write_json  # noqa: E402
from rigcrop.model import RIGCropModel  # noqa: E402
from rigcrop.runtime import get_device, load_checkpoint, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RIG-Crop candidate ranking metrics for GAIC-style JSONL.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-candidates-per-record", type=int, default=0)
    parser.add_argument("--acc-k", default="1,5,10")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    model = RIGCropModel(**cfg.get("model", {})).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    acc_ks = sorted({int(x) for x in args.acc_k.split(",") if x.strip()})
    hits = {k: 0 for k in acc_ks}
    total_records = 0
    total_candidates = 0
    global_pred: List[float] = []
    global_teacher: List[float] = []
    per_image_srcc: List[float] = []

    for rec_idx, rec in enumerate(iter_jsonl(args.jsonl), start=1):
        if args.max_records > 0 and rec_idx > args.max_records:
            break
        candidates = _scored_candidates(rec.get("candidates", []) or [])
        if args.max_candidates_per_record > 0:
            candidates = candidates[: args.max_candidates_per_record]
        if len(candidates) < 2:
            continue

        preds = _predict_candidate_scores(model, rec, candidates, device, args.image_size, args.crop_size, args.batch_size)
        teachers = [float(item["teacher_score"]) for item in candidates]
        cids = [str(item["candidate_id"]) for item in candidates]
        if len(preds) != len(teachers):
            continue

        total_records += 1
        total_candidates += len(candidates)
        global_pred.extend(preds)
        global_teacher.extend(teachers)
        per_image_srcc.append(_spearman(preds, teachers))

        pred_top = cids[max(range(len(preds)), key=lambda i: preds[i])]
        teacher_order = sorted(range(len(teachers)), key=lambda i: teachers[i], reverse=True)
        for k in acc_ks:
            top_teacher = {cids[i] for i in teacher_order[: min(k, len(teacher_order))]}
            hits[k] += int(pred_top in top_teacher)

    metrics: Dict[str, Any] = {
        "jsonl": str(Path(args.jsonl).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_records": total_records,
        "num_candidates": total_candidates,
        "srcc_global": _spearman(global_pred, global_teacher),
        "srcc_mean_per_image": sum(per_image_srcc) / max(len(per_image_srcc), 1),
        "acc_at_k": {str(k): hits[k] / max(total_records, 1) for k in acc_ks},
    }
    if args.out_json:
        write_json(args.out_json, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _scored_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        score = _teacher_score(cand)
        box = cand.get("box")
        if score is None or not isinstance(box, list) or len(box) < 4:
            continue
        item = dict(cand)
        item["candidate_id"] = str(cand.get("candidate_id", idx))
        item["teacher_score"] = float(score)
        out.append(item)
    return out


def _teacher_score(cand: Dict[str, Any]) -> float | None:
    scores = cand.get("scores", {}) if isinstance(cand.get("scores"), dict) else {}
    value = scores.get("mos", scores.get("final_score", cand.get("score", None)))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@torch.no_grad()
def _predict_candidate_scores(
    model: RIGCropModel,
    rec: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    device: torch.device,
    image_size: int,
    crop_size: int,
    batch_size: int,
) -> List[float]:
    img = read_image_rgb(rec["image_path"])
    h, w = img.shape[:2]
    image_tensor = resize_to_tensor(img, image_size).unsqueeze(0).to(device)
    graph = model.encode_graph(image_tensor)
    preds: List[float] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        boxes = [item["box"] for item in batch]
        crops = torch.stack([resize_to_tensor(crop_rgb(img, box), crop_size) for box in boxes]).to(device, non_blocking=True)
        box_feat = torch.tensor(
            [candidate_box_features(normalize_xyxy(box, w, h)) for box in boxes],
            dtype=torch.float32,
            device=device,
        )
        image_batch = image_tensor.expand(len(batch), -1, -1, -1)
        out = model(image_batch, crops, box_feat, graph=_expand_graph(graph, len(batch)))
        score = out.get("score_logit", out["score"])
        preds.extend(float(v) for v in score.detach().cpu().tolist())
    return preds


def _expand_graph(graph: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in graph.items():
        if torch.is_tensor(value) and value.size(0) == 1:
            out[key] = value.expand(batch_size, *value.shape[1:])
        else:
            out[key] = value
    return out


def _spearman(x: List[float], y: List[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    vx = sum((v - mx) ** 2 for v in rx)
    vy = sum((v - my) ** 2 for v in ry)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return float(cov / ((vx * vy) ** 0.5))


def _average_ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0 for _ in values]
    idx = 0
    while idx < len(order):
        end = idx + 1
        while end < len(order) and values[order[end]] == values[order[idx]]:
            end += 1
        avg_rank = (idx + end - 1) / 2.0
        for pos in range(idx, end):
            ranks[order[pos]] = avg_rank
        idx = end
    return ranks


if __name__ == "__main__":
    main()
