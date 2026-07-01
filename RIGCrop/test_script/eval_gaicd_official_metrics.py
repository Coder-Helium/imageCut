#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rigcrop.box_ops import candidate_box_features, normalize_xyxy  # noqa: E402
from rigcrop.image_io import crop_rgb, read_image_rgb, resize_to_tensor  # noqa: E402
from rigcrop.io import iter_jsonl, write_json  # noqa: E402
from rigcrop.model import RIGCropModel  # noqa: E402
from rigcrop.runtime import get_device, load_checkpoint, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RIG-Crop on GAICD-style candidate MOS annotations."
    )
    parser.add_argument("--jsonl", required=True, help="GAICD val/test JSONL with candidate MOS scores.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-candidates-per-record", type=int, default=0)
    parser.add_argument("--pred-topk", default="1,2,3,4", help="K values for AccK/N.")
    parser.add_argument("--gt-topn", default="5,10", help="N values for AccK/N.")
    parser.add_argument("--compute-pairwise", action="store_true", help="Also compute all-pair ranking accuracy.")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    model = RIGCropModel(**cfg.get("model", {})).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    pred_topk = _parse_ints(args.pred_topk)
    gt_topn = _parse_ints(args.gt_topn)
    acc_sum = {(k, n): 0.0 for n in gt_topn for k in pred_topk}
    acc_count = {(k, n): 0 for n in gt_topn for k in pred_topk}

    num_records = 0
    num_candidates = 0
    skipped_records = 0
    global_pred: List[float] = []
    global_teacher: List[float] = []
    per_image_srcc: List[float] = []
    per_image_pcc: List[float] = []
    pairwise_correct = 0
    pairwise_total = 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with torch.inference_mode():
        for rec_idx, rec in enumerate(iter_jsonl(args.jsonl), start=1):
            if args.max_records > 0 and rec_idx > args.max_records:
                break
            candidates = _scored_candidates(rec.get("candidates", []) or [])
            if args.max_candidates_per_record > 0:
                candidates = candidates[: args.max_candidates_per_record]
            if len(candidates) < 2:
                skipped_records += 1
                continue

            preds = _predict_candidate_scores(
                model=model,
                rec=rec,
                candidates=candidates,
                device=device,
                image_size=args.image_size,
                crop_size=args.crop_size,
                batch_size=args.batch_size,
            )
            if len(preds) != len(candidates):
                skipped_records += 1
                continue

            teachers = [float(item["teacher_score"]) for item in candidates]
            pred_order = sorted(range(len(preds)), key=lambda i: preds[i], reverse=True)
            teacher_order = sorted(range(len(teachers)), key=lambda i: teachers[i], reverse=True)

            for n in gt_topn:
                top_teacher = set(teacher_order[: min(n, len(teacher_order))])
                for k in pred_topk:
                    pred_k = pred_order[: min(k, len(pred_order))]
                    if not pred_k:
                        continue
                    hits = sum(1 for idx in pred_k if idx in top_teacher)
                    acc_sum[(k, n)] += hits / len(pred_k)
                    acc_count[(k, n)] += 1

            if args.compute_pairwise:
                ok, total = _pairwise_accuracy_counts(preds, teachers)
                pairwise_correct += ok
                pairwise_total += total

            num_records += 1
            num_candidates += len(candidates)
            global_pred.extend(preds)
            global_teacher.extend(teachers)
            per_image_srcc.append(_spearman(preds, teachers))
            per_image_pcc.append(_pearson(preds, teachers))

    acc_k_over_n = {
        f"Acc{k}/{n}": acc_sum[(k, n)] / max(acc_count[(k, n)], 1)
        for n in gt_topn
        for k in pred_topk
    }
    metrics: Dict[str, Any] = {
        "jsonl": str(Path(args.jsonl).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "num_records": num_records,
        "num_candidates": num_candidates,
        "skipped_records": skipped_records,
        "metrics": {
            **acc_k_over_n,
            "Acc5": _mean_named(acc_k_over_n, [f"Acc{k}/5" for k in pred_topk]),
            "Acc10": _mean_named(acc_k_over_n, [f"Acc{k}/10" for k in pred_topk]),
            "SRCC": _mean(per_image_srcc),
            "PCC": _mean(per_image_pcc),
            "SRCC_global": _spearman(global_pred, global_teacher),
            "PCC_global": _pearson(global_pred, global_teacher),
        },
        "notes": {
            "AccK/N": "For each image, mean overlap between predicted top-K candidates and GT top-N MOS candidates.",
            "SRCC_PCC": "Mean per-image correlations, matching common GAICD reporting.",
            "teacher_score": "scores.mos, fallback scores.final_score, fallback candidate score.",
        },
    }
    if args.compute_pairwise:
        metrics["metrics"]["pairwise_ranking_acc"] = pairwise_correct / max(pairwise_total, 1)
        metrics["metrics"]["pairwise_pairs"] = pairwise_total

    if args.out_json:
        write_json(args.out_json, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _parse_ints(value: str) -> List[int]:
    out = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not out:
        raise ValueError(f"empty integer list: {value!r}")
    return out


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


def _box_is_normalized(box: Sequence[float]) -> bool:
    try:
        vals = [float(v) for v in box[:4]]
    except (TypeError, ValueError):
        return False
    return all(-1e-6 <= v <= 1.0 + 1e-6 for v in vals)


def _box_for_image_crop(box: Sequence[float], width: int, height: int) -> List[float]:
    vals = [float(v) for v in box[:4]]
    if _box_is_normalized(vals):
        return [
            vals[0] * width,
            vals[1] * height,
            vals[2] * width,
            vals[3] * height,
        ]
    return vals


def _box_for_features(box: Sequence[float], width: int, height: int) -> List[float]:
    vals = [float(v) for v in box[:4]]
    if _box_is_normalized(vals):
        return vals
    return normalize_xyxy(vals, width, height)


def _expand_graph(graph: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in graph.items():
        if torch.is_tensor(value) and value.size(0) == 1:
            out[key] = value.expand(batch_size, *value.shape[1:])
        else:
            out[key] = value
    return out


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
    height, width = img.shape[:2]
    image_tensor = resize_to_tensor(img, image_size).unsqueeze(0).to(device)
    graph = model.encode_graph(image_tensor)
    needs_crop = bool(getattr(model, "uses_crop_image", lambda: True)())
    preds: List[float] = []
    batch_size = max(1, int(batch_size))

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        boxes = [item["box"] for item in batch]
        crops = (
            torch.stack(
                [
                    resize_to_tensor(crop_rgb(img, _box_for_image_crop(box, width, height)), crop_size)
                    for box in boxes
                ]
            ).to(device, non_blocking=True)
            if needs_crop
            else None
        )
        box_feat = torch.tensor(
            [candidate_box_features(_box_for_features(box, width, height)) for box in boxes],
            dtype=torch.float32,
            device=device,
        )
        image_batch = image_tensor.expand(len(batch), -1, -1, -1) if needs_crop else None
        out = model(image_batch, crops, box_feat, graph=_expand_graph(graph, len(batch)))
        score = out.get("score_logit", out["score"])
        preds.extend(float(v) for v in score.detach().cpu().tolist())
    return preds


def _pairwise_accuracy_counts(preds: Sequence[float], teachers: Sequence[float]) -> tuple[int, int]:
    correct = 0
    total = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            teacher_diff = teachers[i] - teachers[j]
            if abs(teacher_diff) <= 1e-12:
                continue
            pred_diff = preds[i] - preds[j]
            correct += int(pred_diff * teacher_diff > 0)
            total += 1
    return correct, total


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _mean_named(values: Dict[str, float], keys: Sequence[str]) -> float:
    present = [values[key] for key in keys if key in values]
    return _mean(present)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return float(cov / ((vx * vy) ** 0.5))


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    return _pearson(_average_ranks(x), _average_ranks(y))


def _average_ranks(values: Sequence[float]) -> List[float]:
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
