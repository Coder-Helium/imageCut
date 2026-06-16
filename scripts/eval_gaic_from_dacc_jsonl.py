#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaic_utils import (  # noqa: E402
    annotation_path_for,
    best_annotation,
    iou_xyxy,
    load_gaic_annotations,
    load_jsonl,
    nearest_annotation,
    read_image_size,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DACC-style crop candidates against GAICD MOS annotations.")
    parser.add_argument("--gaic-root", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--candidate-field", default="candidates")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--score-key", default="pred_score", help="Candidate score key under scores. Falls back to final_score.")
    parser.add_argument("--coord-mode", default="auto", choices=["auto", "square1024", "image"])
    parser.add_argument("--annotation-size", type=int, default=1024)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--print-samples", action="store_true", help="Print per-sample metrics to stdout.")
    args = parser.parse_args()

    records = load_jsonl(args.jsonl)
    rows: List[Dict[str, Any]] = []
    for rec in records:
        image_path = Path(rec["image_path"])
        image_w = int(rec.get("image_width") or 0)
        image_h = int(rec.get("image_height") or 0)
        if image_w <= 0 or image_h <= 0:
            image_w, image_h = read_image_size(image_path)

        ann_path = annotation_path_for(args.gaic_root, image_path)
        anns, resolved_mode = load_gaic_annotations(
            ann_path,
            image_w,
            image_h,
            coord_mode=args.coord_mode,
            annotation_size=args.annotation_size,
            clip=True,
        )
        best_gt = best_annotation(anns)
        candidates = _rank_candidates(rec.get(args.candidate_field, []), args.score_key)
        if not candidates:
            continue
        boxes = [cand["box"] for cand in candidates[: max(args.topk, 1)] if cand.get("box")]
        if not boxes:
            continue

        top1_box = boxes[0]
        top1_nearest, top1_nearest_iou = nearest_annotation(top1_box, anns)
        top1_iou_best = iou_xyxy(top1_box, best_gt.box)
        topk_iou_best = max(iou_xyxy(box, best_gt.box) for box in boxes)
        topk_nearest_mos = max(nearest_annotation(box, anns)[0].mos for box in boxes)
        topk_nearest_iou = max(nearest_annotation(box, anns)[1] for box in boxes)
        rows.append(
            {
                "sample_id": rec.get("sample_id", image_path.stem),
                "resolved_coord_mode": resolved_mode,
                "best_gt_box": best_gt.box,
                "best_gt_mos": best_gt.mos,
                "top1_box": top1_box,
                "top1_iou_to_best_gt": top1_iou_best,
                "top1_nearest_gt_iou": top1_nearest_iou,
                "top1_nearest_gt_mos": top1_nearest.mos,
                "topk_oracle_iou_to_best_gt": topk_iou_best,
                "topk_oracle_nearest_gt_iou": topk_nearest_iou,
                "topk_oracle_nearest_gt_mos": topk_nearest_mos,
                "top1_exact_match": top1_iou_best >= 0.999,
                "top1_acc_iou_075": top1_iou_best >= 0.75,
            }
        )

    summary = _summarize(rows, args.topk)
    output = {"summary": summary, "per_sample": rows}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.out_json:
        write_json(args.out_json, output)
    if args.print_samples:
        print(text)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def _rank_candidates(candidates: List[Dict[str, Any]], score_key: str) -> List[Dict[str, Any]]:
    def score(cand: Dict[str, Any]) -> float:
        scores = cand.get("scores", {}) or {}
        if score_key in scores:
            return float(scores.get(score_key) or 0.0)
        if "final_score" in scores:
            return float(scores.get("final_score") or 0.0)
        return float(cand.get("score", 0.0) or 0.0)

    # If all scores are absent or zero, preserve the existing order.
    vals = [score(c) for c in candidates]
    if not vals or max(vals) == min(vals) == 0.0:
        return list(candidates)
    return sorted(candidates, key=score, reverse=True)


def _summarize(rows: List[Dict[str, Any]], topk: int) -> Dict[str, Any]:
    def mean(key: str) -> float:
        if not rows:
            return 0.0
        return float(sum(float(r[key]) for r in rows) / len(rows))

    return {
        "num_samples": len(rows),
        "topk": topk,
        "mean_best_gt_mos": mean("best_gt_mos"),
        "mean_top1_iou_to_best_gt": mean("top1_iou_to_best_gt"),
        "mean_top1_nearest_gt_iou": mean("top1_nearest_gt_iou"),
        "mean_top1_nearest_gt_mos": mean("top1_nearest_gt_mos"),
        "mean_topk_oracle_iou_to_best_gt": mean("topk_oracle_iou_to_best_gt"),
        "mean_topk_oracle_nearest_gt_iou": mean("topk_oracle_nearest_gt_iou"),
        "mean_topk_oracle_nearest_gt_mos": mean("topk_oracle_nearest_gt_mos"),
        "top1_exact_match_rate": mean("top1_exact_match"),
        "top1_acc_iou_075": mean("top1_acc_iou_075"),
    }


if __name__ == "__main__":
    main()
