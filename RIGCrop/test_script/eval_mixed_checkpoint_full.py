#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rigcrop.box_ops import (  # noqa: E402
    area,
    boundary_cut,
    candidate_box_features,
    clip_box,
    coverage,
    intersection_area,
    normalize_xyxy,
    tensor_sanitize_xyxy,
)
from rigcrop.image_io import crop_rgb, read_image_rgb, resize_to_tensor  # noqa: E402
from rigcrop.io import iter_jsonl, write_json  # noqa: E402
from rigcrop.model import RIGCropModel  # noqa: E402
from rigcrop.runtime import get_device, load_checkpoint, load_config  # noqa: E402
from rigcrop.schema import RELATION_POLICIES, ROLES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Full evaluation for the original mixed CPC+GAIC RIGCrop checkpoint. "
            "It reports candidate-ranking metrics plus generated-query crop quality."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Trained mixed checkpoint, e.g. best.pt.")
    parser.add_argument("--config", required=True, help="Model config used by the checkpoint.")
    parser.add_argument("--gaic-jsonl", default="", help="GAICD val/test JSONL. Optional.")
    parser.add_argument("--cpc-jsonl", default="", help="CPC val/test JSONL. Optional.")
    parser.add_argument("--out-json", required=True, help="Summary JSON output.")
    parser.add_argument("--out-jsonl", default="", help="Optional per-image JSONL details.")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-records", type=int, default=0, help="Limit records per dataset; 0 means all.")
    parser.add_argument("--max-candidates-per-record", type=int, default=0)
    parser.add_argument("--max-pairs-per-record", type=int, default=0)
    parser.add_argument("--derive-cpc-pairs-from-scores", action="store_true")
    parser.add_argument("--min-score-gap", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pred-topk", default="1,2,3,4", help="Predicted K values for GAICD AccK/N.")
    parser.add_argument("--gt-topn", default="5,10", help="GT top-N values for GAICD AccK/N.")
    parser.add_argument("--main-threshold", type=float, default=0.95)
    parser.add_argument("--key-threshold", type=float, default=0.90)
    parser.add_argument("--relation-threshold", type=float, default=0.90)
    parser.add_argument("--distractor-threshold", type=float, default=0.50)
    parser.add_argument("--progress-interval", type=int, default=50)
    args = parser.parse_args()

    if not args.gaic_jsonl and not args.cpc_jsonl:
        raise SystemExit("At least one of --gaic-jsonl or --cpc-jsonl is required.")

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    model = RIGCropModel(**cfg.get("model", {})).to(device)
    ckpt = load_checkpoint(args.checkpoint, model)
    model.eval()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    per_record_path = Path(args.out_jsonl) if args.out_jsonl else None
    per_record_fh = None
    if per_record_path is not None:
        per_record_path.parent.mkdir(parents=True, exist_ok=True)
        per_record_fh = per_record_path.open("w", encoding="utf-8")

    try:
        summary: Dict[str, Any] = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "config": str(Path(args.config).resolve()),
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_metrics": ckpt.get("metrics", {}),
            "image_size": args.image_size,
            "crop_size": args.crop_size,
            "batch_size": args.batch_size,
            "metrics": {},
            "notes": {
                "candidate_score_metrics": "Evaluate the model score head on dataset candidate crops.",
                "generated_crop_metrics": "Evaluate model-generated query_boxes from encode_graph, independent of dataset candidate enumeration.",
                "query_score": "Select the generated query box with the model query_score head.",
                "score_head": "Re-score generated query boxes with the crop score head and select the best one.",
                "query_oracle": "Upper-bound over generated query boxes against candidate/MOS targets; not an executable inference policy.",
                "semantic_preservation": "Coverage is measured against rig_targets nodes in normalized xyxy coordinates.",
            },
        }

        with torch.inference_mode():
            if args.gaic_jsonl:
                summary["metrics"]["gaic"] = evaluate_gaic(
                    model=model,
                    jsonl_path=args.gaic_jsonl,
                    device=device,
                    image_size=args.image_size,
                    crop_size=args.crop_size,
                    batch_size=args.batch_size,
                    max_records=args.max_records,
                    max_candidates_per_record=args.max_candidates_per_record,
                    pred_topk=_parse_ints(args.pred_topk),
                    gt_topn=_parse_ints(args.gt_topn),
                    thresholds=_thresholds(args),
                    per_record_fh=per_record_fh,
                    progress_interval=args.progress_interval,
                )

            if args.cpc_jsonl:
                summary["metrics"]["cpc"] = evaluate_cpc(
                    model=model,
                    jsonl_path=args.cpc_jsonl,
                    device=device,
                    image_size=args.image_size,
                    crop_size=args.crop_size,
                    batch_size=args.batch_size,
                    max_records=args.max_records,
                    max_candidates_per_record=args.max_candidates_per_record,
                    max_pairs_per_record=args.max_pairs_per_record,
                    derive_pairs_from_scores=args.derive_cpc_pairs_from_scores,
                    min_score_gap=args.min_score_gap,
                    seed=args.seed,
                    thresholds=_thresholds(args),
                    per_record_fh=per_record_fh,
                    progress_interval=args.progress_interval,
                )
    finally:
        if per_record_fh is not None:
            per_record_fh.close()

    write_json(args.out_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluate_gaic(
    model: RIGCropModel,
    jsonl_path: str,
    device: torch.device,
    image_size: int,
    crop_size: int,
    batch_size: int,
    max_records: int,
    max_candidates_per_record: int,
    pred_topk: Sequence[int],
    gt_topn: Sequence[int],
    thresholds: Dict[str, float],
    per_record_fh: Any,
    progress_interval: int,
) -> Dict[str, Any]:
    acc_sum = {(k, n): 0.0 for n in gt_topn for k in pred_topk}
    acc_count = {(k, n): 0 for n in gt_topn for k in pred_topk}
    per_image_srcc: List[float] = []
    per_image_pcc: List[float] = []
    global_pred: List[float] = []
    global_teacher: List[float] = []
    pairwise_correct = 0
    pairwise_total = 0

    generated = _new_generated_accumulator()
    num_records = 0
    num_candidates = 0
    skipped_records = 0

    for rec_idx, rec in enumerate(iter_jsonl(jsonl_path), start=1):
        if max_records > 0 and rec_idx > max_records:
            break
        candidates = _scored_candidates(rec.get("candidates", []) or [])
        if max_candidates_per_record > 0:
            candidates = candidates[:max_candidates_per_record]
        if len(candidates) < 2:
            skipped_records += 1
            continue

        try:
            record_ctx = _encode_record(model, rec, device, image_size)
        except Exception as exc:  # pragma: no cover - kept for server batch robustness.
            skipped_records += 1
            _log_progress("gaic", rec_idx, max_records, f"skip image load/encode failed: {exc}", progress_interval)
            continue

        pred_scores = _score_candidate_boxes(
            model=model,
            record_ctx=record_ctx,
            candidates=candidates,
            crop_size=crop_size,
            batch_size=batch_size,
        )
        if len(pred_scores) != len(candidates):
            skipped_records += 1
            continue

        teachers = [float(item["teacher_score"]) for item in candidates]
        pred_order = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
        teacher_order = sorted(range(len(teachers)), key=lambda i: teachers[i], reverse=True)

        for n in gt_topn:
            gt_top = set(teacher_order[: min(n, len(teacher_order))])
            for k in pred_topk:
                pred_k = pred_order[: min(k, len(pred_order))]
                if pred_k:
                    acc_sum[(k, n)] += sum(1 for idx in pred_k if idx in gt_top) / len(pred_k)
                    acc_count[(k, n)] += 1

        ok, total = _pairwise_accuracy_counts(pred_scores, teachers)
        pairwise_correct += ok
        pairwise_total += total
        per_image_srcc.append(_spearman(pred_scores, teachers))
        per_image_pcc.append(_pearson(pred_scores, teachers))
        global_pred.extend(pred_scores)
        global_teacher.extend(teachers)

        generated_detail = _evaluate_generated_queries(
            model=model,
            record_ctx=record_ctx,
            candidates=candidates,
            thresholds=thresholds,
        )
        _update_generated_accumulator(generated, generated_detail)

        num_records += 1
        num_candidates += len(candidates)
        _write_record(
            per_record_fh,
            {
                "dataset": "gaic",
                "sample_id": rec.get("sample_id", ""),
                "num_candidates": len(candidates),
                "candidate_srcc": per_image_srcc[-1],
                "candidate_pcc": per_image_pcc[-1],
                "generated": generated_detail,
            },
        )
        _log_progress("gaic", rec_idx, max_records, f"records={num_records}", progress_interval)

    acc_k_over_n = {
        f"Acc{k}/{n}": acc_sum[(k, n)] / max(acc_count[(k, n)], 1)
        for n in gt_topn
        for k in pred_topk
    }
    return {
        "jsonl": str(Path(jsonl_path).resolve()),
        "num_records": num_records,
        "num_candidates": num_candidates,
        "skipped_records": skipped_records,
        "candidate_score_metrics": {
            **acc_k_over_n,
            "Acc5": _mean_named(acc_k_over_n, [f"Acc{k}/5" for k in pred_topk]),
            "Acc10": _mean_named(acc_k_over_n, [f"Acc{k}/10" for k in pred_topk]),
            "SRCC": _mean(per_image_srcc),
            "PCC": _mean(per_image_pcc),
            "SRCC_global": _spearman(global_pred, global_teacher),
            "PCC_global": _pearson(global_pred, global_teacher),
            "pairwise_ranking_acc": pairwise_correct / max(pairwise_total, 1),
            "pairwise_pairs": pairwise_total,
        },
        "generated_crop_metrics": _finalize_generated_accumulator(generated),
    }


def evaluate_cpc(
    model: RIGCropModel,
    jsonl_path: str,
    device: torch.device,
    image_size: int,
    crop_size: int,
    batch_size: int,
    max_records: int,
    max_candidates_per_record: int,
    max_pairs_per_record: int,
    derive_pairs_from_scores: bool,
    min_score_gap: float,
    seed: int,
    thresholds: Dict[str, float],
    per_record_fh: Any,
    progress_interval: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    total_pairs = 0
    total_correct = 0
    weighted_total = 0.0
    weighted_correct = 0.0
    margin_sum = 0.0
    weighted_margin_sum = 0.0
    tie_pairs = 0
    records_used = 0
    records_skipped = 0
    per_image_acc: List[float] = []
    per_image_margin: List[float] = []
    generated = _new_generated_accumulator()

    for rec_idx, rec in enumerate(iter_jsonl(jsonl_path), start=1):
        if max_records > 0 and rec_idx > max_records:
            break
        candidates = _valid_candidates(rec.get("candidates", []) or [])
        if max_candidates_per_record > 0:
            candidates = candidates[:max_candidates_per_record]
        candidate_map = {str(cand["candidate_id"]): cand for cand in candidates}
        pairs = list(rec.get("pairwise_preferences", []) or [])
        if not pairs and derive_pairs_from_scores:
            pairs = _derive_pairs_from_candidate_scores(candidates, min_score_gap)
        pairs = [
            pair
            for pair in pairs
            if str(pair.get("winner")) in candidate_map and str(pair.get("loser")) in candidate_map
        ]
        if max_pairs_per_record > 0 and len(pairs) > max_pairs_per_record:
            pairs = rng.sample(pairs, max_pairs_per_record)
        if len(candidates) < 2 or not pairs:
            records_skipped += 1
            continue

        try:
            record_ctx = _encode_record(model, rec, device, image_size)
        except Exception as exc:  # pragma: no cover - kept for server batch robustness.
            records_skipped += 1
            _log_progress("cpc", rec_idx, max_records, f"skip image load/encode failed: {exc}", progress_interval)
            continue

        pred_scores = _score_candidate_boxes(
            model=model,
            record_ctx=record_ctx,
            candidates=candidates,
            crop_size=crop_size,
            batch_size=batch_size,
        )
        score_map = {
            str(cand["candidate_id"]): float(score)
            for cand, score in zip(candidates, pred_scores)
        }

        image_total = 0
        image_correct = 0
        image_margin_sum = 0.0
        for pair in pairs:
            winner = str(pair.get("winner"))
            loser = str(pair.get("loser"))
            weight = _pair_weight(pair)
            margin = score_map[winner] - score_map[loser]
            correct = margin > 0.0
            tie_pairs += int(abs(margin) <= 1e-12)

            total_pairs += 1
            total_correct += int(correct)
            weighted_total += weight
            weighted_correct += weight if correct else 0.0
            margin_sum += margin
            weighted_margin_sum += weight * margin

            image_total += 1
            image_correct += int(correct)
            image_margin_sum += margin

        if image_total > 0:
            records_used += 1
            per_image_acc.append(image_correct / image_total)
            per_image_margin.append(image_margin_sum / image_total)

        generated_detail = _evaluate_generated_queries(
            model=model,
            record_ctx=record_ctx,
            candidates=_scored_candidates(candidates),
            thresholds=thresholds,
        )
        _update_generated_accumulator(generated, generated_detail)

        _write_record(
            per_record_fh,
            {
                "dataset": "cpc",
                "sample_id": rec.get("sample_id", ""),
                "num_candidates": len(candidates),
                "num_pairs": image_total,
                "pairwise_acc": image_correct / max(image_total, 1),
                "mean_margin": image_margin_sum / max(image_total, 1),
                "generated": generated_detail,
            },
        )
        _log_progress("cpc", rec_idx, max_records, f"records={records_used}", progress_interval)

    pairwise_acc = total_correct / max(total_pairs, 1)
    weighted_pairwise_acc = weighted_correct / max(weighted_total, 1e-12)
    return {
        "jsonl": str(Path(jsonl_path).resolve()),
        "num_records": records_used,
        "skipped_records": records_skipped,
        "num_pairs": total_pairs,
        "candidate_score_metrics": {
            "pairwise_acc": pairwise_acc,
            "swap_error": 1.0 - pairwise_acc,
            "weighted_pairwise_acc": weighted_pairwise_acc,
            "weighted_swap_error": 1.0 - weighted_pairwise_acc,
            "mean_score_margin": margin_sum / max(total_pairs, 1),
            "weighted_mean_score_margin": weighted_margin_sum / max(weighted_total, 1e-12),
            "mean_per_image_pairwise_acc": _mean(per_image_acc),
            "mean_per_image_swap_error": 1.0 - _mean(per_image_acc),
            "mean_per_image_score_margin": _mean(per_image_margin),
            "tie_rate": tie_pairs / max(total_pairs, 1),
        },
        "generated_crop_metrics": _finalize_generated_accumulator(generated),
    }


def _encode_record(
    model: RIGCropModel,
    rec: Dict[str, Any],
    device: torch.device,
    image_size: int,
) -> Dict[str, Any]:
    img = read_image_rgb(rec["image_path"])
    height, width = img.shape[:2]
    image_tensor = resize_to_tensor(img, image_size).unsqueeze(0).to(device)
    graph = model.encode_graph(image_tensor)
    return {
        "record": rec,
        "image": img,
        "height": height,
        "width": width,
        "image_tensor": image_tensor,
        "graph": graph,
    }


def _score_candidate_boxes(
    model: RIGCropModel,
    record_ctx: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    crop_size: int,
    batch_size: int,
) -> List[float]:
    img = record_ctx["image"]
    height = int(record_ctx["height"])
    width = int(record_ctx["width"])
    image_tensor = record_ctx["image_tensor"]
    graph = record_ctx["graph"]
    device = image_tensor.device
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


def _evaluate_generated_queries(
    model: RIGCropModel,
    record_ctx: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    graph = record_ctx["graph"]
    if "query_boxes" not in graph:
        return {"available": False, "reason": "model graph has no query_boxes"}

    query_boxes = tensor_sanitize_xyxy(graph["query_boxes"][0]).detach().cpu().tolist()
    query_scores = graph.get("query_scores")
    query_score_values = (
        [float(v) for v in query_scores[0].detach().cpu().tolist()]
        if torch.is_tensor(query_scores)
        else [0.0 for _ in query_boxes]
    )
    valid_indices = [
        idx
        for idx, box in enumerate(query_boxes)
        if area(box) > 1e-4
    ]
    if not valid_indices:
        return {"available": False, "reason": "no valid generated query boxes"}

    score_head_values = _score_normalized_boxes(
        model=model,
        record_ctx=record_ctx,
        boxes=[query_boxes[idx] for idx in valid_indices],
    )
    query_boxes = [query_boxes[idx] for idx in valid_indices]
    query_score_values = [query_score_values[idx] for idx in valid_indices]

    selections = {
        "query_score": _argmax_index(query_score_values),
        "score_head": _argmax_index(score_head_values),
    }
    detail: Dict[str, Any] = {
        "available": True,
        "num_query_boxes": len(query_boxes),
        "query_oracle": _query_oracle_metrics(
            query_boxes,
            candidates,
            width=int(record_ctx["width"]),
            height=int(record_ctx["height"]),
        ),
    }
    for name, idx in selections.items():
        selected_box = query_boxes[idx]
        detail[name] = {
            **_candidate_alignment_metrics(
                selected_box,
                candidates,
                width=int(record_ctx["width"]),
                height=int(record_ctx["height"]),
            ),
            **_semantic_crop_metrics(record_ctx["record"], selected_box, thresholds),
            "selected_query_index": int(idx),
            "selected_query_score": float(query_score_values[idx]),
            "selected_score_head": float(score_head_values[idx]),
            "crop_area": area(selected_box),
            "crop_box_norm": [float(v) for v in selected_box],
        }
    return detail


def _score_normalized_boxes(
    model: RIGCropModel,
    record_ctx: Dict[str, Any],
    boxes: Sequence[Sequence[float]],
) -> List[float]:
    if not boxes:
        return []
    image_tensor = record_ctx["image_tensor"]
    graph = record_ctx["graph"]
    device = image_tensor.device
    box_feat = torch.tensor(
        [candidate_box_features(box) for box in boxes],
        dtype=torch.float32,
        device=device,
    )
    out = model(None, None, box_feat, graph=_expand_graph(graph, len(boxes)))
    score = out.get("score_logit", out["score"])
    return [float(v) for v in score.detach().cpu().tolist()]


def _candidate_alignment_metrics(
    crop_box: Sequence[float],
    candidates: Sequence[Dict[str, Any]],
    width: int,
    height: int,
) -> Dict[str, Any]:
    scored = _scored_candidates(candidates)
    if not scored:
        return {
            "has_scored_candidates": False,
            "iou_to_best_candidate": 0.0,
            "center_l1_to_best_candidate": 0.0,
            "nearest_candidate_iou": 0.0,
            "nearest_candidate_score": 0.0,
            "nearest_candidate_rank": 0,
            "nearest_in_top5": 0.0,
            "nearest_in_top10": 0.0,
            "nearest_score_ratio_to_best": 0.0,
            "score_delta_vs_full_candidate": 0.0,
            "improves_full_candidate": 0.0,
        }

    candidate_boxes = [_box_for_features(cand["box"], width, height) for cand in scored]
    teacher_scores = [float(cand["teacher_score"]) for cand in scored]
    order = sorted(range(len(scored)), key=lambda i: teacher_scores[i], reverse=True)
    ranks = {idx: rank + 1 for rank, idx in enumerate(order)}
    best_idx = order[0]
    nearest_idx = max(range(len(scored)), key=lambda i: _iou(crop_box, candidate_boxes[i]))
    best_score = teacher_scores[best_idx]
    nearest_score = teacher_scores[nearest_idx]
    full_idx = _find_full_candidate_idx(candidate_boxes)
    full_score = teacher_scores[full_idx] if full_idx is not None else None

    return {
        "has_scored_candidates": True,
        "iou_to_best_candidate": _iou(crop_box, candidate_boxes[best_idx]),
        "center_l1_to_best_candidate": _center_l1(crop_box, candidate_boxes[best_idx]),
        "nearest_candidate_iou": _iou(crop_box, candidate_boxes[nearest_idx]),
        "nearest_candidate_score": nearest_score,
        "nearest_candidate_rank": int(ranks[nearest_idx]),
        "nearest_in_top5": float(ranks[nearest_idx] <= 5),
        "nearest_in_top10": float(ranks[nearest_idx] <= 10),
        "nearest_score_ratio_to_best": nearest_score / best_score if abs(best_score) > 1e-12 else 0.0,
        "score_delta_vs_full_candidate": nearest_score - full_score if full_score is not None else 0.0,
        "improves_full_candidate": float(nearest_score > full_score) if full_score is not None else 0.0,
    }


def _query_oracle_metrics(
    query_boxes: Sequence[Sequence[float]],
    candidates: Sequence[Dict[str, Any]],
    width: int,
    height: int,
) -> Dict[str, Any]:
    scored = _scored_candidates(candidates)
    if not scored:
        return {
            "has_scored_candidates": False,
            "max_iou_to_best_candidate": 0.0,
            "best_nearest_candidate_score": 0.0,
            "best_nearest_candidate_rank": 0,
            "best_nearest_in_top5": 0.0,
            "best_nearest_in_top10": 0.0,
        }
    candidate_boxes = [_box_for_features(cand["box"], width, height) for cand in scored]
    teacher_scores = [float(cand["teacher_score"]) for cand in scored]
    order = sorted(range(len(scored)), key=lambda i: teacher_scores[i], reverse=True)
    ranks = {idx: rank + 1 for rank, idx in enumerate(order)}
    best_idx = order[0]
    max_iou_to_best = max(_iou(box, candidate_boxes[best_idx]) for box in query_boxes)

    nearest_for_query = [
        max(range(len(scored)), key=lambda i: _iou(box, candidate_boxes[i]))
        for box in query_boxes
    ]
    best_nearest_idx = max(nearest_for_query, key=lambda i: teacher_scores[i])
    return {
        "has_scored_candidates": True,
        "max_iou_to_best_candidate": max_iou_to_best,
        "best_nearest_candidate_score": teacher_scores[best_nearest_idx],
        "best_nearest_candidate_rank": int(ranks[best_nearest_idx]),
        "best_nearest_in_top5": float(ranks[best_nearest_idx] <= 5),
        "best_nearest_in_top10": float(ranks[best_nearest_idx] <= 10),
    }


def _semantic_crop_metrics(
    rec: Dict[str, Any],
    crop_box: Sequence[float],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    rig = rec.get("rig_targets", {}) if isinstance(rec.get("rig_targets"), dict) else {}
    nodes = [node for node in rig.get("nodes", []) or [] if _valid_node(node)]
    if not nodes:
        return {
            "has_rig_targets": False,
            "main_subject_coverage": 0.0,
            "main_subject_preserved": 0.0,
            "key_object_mean_coverage": 0.0,
            "key_object_all_preserved": 0.0,
            "non_distractor_mean_coverage": 0.0,
            "distractor_mean_coverage": 0.0,
            "distractor_excluded": 0.0,
            "relation_preserved": 0.0,
            "boundary_cut_mean": 0.0,
            "no_crop_area_ge_0.95": float(area(crop_box) >= 0.95),
        }

    role_groups: Dict[str, List[Dict[str, Any]]] = {role: [] for role in ROLES}
    for node in nodes:
        role_groups.setdefault(str(node.get("role", "padding")), []).append(node)

    cov_by_node = [_node_coverage(node, crop_box) for node in nodes]
    non_distractor_cov = [
        cov
        for node, cov in zip(nodes, cov_by_node)
        if node.get("role") not in {"distractor", "padding"}
    ]
    distractor_cov = [
        cov
        for node, cov in zip(nodes, cov_by_node)
        if node.get("role") == "distractor"
    ]
    main_covs = [_node_coverage(node, crop_box) for node in role_groups.get("main_subject", [])]
    key_covs = [_node_coverage(node, crop_box) for node in role_groups.get("key_object", [])]
    boundary_values = [
        boundary_cut(node.get("bbox_norm", [0, 0, 0, 0]), crop_box)
        for node in nodes
        if node.get("role") not in {"distractor", "padding"}
    ]

    relation_rate = _relation_preservation_rate(rig, nodes, crop_box, thresholds["relation"])
    return {
        "has_rig_targets": True,
        "main_subject_coverage": max(main_covs) if main_covs else 0.0,
        "main_subject_preserved": float(bool(main_covs) and max(main_covs) >= thresholds["main"]),
        "key_object_mean_coverage": _mean(key_covs),
        "key_object_all_preserved": float(bool(key_covs) and min(key_covs) >= thresholds["key"]),
        "non_distractor_mean_coverage": _mean(non_distractor_cov),
        "distractor_mean_coverage": _mean(distractor_cov),
        "distractor_excluded": float(bool(distractor_cov) and _mean(distractor_cov) <= thresholds["distractor"]),
        "relation_preserved": relation_rate,
        "boundary_cut_mean": _mean(boundary_values),
        "no_crop_area_ge_0.95": float(area(crop_box) >= 0.95),
    }


def _relation_preservation_rate(
    rig: Dict[str, Any],
    nodes: Sequence[Dict[str, Any]],
    crop_box: Sequence[float],
    threshold: float,
) -> float:
    relations = rig.get("relations", {}) if isinstance(rig.get("relations"), dict) else {}
    mask = relations.get("mask", []) or []
    policy = relations.get("policy", []) or []
    preserve_ids = {
        RELATION_POLICIES.index("preserve_relation"),
        RELATION_POLICIES.index("optional_preserve"),
        RELATION_POLICIES.index("avoid_cutting"),
        RELATION_POLICIES.index("leave_space"),
    }
    node_ids = {int(node.get("node_id", idx)): node for idx, node in enumerate(nodes)}
    total = 0
    ok = 0
    for i, row in enumerate(mask):
        if not isinstance(row, Sequence):
            continue
        for j, active in enumerate(row):
            if i == j or not active:
                continue
            try:
                policy_id = int(policy[i][j])
            except (IndexError, TypeError, ValueError):
                policy_id = 0
            if policy_id not in preserve_ids:
                continue
            node_i = node_ids.get(i)
            node_j = node_ids.get(j)
            if node_i is None or node_j is None:
                continue
            total += 1
            cov_i = _node_coverage(node_i, crop_box)
            cov_j = _node_coverage(node_j, crop_box)
            ok += int(cov_i >= threshold and cov_j >= threshold)
    return ok / total if total > 0 else 0.0


def _new_generated_accumulator() -> Dict[str, Any]:
    return {
        "query_score": {},
        "score_head": {},
        "query_oracle": {},
        "num_available": 0,
        "num_unavailable": 0,
    }


def _update_generated_accumulator(acc: Dict[str, Any], detail: Dict[str, Any]) -> None:
    if not detail.get("available"):
        acc["num_unavailable"] += 1
        return
    acc["num_available"] += 1
    for section in ["query_score", "score_head"]:
        for key, value in detail.get(section, {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                _append_metric(acc[section], key, float(value))
    for key, value in detail.get("query_oracle", {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            _append_metric(acc["query_oracle"], key, float(value))


def _finalize_generated_accumulator(acc: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "num_available": acc["num_available"],
        "num_unavailable": acc["num_unavailable"],
    }
    for section in ["query_score", "score_head", "query_oracle"]:
        out[section] = {
            key: _mean(values)
            for key, values in sorted(acc[section].items())
        }
    return out


def _append_metric(store: Dict[str, List[float]], key: str, value: float) -> None:
    store.setdefault(key, []).append(value)


def _valid_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        box = cand.get("box")
        if not isinstance(box, list) or len(box) < 4:
            continue
        item = dict(cand)
        item["candidate_id"] = str(cand.get("candidate_id", idx))
        out.append(item)
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
    for key in [
        "mos",
        "final_score",
        "cpc_raw_score",
        "score",
        "aesthetic_score",
        "composition_score",
        "technical_score",
    ]:
        value = scores.get(key, None)
        if value is not None:
            parsed = _safe_float(value)
            if parsed is not None:
                return parsed
    return _safe_float(cand.get("score", None))


def _derive_pairs_from_candidate_scores(candidates: Sequence[Dict[str, Any]], min_score_gap: float) -> List[Dict[str, Any]]:
    scored: List[tuple[str, float]] = []
    raw_scores: List[float] = []
    for cand in candidates:
        score = _teacher_score(cand)
        if score is None:
            continue
        cid = str(cand.get("candidate_id"))
        scored.append((cid, score))
        raw_scores.append(score)
    if not scored:
        return []
    lo = min(raw_scores)
    hi = max(raw_scores)

    def unit(v: float) -> float:
        if 0.0 <= lo and hi <= 1.0:
            return v
        if 1.0 <= lo and hi <= 5.0:
            return (v - 1.0) / 4.0
        return 0.5 if hi <= lo else (v - lo) / (hi - lo)

    pairs: List[Dict[str, Any]] = []
    for i, (cid_a, score_a) in enumerate(scored):
        for cid_b, score_b in scored[i + 1 :]:
            diff = unit(score_a) - unit(score_b)
            if abs(diff) < min_score_gap:
                continue
            if diff > 0:
                pairs.append({"winner": cid_a, "loser": cid_b, "weight": abs(diff), "source": "score_derived_pair"})
            else:
                pairs.append({"winner": cid_b, "loser": cid_a, "weight": abs(diff), "source": "score_derived_pair"})
    return pairs


def _valid_node(node: Dict[str, Any]) -> bool:
    return bool(node.get("valid")) and bool(node.get("has_box")) and area(node.get("bbox_norm", [0, 0, 0, 0])) > 1e-8


def _node_coverage(node: Dict[str, Any], crop_box: Sequence[float]) -> float:
    return coverage(node.get("bbox_norm", [0, 0, 0, 0]), crop_box)


def _pair_weight(pair: Dict[str, Any]) -> float:
    value = _safe_float(pair.get("weight", 1.0))
    return max(0.0, value if value is not None else 1.0)


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
        return clip_box(vals)
    return normalize_xyxy(vals, width, height)


def _expand_graph(graph: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in graph.items():
        if torch.is_tensor(value) and value.size(0) == 1:
            out[key] = value.expand(batch_size, *value.shape[1:])
        else:
            out[key] = value
    return out


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    denom = area(a) + area(b) - intersection_area(a, b)
    return intersection_area(a, b) / denom if denom > 1e-12 else 0.0


def _center_l1(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = clip_box(a)
    bx1, by1, bx2, by2 = clip_box(b)
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    return (abs(acx - bcx) + abs(acy - bcy)) / 2.0


def _find_full_candidate_idx(candidate_boxes: Sequence[Sequence[float]]) -> int | None:
    full = [0.0, 0.0, 1.0, 1.0]
    best_idx: int | None = None
    best_iou = 0.0
    for idx, box in enumerate(candidate_boxes):
        iou = _iou(box, full)
        if iou > best_iou:
            best_idx = idx
            best_iou = iou
    return best_idx if best_iou >= 0.98 else None


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


def _argmax_index(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def _parse_ints(value: str) -> List[int]:
    out = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not out:
        raise ValueError(f"empty integer list: {value!r}")
    return out


def _thresholds(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "main": float(args.main_threshold),
        "key": float(args.key_threshold),
        "relation": float(args.relation_threshold),
        "distractor": float(args.distractor_threshold),
    }


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


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_record(fh: Any, payload: Dict[str, Any]) -> None:
    if fh is None:
        return
    fh.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _log_progress(dataset: str, idx: int, max_records: int, message: str, interval: int) -> None:
    if interval <= 0:
        return
    if idx == 1 or idx % interval == 0:
        suffix = f"/{max_records}" if max_records > 0 else ""
        print(f"[eval-{dataset}] record={idx}{suffix} {message}", flush=True)


if __name__ == "__main__":
    main()
