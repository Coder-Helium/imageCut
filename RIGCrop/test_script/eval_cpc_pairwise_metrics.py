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

from rigcrop.box_ops import candidate_box_features, normalize_xyxy  # noqa: E402
from rigcrop.image_io import crop_rgb, read_image_rgb, resize_to_tensor  # noqa: E402
from rigcrop.io import iter_jsonl, write_json  # noqa: E402
from rigcrop.model import RIGCropModel  # noqa: E402
from rigcrop.runtime import get_device, load_checkpoint, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RIG-Crop on CPC-style pairwise crop preferences.")
    parser.add_argument("--jsonl", required=True, help="CPC JSONL with candidates and pairwise_preferences.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-pairs-per-record", type=int, default=0)
    parser.add_argument("--derive-pairs-from-scores", action="store_true")
    parser.add_argument("--min-score-gap", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    model = RIGCropModel(**cfg.get("model", {})).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    rng = random.Random(args.seed)
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
    per_image_swap: List[float] = []
    per_image_margin: List[float] = []

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with torch.inference_mode():
        for rec_idx, rec in enumerate(iter_jsonl(args.jsonl), start=1):
            if args.max_records > 0 and rec_idx > args.max_records:
                break
            candidates = _valid_candidates(rec.get("candidates", []) or [])
            candidate_map = {str(cand["candidate_id"]): cand for cand in candidates}
            pairs = list(rec.get("pairwise_preferences", []) or [])
            if not pairs and args.derive_pairs_from_scores:
                pairs = _derive_pairs_from_candidate_scores(candidates, args.min_score_gap)
            pairs = [
                pair
                for pair in pairs
                if str(pair.get("winner")) in candidate_map and str(pair.get("loser")) in candidate_map
            ]
            if args.max_pairs_per_record > 0 and len(pairs) > args.max_pairs_per_record:
                pairs = rng.sample(pairs, args.max_pairs_per_record)
            if len(candidates) < 2 or not pairs:
                records_skipped += 1
                continue

            pred_scores = _predict_candidate_scores(
                model=model,
                rec=rec,
                candidates=candidates,
                device=device,
                image_size=args.image_size,
                crop_size=args.crop_size,
                batch_size=args.batch_size,
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
                image_acc = image_correct / image_total
                per_image_acc.append(image_acc)
                per_image_swap.append(1.0 - image_acc)
                per_image_margin.append(image_margin_sum / image_total)

    pairwise_acc = total_correct / max(total_pairs, 1)
    weighted_pairwise_acc = weighted_correct / max(weighted_total, 1e-12)
    metrics: Dict[str, Any] = {
        "jsonl": str(Path(args.jsonl).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "num_records": records_used,
        "skipped_records": records_skipped,
        "num_pairs": total_pairs,
        "metrics": {
            "pairwise_acc": pairwise_acc,
            "swap_error": 1.0 - pairwise_acc,
            "weighted_pairwise_acc": weighted_pairwise_acc,
            "weighted_swap_error": 1.0 - weighted_pairwise_acc,
            "mean_score_margin": margin_sum / max(total_pairs, 1),
            "weighted_mean_score_margin": weighted_margin_sum / max(weighted_total, 1e-12),
            "mean_per_image_pairwise_acc": _mean(per_image_acc),
            "mean_per_image_swap_error": _mean(per_image_swap),
            "mean_per_image_score_margin": _mean(per_image_margin),
            "tie_rate": tie_pairs / max(total_pairs, 1),
        },
        "notes": {
            "pairwise_acc": "mean(score(winner_crop) > score(loser_crop)) over CPC pairwise preferences.",
            "swap_error": "1 - pairwise_acc, matching CPC-style preference inversion error.",
            "weighted_metrics": "Use pair weight when present; otherwise weight=1.",
        },
    }
    if args.out_json:
        write_json(args.out_json, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


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


def _pair_weight(pair: Dict[str, Any]) -> float:
    try:
        return max(0.0, float(pair.get("weight", 1.0)))
    except (TypeError, ValueError):
        return 1.0


def _candidate_score(cand: Dict[str, Any]) -> float | None:
    scores = cand.get("scores", {}) if isinstance(cand.get("scores"), dict) else {}
    value = scores.get("final_score", scores.get("mos", cand.get("score", None)))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _derive_pairs_from_candidate_scores(candidates: Sequence[Dict[str, Any]], min_score_gap: float) -> List[Dict[str, Any]]:
    scored: List[tuple[str, float]] = []
    raw_scores: List[float] = []
    for cand in candidates:
        score = _candidate_score(cand)
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


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
