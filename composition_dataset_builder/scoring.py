from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .geometry import BBox, contains, intersection_area, normalize_score_1_5
from .io_utils import crop_image
from .schema import Candidate, MaskRecord


class AestheticScorer:
    def score(self, image_bgr) -> float:
        return 3.5


class TorchEATAestheticScorer(AestheticScorer):
    def __init__(self, model_path: str, project_root: str = "caption-rule-co"):
        import torch

        root = Path(project_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from TorchEATPredictor import TorchEATPredictor
        except Exception as exc:
            raise RuntimeError("TorchEATPredictor import failed") from exc
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = TorchEATPredictor(model_path=model_path, device=device)

    def score(self, image_bgr) -> float:
        value = float(self.model.evaluate(image_bgr))
        # TorchEAT scales can differ by checkpoint; keep it bounded but not over-normalized.
        if 1.0 <= value <= 5.0:
            return value
        if 0.0 <= value <= 1.0:
            return normalize_score_1_5(value)
        return max(1.0, min(5.0, value))


def create_aesthetic_scorer(kind: str, model_path: str = "", project_root: str = "caption-rule-co") -> AestheticScorer:
    kind = (kind or "none").lower()
    if kind in {"none", "neutral"}:
        return AestheticScorer()
    if kind in {"torcheat", "torch_eat"}:
        if not model_path:
            raise ValueError("--aesthetic-model is required for TorchEAT scoring")
        return TorchEATAestheticScorer(model_path=model_path, project_root=project_root)
    raise ValueError(f"Unknown aesthetic scorer: {kind}")


def score_candidates(
    image_bgr,
    candidates: List[Candidate],
    masks: Dict[str, List[MaskRecord]],
    crop_state_graph: Dict[str, Any],
    aesthetic_scorer: Optional[AestheticScorer] = None,
) -> List[Candidate]:
    aesthetic_scorer = aesthetic_scorer or AestheticScorer()
    scored: List[Candidate] = []
    for cand in candidates:
        features = compute_candidate_features(cand.box, masks, crop_state_graph, image_bgr.shape[1], image_bgr.shape[0])
        scores = compute_scores(features, cand, crop_state_graph)
        crop = crop_image(image_bgr, cand.box)
        scores["aesthetic_score"] = float(aesthetic_scorer.score(crop))
        scores["vlm_preference"] = 3.5
        scores["final_score"] = final_score(scores)
        cand.features = features
        cand.scores = scores
        cand.quality_label = quality_label(scores["final_score"])
        scored.append(cand)

    scored.sort(key=lambda c: c.scores.get("final_score", 0.0), reverse=True)
    for idx, cand in enumerate(scored, start=1):
        cand.rank = idx
    return scored


def compute_candidate_features(box: BBox, masks: Dict[str, List[MaskRecord]], graph: Dict[str, Any], image_w: int, image_h: int) -> Dict[str, Any]:
    preserve = masks.get("preserve_masks", [])
    relation = masks.get("relation_masks", [])
    env = masks.get("environment_masks", [])
    distractors = masks.get("distractor_masks", [])

    subject_coverage = weighted_coverage(box, preserve)
    key_object_coverage = weighted_coverage(box, relation)
    env_coverage = weighted_coverage(box, env)
    relation_boxes = preserve + relation
    relation_coverage = weighted_coverage(box, relation_boxes)
    distractor_kept = weighted_coverage(box, distractors)
    distractor_removed = 1.0 - distractor_kept if distractors else 0.5

    subject_position_type = "unknown"
    subject_center_norm = [0.5, 0.5]
    margins = {"margin_top": 0.0, "margin_bottom": 0.0, "margin_left": 0.0, "margin_right": 0.0}
    if graph.get("subject", {}).get("bbox"):
        sb = BBox.from_seq(graph["subject"]["bbox"])
        cx, cy = sb.center()
        subject_center_norm = [
            (cx - box.x1) / max(box.w(), 1.0),
            (cy - box.y1) / max(box.h(), 1.0),
        ]
        x = subject_center_norm[0]
        if 0.42 <= x <= 0.58:
            subject_position_type = "center"
        elif 0.25 <= x < 0.42:
            subject_position_type = "left_third"
        elif 0.58 < x <= 0.75:
            subject_position_type = "right_third"
        else:
            subject_position_type = "edge"
        margins = {
            "margin_top": max(0.0, (sb.y1 - box.y1) / max(box.h(), 1.0)),
            "margin_bottom": max(0.0, (box.y2 - sb.y2) / max(box.h(), 1.0)),
            "margin_left": max(0.0, (sb.x1 - box.x1) / max(box.w(), 1.0)),
            "margin_right": max(0.0, (box.x2 - sb.x2) / max(box.w(), 1.0)),
        }

    return {
        "area_ratio": box.area() / max(float(image_w * image_h), 1.0),
        "subject_coverage": subject_coverage,
        "key_object_coverage": key_object_coverage,
        "relation_coverage": relation_coverage,
        "environment_coverage": env_coverage,
        "distractor_removed_ratio": distractor_removed,
        "subject_center_norm": subject_center_norm,
        "subject_position_type": subject_position_type,
        **margins,
    }


def compute_scores(features: Dict[str, Any], cand: Candidate, graph: Dict[str, Any]) -> Dict[str, float]:
    subject_integrity = coverage_to_score(features.get("subject_coverage", 1.0))
    key_object_integrity = coverage_to_score(features.get("key_object_coverage", 1.0))
    relation_preservation = coverage_to_score(features.get("relation_coverage", 1.0))
    environment_preservation = coverage_to_score(features.get("environment_coverage", 0.7))
    distractor_removal = normalize_score_1_5(features.get("distractor_removed_ratio", 0.5))
    composition_position = composition_position_score(features)
    penalty = 0.0
    if cand.source == "negative_synthetic":
        penalty += 1.2
    if features.get("subject_coverage", 1.0) < 0.90:
        penalty += 1.5
    if features.get("key_object_coverage", 1.0) < 0.90 and graph.get("key_objects"):
        penalty += 1.0
    return {
        "subject_integrity": subject_integrity,
        "key_object_integrity": key_object_integrity,
        "relation_preservation": relation_preservation,
        "composition_position": composition_position,
        "environment_preservation": environment_preservation,
        "distractor_removal": distractor_removal,
        "penalty": penalty,
    }


def final_score(scores: Dict[str, float]) -> float:
    weights = {
        "subject_integrity": 0.25,
        "key_object_integrity": 0.15,
        "relation_preservation": 0.15,
        "composition_position": 0.15,
        "environment_preservation": 0.05,
        "distractor_removal": 0.10,
        "aesthetic_score": 0.10,
        "vlm_preference": 0.05,
    }
    total = sum(scores.get(k, 3.0) * w for k, w in weights.items())
    total -= scores.get("penalty", 0.0)
    return float(max(1.0, min(5.0, total)))


def weighted_coverage(crop_box: BBox, mask_records: List[MaskRecord]) -> float:
    if not mask_records:
        return 1.0
    numer = 0.0
    denom = 0.0
    for rec in mask_records:
        weight = max(0.05, rec.importance)
        denom += weight
        numer += weight * box_coverage(crop_box, rec.bbox)
    return numer / max(denom, 1e-6)


def box_coverage(crop_box: BBox, target_box: BBox) -> float:
    return intersection_area(crop_box, target_box) / max(target_box.area(), 1.0)


def coverage_to_score(value: float) -> float:
    value = float(value)
    if value >= 0.98:
        return 5.0
    if value >= 0.95:
        return 4.5
    if value >= 0.90:
        return 3.7
    if value >= 0.80:
        return 2.5
    return 1.0


def composition_position_score(features: Dict[str, Any]) -> float:
    pos = features.get("subject_position_type")
    if pos in {"center", "left_third", "right_third"}:
        score = 4.4
    elif pos == "edge":
        score = 2.4
    else:
        score = 3.5
    mt = features.get("margin_top", 0.05)
    mb = features.get("margin_bottom", 0.05)
    if mt < 0.02 or mb < 0.02:
        score -= 0.7
    return float(max(1.0, min(5.0, score)))


def quality_label(score: float) -> str:
    if score >= 4.5:
        return "excellent"
    if score >= 3.8:
        return "good"
    if score >= 3.0:
        return "fair"
    if score >= 2.0:
        return "poor"
    return "bad"

