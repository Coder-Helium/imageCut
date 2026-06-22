from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .box_ops import normalize_xyxy
from .vocab import ACTION_VOCAB, ISSUE_VOCAB, aspect_to_float, index_or_unknown


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def read_image_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2.imread failed: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_to_tensor(img_rgb: np.ndarray, size: int, normalize: bool = True) -> torch.Tensor:
    img = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    if normalize:
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor


def crop_rgb(img_rgb: np.ndarray, box: List[int]) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return img_rgb.copy()
    return img_rgb[y1:y2, x1:x2].copy()


def score_to_unit(score: float) -> float:
    return max(0.0, min(1.0, (float(score) - 1.0) / 4.0))


def unit_to_score(value: float) -> float:
    return 1.0 + 4.0 * float(value)


class CropRankerDataset(Dataset):
    """Flattened candidate-level dataset for image + candidate crop scoring."""

    def __init__(
        self,
        jsonl_path: str | Path,
        image_size: int = 224,
        crop_size: int = 224,
        max_records: Optional[int] = None,
        max_candidates_per_record: Optional[int] = None,
        min_score: float = 0.0,
    ) -> None:
        self.records = load_jsonl(jsonl_path)
        if max_records is not None:
            self.records = self.records[:max_records]
        self.image_size = image_size
        self.crop_size = crop_size
        self.items: List[tuple[int, int]] = []
        for ridx, rec in enumerate(self.records):
            candidates = rec.get("candidates", [])
            if max_candidates_per_record is not None:
                candidates = candidates[:max_candidates_per_record]
            for cidx, cand in enumerate(candidates):
                score = float(cand.get("scores", {}).get("final_score", cand.get("score", 0.0)) or 0.0)
                if score >= min_score:
                    self.items.append((ridx, cidx))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ridx, cidx = self.items[idx]
        rec = self.records[ridx]
        cand = rec["candidates"][cidx]
        img = read_image_rgb(rec["image_path"])
        h, w = img.shape[:2]
        box = cand["box"]
        full_tensor = resize_to_tensor(img, self.image_size)
        crop_tensor = resize_to_tensor(crop_rgb(img, box), self.crop_size)
        norm_box = normalize_xyxy(box, w, h)
        score = float(cand.get("scores", {}).get("final_score", cand.get("score", 3.0)))
        action = index_or_unknown(ACTION_VOCAB, cand.get("action", rec.get("best_action", "unknown")))
        issue = index_or_unknown(ISSUE_VOCAB, cand.get("issue", rec.get("main_issue", "unknown")))
        box_feat = torch.tensor(
            [
                norm_box[0],
                norm_box[1],
                norm_box[2],
                norm_box[3],
                max(0.0, norm_box[2] - norm_box[0]),
                max(0.0, norm_box[3] - norm_box[1]),
                float(cand.get("features", {}).get("subject_coverage", 1.0)),
                float(cand.get("features", {}).get("relation_coverage", 1.0)),
            ],
            dtype=torch.float32,
        )
        return {
            "image": full_tensor,
            "crop": crop_tensor,
            "box_feat": box_feat,
            "score": torch.tensor(score_to_unit(score), dtype=torch.float32),
            "score_raw": torch.tensor(score, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.long),
            "issue": torch.tensor(issue, dtype=torch.long),
            "sample_id": rec.get("sample_id", ""),
        }


class PairwiseCropRankerDataset(Dataset):
    """Pairwise crop preference dataset.

    Each item is one preference edge from ``pairwise_preferences``:
    winner crop should receive a higher score than loser crop.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        image_size: int = 224,
        crop_size: int = 224,
        max_records: Optional[int] = None,
        max_pairs_per_record: Optional[int] = None,
    ) -> None:
        self.records = load_jsonl(jsonl_path)
        if max_records is not None:
            self.records = self.records[:max_records]
        self.image_size = image_size
        self.crop_size = crop_size
        self.items: List[tuple[int, int]] = []
        for ridx, rec in enumerate(self.records):
            pairs = rec.get("pairwise_preferences", []) or []
            if max_pairs_per_record is not None:
                pairs = pairs[:max_pairs_per_record]
            for pidx, _ in enumerate(pairs):
                self.items.append((ridx, pidx))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ridx, pidx = self.items[idx]
        rec = self.records[ridx]
        pref = rec["pairwise_preferences"][pidx]
        candidates = {str(c.get("candidate_id")): c for c in rec.get("candidates", [])}
        winner = candidates[str(pref["winner"])]
        loser = candidates[str(pref["loser"])]
        img = read_image_rgb(rec["image_path"])
        h, w = img.shape[:2]
        full_tensor = resize_to_tensor(img, self.image_size)
        winner_box = winner["box"]
        loser_box = loser["box"]
        return {
            "image": full_tensor,
            "winner_crop": resize_to_tensor(crop_rgb(img, winner_box), self.crop_size),
            "loser_crop": resize_to_tensor(crop_rgb(img, loser_box), self.crop_size),
            "winner_box_feat": _candidate_box_feat(winner, winner_box, w, h),
            "loser_box_feat": _candidate_box_feat(loser, loser_box, w, h),
            "weight": torch.tensor(float(pref.get("weight", 1.0)), dtype=torch.float32),
            "sample_id": rec.get("sample_id", ""),
            "winner": str(pref["winner"]),
            "loser": str(pref["loser"]),
        }


def _candidate_box_feat(cand: Dict[str, Any], box: List[int], image_w: int, image_h: int) -> torch.Tensor:
    norm_box = normalize_xyxy(box, image_w, image_h)
    return torch.tensor(
        [
            norm_box[0],
            norm_box[1],
            norm_box[2],
            norm_box[3],
            max(0.0, norm_box[2] - norm_box[0]),
            max(0.0, norm_box[3] - norm_box[1]),
            float(cand.get("features", {}).get("subject_coverage", 1.0)),
            float(cand.get("features", {}).get("relation_coverage", 1.0)),
        ],
        dtype=torch.float32,
    )


class DACCGeneratorDataset(Dataset):
    """Image-level dataset for top-k crop/action/issue generation."""

    def __init__(
        self,
        jsonl_path: str | Path,
        image_size: int = 384,
        max_records: Optional[int] = None,
        max_targets: int = 8,
        min_positive_score: float = 3.5,
    ) -> None:
        self.records = load_jsonl(jsonl_path)
        if max_records is not None:
            self.records = self.records[:max_records]
        self.image_size = image_size
        self.max_targets = max_targets
        self.min_positive_score = min_positive_score

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img = read_image_rgb(rec["image_path"])
        h, w = img.shape[:2]
        image = resize_to_tensor(img, self.image_size)
        candidates = rec.get("candidates", [])
        positives = [
            c for c in candidates
            if float(c.get("scores", {}).get("final_score", c.get("score", 0.0)) or 0.0) >= self.min_positive_score
        ]
        if not positives and rec.get("best_crop"):
            positives = [
                {
                    "box": rec["best_crop"],
                    "scores": {"final_score": rec.get("best_score", 4.0) or 4.0},
                    "action": rec.get("best_action", "unknown"),
                    "issue": rec.get("main_issue", "unknown"),
                }
            ]
        positives = positives[: self.max_targets]
        n = len(positives)
        boxes = torch.zeros(self.max_targets, 4, dtype=torch.float32)
        scores = torch.zeros(self.max_targets, dtype=torch.float32)
        actions = torch.full((self.max_targets,), index_or_unknown(ACTION_VOCAB, "unknown"), dtype=torch.long)
        issues = torch.full((self.max_targets,), index_or_unknown(ISSUE_VOCAB, "unknown"), dtype=torch.long)
        mask = torch.zeros(self.max_targets, dtype=torch.bool)
        for i, cand in enumerate(positives):
            boxes[i] = torch.tensor(normalize_xyxy(cand["box"], w, h), dtype=torch.float32)
            score = float(cand.get("scores", {}).get("final_score", cand.get("score", 3.0)))
            scores[i] = score_to_unit(score)
            actions[i] = index_or_unknown(ACTION_VOCAB, cand.get("action", rec.get("best_action", "unknown")))
            issues[i] = index_or_unknown(ISSUE_VOCAB, cand.get("issue", rec.get("main_issue", "unknown")))
            mask[i] = True

        # Normalized box ratio = (crop_w / image_w) / (crop_h / image_h)
        #                      = target_aspect / image_aspect.
        aspect = aspect_to_float(rec.get("target_aspect_ratio", "original"), w, h) / max(w / max(float(h), 1.0), 1e-6)
        return {
            "image": image,
            "target_boxes": boxes,
            "target_scores": scores,
            "target_actions": actions,
            "target_issues": issues,
            "target_mask": mask,
            "aspect": torch.tensor([aspect], dtype=torch.float32),
            "sample_id": rec.get("sample_id", ""),
            "image_size": torch.tensor([w, h], dtype=torch.float32),
        }
