from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .geometry import BBox, clip_box
from .io_utils import ensure_dir, safe_stem
from .schema import Detection, MaskRecord


class Segmenter:
    def segment(
        self,
        image_bgr,
        image_path: str,
        detections: List[Detection],
        vlm_understanding: Dict[str, Any],
        mask_dir: str,
    ) -> Dict[str, List[MaskRecord]]:
        raise NotImplementedError


class BBoxSegmenter(Segmenter):
    """Fast executable fallback: stores rectangular masks from detection boxes."""

    def segment(
        self,
        image_bgr,
        image_path: str,
        detections: List[Detection],
        vlm_understanding: Dict[str, Any],
        mask_dir: str,
    ) -> Dict[str, List[MaskRecord]]:
        h, w = image_bgr.shape[:2]
        out = {"preserve_masks": [], "relation_masks": [], "environment_masks": [], "distractor_masks": []}
        ensure_dir(mask_dir)
        roles = _role_map(vlm_understanding)

        for idx, det in enumerate(detections):
            role = roles.get(det.name.lower(), "relation_masks")
            if det.name.lower() in {"person", "man", "woman", "boy", "girl", "people", "main subject"}:
                role = "preserve_masks"
            if "background" in det.source:
                role = "environment_masks"
            if "distractor" in det.source:
                role = "distractor_masks"
            bbox = clip_box(det.bbox, w, h)
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = bbox.to_xyxy_int()
            mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
            mask_id = f"{safe_stem(image_path)}_{idx:02d}_{det.name.replace(' ', '_')}"
            rel_path = f"{mask_id}.png"
            abs_path = str(Path(mask_dir) / rel_path)
            cv2.imwrite(abs_path, mask)
            area = int((mask > 0).sum())
            out[role].append(
                MaskRecord(
                    mask_id=mask_id,
                    name=det.name,
                    category=det.name,
                    role=role.replace("_masks", ""),
                    bbox=bbox,
                    mask_path=abs_path,
                    area=area,
                    area_ratio=area / max(float(w * h), 1.0),
                    confidence=det.confidence,
                    source="bbox_mask",
                    importance=_importance_for(det.name, vlm_understanding),
                )
            )
        return out


class SamBoxSegmenter(BBoxSegmenter):
    """Optional SAM adapter. Falls back to rectangular masks if SAM is unavailable."""

    def __init__(self, checkpoint: str = "", model_type: str = "vit_h", device: str = "cpu"):
        self.available = False
        self.predictor = None
        try:
            from segment_anything import SamPredictor, sam_model_registry

            if checkpoint:
                sam = sam_model_registry[model_type](checkpoint=checkpoint)
                sam.to(device=device)
                self.predictor = SamPredictor(sam)
                self.available = True
        except Exception:
            self.available = False

    def segment(self, image_bgr, image_path: str, detections: List[Detection], vlm_understanding: Dict[str, Any], mask_dir: str):
        if not self.available or self.predictor is None:
            return super().segment(image_bgr, image_path, detections, vlm_understanding, mask_dir)

        h, w = image_bgr.shape[:2]
        out = {"preserve_masks": [], "relation_masks": [], "environment_masks": [], "distractor_masks": []}
        ensure_dir(mask_dir)
        roles = _role_map(vlm_understanding)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        for idx, det in enumerate(detections):
            role = roles.get(det.name.lower(), "relation_masks")
            if det.name.lower() in {"person", "man", "woman", "boy", "girl", "people", "main subject"}:
                role = "preserve_masks"
            bbox = clip_box(det.bbox, w, h)
            box_arr = np.array(bbox.to_xyxy_int(), dtype=np.float32)
            masks, scores, _ = self.predictor.predict(box=box_arr, multimask_output=True)
            best_idx = int(np.argmax(scores))
            mask = (masks[best_idx].astype(np.uint8) * 255)
            mask_id = f"{safe_stem(image_path)}_{idx:02d}_{det.name.replace(' ', '_')}"
            abs_path = str(Path(mask_dir) / f"{mask_id}.png")
            cv2.imwrite(abs_path, mask)
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                mb = BBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            else:
                mb = bbox
            area = int((mask > 0).sum())
            out[role].append(
                MaskRecord(
                    mask_id=mask_id,
                    name=det.name,
                    category=det.name,
                    role=role.replace("_masks", ""),
                    bbox=mb,
                    mask_path=abs_path,
                    area=area,
                    area_ratio=area / max(float(w * h), 1.0),
                    confidence=float(scores[best_idx]),
                    source="sam_box",
                    importance=_importance_for(det.name, vlm_understanding),
                )
            )
        return out


def create_segmenter(kind: str, sam_checkpoint: str = "", sam_model_type: str = "vit_h", device: str = "cpu") -> Segmenter:
    kind = (kind or "bbox").lower()
    if kind in {"none", "bbox", "rect"}:
        return BBoxSegmenter()
    if kind in {"sam", "sam_box"}:
        return SamBoxSegmenter(checkpoint=sam_checkpoint, model_type=sam_model_type, device=device)
    raise ValueError(f"Unknown segmenter: {kind}")


def _role_map(vlm: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    main = vlm.get("main_subject")
    if isinstance(main, dict):
        mapping[str(main.get("name", "")).lower()] = "preserve_masks"
        mapping[str(main.get("category", "")).lower()] = "preserve_masks"
    for obj in vlm.get("key_objects", []) or []:
        if isinstance(obj, dict):
            mapping[str(obj.get("name", "")).lower()] = "relation_masks"
            mapping[str(obj.get("category", "")).lower()] = "relation_masks"
    for obj in vlm.get("important_background", []) or []:
        if isinstance(obj, dict):
            mapping[str(obj.get("name", "")).lower()] = "environment_masks"
    for obj in vlm.get("distractors", []) or []:
        if isinstance(obj, dict):
            mapping[str(obj.get("name", "")).lower()] = "distractor_masks"
    return mapping


def _importance_for(name: str, vlm: Dict[str, Any]) -> float:
    lname = name.lower()
    main = vlm.get("main_subject")
    if isinstance(main, dict) and lname in {str(main.get("name", "")).lower(), str(main.get("category", "")).lower()}:
        return _as_float(main.get("importance", 1.0), 1.0)
    for key, default in [("key_objects", 0.8), ("important_background", 0.45), ("distractors", 0.2)]:
        for obj in vlm.get(key, []) or []:
            if not isinstance(obj, dict):
                continue
            if lname in {str(obj.get("name", "")).lower(), str(obj.get("category", "")).lower()}:
                return _as_float(obj.get("importance", default), default)
    return 0.6


def _as_float(value, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"very high", "critical", "essential", "极高", "非常高", "最高", "关键", "必须保留"}:
        return 1.0
    if text in {"high", "important", "major", "高", "重要", "主要", "较高"}:
        return 0.85
    if text in {"medium", "moderate", "normal", "中", "中等", "一般", "普通", "适中"}:
        return 0.55
    if text in {"low", "minor", "低", "较低", "次要", "不太重要"}:
        return 0.25
    if text in {"none", "irrelevant", "ignore", "无", "不重要", "忽略", "无需保留"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return default
