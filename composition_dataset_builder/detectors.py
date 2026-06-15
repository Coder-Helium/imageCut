from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .geometry import BBox
from .schema import Detection


class Detector:
    def detect(self, image_path: str, vlm_understanding: Dict[str, Any], image_w: int, image_h: int) -> List[Detection]:
        raise NotImplementedError


class NoopDetector(Detector):
    def detect(self, image_path: str, vlm_understanding: Dict[str, Any], image_w: int, image_h: int) -> List[Detection]:
        detections = detections_from_vlm_boxes(vlm_understanding, image_w, image_h)
        if detections:
            return detections
        return fallback_subject_detection(vlm_understanding, image_w, image_h)


class YOLODetector(Detector):
    def __init__(self, model_path: str, conf: float = 0.15):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("ultralytics is required for YOLODetector") from exc
        self.model = YOLO(model_path, verbose=False)
        if hasattr(self.model, "overrides"):
            self.model.overrides["verbose"] = False
        self.conf = conf

    def detect(self, image_path: str, vlm_understanding: Dict[str, Any], image_w: int, image_h: int) -> List[Detection]:
        results = self.model(image_path, verbose=False)
        detections: List[Detection] = []
        if not results:
            return detections_from_vlm_boxes(vlm_understanding, image_w, image_h)
        result = results[0]
        names = result.names
        if result.boxes is None or len(result.boxes) == 0:
            fallback = detections_from_vlm_boxes(vlm_understanding, image_w, image_h)
            return fallback or fallback_subject_detection(vlm_understanding, image_w, image_h)
        for box in result.boxes:
            score = float(box.conf[0].detach().cpu().item())
            if score < self.conf:
                continue
            cls_id = int(box.cls[0].detach().cpu().item())
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            detections.append(
                Detection(
                    name=str(names.get(cls_id, cls_id)),
                    bbox=BBox.from_seq(xyxy),
                    confidence=score,
                    source="yolo",
                )
            )
        detections.extend(detections_from_vlm_boxes(vlm_understanding, image_w, image_h, only_missing=True, existing=detections))
        return detections


def create_detector(kind: str, model_path: Optional[str] = None, conf: float = 0.15) -> Detector:
    kind = (kind or "none").lower()
    if kind in {"none", "noop", "vlm"}:
        return NoopDetector()
    if kind == "yolo":
        if not model_path:
            raise ValueError("--yolo-model is required when --detector yolo")
        return YOLODetector(model_path=model_path, conf=conf)
    raise ValueError(f"Unknown detector: {kind}")


def detections_from_vlm_boxes(
    vlm: Dict[str, Any],
    image_w: int,
    image_h: int,
    only_missing: bool = False,
    existing: Optional[List[Detection]] = None,
) -> List[Detection]:
    existing_names = {d.name.lower() for d in existing or []}
    out: List[Detection] = []

    def add_obj(obj: Dict[str, Any], role: str) -> None:
        name = str(obj.get("name") or obj.get("category") or role)
        if only_missing and name.lower() in existing_names:
            return
        bbox = obj.get("bbox") or obj.get("box")
        if bbox is None and obj.get("bbox_norm") is not None:
            x1, y1, x2, y2 = obj["bbox_norm"]
            bbox = [x1 * image_w, y1 * image_h, x2 * image_w, y2 * image_h]
        if bbox is None:
            return
        try:
            out.append(
                Detection(
                    name=name,
                    bbox=BBox.from_seq(bbox),
                    confidence=float(obj.get("confidence", 0.65)),
                    source=f"vlm_{role}",
                )
            )
        except Exception:
            return

    main = vlm.get("main_subject")
    if isinstance(main, dict):
        add_obj(main, "main_subject")
    for key in ["key_objects", "important_background", "distractors"]:
        for obj in vlm.get(key, []) or []:
            if isinstance(obj, dict):
                add_obj(obj, key)
    return out


def fallback_subject_detection(vlm: Dict[str, Any], image_w: int, image_h: int) -> List[Detection]:
    """Low-confidence geometry fallback so the MVP can still emit subject-aware samples."""
    main = vlm.get("main_subject") if isinstance(vlm, dict) else None
    if not isinstance(main, dict):
        return []
    name = str(main.get("category") or main.get("name") or "main subject")
    if not name:
        name = "main subject"
    bw = image_w * 0.34
    bh = image_h * 0.62
    cx = image_w * 0.52
    cy = image_h * 0.52
    return [
        Detection(
            name=name,
            bbox=BBox(cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0),
            confidence=0.25,
            source="fallback_center_subject",
        )
    ]
