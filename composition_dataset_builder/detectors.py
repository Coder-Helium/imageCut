from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .geometry import BBox, clip_box
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


class YOLOWorldDetector(Detector):
    """Open-vocabulary detector using VLM object names as class prompts."""

    def __init__(self, model_path: str = "yolov8s-world.pt", conf: float = 0.10):
        try:
            from ultralytics import YOLOWorld
        except Exception as exc:
            raise RuntimeError("ultralytics with YOLOWorld support is required for YOLOWorldDetector") from exc
        self.model = YOLOWorld(model_path)
        if hasattr(self.model, "overrides"):
            self.model.overrides["verbose"] = False
        self.conf = conf

    def detect(self, image_path: str, vlm_understanding: Dict[str, Any], image_w: int, image_h: int) -> List[Detection]:
        classes = _classes_from_vlm(vlm_understanding)
        if hasattr(self.model, "set_classes") and classes:
            self.model.set_classes(classes)
        results = self.model.predict(image_path, conf=self.conf, verbose=False)
        detections: List[Detection] = []
        if not results:
            return fallback_subject_detection(vlm_understanding, image_w, image_h)
        result = results[0]
        names = result.names
        if result.boxes is None or len(result.boxes) == 0:
            return fallback_subject_detection(vlm_understanding, image_w, image_h)
        for box in result.boxes:
            score = float(box.conf[0].detach().cpu().item())
            cls_id = int(box.cls[0].detach().cpu().item())
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            bbox = clip_box(BBox.from_seq(xyxy), image_w, image_h)
            if bbox.area() < 4.0:
                continue
            detections.append(
                Detection(
                    name=str(names.get(cls_id, cls_id)),
                    bbox=bbox,
                    confidence=score,
                    source="yolo_world",
                )
            )
        return detections or fallback_subject_detection(vlm_understanding, image_w, image_h)


def create_detector(kind: str, model_path: Optional[str] = None, conf: float = 0.15) -> Detector:
    kind = (kind or "none").lower()
    if kind in {"none", "noop", "vlm"}:
        return NoopDetector()
    if kind == "yolo":
        if not model_path:
            raise ValueError("--yolo-model is required when --detector yolo")
        return YOLODetector(model_path=model_path, conf=conf)
    if kind in {"yolo_world", "yoloworld", "open_vocab_yolo"}:
        return YOLOWorldDetector(model_path=model_path or "yolov8s-world.pt", conf=conf)
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
        bbox = _bbox_from_vlm_object(obj, image_w, image_h)
        if bbox is None:
            return
        bbox = clip_box(bbox, image_w, image_h)
        if bbox.area() < 4.0:
            return
        out.append(
            Detection(
                name=name,
                bbox=bbox,
                confidence=_as_float(obj.get("confidence", 0.65), 0.65),
                source=f"vlm_{role}",
            )
        )

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


def _classes_from_vlm(vlm: Dict[str, Any]) -> List[str]:
    classes: List[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip().lower()
        if not text or len(text) > 40:
            return
        if text in {"main subject", "subject", "object", "scene", "background"}:
            return
        if text not in classes:
            classes.append(text)

    main = vlm.get("main_subject") if isinstance(vlm, dict) else None
    if isinstance(main, dict):
        add(main.get("name"))
        add(main.get("category"))
    elif isinstance(main, str):
        add(main)
    for key in ["key_objects", "important_background", "distractors"]:
        for obj in vlm.get(key, []) or []:
            if isinstance(obj, dict):
                add(obj.get("name"))
                add(obj.get("category"))
    if not any(x in classes for x in ["person", "man", "woman", "girl", "boy"]):
        semantic = str(vlm.get("semantic_type", "")).lower()
        if "person" in semantic or "portrait" in semantic:
            classes.extend(["person", "woman", "man"])
    return classes[:20]


def _bbox_from_vlm_object(obj: Dict[str, Any], image_w: int, image_h: int) -> Optional[BBox]:
    raw_box = obj.get("bbox") or obj.get("box")
    if raw_box is not None:
        values = _float_sequence(raw_box)
        if values is None:
            return None
        if max(values) <= 1.5:
            return _box_from_values(values, image_w, image_h, normalized=True)
        return _box_from_values(values, image_w, image_h, normalized=False)

    raw_norm = obj.get("bbox_norm")
    if raw_norm is None:
        return None
    values = _float_sequence(raw_norm)
    if values is None:
        return None
    if max(values) > 1.5 and max(values) <= 1000.0 and min(values) >= 0.0:
        # Qwen3-VL grounding returns relative coordinates in a 0-1000 system.
        values = [v / 1000.0 for v in values]
        return _box_from_values(values, image_w, image_h, normalized=True)
    if max(values) > 1.5:
        return None
    return _box_from_values(values, image_w, image_h, normalized=True)


def _box_from_values(values: List[float], image_w: int, image_h: int, normalized: bool) -> BBox:
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        # Some VLMs return [x, y, width, height] despite being asked for xyxy.
        x2 = x1 + max(0.0, values[2])
        y2 = y1 + max(0.0, values[3])
    if normalized:
        return BBox(x1 * image_w, y1 * image_h, x2 * image_w, y2 * image_h)
    return BBox(x1, y1, x2, y2)


def _float_sequence(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    out: List[float] = []
    for item in value:
        number = _as_float_optional(item)
        if number is None:
            return None
        out.append(number)
    return out


def _as_float_optional(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _as_float(value: Any, default: float) -> float:
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
