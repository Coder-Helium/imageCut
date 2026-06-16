from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class GaicAnnotation:
    raw_box: List[float]
    box: List[int]
    mos: float


def iter_split_images(gaic_root: str | Path, split: str) -> List[Path]:
    image_dir = Path(gaic_root) / "images" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing GAIC image split: {image_dir}")
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def read_image_size(image_path: str | Path) -> Tuple[int, int]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2.imread failed: {image_path}")
    h, w = img.shape[:2]
    return int(w), int(h)


def annotation_path_for(gaic_root: str | Path, image_path: str | Path) -> Path:
    return Path(gaic_root) / "annotations" / f"{Path(image_path).stem}.txt"


def read_raw_annotations(path: str | Path) -> List[Tuple[List[float], float]]:
    records: List[Tuple[List[float], float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Bad GAIC annotation line {line_no} in {path}: {line!r}")
            values = [float(x) for x in parts[:5]]
            records.append((values[:4], values[4]))
    return records


def resolve_coord_mode(
    raw_boxes: Sequence[Sequence[float]],
    image_w: int,
    image_h: int,
    coord_mode: str = "auto",
    annotation_size: int = 1024,
) -> str:
    mode = _normalize_coord_mode(coord_mode)
    if mode in {"gaic_yxyx", "image_xyxy", "square1024_xyxy"}:
        return mode
    if mode != "auto":
        raise ValueError(f"Unknown coord mode: {coord_mode}")

    # Official GAICD Matlab code stores boxes as image matrix coordinates:
    # [row1, col1, row2, col2] = [y1, x1, y2, x2].
    # Prefer this mode when raw boxes fit the image under row/column semantics.
    if _all_fit_gaic_yxyx(raw_boxes, image_w, image_h):
        return "gaic_yxyx"

    # Fallback for DACC-style or already converted xyxy files.
    if _all_fit_image_xyxy(raw_boxes, image_w, image_h):
        return "image_xyxy"

    # Legacy fallback for earlier exports that were interpreted as an xyxy box
    # in a 1024 square coordinate space.
    for box in raw_boxes:
        x1, y1, x2, y2 = box
        if x1 < 0 or y1 < 0 or x2 > image_w or y2 > image_h:
            return "square1024_xyxy"
    return "image_xyxy"


def _normalize_coord_mode(coord_mode: str) -> str:
    mode = coord_mode.lower()
    aliases = {
        "image": "image_xyxy",
        "xyxy": "image_xyxy",
        "square1024": "square1024_xyxy",
        "square": "square1024_xyxy",
        "gaic": "gaic_yxyx",
        "yxyx": "gaic_yxyx",
        "matlab": "gaic_yxyx",
    }
    return aliases.get(mode, mode)


def _all_fit_gaic_yxyx(raw_boxes: Sequence[Sequence[float]], image_w: int, image_h: int) -> bool:
    if not raw_boxes:
        return False
    for box in raw_boxes:
        y1, x1, y2, x2 = [float(v) for v in box]
        if y1 < 0 or x1 < 0 or y2 > image_h or x2 > image_w or y2 <= y1 or x2 <= x1:
            return False
    return True


def _all_fit_image_xyxy(raw_boxes: Sequence[Sequence[float]], image_w: int, image_h: int) -> bool:
    if not raw_boxes:
        return False
    for box in raw_boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        if x1 < 0 or y1 < 0 or x2 > image_w or y2 > image_h or x2 <= x1 or y2 <= y1:
            return False
    return True


def convert_raw_box(
    raw_box: Sequence[float],
    image_w: int,
    image_h: int,
    mode: str,
    annotation_size: int = 1024,
    clip: bool = True,
) -> List[int]:
    mode = _normalize_coord_mode(mode)
    if mode == "gaic_yxyx":
        y1, x1, y2, x2 = [float(v) for v in raw_box]
    else:
        x1, y1, x2, y2 = [float(v) for v in raw_box]

    if mode == "square1024_xyxy":
        sx = image_w / float(annotation_size)
        sy = image_h / float(annotation_size)
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
    elif mode not in {"image_xyxy", "gaic_yxyx"}:
        raise ValueError(f"Unknown resolved coord mode: {mode}")

    box = [int(round(v)) for v in [x1, y1, x2, y2]]
    if clip:
        box = clip_xyxy(box, image_w, image_h)
    return box


def clip_xyxy(box: Sequence[int | float], image_w: int, image_h: int) -> List[int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w, x2))
    y2 = max(0, min(image_h, y2))
    return [x1, y1, x2, y2]


def valid_box(box: Sequence[int | float], min_size: int = 2) -> bool:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def load_gaic_annotations(
    annotation_path: str | Path,
    image_w: int,
    image_h: int,
    coord_mode: str = "auto",
    annotation_size: int = 1024,
    clip: bool = True,
) -> Tuple[List[GaicAnnotation], str]:
    raw_records = read_raw_annotations(annotation_path)
    raw_boxes = [box for box, _ in raw_records]
    resolved_mode = resolve_coord_mode(raw_boxes, image_w, image_h, coord_mode, annotation_size)
    anns: List[GaicAnnotation] = []
    for raw_box, mos in raw_records:
        box = convert_raw_box(raw_box, image_w, image_h, resolved_mode, annotation_size, clip=clip)
        if valid_box(box):
            anns.append(GaicAnnotation(raw_box=list(raw_box), box=box, mos=float(mos)))
    return anns, resolved_mode


def quality_label(mos: float) -> str:
    if mos >= 4.0:
        return "good"
    if mos >= 3.0:
        return "fair"
    if mos >= 2.0:
        return "poor"
    return "bad"


def box_features(box: Sequence[int], image_w: int, image_h: int) -> Dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    image_area = max(float(image_w * image_h), 1.0)
    return {
        "x1_norm": x1 / max(float(image_w), 1.0),
        "y1_norm": y1 / max(float(image_h), 1.0),
        "x2_norm": x2 / max(float(image_w), 1.0),
        "y2_norm": y2 / max(float(image_h), 1.0),
        "crop_area_ratio": (bw * bh) / image_area,
        "crop_aspect": bw / max(bh, 1.0),
        "subject_coverage": 1.0,
        "relation_coverage": 1.0,
    }


def build_gaic_candidates(annotations: Sequence[GaicAnnotation], image_w: int, image_h: int) -> List[Dict[str, Any]]:
    sorted_anns = sorted(annotations, key=lambda item: item.mos, reverse=True)
    candidates: List[Dict[str, Any]] = []
    for idx, ann in enumerate(sorted_anns, start=1):
        candidates.append(
            {
                "candidate_id": f"gaic_{idx:03d}",
                "box": ann.box,
                "box_format": "xyxy",
                "source": "gaic_anchor",
                "action": "unknown",
                "issue": "gaic_mos_supervision",
                "reason": "GAICD human MOS annotated crop candidate.",
                "features": box_features(ann.box, image_w, image_h),
                "scores": {"final_score": round(float(ann.mos), 4), "mos": round(float(ann.mos), 4)},
                "rank": idx,
                "quality_label": quality_label(float(ann.mos)),
                "gaic_original_box": [round(float(v), 4) for v in ann.raw_box],
                "gaic_original_box_format": "y1_x1_y2_x2",
            }
        )
    return candidates


def iou_xyxy(box_a: Sequence[int | float], box_b: Sequence[int | float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def best_annotation(annotations: Sequence[GaicAnnotation]) -> GaicAnnotation:
    if not annotations:
        raise ValueError("No valid GAIC annotations")
    return max(annotations, key=lambda item: item.mos)


def nearest_annotation(box: Sequence[int | float], annotations: Sequence[GaicAnnotation]) -> Tuple[GaicAnnotation, float]:
    if not annotations:
        raise ValueError("No valid GAIC annotations")
    best = annotations[0]
    best_iou = -1.0
    for ann in annotations:
        val = iou_xyxy(box, ann.box)
        if val > best_iou:
            best = ann
            best_iou = val
    return best, best_iou


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dump_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count
