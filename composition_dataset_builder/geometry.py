from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return self.w() * self.h()

    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def to_xyxy_int(self) -> List[int]:
        return [int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))]

    def to_xywh_int(self) -> List[int]:
        x1, y1, x2, y2 = self.to_xyxy_int()
        return [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]

    @classmethod
    def from_seq(cls, values: Sequence[float]) -> "BBox":
        if len(values) != 4:
            raise ValueError(f"bbox requires 4 values, got {len(values)}")
        x1, y1, x2, y2 = [float(v) for v in values]
        return cls(x1, y1, x2, y2)


def parse_aspect_ratio(value: str, image_w: int, image_h: int) -> float:
    value = str(value).strip().lower()
    if value in {"original", "orig", "image"}:
        return image_w / max(float(image_h), 1.0)
    if ":" in value:
        a, b = value.split(":", 1)
        return float(a) / max(float(b), 1e-6)
    if "/" in value:
        a, b = value.split("/", 1)
        return float(a) / max(float(b), 1e-6)
    return float(value)


def aspect_name_to_safe(value: str) -> str:
    return str(value).replace(":", "x").replace("/", "x").replace(".", "p")


def clip_box(box: BBox, w: int, h: int) -> BBox:
    return BBox(
        max(0.0, min(float(w), box.x1)),
        max(0.0, min(float(h), box.y1)),
        max(0.0, min(float(w), box.x2)),
        max(0.0, min(float(h), box.y2)),
    )


def shift_box_into_image(box: BBox, w: int, h: int) -> BBox:
    bw = box.w()
    bh = box.h()
    if bw <= 0 or bh <= 0:
        return clip_box(box, w, h)
    if bw > w or bh > h:
        return fit_box_to_canvas(box, w, h)

    dx = 0.0
    dy = 0.0
    if box.x1 < 0:
        dx = -box.x1
    elif box.x2 > w:
        dx = w - box.x2
    if box.y1 < 0:
        dy = -box.y1
    elif box.y2 > h:
        dy = h - box.y2
    return BBox(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy)


def fit_box_to_canvas(box: BBox, w: int, h: int) -> BBox:
    bw = max(1.0, box.w())
    bh = max(1.0, box.h())
    scale = min(w / bw, h / bh, 1.0)
    cx, cy = box.center()
    nw = bw * scale
    nh = bh * scale
    out = BBox(cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0)
    return clip_box(shift_box_into_image(out, w, h), w, h)


def make_aspect_box_from_center(cx: float, cy: float, width: float, aspect: float, canvas_w: int, canvas_h: int) -> BBox:
    aspect = max(float(aspect), 1e-6)
    width = min(max(2.0, float(width)), float(canvas_w))
    height = width / aspect
    if height > canvas_h:
        height = float(canvas_h)
        width = height * aspect
    box = BBox(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
    return shift_box_into_image(box, canvas_w, canvas_h)


def fit_box_to_aspect(box: BBox, aspect: float, canvas_w: int, canvas_h: int, padding: float = 0.0) -> BBox:
    cx, cy = box.center()
    bw = max(2.0, box.w() * (1.0 + padding))
    bh = max(2.0, box.h() * (1.0 + padding))
    cur = bw / max(bh, 1e-6)
    if cur < aspect:
        width = bh * aspect
        height = bh
    else:
        width = bw
        height = bw / aspect
    if width > canvas_w or height > canvas_h:
        scale = min(canvas_w / width, canvas_h / height)
        width *= scale
        height *= scale
    out = BBox(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
    return shift_box_into_image(out, canvas_w, canvas_h)


def union_boxes(boxes: Iterable[BBox]) -> BBox:
    boxes = list(boxes)
    if not boxes:
        raise ValueError("union_boxes requires at least one box")
    return BBox(
        min(b.x1 for b in boxes),
        min(b.y1 for b in boxes),
        max(b.x2 for b in boxes),
        max(b.y2 for b in boxes),
    )


def intersection_area(a: BBox, b: BBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    inter = intersection_area(a, b)
    denom = a.area() + b.area() - inter
    return inter / denom if denom > 0 else 0.0


def contains(outer: BBox, inner: BBox, eps: float = 0.0) -> bool:
    return (
        outer.x1 <= inner.x1 + eps
        and outer.y1 <= inner.y1 + eps
        and outer.x2 >= inner.x2 - eps
        and outer.y2 >= inner.y2 - eps
    )


def dedup_boxes(boxes: Iterable[BBox], eps_px: int = 3) -> List[BBox]:
    out: List[BBox] = []
    seen = set()
    for b in boxes:
        x1, y1, x2, y2 = b.to_xyxy_int()
        key = (
            int(round(x1 / eps_px)),
            int(round(y1 / eps_px)),
            int(round(x2 / eps_px)),
            int(round(y2 / eps_px)),
        )
        if key in seen:
            continue
        seen.add(key)
        if b.w() > 2 and b.h() > 2:
            out.append(b)
    return out


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_score_1_5(value: float, source_min: float = 0.0, source_max: float = 1.0) -> float:
    if source_max <= source_min:
        return 3.0
    t = clamp01((float(value) - source_min) / (source_max - source_min))
    return 1.0 + 4.0 * t

