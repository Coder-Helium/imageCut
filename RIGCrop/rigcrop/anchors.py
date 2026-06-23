from __future__ import annotations

from typing import List, Sequence


def generate_anchors(
    image_w: int,
    image_h: int,
    scales: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6),
    aspect_ratios: Sequence[float] = (1.0, 4.0 / 3.0, 3.0 / 4.0, 16.0 / 9.0, 9.0 / 16.0),
    grid: int = 5,
) -> List[List[int]]:
    anchors: List[List[int]] = []
    full_area = float(image_w * image_h)
    for scale in scales:
        area = full_area * float(scale)
        for ratio in aspect_ratios:
            crop_w = min(float(image_w), (area * ratio) ** 0.5)
            crop_h = min(float(image_h), crop_w / max(float(ratio), 1e-6))
            if crop_h > image_h:
                crop_h = float(image_h)
                crop_w = min(float(image_w), crop_h * ratio)
            if crop_w < 4 or crop_h < 4:
                continue
            xs = _positions(image_w, crop_w, grid)
            ys = _positions(image_h, crop_h, grid)
            for x1 in xs:
                for y1 in ys:
                    x2 = min(image_w, x1 + crop_w)
                    y2 = min(image_h, y1 + crop_h)
                    anchors.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
    return _dedupe(anchors)


def _positions(total: int, size: float, grid: int) -> List[float]:
    if size >= total:
        return [0.0]
    if grid <= 1:
        return [(total - size) / 2.0]
    return [i * (total - size) / float(grid - 1) for i in range(grid)]


def _dedupe(boxes: List[List[int]]) -> List[List[int]]:
    seen = set()
    out = []
    for box in boxes:
        key = tuple(box)
        if key not in seen and box[2] > box[0] and box[3] > box[1]:
            seen.add(key)
            out.append(box)
    return out
