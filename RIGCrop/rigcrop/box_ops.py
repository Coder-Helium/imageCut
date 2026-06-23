from __future__ import annotations

from typing import Iterable, List, Sequence

import torch


def clip_box(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return [x1, y1, x2, y2]


def valid_box(box: Sequence[float], min_size: float = 1e-4) -> bool:
    if len(box) < 4:
        return False
    x1, y1, x2, y2 = clip_box(box)
    return (x2 - x1) > min_size and (y2 - y1) > min_size


def normalize_xyxy(box: Sequence[float], image_w: int, image_h: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return clip_box([x1 / max(float(image_w), 1.0), y1 / max(float(image_h), 1.0), x2 / max(float(image_w), 1.0), y2 / max(float(image_h), 1.0)])


def area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = clip_box(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = clip_box(a)
    bx1, by1, bx2, by2 = clip_box(b)
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def coverage(node_box: Sequence[float], crop_box: Sequence[float]) -> float:
    node_area = area(node_box)
    if node_area <= 1e-8:
        return 0.0
    return max(0.0, min(1.0, intersection_area(node_box, crop_box) / node_area))


def boundary_cut(node_box: Sequence[float], crop_box: Sequence[float]) -> float:
    cov = coverage(node_box, crop_box)
    if cov <= 1e-6 or cov >= 1.0 - 1e-6:
        return 0.0
    return min(1.0, 4.0 * cov * (1.0 - cov))


def tensor_sanitize_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(boxes[..., 0], boxes[..., 2])
    y1 = torch.minimum(boxes[..., 1], boxes[..., 3])
    x2 = torch.maximum(boxes[..., 0], boxes[..., 2])
    y2 = torch.maximum(boxes[..., 1], boxes[..., 3])
    return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)


def tensor_area(boxes: torch.Tensor) -> torch.Tensor:
    boxes = tensor_sanitize_xyxy(boxes)
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)


def tensor_coverage(node_boxes: torch.Tensor, crop_boxes: torch.Tensor) -> torch.Tensor:
    """Coverage of each node by one crop per batch.

    Args:
        node_boxes: B x M x 4 normalized xyxy.
        crop_boxes: B x 4 normalized xyxy.
    Returns:
        B x M coverage values.
    """
    node_boxes = tensor_sanitize_xyxy(node_boxes)
    crop_boxes = tensor_sanitize_xyxy(crop_boxes).unsqueeze(1)
    lt = torch.maximum(node_boxes[..., :2], crop_boxes[..., :2])
    rb = torch.minimum(node_boxes[..., 2:], crop_boxes[..., 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / tensor_area(node_boxes).clamp(min=1e-6)


def candidate_box_features(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = clip_box(box)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [x1, y1, x2, y2, w, h, w * h, w / max(h, 1e-6)]


def denormalize_xyxy(box: Iterable[float], image_w: int, image_h: int) -> List[int]:
    x1, y1, x2, y2 = clip_box(list(box))
    return [
        int(round(x1 * image_w)),
        int(round(y1 * image_h)),
        int(round(x2 * image_w)),
        int(round(y2 * image_h)),
    ]
