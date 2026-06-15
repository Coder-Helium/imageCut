from __future__ import annotations

import torch


def sanitize_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(boxes[..., 0], boxes[..., 2])
    y1 = torch.minimum(boxes[..., 1], boxes[..., 3])
    x2 = torch.maximum(boxes[..., 0], boxes[..., 2])
    y2 = torch.maximum(boxes[..., 1], boxes[..., 3])
    out = torch.stack([x1, y1, x2, y2], dim=-1)
    return out.clamp(0.0, 1.0)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            (x2 - x1).clamp(min=0.0),
            (y2 - y1).clamp(min=0.0),
        ],
        dim=-1,
    )


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    boxes = sanitize_xyxy(boxes)
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    boxes1 = sanitize_xyxy(boxes1)
    boxes2 = sanitize_xyxy(boxes2)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    boxes1 = sanitize_xyxy(boxes1)
    boxes2 = sanitize_xyxy(boxes2)
    iou = box_iou(boxes1, boxes2)
    lt = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    area_c = wh[..., 0] * wh[..., 1]

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    inter = iou * (area1[:, None] + area2[None, :]) / (1.0 + iou).clamp(min=1e-6)
    union = area1[:, None] + area2[None, :] - inter
    return iou - (area_c - union) / area_c.clamp(min=1e-6)


def normalize_xyxy(box: list[float] | tuple[float, float, float, float], image_w: int, image_h: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        max(0.0, min(1.0, x1 / max(float(image_w), 1.0))),
        max(0.0, min(1.0, y1 / max(float(image_h), 1.0))),
        max(0.0, min(1.0, x2 / max(float(image_w), 1.0))),
        max(0.0, min(1.0, y2 / max(float(image_h), 1.0))),
    ]


def denormalize_xyxy(box: torch.Tensor, image_w: int, image_h: int) -> torch.Tensor:
    scale = torch.tensor([image_w, image_h, image_w, image_h], dtype=box.dtype, device=box.device)
    return sanitize_xyxy(box) * scale

