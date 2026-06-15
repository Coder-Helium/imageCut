from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch

from .box_ops import box_iou


def spearmanr_np(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return 0.0
    rx = rankdata(np.asarray(x, dtype=np.float64))
    ry = rankdata(np.asarray(y, dtype=np.float64))
    sx = rx.std()
    sy = ry.std()
    if sx < 1e-8 or sy < 1e-8:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def top1_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, target_mask: torch.Tensor) -> float:
    if pred_boxes.ndim == 3:
        pred = pred_boxes[:, 0]
    else:
        pred = pred_boxes
    vals = []
    for i in range(pred.shape[0]):
        valid = target_mask[i]
        if not valid.any():
            continue
        ious = box_iou(pred[i:i + 1], target_boxes[i][valid]).squeeze(0)
        vals.append(float(ious.max().detach().cpu()))
    return float(np.mean(vals)) if vals else 0.0


def acc_at_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, target_mask: torch.Tensor, threshold: float = 0.75) -> float:
    if pred_boxes.ndim == 3:
        pred = pred_boxes[:, 0]
    else:
        pred = pred_boxes
    hits = []
    for i in range(pred.shape[0]):
        valid = target_mask[i]
        if not valid.any():
            continue
        ious = box_iou(pred[i:i + 1], target_boxes[i][valid]).squeeze(0)
        hits.append(float(ious.max().detach().cpu()) >= threshold)
    return float(np.mean(hits)) if hits else 0.0

