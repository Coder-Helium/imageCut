from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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


def crop_rgb(img_rgb: np.ndarray, box: Sequence[int | float]) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return img_rgb.copy()
    return img_rgb[y1:y2, x1:x2].copy()


def draw_boxes(img_rgb: np.ndarray, boxes: List[Sequence[int]], labels: List[str]) -> np.ndarray:
    out = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
    colors = [(0, 0, 255), (0, 180, 0), (255, 120, 0), (180, 0, 180), (0, 160, 220)]
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
        color = colors[idx % len(colors)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if idx < len(labels):
            text = labels[idx]
            cv2.putText(out, text, (x1, max(18, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
