from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .io_utils import ensure_dir
from .schema import Candidate, MaskRecord


def draw_sample_visualization(image_bgr, candidates: List[Candidate], masks: Dict[str, List[MaskRecord]], out_path: str, title: str = "") -> None:
    vis = image_bgr.copy()
    overlay = vis.copy()
    colors = {
        "preserve_masks": (0, 220, 0),
        "relation_masks": (255, 0, 255),
        "environment_masks": (255, 180, 0),
        "distractor_masks": (0, 180, 255),
    }
    for group, records in masks.items():
        color = colors.get(group, (200, 200, 200))
        for rec in records:
            mask = cv2.imread(rec.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape[:2] == vis.shape[:2]:
                overlay[mask > 0] = color
            x1, y1, x2, y2 = rec.bbox.to_xyxy_int()
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            _text(vis, rec.name, (x1, max(18, y1 + 18)), color)
    vis = cv2.addWeighted(overlay, 0.22, vis, 0.78, 0)

    for cand in candidates[:8]:
        x1, y1, x2, y2 = cand.box.to_xyxy_int()
        color = (0, 0, 255) if cand.rank == 1 else (0, 255, 0)
        thickness = 3 if cand.rank == 1 else 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label = f"#{cand.rank} {cand.scores.get('final_score', 0):.2f}"
        _text(vis, label, (x1, max(20, y1 - 6)), color)

    if title:
        _text(vis, title, (10, 28), (255, 255, 255), bg=(0, 0, 0))
    ensure_dir(Path(out_path).parent)
    cv2.imwrite(out_path, vis)


def _text(img, text: str, org, color, bg=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    x, y = org
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - 5), (x + tw + 4, y + 4), bg, -1)
    cv2.putText(img, text, (x + 2, y), font, scale, color, thickness, cv2.LINE_AA)

