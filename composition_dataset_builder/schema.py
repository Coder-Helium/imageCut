from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .geometry import BBox


@dataclass
class Detection:
    name: str
    bbox: BBox
    confidence: float = 1.0
    source: str = "unknown"

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.name,
            "bbox": self.bbox.to_xyxy_int(),
            "confidence": float(self.confidence),
            "source": self.source,
        }


@dataclass
class MaskRecord:
    mask_id: str
    name: str
    category: str
    role: str
    bbox: BBox
    mask_path: str
    area: int
    area_ratio: float
    confidence: float
    source: str
    importance: float = 1.0

    def to_json(self) -> Dict[str, Any]:
        return {
            "mask_id": self.mask_id,
            "name": self.name,
            "category": self.category,
            "role": self.role,
            "bbox": self.bbox.to_xyxy_int(),
            "mask_path": self.mask_path,
            "area": int(self.area),
            "area_ratio": float(self.area_ratio),
            "confidence": float(self.confidence),
            "source": self.source,
            "importance": float(self.importance),
        }


@dataclass
class Candidate:
    candidate_id: str
    box: BBox
    source: str
    action: str
    issue: str
    reason: str
    features: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    rank: Optional[int] = None
    quality_label: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "box": self.box.to_xyxy_int(),
            "box_format": "xyxy",
            "source": self.source,
            "action": self.action,
            "issue": self.issue,
            "reason": self.reason,
            "features": self.features,
            "scores": self.scores,
            "rank": self.rank,
            "quality_label": self.quality_label,
        }

