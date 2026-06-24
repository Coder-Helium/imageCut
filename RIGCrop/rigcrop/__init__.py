"""RIGCrop / RIGFormer workspace package.

This package is intentionally local to the ``RIGCrop`` workspace.  It consumes
the DACC-style JSONL files produced by the existing repository pipeline and adds
RIG-Crop targets without mutating the original metadata.
"""

from .backbones import build_visual_backbone
from .model import RIGCropModel
from .schema import ACTIONS, RELATION_POLICIES, ROLES

__all__ = ["ACTIONS", "RELATION_POLICIES", "RIGCropModel", "ROLES", "build_visual_backbone"]
