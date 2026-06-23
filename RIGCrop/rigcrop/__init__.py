"""RIG-Crop workspace package.

This package is intentionally local to the ``RIGCrop`` workspace.  It consumes
the DACC-style JSONL files produced by the existing repository pipeline and adds
RIG-Crop targets without mutating the original metadata.
"""

from .schema import ACTIONS, RELATION_POLICIES, ROLES

__all__ = ["ACTIONS", "RELATION_POLICIES", "ROLES"]
