from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .box_ops import candidate_box_features, normalize_xyxy
from .image_io import crop_rgb, read_image_rgb, resize_to_tensor
from .io import load_jsonl
from .schema import ACTIONS, RELATION_POLICIES, ROLES, compact_rig_record


def score_to_unit(score: float) -> float:
    return max(0.0, min(1.0, (float(score) - 1.0) / 4.0))


class RIGPairwiseDataset(Dataset):
    """Pairwise crop dataset with RIG graph targets.

    This dataset strictly consumes the current DACC-style JSONL records:
    - CPC records: use ``pairwise_preferences``.
    - GAIC records: derive pairs from candidate MOS/final_score if no explicit
      pairwise preferences are present.
    - Qwen/VLM middle state: consumed through pre-built ``rig_targets``.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        image_size: int = 384,
        crop_size: int = 224,
        max_records: Optional[int] = None,
        max_pairs_per_record: Optional[int] = None,
        max_nodes: int = 8,
        min_score_gap: float = 0.05,
        derive_pairs_from_scores: bool = True,
        seed: int = 42,
        image_cache_size: int = 8,
        compact_records: bool = True,
        keep_raw_middle_state: bool = False,
        keep_node_text: bool = False,
    ) -> None:
        records = load_jsonl(jsonl_path, max_records=max_records or 0)
        if compact_records:
            records = [
                compact_rig_record(
                    rec,
                    max_nodes=max_nodes,
                    build_if_missing=False,
                    keep_raw_middle_state=keep_raw_middle_state,
                    keep_node_text=keep_node_text,
                )
                for rec in records
            ]
        self.records = records
        self.image_size = image_size
        self.crop_size = crop_size
        self.max_nodes = max_nodes
        self.image_cache_size = max(0, int(image_cache_size or 0))
        self._image_cache: OrderedDict[int, Any] = OrderedDict()
        self._image_tensor_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._candidate_maps: List[Dict[str, Any]] = [
            {str(c.get("candidate_id")): c for c in rec.get("candidates", [])}
            for rec in self.records
        ]
        self.items: List[Tuple[int, Dict[str, Any]]] = []
        rng = random.Random(seed)
        for ridx, rec in enumerate(self.records):
            pairs = list(rec.get("pairwise_preferences", []) or [])
            if not pairs and derive_pairs_from_scores:
                pairs = _derive_pairs_from_candidate_scores(rec, min_score_gap=min_score_gap)
            if max_pairs_per_record is not None and len(pairs) > max_pairs_per_record:
                pairs = rng.sample(pairs, max_pairs_per_record)
            for pair in pairs:
                self.items.append((ridx, pair))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ridx, pref = self.items[idx]
        rec = self.records[ridx]
        candidates = self._candidate_maps[ridx]
        winner = candidates[str(pref["winner"])]
        loser = candidates[str(pref["loser"])]
        img, full_tensor = self._load_image(ridx, rec)
        h, w = img.shape[:2]
        winner_box = winner["box"]
        loser_box = loser["box"]
        rig = rec.get("rig_targets", {}) if isinstance(rec.get("rig_targets"), dict) else {}
        utilities = rig.get("candidate_utilities", {}) if isinstance(rig.get("candidate_utilities"), dict) else {}
        return {
            "image": full_tensor,
            "winner_crop": resize_to_tensor(crop_rgb(img, winner_box), self.crop_size),
            "loser_crop": resize_to_tensor(crop_rgb(img, loser_box), self.crop_size),
            "winner_box_feat": torch.tensor(candidate_box_features(normalize_xyxy(winner_box, w, h)), dtype=torch.float32),
            "loser_box_feat": torch.tensor(candidate_box_features(normalize_xyxy(loser_box, w, h)), dtype=torch.float32),
            "weight": torch.tensor(float(pref.get("weight", 1.0)), dtype=torch.float32),
            "winner_utility": torch.tensor(_utility_for_candidate(utilities, str(pref["winner"])), dtype=torch.float32),
            "loser_utility": torch.tensor(_utility_for_candidate(utilities, str(pref["loser"])), dtype=torch.float32),
            **_rig_target_tensors(rig, self.max_nodes),
            "sample_id": rec.get("sample_id", ""),
            "winner": str(pref["winner"]),
            "loser": str(pref["loser"]),
        }

    def _load_image(self, ridx: int, rec: Dict[str, Any]) -> Tuple[Any, torch.Tensor]:
        if self.image_cache_size <= 0:
            img = read_image_rgb(rec["image_path"])
            return img, resize_to_tensor(img, self.image_size)

        img = self._image_cache.get(ridx)
        full_tensor = self._image_tensor_cache.get(ridx)
        if img is not None and full_tensor is not None:
            self._image_cache.move_to_end(ridx)
            self._image_tensor_cache.move_to_end(ridx)
            return img, full_tensor

        img = read_image_rgb(rec["image_path"])
        full_tensor = resize_to_tensor(img, self.image_size)
        self._image_cache[ridx] = img
        self._image_tensor_cache[ridx] = full_tensor
        self._image_cache.move_to_end(ridx)
        self._image_tensor_cache.move_to_end(ridx)
        while len(self._image_cache) > self.image_cache_size:
            self._image_cache.popitem(last=False)
        while len(self._image_tensor_cache) > self.image_cache_size:
            self._image_tensor_cache.popitem(last=False)
        return img, full_tensor


def _derive_pairs_from_candidate_scores(rec: Dict[str, Any], min_score_gap: float = 0.05) -> List[Dict[str, Any]]:
    candidates = rec.get("candidates", []) or []
    scored: List[Tuple[str, float]] = []
    raw_scores = []
    for cand in candidates:
        cid = str(cand.get("candidate_id"))
        scores = cand.get("scores", {}) if isinstance(cand.get("scores"), dict) else {}
        value = scores.get("mos", scores.get("final_score", cand.get("score", None)))
        if value is None:
            continue
        raw_scores.append(float(value))
        scored.append((cid, float(value)))
    if not scored:
        return []
    lo = min(raw_scores)
    hi = max(raw_scores)

    def unit(v: float) -> float:
        if 0.0 <= lo and hi <= 1.0:
            return v
        if 1.0 <= lo and hi <= 5.0:
            return (v - 1.0) / 4.0
        return 0.5 if hi <= lo else (v - lo) / (hi - lo)

    pairs: List[Dict[str, Any]] = []
    for i, (cid_a, score_a) in enumerate(scored):
        for cid_b, score_b in scored[i + 1 :]:
            diff = unit(score_a) - unit(score_b)
            if abs(diff) < min_score_gap:
                continue
            if diff > 0:
                pairs.append({"winner": cid_a, "loser": cid_b, "weight": abs(diff), "source": "score_derived_pair"})
            else:
                pairs.append({"winner": cid_b, "loser": cid_a, "weight": abs(diff), "source": "score_derived_pair"})
    return pairs


def _utility_for_candidate(utilities: Dict[str, Any], candidate_id: str) -> float:
    item = utilities.get(candidate_id, {}) if isinstance(utilities, dict) else {}
    try:
        return float(item.get("utility_unit", item.get("utility_raw", 0.5)))
    except (TypeError, ValueError):
        return 0.5


def _rig_target_tensors(rig: Dict[str, Any], max_nodes: int) -> Dict[str, torch.Tensor]:
    nodes = list(rig.get("nodes", []) or [])[:max_nodes]
    while len(nodes) < max_nodes:
        nodes.append({})
    boxes = torch.zeros(max_nodes, 4, dtype=torch.float32)
    roles = torch.full((max_nodes,), ROLES.index("padding"), dtype=torch.long)
    importance = torch.zeros(max_nodes, dtype=torch.float32)
    valid = torch.zeros(max_nodes, dtype=torch.float32)
    has_box = torch.zeros(max_nodes, dtype=torch.float32)
    for idx, node in enumerate(nodes):
        box = node.get("bbox_norm", [0, 0, 0, 0])
        if isinstance(box, list) and len(box) >= 4:
            boxes[idx] = torch.tensor([float(v) for v in box[:4]], dtype=torch.float32)
        role_id = int(node.get("role_id", ROLES.index("padding")) or 0)
        roles[idx] = max(0, min(len(ROLES) - 1, role_id))
        importance[idx] = float(node.get("importance", 0.0) or 0.0)
        valid[idx] = 1.0 if node.get("valid") else 0.0
        has_box[idx] = 1.0 if node.get("has_box") else 0.0

    relations = rig.get("relations", {}) if isinstance(rig.get("relations"), dict) else {}
    policy = torch.zeros(max_nodes, max_nodes, dtype=torch.long)
    rel_weight = torch.zeros(max_nodes, max_nodes, dtype=torch.float32)
    rel_mask = torch.zeros(max_nodes, max_nodes, dtype=torch.float32)
    for i in range(max_nodes):
        for j in range(max_nodes):
            try:
                policy[i, j] = int(relations.get("policy", [])[i][j])
                rel_weight[i, j] = float(relations.get("weight", [])[i][j])
                rel_mask[i, j] = 1.0 if relations.get("mask", [])[i][j] else 0.0
            except (IndexError, TypeError, ValueError):
                continue

    action_targets = rig.get("action_targets", {}) if isinstance(rig.get("action_targets"), dict) else {}
    action = torch.zeros(len(ACTIONS), dtype=torch.float32)
    values = action_targets.get("multi_hot", [])
    if isinstance(values, list):
        for idx, value in enumerate(values[: len(ACTIONS)]):
            action[idx] = float(value)

    return {
        "node_boxes": boxes,
        "node_roles": roles,
        "node_importance": importance,
        "node_valid": valid,
        "node_has_box": has_box,
        "relation_policy": policy.clamp(0, len(RELATION_POLICIES) - 1),
        "relation_weight": rel_weight,
        "relation_mask": rel_mask,
        "action_targets": action,
    }
