from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Sequence

from .box_ops import boundary_cut, candidate_box_features, clip_box, coverage, normalize_xyxy, valid_box


ROLES = ["main_subject", "key_object", "important_background", "distractor", "padding"]
ROLE_TO_ID = {name: idx for idx, name in enumerate(ROLES)}

RELATION_POLICIES = [
    "none",
    "preserve_relation",
    "optional_preserve",
    "avoid_cutting",
    "leave_space",
    "distractor_exclusion",
]
RELATION_TO_ID = {name: idx for idx, name in enumerate(RELATION_POLICIES)}

ACTIONS = [
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "zoom_in",
    "zoom_out",
    "place_subject_center",
    "place_subject_left_third",
    "place_subject_right_third",
    "preserve_relation",
    "remove_distractor",
    "keep_environment",
    "keep_full_body",
    "keep_upper_body",
    "fallback_full_image",
    "no_crop_needed",
]
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTIONS)}


def build_rig_targets(rec: Dict[str, Any], max_nodes: int = 8) -> Dict[str, Any]:
    middle = _middle_state(rec)
    image_w = int(rec.get("image_width", 0) or 0)
    image_h = int(rec.get("image_height", 0) or 0)
    nodes = _extract_nodes(middle, max_nodes=max_nodes, image_w=image_w, image_h=image_h)
    relations = _build_relations(nodes, middle)
    utilities = _candidate_utilities(rec, nodes, relations)
    actions = _action_targets(middle)
    flags = _quality_flags(nodes, relations, middle)
    return {
        "version": "rig_targets_v1",
        "source_middle_state": str(middle.get("source", rec.get("quality_flags", {}).get("semantic_teacher_source", "unknown"))),
        "crop_supervision_source": _crop_supervision_source(rec),
        "max_nodes": int(max_nodes),
        "roles": ROLES,
        "relation_policies": RELATION_POLICIES,
        "actions": ACTIONS,
        "nodes": nodes,
        "relations": relations,
        "candidate_utilities": utilities,
        "action_targets": actions,
        "graph_quality_flags": flags,
    }


def compact_rig_record(
    rec: Dict[str, Any],
    max_nodes: int = 12,
    build_if_missing: bool = False,
    keep_raw_middle_state: bool = False,
    keep_node_text: bool = False,
) -> Dict[str, Any]:
    """Compile a DACC/RIG record into the compact training-time schema.

    The raw VLM middle state may contain long descriptions, free-form reasons,
    prompt artifacts, and other audit-only fields. Training only needs stable
    numeric/categorical supervision, so this function keeps DACC crop labels
    plus a compact ``rig_targets`` block.
    """

    out: Dict[str, Any] = {}
    for key in [
        "sample_id",
        "image_path",
        "rel_path",
        "image_width",
        "image_height",
        "width",
        "height",
        "best_crop",
        "best_score",
    ]:
        if key in rec:
            out[key] = rec[key]

    for key in ["cpc_supervision", "gaic_supervision", "quality_flags"]:
        value = rec.get(key)
        if isinstance(value, dict) and value:
            out[key] = _compact_primitive_dict(value)

    out["candidates"] = [_compact_candidate(cand) for cand in rec.get("candidates", []) or [] if isinstance(cand, dict)]
    if rec.get("pairwise_preferences"):
        out["pairwise_preferences"] = [
            _compact_pairwise_preference(pref)
            for pref in rec.get("pairwise_preferences", []) or []
            if isinstance(pref, dict)
        ]

    rig = rec.get("rig_targets")
    if isinstance(rig, dict) and rig:
        out["rig_targets"] = compact_rig_targets(rig, max_nodes=max_nodes, keep_node_text=keep_node_text)
    elif build_if_missing:
        out["rig_targets"] = compact_rig_targets(build_rig_targets(rec, max_nodes=max_nodes), max_nodes=max_nodes, keep_node_text=keep_node_text)

    if keep_raw_middle_state:
        for key in ["composition_middle_state", "vlm_understanding"]:
            value = rec.get(key)
            if isinstance(value, dict) and value:
                out[key] = value
    return out


def compact_rig_targets(rig: Dict[str, Any], max_nodes: int = 12, keep_node_text: bool = False) -> Dict[str, Any]:
    """Keep only trainable RIG supervision and audit scalars."""

    out: Dict[str, Any] = {}
    for key in ["version", "source_middle_state", "crop_supervision_source", "max_nodes"]:
        if key in rig:
            out[key] = rig[key]
    out["roles"] = list(rig.get("roles", ROLES) or ROLES)
    out["relation_policies"] = list(rig.get("relation_policies", RELATION_POLICIES) or RELATION_POLICIES)
    out["actions"] = list(rig.get("actions", ACTIONS) or ACTIONS)
    out["nodes"] = [_compact_node(node, keep_text=keep_node_text) for node in list(rig.get("nodes", []) or [])[:max_nodes]]
    while len(out["nodes"]) < max_nodes:
        out["nodes"].append(_compact_node({}, keep_text=keep_node_text, node_id=len(out["nodes"])))
    out["relations"] = _compact_relations(rig.get("relations", {}), max_nodes=max_nodes)
    out["candidate_utilities"] = {
        str(cid): _compact_candidate_utility(item)
        for cid, item in (rig.get("candidate_utilities", {}) or {}).items()
        if isinstance(item, dict)
    }
    actions = rig.get("action_targets", {}) if isinstance(rig.get("action_targets"), dict) else {}
    out["action_targets"] = {"multi_hot": _compact_float_list(actions.get("multi_hot", []), len(ACTIONS))}
    flags = rig.get("graph_quality_flags", {}) if isinstance(rig.get("graph_quality_flags"), dict) else {}
    out["graph_quality_flags"] = _compact_primitive_dict(flags)
    return out


def audit_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counters: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    relation_texts: Counter[str] = Counter()
    action_texts: Counter[str] = Counter()
    for rec in records:
        counters["records"] += 1
        candidates = rec.get("candidates", []) or []
        pairs = rec.get("pairwise_preferences", []) or []
        counters["candidates"] += len(candidates)
        counters["pairwise_preferences"] += len(pairs)
        if rec.get("cpc_supervision"):
            sources["cpc"] += 1
        if rec.get("gaic_supervision"):
            sources["gaic"] += 1
        middle = _middle_state(rec)
        if middle:
            counters["has_middle_state"] += 1
        nodes = _extract_nodes(middle, max_nodes=16, image_w=_rec_image_w(rec), image_h=_rec_image_h(rec)) if isinstance(middle, dict) else []
        main_nodes = [node for node in nodes if node.get("valid") and node.get("role") == "main_subject"]
        if main_nodes:
            counters["has_main_subject"] += 1
            if any(node.get("has_box") for node in main_nodes):
                counters["has_main_subject_bbox"] += 1
            if any(_as_float(node.get("importance"), None) is not None for node in main_nodes):
                counters["has_main_subject_importance"] += 1
        for key in ["key_objects", "important_background", "distractors"]:
            values = middle.get(key, []) if isinstance(middle, dict) else []
            if isinstance(values, list) and values:
                counters[f"has_{key}"] += 1
                if any(isinstance(v, dict) and _entity_bbox_norm(v, _rec_image_w(rec), _rec_image_h(rec)) is not None for v in values):
                    counters[f"has_{key}_bbox"] += 1
        for obj in middle.get("key_objects", []) or []:
            if isinstance(obj, dict) and obj.get("relation_to_subject"):
                relation_texts[str(obj["relation_to_subject"]).strip().lower()] += 1
        intent = middle.get("composition_intent", {}) if isinstance(middle.get("composition_intent"), dict) else {}
        for action in _text_values(intent.get("suggested_actions", [])):
            action_texts[str(action).strip().lower()] += 1
    total = max(counters["records"], 1)
    return {
        "records": counters["records"],
        "supervision_sources": dict(sources),
        "total_candidates": counters["candidates"],
        "total_pairwise_preferences": counters["pairwise_preferences"],
        "rates": {
            "middle_state": counters["has_middle_state"] / total,
            "main_subject": counters["has_main_subject"] / total,
            "main_subject_bbox": counters["has_main_subject_bbox"] / total,
            "main_subject_importance": counters["has_main_subject_importance"] / total,
            "key_objects": counters["has_key_objects"] / total,
            "key_objects_bbox": counters["has_key_objects_bbox"] / total,
            "important_background": counters["has_important_background"] / total,
            "important_background_bbox": counters["has_important_background_bbox"] / total,
            "distractors": counters["has_distractors"] / total,
            "distractors_bbox": counters["has_distractors_bbox"] / total,
        },
        "top_relation_to_subject": relation_texts.most_common(30),
        "top_suggested_actions": action_texts.most_common(30),
    }


def _compact_candidate(cand: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ["candidate_id", "box", "score", "box_format", "source", "rel_path"]:
        if key in cand:
            out[key] = cand[key]
    scores = cand.get("scores")
    if isinstance(scores, dict) and scores:
        out["scores"] = _compact_scores(scores)
    return out


def _compact_scores(scores: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "mos",
        "final_score",
        "cpc_raw_score",
        "score",
        "aesthetic_score",
        "technical_score",
        "composition_score",
    ]
    out: Dict[str, Any] = {}
    for key in keep:
        if key in scores:
            out[key] = scores[key]
    if not out:
        out.update(_compact_primitive_dict(scores))
    return out


def _compact_pairwise_preference(pref: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ["winner", "loser", "weight", "source"]:
        if key in pref:
            out[key] = pref[key]
    return out


def _compact_node(node: Dict[str, Any], keep_text: bool = False, node_id: int | None = None) -> Dict[str, Any]:
    role = str(node.get("role", "padding") or "padding")
    if role not in ROLE_TO_ID:
        role = "padding"
    out: Dict[str, Any] = {
        "node_id": int(node.get("node_id", node_id if node_id is not None else 0) or 0),
        "role": role,
        "role_id": int(node.get("role_id", ROLE_TO_ID[role]) or ROLE_TO_ID[role]),
        "importance": _as_float(node.get("importance", 0.0), 0.0),
        "bbox_norm": _compact_box(node.get("bbox_norm", [0.0, 0.0, 0.0, 0.0])),
        "has_box": bool(node.get("has_box", False)),
        "valid": bool(node.get("valid", False)),
    }
    if keep_text:
        for key in ["name", "category", "description", "relation_to_subject", "promoted_from"]:
            if node.get(key):
                out[key] = _text(node.get(key))
    return out


def _compact_relations(relations: Any, max_nodes: int) -> Dict[str, Any]:
    if not isinstance(relations, dict):
        relations = {}
    return {
        "policy": _compact_int_matrix(relations.get("policy", []), max_nodes=max_nodes),
        "weight": _compact_float_matrix(relations.get("weight", []), max_nodes=max_nodes),
        "mask": _compact_bool_matrix(relations.get("mask", []), max_nodes=max_nodes),
    }


def _compact_candidate_utility(item: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "utility_raw",
        "utility_unit",
        "node_keep",
        "relation_keep",
        "boundary_cut_penalty",
        "distractor_penalty",
        "subject_position_score",
    ]
    out: Dict[str, Any] = {}
    for key in keep:
        if key in item:
            out[key] = _as_float(item.get(key), 0.0)
    return out


def _compact_primitive_dict(value: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)) or item is None:
            out[str(key)] = item
    return out


def _compact_box(value: Any) -> List[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    try:
        return [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return [0.0, 0.0, 0.0, 0.0]


def _compact_float_list(values: Any, length: int) -> List[float]:
    out = [0.0 for _ in range(length)]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return out
    for idx, value in enumerate(values[:length]):
        try:
            out[idx] = float(value)
        except (TypeError, ValueError):
            out[idx] = 0.0
    return out


def _compact_int_matrix(values: Any, max_nodes: int) -> List[List[int]]:
    out = [[0 for _ in range(max_nodes)] for _ in range(max_nodes)]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return out
    for i, row in enumerate(values[:max_nodes]):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            continue
        for j, value in enumerate(row[:max_nodes]):
            try:
                out[i][j] = int(value)
            except (TypeError, ValueError):
                out[i][j] = 0
    return out


def _compact_float_matrix(values: Any, max_nodes: int) -> List[List[float]]:
    out = [[0.0 for _ in range(max_nodes)] for _ in range(max_nodes)]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return out
    for i, row in enumerate(values[:max_nodes]):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            continue
        for j, value in enumerate(row[:max_nodes]):
            try:
                out[i][j] = float(value)
            except (TypeError, ValueError):
                out[i][j] = 0.0
    return out


def _compact_bool_matrix(values: Any, max_nodes: int) -> List[List[bool]]:
    out = [[False for _ in range(max_nodes)] for _ in range(max_nodes)]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return out
    for i, row in enumerate(values[:max_nodes]):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            continue
        for j, value in enumerate(row[:max_nodes]):
            out[i][j] = bool(value)
    return out


def _middle_state(rec: Dict[str, Any]) -> Dict[str, Any]:
    middle = rec.get("composition_middle_state")
    if isinstance(middle, dict) and middle:
        return middle
    understanding = rec.get("vlm_understanding")
    return understanding if isinstance(understanding, dict) else {}


def _extract_nodes(middle: Dict[str, Any], max_nodes: int, image_w: int = 0, image_h: int = 0) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    main = middle.get("main_subject")
    key_values = middle.get("key_objects", [])
    key_entities = [item for item in key_values if isinstance(item, dict)] if isinstance(key_values, list) else []
    promoted_key_idx: int | None = None
    if isinstance(main, dict):
        nodes.append(_entity_to_node(main, "main_subject", image_w=image_w, image_h=image_h))
    elif key_entities:
        promoted_key_idx = _select_promoted_main_index(key_entities, _text(main), image_w=image_w, image_h=image_h)
        if promoted_key_idx is not None:
            promoted = _entity_to_node(key_entities[promoted_key_idx], "main_subject", image_w=image_w, image_h=image_h)
            promoted["promoted_from"] = "key_objects"
            nodes.append(promoted)
    nodes.extend(
        _entity_to_node(v, "key_object", image_w=image_w, image_h=image_h)
        for idx, v in enumerate(key_entities)
        if idx != promoted_key_idx
    )
    nodes.extend(_list_to_nodes(middle.get("important_background", []), "important_background", image_w=image_w, image_h=image_h))
    nodes.extend(_list_to_nodes(middle.get("distractors", []), "distractor", image_w=image_w, image_h=image_h))
    nodes = sorted(nodes, key=lambda item: (item["role"] != "main_subject", -float(item["importance"])))
    nodes = nodes[:max_nodes]
    for idx, node in enumerate(nodes):
        node["node_id"] = idx
    while len(nodes) < max_nodes:
        nodes.append(
            {
                "node_id": len(nodes),
                "role": "padding",
                "role_id": ROLE_TO_ID["padding"],
                "name": "",
                "category": "",
                "description": "",
                "importance": 0.0,
                "bbox_norm": [0.0, 0.0, 0.0, 0.0],
                "has_box": False,
                "valid": False,
            }
        )
    return nodes


def _list_to_nodes(values: Any, role: str, image_w: int = 0, image_h: int = 0) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    return [_entity_to_node(v, role, image_w=image_w, image_h=image_h) for v in values if isinstance(v, dict)]


def _entity_to_node(entity: Dict[str, Any], role: str, image_w: int = 0, image_h: int = 0) -> Dict[str, Any]:
    bbox = _entity_bbox_norm(entity, image_w=image_w, image_h=image_h)
    has_box = bbox is not None
    return {
        "node_id": -1,
        "role": role,
        "role_id": ROLE_TO_ID[role],
        "name": _text(entity.get("name")),
        "category": _text(entity.get("category")),
        "description": _text(entity.get("description")),
        "relation_to_subject": _text(entity.get("relation_to_subject")),
        "importance": _as_float(entity.get("importance"), _default_importance(role)),
        "bbox_norm": bbox if has_box else [0.0, 0.0, 0.0, 0.0],
        "has_box": bool(has_box),
        "valid": True,
    }


def _select_promoted_main_index(key_entities: List[Dict[str, Any]], main_hint: str = "", image_w: int = 0, image_h: int = 0) -> int | None:
    best_idx: int | None = None
    best_score = -1e9
    hint_tokens = _tokens(main_hint)
    for idx, entity in enumerate(key_entities):
        score = _as_float(entity.get("importance"), _default_importance("key_object"))
        if _entity_bbox_norm(entity, image_w=image_w, image_h=image_h) is not None:
            score += 2.0
        signature = _node_signature(entity)
        relation = _text(entity.get("relation_to_subject")).lower()
        name_tokens = _tokens(entity.get("name")) | _tokens(entity.get("category")) | _tokens(entity.get("description"))
        if hint_tokens and (hint_tokens & name_tokens):
            score += 3.0
        if any(word in relation for word in ["main", "primary", "central focal", "primary focal", "core"]):
            score += 4.0
        elif "focal" in relation:
            score += 2.0
        elif "subject" in relation:
            score += 1.0
        if signature & {"main", "primary", "subject", "focal"}:
            score += 0.5
        if any(word in relation for word in ["secondary", "tertiary", "background", "environment"]):
            score -= 2.5
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx


def _build_relations(nodes: List[Dict[str, Any]], middle: Dict[str, Any]) -> Dict[str, Any]:
    n = len(nodes)
    policy = [[RELATION_TO_ID["none"] for _ in range(n)] for _ in range(n)]
    weights = [[0.0 for _ in range(n)] for _ in range(n)]
    mask = [[False for _ in range(n)] for _ in range(n)]
    preserve_text = _intent_text_set(middle, "preserve")
    optional_text = _intent_text_set(middle, "optional_preserve")
    avoid_text = _intent_text_set(middle, "avoid_cutting")
    main_indices = [i for i, node in enumerate(nodes) if node["role"] == "main_subject" and node["valid"]]
    if not main_indices:
        return {"policy": policy, "weight": weights, "mask": mask}
    main_idx = main_indices[0]
    for j, node in enumerate(nodes):
        if j == main_idx or not node["valid"]:
            continue
        rel = _relation_for_node(node, preserve_text, optional_text, avoid_text)
        weight = math.sqrt(max(0.0, float(nodes[main_idx]["importance"])) * max(0.0, float(node["importance"])))
        policy[main_idx][j] = RELATION_TO_ID[rel]
        policy[j][main_idx] = RELATION_TO_ID[rel]
        weights[main_idx][j] = weight
        weights[j][main_idx] = weight
        mask[main_idx][j] = True
        mask[j][main_idx] = True
    return {"policy": policy, "weight": weights, "mask": mask}


def _relation_for_node(node: Dict[str, Any], preserve_text: set[str], optional_text: set[str], avoid_text: set[str]) -> str:
    if node["role"] == "distractor":
        return "distractor_exclusion"
    signature = _node_signature(node)
    if signature & avoid_text:
        return "avoid_cutting"
    if signature & preserve_text:
        return "preserve_relation"
    if signature & optional_text:
        return "optional_preserve"
    if node["role"] == "important_background":
        return "leave_space"
    if node["role"] == "key_object":
        return "preserve_relation" if float(node["importance"]) >= 0.65 else "optional_preserve"
    return "none"


def _candidate_utilities(rec: Dict[str, Any], nodes: List[Dict[str, Any]], relations: Dict[str, Any]) -> Dict[str, Any]:
    image_w = int(rec.get("image_width", 0) or 0)
    image_h = int(rec.get("image_height", 0) or 0)
    utilities: Dict[str, Any] = {}
    raw_values: List[float] = []
    for cand in rec.get("candidates", []) or []:
        cid = str(cand.get("candidate_id", len(utilities)))
        crop = normalize_xyxy(cand.get("box", [0, 0, image_w, image_h]), image_w, image_h)
        item = _utility_for_crop(crop, nodes, relations)
        item["box_feat"] = candidate_box_features(crop)
        utilities[cid] = item
        raw_values.append(float(item["utility_raw"]))
    lo = min(raw_values) if raw_values else 0.0
    hi = max(raw_values) if raw_values else 1.0
    for item in utilities.values():
        raw = float(item["utility_raw"])
        item["utility_unit"] = 0.5 if hi <= lo else max(0.0, min(1.0, (raw - lo) / (hi - lo)))
    return utilities


def _utility_for_crop(crop: Sequence[float], nodes: List[Dict[str, Any]], relations: Dict[str, Any]) -> Dict[str, float]:
    node_keep = 0.0
    cut_penalty = 0.0
    distractor_penalty = 0.0
    main_position = 0.0
    node_cov: List[float] = []
    for node in nodes:
        if not node["valid"] or not node["has_box"]:
            node_cov.append(0.0)
            continue
        cov = coverage(node["bbox_norm"], crop)
        node_cov.append(cov)
        imp = float(node["importance"])
        if node["role"] == "distractor":
            distractor_penalty += imp * cov
        else:
            node_keep += imp * cov
            cut_penalty += imp * boundary_cut(node["bbox_norm"], crop)
        if node["role"] == "main_subject":
            main_position = _subject_position_score(node["bbox_norm"], crop)
    rel_keep = 0.0
    rel_weight = relations.get("weight", [])
    rel_policy = relations.get("policy", [])
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            try:
                policy_id = int(rel_policy[i][j])
                weight = float(rel_weight[i][j])
            except (IndexError, TypeError, ValueError):
                continue
            if policy_id == RELATION_TO_ID["none"] or weight <= 0.0:
                continue
            if policy_id == RELATION_TO_ID["distractor_exclusion"]:
                continue
            rel_keep += weight * min(node_cov[i], node_cov[j])
    utility = node_keep + 0.7 * rel_keep + 0.2 * main_position - 0.7 * cut_penalty - 0.5 * distractor_penalty
    return {
        "utility_raw": round(float(utility), 6),
        "node_keep": round(float(node_keep), 6),
        "relation_keep": round(float(rel_keep), 6),
        "boundary_cut_penalty": round(float(cut_penalty), 6),
        "distractor_penalty": round(float(distractor_penalty), 6),
        "subject_position_score": round(float(main_position), 6),
    }


def _subject_position_score(node_box: Sequence[float], crop: Sequence[float]) -> float:
    cov = coverage(node_box, crop)
    if cov <= 0.0:
        return 0.0
    nx1, ny1, nx2, ny2 = clip_box(node_box)
    cx = (nx1 + nx2) / 2.0
    cy = (ny1 + ny2) / 2.0
    x1, y1, x2, y2 = clip_box(crop)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    rx = (cx - x1) / max(x2 - x1, 1e-6)
    ry = (cy - y1) / max(y2 - y1, 1e-6)
    center_score = 1.0 - min(1.0, ((rx - 0.5) ** 2 + (ry - 0.5) ** 2) ** 0.5 * 2.0)
    third_score = max(1.0 - abs(rx - 1.0 / 3.0) * 3.0, 1.0 - abs(rx - 2.0 / 3.0) * 3.0, 0.0)
    return max(0.0, min(1.0, cov * max(center_score, third_score)))


def _action_targets(middle: Dict[str, Any]) -> Dict[str, Any]:
    intent = middle.get("composition_intent", {}) if isinstance(middle.get("composition_intent"), dict) else {}
    actions = [str(a).strip() for a in _text_values(intent.get("suggested_actions", []))]
    multi_hot = [0.0 for _ in ACTIONS]
    unknown: List[str] = []
    for action in actions:
        if action in ACTION_TO_ID:
            multi_hot[ACTION_TO_ID[action]] = 1.0
        elif action:
            unknown.append(action)
    return {"multi_hot": multi_hot, "labels": actions, "unknown_labels": unknown}


def _quality_flags(nodes: List[Dict[str, Any]], relations: Dict[str, Any], middle: Dict[str, Any]) -> Dict[str, Any]:
    valid_nodes = [node for node in nodes if node["valid"]]
    boxed_nodes = [node for node in valid_nodes if node["has_box"]]
    relation_mask = relations.get("mask", [])
    relation_count = sum(1 for row in relation_mask for item in row if item) // 2
    return {
        "valid_node_count": len(valid_nodes),
        "boxed_node_count": len(boxed_nodes),
        "has_main_subject": any(node["role"] == "main_subject" for node in valid_nodes),
        "has_main_subject_box": any(node["role"] == "main_subject" and node["has_box"] for node in valid_nodes),
        "relation_count": relation_count,
        "has_composition_intent": isinstance(middle.get("composition_intent"), dict) and bool(middle.get("composition_intent")),
    }


def _intent_text_set(middle: Dict[str, Any], key: str) -> set[str]:
    intent = middle.get("composition_intent", {}) if isinstance(middle.get("composition_intent"), dict) else {}
    out: set[str] = set()
    for value in _text_values(intent.get(key, [])):
        out.update(_tokens(value))
    return out


def _text_values(value: Any) -> List[str]:
    """Flatten loose VLM fields into strings.

    Qwen sometimes returns booleans or dictionaries for fields prompted as
    string arrays. A bare boolean has no text; a keyed boolean dictionary like
    {"person": true} still exposes "person" as a usable token.
    """
    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, dict):
        out: List[str] = []
        for key, item in value.items():
            if isinstance(item, bool):
                if item:
                    out.extend(_text_values(key))
            else:
                out.extend(_text_values(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(_text_values(item))
        return out
    return [str(value)]


def _node_signature(node: Dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ["name", "category", "description", "relation_to_subject"]:
        out.update(_tokens(node.get(key, "")))
    return out


def _tokens(value: Any) -> set[str]:
    text = str(value or "").strip().lower()
    if not text:
        return set()
    parts = {text}
    for token in text.replace("/", " ").replace("_", " ").replace("-", " ").split():
        if token:
            parts.add(token)
    return parts


def _crop_supervision_source(rec: Dict[str, Any]) -> str:
    if rec.get("cpc_supervision"):
        return "cpc_pairwise_preference"
    if rec.get("gaic_supervision"):
        return "gaicd_human_mos"
    if rec.get("pairwise_preferences"):
        return "pairwise_preference"
    return "unknown"


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) < 4:
        return False
    try:
        box = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return False
    return valid_box(box, min_size=1e-4)


def _entity_bbox_norm(entity: Dict[str, Any], image_w: int = 0, image_h: int = 0) -> List[float] | None:
    for key in ["bbox_norm", "bbox", "box", "bbox_xyxy"]:
        if key not in entity:
            continue
        box = _box_to_norm(entity.get(key), normalized_hint=(key == "bbox_norm"), image_w=image_w, image_h=image_h)
        if box is not None:
            return box
    return None


def _box_to_norm(value: Any, normalized_hint: bool, image_w: int = 0, image_h: int = 0) -> List[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    try:
        vals = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        return None

    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        vals[2] = vals[0] + max(0.0, vals[2])
        vals[3] = vals[1] + max(0.0, vals[3])

    max_val = max(vals)
    min_val = min(vals)
    if normalized_hint and 1.5 < max_val <= 1000.0 and min_val >= 0.0:
        vals = [v / 1000.0 for v in vals]
    elif 0.0 <= min_val and max_val <= 1.5:
        pass
    elif image_w > 0 and image_h > 0:
        vals = [vals[0] / image_w, vals[1] / image_h, vals[2] / image_w, vals[3] / image_h]
    else:
        return None

    box = clip_box(vals)
    return box if valid_box(box, min_size=1e-4) else None


def _rec_image_w(rec: Dict[str, Any]) -> int:
    return int(rec.get("image_width", 0) or rec.get("width", 0) or 0)


def _rec_image_h(rec: Dict[str, Any]) -> int:
    return int(rec.get("image_height", 0) or rec.get("height", 0) or 0)


def _as_float(value: Any, default: float | None = 0.0) -> float:
    try:
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            raise ValueError
        return max(0.0, min(1.0, parsed))
    except (TypeError, ValueError):
        return 0.0 if default is None else float(default)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _default_importance(role: str) -> float:
    return {
        "main_subject": 1.0,
        "key_object": 0.75,
        "important_background": 0.45,
        "distractor": 0.35,
    }.get(role, 0.0)
