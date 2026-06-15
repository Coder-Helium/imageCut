from __future__ import annotations

from typing import Any, Dict, List, Optional

from .geometry import BBox, union_boxes
from .schema import Detection, MaskRecord


def build_crop_state_graph(
    image_w: int,
    image_h: int,
    vlm_understanding: Dict[str, Any],
    detections: List[Detection],
    masks: Dict[str, List[MaskRecord]],
) -> Dict[str, Any]:
    subject_mask = _first(masks.get("preserve_masks", []))
    subject_box = subject_mask.bbox if subject_mask else _select_subject_box(detections, image_w, image_h)
    subject_name = subject_mask.name if subject_mask else _subject_name(vlm_understanding)

    key_masks = masks.get("relation_masks", []) or []
    env_masks = masks.get("environment_masks", []) or []
    distractor_masks = masks.get("distractor_masks", []) or []

    preserve_boxes = [subject_box] if subject_box else []
    preserve_boxes += [m.bbox for m in key_masks if m.importance >= 0.5]
    preserve_box = union_boxes(preserve_boxes) if preserve_boxes else BBox(0, 0, image_w, image_h)

    relation_box = union_boxes([subject_box] + [m.bbox for m in key_masks]) if subject_box and key_masks else preserve_box
    important_boxes = preserve_boxes + [m.bbox for m in env_masks if m.importance >= 0.55]
    important_region_box = union_boxes(important_boxes) if important_boxes else preserve_box

    subject = _subject_state(subject_name, subject_box, image_w, image_h) if subject_box else {}
    key_objects = [_mask_state(m, image_w, image_h) for m in key_masks]
    background = {
        "important_regions": [_mask_state(m, image_w, image_h) for m in env_masks],
        "environment_importance": _environment_importance(env_masks),
    }
    distractors = [_distractor_state(m, image_w, image_h) for m in distractor_masks]

    issues = infer_issues(subject_box, preserve_box, distractor_masks, image_w, image_h, vlm_understanding)
    actions = sorted({a for issue in issues for a in issue.get("suggested_actions", [])})
    if not actions:
        actions = vlm_understanding.get("composition_intent", {}).get("suggested_actions", []) or ["place_subject_center"]

    return {
        "subject": subject,
        "key_objects": key_objects,
        "union_regions": {
            "preserve_box": preserve_box.to_xyxy_int(),
            "relation_box": relation_box.to_xyxy_int(),
            "important_region_box": important_region_box.to_xyxy_int(),
        },
        "background": background,
        "distractors": distractors,
        "issues": issues,
        "suggested_actions": actions,
    }


def infer_issues(
    subject_box: Optional[BBox],
    preserve_box: BBox,
    distractors: List[MaskRecord],
    image_w: int,
    image_h: int,
    vlm_understanding: Dict[str, Any],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    if subject_box:
        margin_l = subject_box.x1 / max(image_w, 1)
        margin_r = (image_w - subject_box.x2) / max(image_w, 1)
        margin_t = subject_box.y1 / max(image_h, 1)
        margin_b = (image_h - subject_box.y2) / max(image_h, 1)
        cx = (subject_box.x1 + subject_box.x2) / 2.0 / max(image_w, 1)
        cy = (subject_box.y1 + subject_box.y2) / 2.0 / max(image_h, 1)
        area_ratio = subject_box.area() / max(float(image_w * image_h), 1.0)

        if margin_t < 0.04:
            issues.append(_issue("subject_top_too_tight", 1.0 - margin_t / 0.04, ["move_up", "zoom_out"]))
        if margin_b < 0.04:
            issues.append(_issue("subject_bottom_too_tight", 1.0 - margin_b / 0.04, ["move_down", "zoom_out"]))
        if margin_l < 0.035:
            issues.append(_issue("subject_left_too_tight", 1.0 - margin_l / 0.035, ["move_left", "zoom_out"]))
        if margin_r < 0.035:
            issues.append(_issue("subject_right_too_tight", 1.0 - margin_r / 0.035, ["move_right", "zoom_out"]))
        if cx < 0.28:
            issues.append(_issue("subject_too_left", 0.28 - cx, ["move_left", "place_subject_center"]))
        elif cx > 0.72:
            issues.append(_issue("subject_too_right", cx - 0.72, ["move_right", "place_subject_center"]))
        if area_ratio < 0.04:
            issues.append(_issue("subject_too_small", min(1.0, (0.04 - area_ratio) / 0.04), ["zoom_in"]))
        elif area_ratio > 0.65:
            issues.append(_issue("subject_too_large", min(1.0, (area_ratio - 0.65) / 0.35), ["zoom_out"]))

    if preserve_box.area() > 0 and preserve_box.area() / max(float(image_w * image_h), 1.0) > 0.70:
        issues.append(_issue("large_relation_region", 0.5, ["preserve_relation", "fallback_full_image"]))

    for m in distractors:
        if m.importance <= 0.35:
            issues.append(_issue("background_too_distracting", min(1.0, m.area_ratio * 4.0), ["remove_distractor", "zoom_in"]))

    initial = vlm_understanding.get("composition_intent", {}).get("initial_issue")
    if initial and initial not in {"unknown", "unknown_issue"}:
        issues.append(
            {
                "issue_type": initial,
                "severity": 0.5,
                "evidence": "vlm_composition_intent",
                "suggested_actions": vlm_understanding.get("composition_intent", {}).get("suggested_actions", []),
            }
        )

    if not issues:
        issues.append(_issue("already_good_composition", 0.25, ["no_crop_needed", "place_subject_center"]))
    return issues


def _issue(issue_type: str, severity: float, actions: List[str]) -> Dict[str, Any]:
    return {
        "issue_type": issue_type,
        "severity": max(0.0, min(1.0, float(severity))),
        "evidence": "geometry",
        "suggested_actions": actions,
    }


def _select_subject_box(detections: List[Detection], image_w: int, image_h: int) -> Optional[BBox]:
    if not detections:
        return None
    people = [d for d in detections if d.name.lower() in {"person", "man", "woman", "boy", "girl", "people"}]
    pool = people or detections
    return max(pool, key=lambda d: d.bbox.area()).bbox


def _subject_name(vlm: Dict[str, Any]) -> str:
    main = vlm.get("main_subject")
    if isinstance(main, dict):
        return str(main.get("name") or main.get("category") or "main subject")
    return "main subject"


def _subject_state(name: str, box: BBox, image_w: int, image_h: int) -> Dict[str, Any]:
    cx, cy = box.center()
    edge_touch = []
    if box.x1 / max(image_w, 1) < 0.02:
        edge_touch.append("left")
    if (image_w - box.x2) / max(image_w, 1) < 0.02:
        edge_touch.append("right")
    if box.y1 / max(image_h, 1) < 0.02:
        edge_touch.append("top")
    if (image_h - box.y2) / max(image_h, 1) < 0.02:
        edge_touch.append("bottom")
    return {
        "name": name,
        "bbox": box.to_xyxy_int(),
        "center": [int(round(cx)), int(round(cy))],
        "center_norm": [cx / max(image_w, 1), cy / max(image_h, 1)],
        "area_ratio": box.area() / max(float(image_w * image_h), 1.0),
        "edge_touch": edge_touch,
        "completeness": 1.0 if not edge_touch else 0.85,
        "importance": 1.0,
    }


def _mask_state(mask: MaskRecord, image_w: int, image_h: int) -> Dict[str, Any]:
    cx, cy = mask.bbox.center()
    return {
        "name": mask.name,
        "bbox": mask.bbox.to_xyxy_int(),
        "center": [int(round(cx)), int(round(cy))],
        "center_norm": [cx / max(image_w, 1), cy / max(image_h, 1)],
        "area_ratio": mask.area_ratio,
        "completeness": 1.0,
        "importance": mask.importance,
        "relation": mask.role,
    }


def _distractor_state(mask: MaskRecord, image_w: int, image_h: int) -> Dict[str, Any]:
    state = _mask_state(mask, image_w, image_h)
    cx, cy = mask.bbox.center()
    if cx < image_w * 0.33:
        loc = "left_edge"
    elif cx > image_w * 0.67:
        loc = "right_edge"
    elif cy < image_h * 0.33:
        loc = "top_edge"
    elif cy > image_h * 0.67:
        loc = "bottom_edge"
    else:
        loc = "center"
    state["location"] = loc
    state["removal_priority"] = max(0.0, min(1.0, 1.0 - mask.importance))
    return state


def _environment_importance(env_masks: List[MaskRecord]) -> str:
    if not env_masks:
        return "low"
    score = max(m.importance for m in env_masks)
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _first(items):
    return items[0] if items else None

