from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from .geometry import BBox, dedup_boxes, fit_box_to_aspect, make_aspect_box_from_center, parse_aspect_ratio, shift_box_into_image, union_boxes
from .schema import Candidate


def generate_candidates(
    image_w: int,
    image_h: int,
    target_aspect: str,
    crop_state_graph: Dict[str, Any],
    max_candidates: int = 80,
    grid_size: int = 12,
) -> List[Candidate]:
    aspect = parse_aspect_ratio(target_aspect, image_w, image_h)
    raw: List[Tuple[BBox, str, str, str, str]] = []

    raw.extend((b, "grid_anchor", "unknown", "gaic_grid_anchor", "GAIC-style grid anchor candidate") for b in grid_anchor_candidates(image_w, image_h, aspect, grid_size))
    raw.extend(direction_candidates(image_w, image_h, aspect, crop_state_graph))
    raw.extend(mask_protection_candidates(image_w, image_h, aspect, crop_state_graph))
    raw.extend(negative_candidates(image_w, image_h, aspect, crop_state_graph))

    # Always include full image if it matches the target after fitting.
    raw.append((fit_box_to_aspect(BBox(0, 0, image_w, image_h), aspect, image_w, image_h), "fallback", "fallback_full_image", "fallback_full_image", "Full image fallback"))

    boxes = dedup_boxes([x[0] for x in raw])
    by_key = {tuple(b.to_xyxy_int()): b for b in boxes}
    deduped: List[Tuple[BBox, str, str, str, str]] = []
    seen = set()
    for b, source, action, issue, reason in raw:
        key = tuple(b.to_xyxy_int())
        if key in seen or key not in by_key:
            continue
        seen.add(key)
        deduped.append((by_key[key], source, action, issue, reason))

    out: List[Candidate] = []
    for idx, (box, source, action, issue, reason) in enumerate(deduped[:max_candidates]):
        out.append(
            Candidate(
                candidate_id=f"cand_{idx:04d}",
                box=box,
                source=source,
                action=action,
                issue=issue,
                reason=reason,
            )
        )
    return out


def grid_anchor_candidates(image_w: int, image_h: int, aspect: float, grid_size: int = 12) -> List[BBox]:
    boxes: List[BBox] = []
    # Fixed-aspect center grid is more useful for product ratios than pure corner enumeration.
    scales = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50]
    xs = [0.25, 0.333, 0.5, 0.667, 0.75]
    ys = [0.25, 0.333, 0.5, 0.667, 0.75]
    max_width = min(float(image_w), float(image_h) * aspect)
    if max_width <= 2:
        return []
    for s in scales:
        width = max_width * s
        for nx in xs:
            for ny in ys:
                boxes.append(make_aspect_box_from_center(nx * image_w, ny * image_h, width, aspect, image_w, image_h))
    return boxes


def mask_protection_candidates(image_w: int, image_h: int, aspect: float, graph: Dict[str, Any]) -> List[Tuple[BBox, str, str, str, str]]:
    out: List[Tuple[BBox, str, str, str, str]] = []
    union_regions = graph.get("union_regions", {})
    for key, action in [
        ("preserve_box", "keep_full_body"),
        ("relation_box", "preserve_relation"),
        ("important_region_box", "keep_environment"),
    ]:
        box_values = union_regions.get(key)
        if not box_values:
            continue
        base = BBox.from_seq(box_values)
        for padding in [0.08, 0.18, 0.32, 0.50]:
            out.append(
                (
                    fit_box_to_aspect(base, aspect, image_w, image_h, padding=padding),
                    "mask_protection",
                    action,
                    key,
                    f"Protect {key} with padding {padding:.2f}",
                )
            )
    return out


def direction_candidates(image_w: int, image_h: int, aspect: float, graph: Dict[str, Any]) -> List[Tuple[BBox, str, str, str, str]]:
    subject_box = None
    if graph.get("subject", {}).get("bbox"):
        subject_box = BBox.from_seq(graph["subject"]["bbox"])
    if subject_box is None:
        return []

    base = fit_box_to_aspect(subject_box, aspect, image_w, image_h, padding=0.45)
    bw = base.w()
    bh = base.h()
    out: List[Tuple[BBox, str, str, str, str]] = []
    action_to_shift = {
        "move_left": (-0.12, 0.0),
        "move_right": (0.12, 0.0),
        "move_up": (0.0, -0.12),
        "move_down": (0.0, 0.12),
        "place_subject_center": (0.0, 0.0),
        "place_subject_left_third": (0.18, 0.0),
        "place_subject_right_third": (-0.18, 0.0),
    }

    graph_actions = graph.get("suggested_actions", []) or ["place_subject_center"]
    graph_issues = graph.get("issues", []) or [{"issue_type": "unknown_issue"}]
    issue = graph_issues[0].get("issue_type", "unknown_issue")
    for action in graph_actions:
        if action in action_to_shift:
            dx, dy = action_to_shift[action]
            moved = BBox(base.x1 + dx * bw, base.y1 + dy * bh, base.x2 + dx * bw, base.y2 + dy * bh)
            out.append((shift_box_into_image(moved, image_w, image_h), "direction_rule", action, issue, f"Apply directional action {action}"))
        elif action == "zoom_in":
            out.append((fit_box_to_aspect(subject_box, aspect, image_w, image_h, padding=0.18), "direction_rule", action, issue, "Zoom in around subject"))
        elif action == "zoom_out":
            out.append((fit_box_to_aspect(subject_box, aspect, image_w, image_h, padding=0.85), "direction_rule", action, issue, "Zoom out around subject"))
        elif action in {"preserve_relation", "keep_environment", "keep_full_body"}:
            # Generated by mask_protection_candidates; keep a light duplicate around subject.
            out.append((fit_box_to_aspect(subject_box, aspect, image_w, image_h, padding=0.55), "direction_rule", action, issue, f"Directional preservation action {action}"))
    return out


def negative_candidates(image_w: int, image_h: int, aspect: float, graph: Dict[str, Any]) -> List[Tuple[BBox, str, str, str, str]]:
    subject = graph.get("subject", {})
    if not subject.get("bbox"):
        return []
    box = BBox.from_seq(subject["bbox"])
    cx, cy = box.center()
    # Deliberately too tight candidates; scoring should push them down.
    widths = [box.w() * 0.65, box.w() * 0.85]
    out: List[Tuple[BBox, str, str, str, str]] = []
    for i, width in enumerate(widths):
        bad = make_aspect_box_from_center(cx, cy, width, aspect, image_w, image_h)
        out.append((bad, "negative_synthetic", "bad_crop", "subject_cut_or_too_tight", "Synthetic bad crop for ranker negatives"))
    return out

