from __future__ import annotations

ACTION_VOCAB = [
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
    "semantic_rule",
    "bad_crop",
    "unknown",
]

ISSUE_VOCAB = [
    "subject_too_left",
    "subject_too_right",
    "subject_too_high",
    "subject_too_low",
    "subject_too_small",
    "subject_too_large",
    "subject_top_too_tight",
    "subject_bottom_too_tight",
    "subject_left_too_tight",
    "subject_right_too_tight",
    "key_object_cut",
    "relation_object_missing",
    "background_too_distracting",
    "important_environment_cut",
    "empty_space_imbalance",
    "bad_aspect_ratio",
    "subject_object_relation_should_be_preserved",
    "large_relation_region",
    "already_good_composition",
    "gaic_grid_anchor",
    "preserve_box",
    "relation_box",
    "important_region_box",
    "subject_cut_or_too_tight",
    "fallback_full_image",
    "unknown_issue",
    "unknown",
]


def index_or_unknown(vocab: list[str], value: str, unknown: str = "unknown") -> int:
    value = str(value or unknown)
    if value in vocab:
        return vocab.index(value)
    if unknown in vocab:
        return vocab.index(unknown)
    return len(vocab) - 1


def aspect_to_float(value: str, image_w: int, image_h: int) -> float:
    value = str(value or "original").lower()
    if value in {"original", "orig", "image"}:
        return image_w / max(float(image_h), 1.0)
    if ":" in value:
        a, b = value.split(":", 1)
        return float(a) / max(float(b), 1e-6)
    if "/" in value:
        a, b = value.split("/", 1)
        return float(a) / max(float(b), 1e-6)
    return float(value)

