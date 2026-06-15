from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


def route_caption(caption: str, caption_rule_root: str | None = None) -> Dict[str, Any]:
    if caption_rule_root:
        root = Path(caption_rule_root).resolve()
    else:
        root = Path(__file__).resolve().parents[1] / "caption-rule-co"

    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from semantic.semantic_router import route_caption_to_semantic_type

        return route_caption_to_semantic_type(caption or "")
    except Exception:
        text = (caption or "").lower()
        has_person = any(w in text for w in ["person", "man", "woman", "boy", "girl", "people"])
        has_action = any(w in text for w in ["jump", "run", "dance", "ride", "skate", "play"])
        has_animal = any(w in text for w in ["dog", "cat", "horse", "bird", "animal"])
        has_object = any(w in text for w in ["camera", "phone", "bag", "cup", "book", "guitar"])

        if has_person and has_animal:
            semantic_type = "person_animal"
            group = "C"
        elif has_person and has_object:
            semantic_type = "person_holding_object"
            group = "C"
        elif has_person and has_action:
            semantic_type = "single_action_portrait"
            group = "A"
        elif has_person:
            semantic_type = "single_static_portrait"
            group = "A"
        else:
            semantic_type = "landscape_scene"
            group = "E"

        return {
            "main_group": group,
            "semantic_type": semantic_type,
            "clean_caption": text,
            "reason": "fallback keyword router",
        }

