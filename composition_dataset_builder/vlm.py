from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .semantic import route_caption


DEFAULT_ACTIONS = ["preserve_relation", "place_subject_center"]


def _fallback_understanding(image_path: str, caption: str, semantic_type: str) -> Dict[str, Any]:
    has_person = semantic_type.startswith("single") or semantic_type.startswith("person") or "portrait" in semantic_type
    main_name = "person" if has_person else "main subject"
    text = _short_english_caption(caption, main_name)
    preserve = [main_name]
    key_objects: List[Dict[str, Any]] = []

    for word in ["camera", "phone", "dog", "cat", "car", "bicycle", "bag", "book", "cup", "guitar"]:
        if word in text.lower():
            key_objects.append(
                {
                    "name": word,
                    "category": "object" if word not in {"dog", "cat"} else "animal",
                    "relation_to_subject": "related_to_subject",
                    "importance": 0.75,
                }
            )
            preserve.append(word)

    return {
        "caption": text,
        "semantic_type": semantic_type,
        "main_subject": {
            "name": main_name,
            "category": "person" if has_person else "subject",
            "description": text,
            "importance": 1.0,
        },
        "key_objects": key_objects,
        "important_background": [],
        "distractors": [],
        "composition_intent": {
            "preserve": preserve,
            "optional_preserve": [],
            "avoid_cutting": ["head", "hands", "feet"] if has_person else [],
            "leave_space_direction": "unknown",
            "preferred_subject_position": "center",
            "initial_issue": "unknown_issue",
            "suggested_actions": DEFAULT_ACTIONS,
        },
        "source": "heuristic",
    }


class VLMProvider:
    def understand(self, image_path: str, caption: str, semantic_info: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class HeuristicVLMProvider(VLMProvider):
    def understand(self, image_path: str, caption: str, semantic_info: Dict[str, Any]) -> Dict[str, Any]:
        parsed = _fallback_understanding(image_path, caption, semantic_info.get("semantic_type", "unknown"))
        return _finalize_understanding(parsed, caption, semantic_info, "heuristic")


class PrecomputedVLMProvider(VLMProvider):
    def __init__(self, records_path: str):
        self.records: Dict[str, Dict[str, Any]] = {}
        path = Path(records_path)
        if path.suffix.lower() == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    key = rec.get("image_path") or rec.get("rel_path") or rec.get("id")
                    if key:
                        self.records[str(key)] = rec
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.records = {str(k): v for k, v in data.items()}
            else:
                for rec in data:
                    key = rec.get("image_path") or rec.get("rel_path") or rec.get("id")
                    if key:
                        self.records[str(key)] = rec

    def understand(self, image_path: str, caption: str, semantic_info: Dict[str, Any]) -> Dict[str, Any]:
        keys = [image_path, Path(image_path).name]
        for key in keys:
            if key in self.records:
                rec = self.records[key]
                return _finalize_understanding(rec.get("vlm_understanding", rec), caption, semantic_info, "precomputed")
        parsed = _fallback_understanding(image_path, caption, semantic_info.get("semantic_type", "unknown"))
        return _finalize_understanding(parsed, caption, semantic_info, "heuristic")


class QwenDashScopeVLMProvider(VLMProvider):
    """OpenAI-compatible DashScope client for Qwen-VL style teacher output."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 90,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        self.model = model or os.getenv("QWEN_VL_MODEL", "qwen3-vl-flash")
        self.base_url = (base_url or os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("Qwen provider requires DASHSCOPE_API_KEY or QWEN_API_KEY")

    def understand(self, image_path: str, caption: str, semantic_info: Dict[str, Any]) -> Dict[str, Any]:
        image_url = _image_to_data_url(image_path)
        prompt = _build_qwen_prompt(caption, semantic_info)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        parsed = json.loads(content)
        return _finalize_understanding(parsed, caption, semantic_info, "qwen_dashscope")


class OpenAIResponsesVLMProvider(VLMProvider):
    """OpenAI /v1/responses provider for structured image understanding."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 90,
        image_detail: str = "auto",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_VLM_MODEL", "gpt-4.1-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        self.image_detail = image_detail
        if not self.api_key:
            raise RuntimeError("OpenAI provider requires OPENAI_API_KEY")

    def understand(self, image_path: str, caption: str, semantic_info: Dict[str, Any]) -> Dict[str, Any]:
        image_url = _image_to_data_url(image_path)
        prompt = _build_openai_prompt(caption, semantic_info)
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url, "detail": self.image_detail},
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "composition_understanding",
                    "schema": _composition_understanding_schema(),
                    "strict": False,
                }
            },
            "temperature": 0.1,
        }
        resp = requests.post(
            f"{self.base_url}/responses",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = _extract_response_text(data)
        parsed = json.loads(text)
        return _finalize_understanding(
            parsed,
            caption,
            semantic_info,
            "openai_responses",
            response_id_key="_openai_response_id",
            response_id=data.get("id", ""),
        )


def create_vlm_provider(kind: str, precomputed_path: Optional[str] = None, **kwargs: Any) -> VLMProvider:
    kind = (kind or "heuristic").lower()
    if kind in {"none", "heuristic", "fallback"}:
        return HeuristicVLMProvider()
    if kind == "precomputed":
        if not precomputed_path:
            raise ValueError("--vlm-precomputed is required for precomputed VLM provider")
        return PrecomputedVLMProvider(precomputed_path)
    if kind in {"qwen", "qwen_dashscope", "dashscope"}:
        return QwenDashScopeVLMProvider(**kwargs)
    if kind in {"openai", "openai_responses", "responses"}:
        return OpenAIResponsesVLMProvider(**kwargs)
    raise ValueError(f"Unknown VLM provider: {kind}")


def _image_to_data_url(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_qwen_prompt(caption: str, semantic_info: Dict[str, Any]) -> str:
    return f"""
You are building a dataset for direction-aware image composition cropping.
Return valid JSON only. Do not output markdown or explanations.

All natural-language values must be English only. Do not output Chinese text anywhere.
The caption must be concise, focus on the key subject/action/context, and use at most 10 English words.

Known caption: {caption or ""}
Initial semantic_type: {semantic_info.get("semantic_type", "unknown")}

Analyze the image for cropping and output these fields:
- caption: one short English caption, at most 10 words.
- semantic_type: choose one from single_static_portrait, single_action_portrait, double_relation,
  group_portrait, person_holding_object, person_animal, person_vehicle, person_nature,
  person_architecture, person_street, nonhuman_subject, food_subject, product_subject,
  landscape_scene, architecture_scene, detail_scene, unknown.
- main_subject: name/category/description/importance.
- key_objects: array of name/category/relation_to_subject/importance.
- important_background: array of name/category/importance.
- distractors: array of name/category/importance/location.
- composition_intent:
  preserve, optional_preserve, avoid_cutting, leave_space_direction,
  preferred_subject_position, initial_issue, suggested_actions.

All importance and confidence fields must be numbers from 0 to 1, such as 1.0, 0.85, 0.55, 0.25.
Do not output text grades such as "high", "medium", or "low".

For key_objects, include only objects that materially affect cropping, such as people, animals,
held objects, vehicles, or strong relation objects. Do not list tiny accessories, reflections,
clothing details, or guessed objects unless they are central to the composition.

suggested_actions must come from:
move_left, move_right, move_up, move_down, zoom_in, zoom_out,
place_subject_center, place_subject_left_third, place_subject_right_third,
preserve_relation, remove_distractor, keep_environment, keep_full_body,
keep_upper_body, fallback_full_image, no_crop_needed

If the subject or key objects can be localized, add bbox_norm: [x1, y1, x2, y2],
normalized to [0, 1]. Use decimal numbers only, for example [0.18, 0.22, 0.74, 0.91].
Do not use pixel coordinates such as [394, 555, 602, 998].
""".strip()


def _build_openai_prompt(caption: str, semantic_info: Dict[str, Any]) -> str:
    return f"""
You are building a dataset for direction-aware image composition cropping.
Return only JSON matching the requested schema.

All natural-language values must be English only. Do not output Chinese text anywhere.
The caption must be concise, focus on the key subject/action/context, and use at most 10 English words.

Known caption: {caption or ""}
Initial semantic_type: {semantic_info.get("semantic_type", "unknown")}

Analyze the image for cropping. Identify:
1. main subject,
2. key objects that should be preserved,
3. important background regions,
4. distractors or removable regions,
5. composition issue,
6. corrective crop actions.

Use normalized bbox coordinates [x1, y1, x2, y2] in [0, 1] when location is visually clear.
Use decimal numbers only, for example [0.18, 0.22, 0.74, 0.91], not pixel coordinates.
Suggested actions must come from:
move_left, move_right, move_up, move_down, zoom_in, zoom_out,
place_subject_center, place_subject_left_third, place_subject_right_third,
preserve_relation, remove_distractor, keep_environment, keep_full_body,
keep_upper_body, fallback_full_image, no_crop_needed.

semantic_type must be one of:
single_static_portrait, single_action_portrait, double_relation, group_portrait,
person_holding_object, person_animal, person_vehicle, person_nature,
person_architecture, person_street, nonhuman_subject, food_subject,
product_subject, landscape_scene, architecture_scene, detail_scene, unknown.

All importance and confidence fields must be numbers from 0 to 1, such as 1.0, 0.85, 0.55, 0.25.
""".strip()


def _composition_understanding_schema() -> Dict[str, Any]:
    entity_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "category": {"type": "string"},
            "description": {"type": "string"},
            "importance": {"type": "number"},
            "relation_to_subject": {"type": "string"},
            "location": {"type": "string"},
            "bbox_norm": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
        },
        "additionalProperties": True,
    }
    return {
        "type": "object",
        "properties": {
            "caption": {"type": "string"},
            "semantic_type": {"type": "string"},
            "main_subject": entity_schema,
            "key_objects": {"type": "array", "items": entity_schema},
            "important_background": {"type": "array", "items": entity_schema},
            "distractors": {"type": "array", "items": entity_schema},
            "composition_intent": {
                "type": "object",
                "properties": {
                    "preserve": {"type": "array", "items": {"type": "string"}},
                    "optional_preserve": {"type": "array", "items": {"type": "string"}},
                    "avoid_cutting": {"type": "array", "items": {"type": "string"}},
                    "leave_space_direction": {"type": "string"},
                    "preferred_subject_position": {"type": "string"},
                    "initial_issue": {"type": "string"},
                    "suggested_actions": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "preserve",
                    "optional_preserve",
                    "avoid_cutting",
                    "leave_space_direction",
                    "preferred_subject_position",
                    "initial_issue",
                    "suggested_actions",
                ],
                "additionalProperties": True,
            },
        },
        "required": [
            "caption",
            "semantic_type",
            "main_subject",
            "key_objects",
            "important_background",
            "distractors",
            "composition_intent",
        ],
        "additionalProperties": True,
    }


def _extract_response_text(data: Dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks: List[str] = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict):
                if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                    chunks.append(content["text"])
                elif isinstance(content.get("refusal"), str):
                    raise RuntimeError(f"OpenAI response refusal: {content['refusal']}")
    if not chunks:
        raise RuntimeError(f"Could not extract output_text from OpenAI response: {data.keys()}")
    return "".join(chunks).strip()


def _finalize_understanding(
    parsed: Dict[str, Any],
    caption: str,
    semantic_info: Dict[str, Any],
    source: str,
    response_id_key: str = "",
    response_id: str = "",
) -> Dict[str, Any]:
    parsed = dict(parsed or {})
    parsed.setdefault("semantic_type", semantic_info.get("semantic_type", "unknown"))
    parsed["caption"] = _short_english_caption(parsed.get("caption") or caption, _subject_caption(parsed))
    _normalize_numeric_fields(parsed)
    parsed["source"] = source
    if response_id_key:
        parsed[response_id_key] = response_id
    return parsed


def _short_english_caption(value: Any, fallback: str = "main subject") -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text or _contains_cjk(text):
        text = fallback
    text = re.sub(r"[^A-Za-z0-9&'.,:;()/-]+", " ", text).strip()
    words = text.split()
    if not words:
        return fallback
    return " ".join(words[:10]).strip(" ,.;:")


def _subject_caption(parsed: Dict[str, Any]) -> str:
    main = parsed.get("main_subject") if isinstance(parsed, dict) else None
    if isinstance(main, dict):
        for key in ["description", "name", "category"]:
            value = str(main.get(key, "")).strip()
            if value and not _contains_cjk(value):
                return value
    semantic_type = str(parsed.get("semantic_type", "")).lower() if isinstance(parsed, dict) else ""
    if "portrait" in semantic_type or semantic_type.startswith("person"):
        return "person portrait"
    if "landscape" in semantic_type:
        return "landscape scene"
    if "architecture" in semantic_type:
        return "architecture scene"
    if "food" in semantic_type:
        return "food subject"
    if "product" in semantic_type:
        return "product subject"
    return "main subject"


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _normalize_numeric_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key in {"importance", "confidence"}:
                value[key] = _importance_to_float(item)
            else:
                _normalize_numeric_fields(item)
    elif isinstance(value, list):
        for item in value:
            _normalize_numeric_fields(item)


def _importance_to_float(value: Any, default: float = 0.65) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    text = str(value).strip().lower()
    if text in {"very high", "critical", "essential", "extreme", "top"}:
        return 1.0
    if text in {"high", "important", "major", "strong"}:
        return 0.85
    if text in {"medium", "moderate", "normal", "average"}:
        return 0.55
    if text in {"low", "minor", "weak"}:
        return 0.25
    if text in {"none", "irrelevant", "ignore"}:
        return 0.0
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        return default
