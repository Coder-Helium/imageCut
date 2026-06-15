from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .semantic import route_caption


DEFAULT_ACTIONS = ["preserve_relation", "place_subject_center"]


def _fallback_understanding(image_path: str, caption: str, semantic_type: str) -> Dict[str, Any]:
    text = caption or Path(image_path).stem
    has_person = semantic_type.startswith("single") or semantic_type.startswith("person") or "portrait" in semantic_type
    main_name = "person" if has_person else "main subject"
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
        return _fallback_understanding(image_path, caption, semantic_info.get("semantic_type", "unknown"))


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
                return rec.get("vlm_understanding", rec)
        return _fallback_understanding(image_path, caption, semantic_info.get("semantic_type", "unknown"))


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
        self.model = model or os.getenv("QWEN_VL_MODEL", "qwen-vl-plus")
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
        parsed.setdefault("caption", caption)
        parsed.setdefault("semantic_type", semantic_info.get("semantic_type", "unknown"))
        parsed["source"] = "qwen_dashscope"
        return parsed


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
        parsed.setdefault("caption", caption)
        parsed.setdefault("semantic_type", semantic_info.get("semantic_type", "unknown"))
        parsed["source"] = "openai_responses"
        parsed["_openai_response_id"] = data.get("id", "")
        return parsed


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
你是图像构图裁剪数据集构建助手。请只输出严格 JSON，不要输出解释文字。

已知 caption: {caption or ""}
初步 semantic_type: {semantic_info.get("semantic_type", "unknown")}

请分析图片，并输出以下字段：
- caption: 一句客观描述
- semantic_type: 从 single_static_portrait, single_action_portrait, double_relation, group_portrait,
  person_holding_object, person_animal, person_vehicle, person_nature, person_architecture,
  person_street, nonhuman_subject, food_subject, product_subject, landscape_scene,
  architecture_scene, detail_scene, unknown 中选一个
- main_subject: name/category/description/importance
- key_objects: 数组，每个包含 name/category/relation_to_subject/importance
- important_background: 数组，每个包含 name/category/importance
- distractors: 数组，每个包含 name/category/importance/location
- composition_intent:
  preserve, optional_preserve, avoid_cutting, leave_space_direction,
  preferred_subject_position, initial_issue, suggested_actions

suggested_actions 只能从以下集合中选择：
move_left, move_right, move_up, move_down, zoom_in, zoom_out,
place_subject_center, place_subject_left_third, place_subject_right_third,
preserve_relation, remove_distractor, keep_environment, keep_full_body,
keep_upper_body, fallback_full_image, no_crop_needed

如果能判断主体或关键物大概位置，可在对应对象中加入 bbox_norm: [x1,y1,x2,y2]，坐标归一化到 0 到 1。
""".strip()


def _build_openai_prompt(caption: str, semantic_info: Dict[str, Any]) -> str:
    return f"""
You are building a dataset for direction-aware image composition cropping.
Return only JSON matching the requested schema.

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
