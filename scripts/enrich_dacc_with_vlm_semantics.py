#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from composition_dataset_builder.semantic import route_caption  # noqa: E402
from enrich_gaic_with_vlm_semantics import (  # noqa: E402
    _append_jsonl,
    _first_action,
    _initial_issue,
    _iter_jsonl,
    _load_done_ids,
    _make_provider,
    _resolve_vis_dir,
    _safe_file_name,
    _unlink_if_exists,
    draw_enrichment_visualization,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add VLM semantic middle states to any DACC-style JSONL records.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--failed-jsonl", default="")
    parser.add_argument(
        "--vlm",
        default="heuristic",
        choices=[
            "heuristic",
            "precomputed",
            "qwen",
            "qwen_dashscope",
            "dashscope",
            "local_qwen",
            "qwen_local",
            "local_qwen_transformers",
            "openai",
            "openai_responses",
            "responses",
        ],
    )
    parser.add_argument("--vlm-precomputed", default="")
    parser.add_argument("--qwen-model", default="")
    parser.add_argument("--qwen-base-url", default="")
    parser.add_argument("--local-qwen-model", default="")
    parser.add_argument("--local-qwen-device-map", default="auto")
    parser.add_argument("--local-qwen-dtype", default="float16", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--local-qwen-attn", default="sdpa")
    parser.add_argument("--local-qwen-max-new-tokens", type=int, default=768)
    parser.add_argument("--local-qwen-min-pixels", type=int, default=262144)
    parser.add_argument("--local-qwen-max-pixels", type=int, default=1048576)
    parser.add_argument("--openai-model", default="")
    parser.add_argument("--openai-base-url", default="")
    parser.add_argument("--openai-image-detail", default="auto")
    parser.add_argument("--caption-rule-root", default="")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--propagate-sample-action", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis-dir", default="")
    parser.add_argument("--vis-topk", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    out_path = Path(args.out_jsonl)
    failed_path = Path(args.failed_jsonl) if args.failed_jsonl else out_path.with_suffix(out_path.suffix + ".failed.jsonl")
    vis_dir = _resolve_vis_dir(out_path, args.vis_dir) if args.visualize else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        _unlink_if_exists(out_path)
        _unlink_if_exists(failed_path)
    elif out_path.exists() and not args.resume:
        raise FileExistsError(f"{out_path} exists. Use --resume or --overwrite.")

    done_ids = _load_done_ids(out_path) if args.resume else set()
    provider = _make_provider(args)
    records = list(_iter_jsonl(input_path))
    if args.max_records > 0:
        records = records[: args.max_records]

    processed = skipped = failed = 0
    for idx, rec in enumerate(records, start=1):
        sample_id = str(rec.get("sample_id") or Path(rec.get("image_path", "")).stem or idx)
        if sample_id in done_ids:
            skipped += 1
            continue
        try:
            enriched = enrich_record(rec, provider, args.caption_rule_root, args.propagate_sample_action)
            if vis_dir is not None:
                vis_path = vis_dir / f"{_safe_file_name(sample_id)}.jpg"
                draw_enrichment_visualization(enriched, vis_path, topk=args.vis_topk)
                enriched["visualization_path"] = str(vis_path.resolve())
            _append_jsonl(out_path, enriched)
            processed += 1
            print(f"[{idx}/{len(records)}] {sample_id} ok: {enriched.get('caption', '')}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            _append_jsonl(failed_path, {"sample_id": sample_id, "image_path": rec.get("image_path", ""), "error": repr(exc)})
            print(f"[{idx}/{len(records)}] {sample_id} ERROR: {repr(exc)}")
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    summary = {
        "input_jsonl": str(input_path.resolve()),
        "out_jsonl": str(out_path.resolve()),
        "failed_jsonl": str(failed_path.resolve()),
        "vlm": args.vlm,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "visualize": args.visualize,
        "vis_dir": str(vis_dir.resolve()) if vis_dir is not None else "",
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def enrich_record(rec: Dict[str, Any], provider: Any, caption_rule_root: str = "", propagate_sample_action: bool = False) -> Dict[str, Any]:
    image_path = str(rec["image_path"])
    seed_caption = str(rec.get("caption") or rec.get("source_caption") or "").strip()
    seed_info = rec.get("semantic_info") if isinstance(rec.get("semantic_info"), dict) and rec.get("semantic_info") else {}
    if not seed_info:
        seed_info = route_caption(seed_caption, caption_rule_root or None)

    understanding = provider.understand(image_path, seed_caption, seed_info)
    final_caption = str(understanding.get("caption") or seed_caption or "").strip()
    routed_info = route_caption(final_caption, caption_rule_root or None)
    semantic_type = str(understanding.get("semantic_type") or routed_info.get("semantic_type") or "unknown")
    routed_info["semantic_type"] = semantic_type

    action = _first_action(understanding)
    issue = _initial_issue(understanding)
    out = dict(rec)
    out["source_caption"] = rec.get("source_caption", seed_caption)
    out["caption"] = final_caption
    out["semantic_type"] = semantic_type
    out["semantic_info"] = routed_info
    out["vlm_understanding"] = understanding
    out["composition_middle_state"] = build_middle_state(understanding, rec, final_caption, semantic_type)
    out["dataset_supervision"] = build_dataset_supervision(rec)
    out["best_action"] = action or rec.get("best_action", "unknown")
    out["main_issue"] = issue or rec.get("main_issue", "unknown")
    out["quality_flags"] = dict(rec.get("quality_flags", {}) or {})
    out["quality_flags"]["has_vlm_middle_state"] = True
    out["quality_flags"]["semantic_teacher_source"] = understanding.get("source", "unknown")
    if propagate_sample_action:
        out["candidates"] = _propagate_action(out.get("candidates", []), out["best_action"], out["main_issue"])
    return out


def build_middle_state(understanding: Dict[str, Any], rec: Dict[str, Any], caption: str, semantic_type: str) -> Dict[str, Any]:
    intent = understanding.get("composition_intent", {}) if isinstance(understanding.get("composition_intent"), dict) else {}
    return {
        "source": understanding.get("source", "unknown"),
        "teacher_role": "semantic_middle_state_only",
        "crop_supervision_source": _crop_supervision_source(rec),
        "caption": caption,
        "semantic_type": semantic_type,
        "main_subject": understanding.get("main_subject", {}),
        "key_objects": understanding.get("key_objects", []),
        "important_background": understanding.get("important_background", []),
        "distractors": understanding.get("distractors", []),
        "composition_intent": intent,
        "suggested_action": _first_action(understanding),
        "initial_issue": _initial_issue(understanding),
        "best_crop": rec.get("best_crop", []),
        "best_score": rec.get("best_score", None),
        "num_candidates": len(rec.get("candidates", []) or []),
        "num_pairwise_preferences": len(rec.get("pairwise_preferences", []) or []),
    }


def build_dataset_supervision(rec: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(rec.get("cpc_supervision"), dict):
        return dict(rec["cpc_supervision"])
    if isinstance(rec.get("gaic_supervision"), dict):
        return dict(rec["gaic_supervision"])
    return {
        "source": rec.get("dataset", "unknown"),
        "num_candidates": len(rec.get("candidates", []) or []),
        "num_pairwise_preferences": len(rec.get("pairwise_preferences", []) or []),
    }


def _crop_supervision_source(rec: Dict[str, Any]) -> str:
    if rec.get("cpc_supervision"):
        return "cpc_pairwise_preference"
    if rec.get("gaic_supervision"):
        return "gaicd_human_mos"
    if rec.get("pairwise_preferences"):
        return "pairwise_preference"
    return "unknown"


def _propagate_action(candidates: Any, action: str, issue: str) -> Any:
    if not isinstance(candidates, list):
        return candidates
    out = []
    for cand in candidates:
        if not isinstance(cand, dict):
            out.append(cand)
            continue
        item = dict(cand)
        if item.get("action", "unknown") == "unknown":
            item["action"] = action
        if item.get("issue", "unknown") in {"unknown", "cpc_pairwise_supervision", "gaic_mos_supervision"}:
            item["issue"] = issue
        out.append(item)
    return out


if __name__ == "__main__":
    main()
