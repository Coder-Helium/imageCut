#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from composition_dataset_builder.semantic import route_caption  # noqa: E402
from composition_dataset_builder.vlm import create_vlm_provider  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Add VLM semantic middle states to GAICD/DACC JSONL records.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--failed-jsonl", default="")
    parser.add_argument("--vlm", default="heuristic", choices=["heuristic", "precomputed", "qwen", "qwen_dashscope", "dashscope", "openai", "openai_responses", "responses"])
    parser.add_argument("--vlm-precomputed", default="")
    parser.add_argument("--qwen-model", default="")
    parser.add_argument("--qwen-base-url", default="")
    parser.add_argument("--openai-model", default="")
    parser.add_argument("--openai-base-url", default="")
    parser.add_argument("--openai-image-detail", default="auto")
    parser.add_argument("--caption-rule-root", default="")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--propagate-sample-action", action="store_true", help="Fill unknown candidate action/issue from sample-level VLM intent.")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    out_path = Path(args.out_jsonl)
    failed_path = Path(args.failed_jsonl) if args.failed_jsonl else out_path.with_suffix(out_path.suffix + ".failed.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

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

    processed = 0
    skipped = 0
    failed = 0
    for idx, rec in enumerate(records, start=1):
        sample_id = str(rec.get("sample_id") or Path(rec.get("image_path", "")).stem or idx)
        if sample_id in done_ids:
            skipped += 1
            continue
        try:
            enriched = enrich_record(rec, provider, args.caption_rule_root, args.propagate_sample_action)
            _append_jsonl(out_path, enriched)
            processed += 1
            print(f"[{idx}/{len(records)}] {sample_id} ok: {enriched.get('caption', '')}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            err = {"sample_id": sample_id, "image_path": rec.get("image_path", ""), "error": repr(exc)}
            _append_jsonl(failed_path, err)
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
    out["best_action"] = action or rec.get("best_action", "unknown")
    out["main_issue"] = issue or rec.get("main_issue", "unknown")
    out["gaic_supervision"] = _gaic_supervision(rec)
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
        "crop_supervision_source": "gaicd_human_mos",
        "caption": caption,
        "semantic_type": semantic_type,
        "main_subject": understanding.get("main_subject", {}),
        "key_objects": understanding.get("key_objects", []),
        "important_background": understanding.get("important_background", []),
        "distractors": understanding.get("distractors", []),
        "composition_intent": intent,
        "suggested_action": _first_action(understanding),
        "initial_issue": _initial_issue(understanding),
        "gaic_best_crop": rec.get("best_crop", []),
        "gaic_best_score": rec.get("best_score", None),
        "gaic_num_candidates": len(rec.get("candidates", []) or []),
    }


def _make_provider(args: argparse.Namespace) -> Any:
    kwargs: Dict[str, Any] = {}
    if args.vlm in {"precomputed"}:
        return create_vlm_provider(args.vlm, precomputed_path=args.vlm_precomputed)
    if args.vlm in {"qwen", "qwen_dashscope", "dashscope"}:
        if args.qwen_model:
            kwargs["model"] = args.qwen_model
        if args.qwen_base_url:
            kwargs["base_url"] = args.qwen_base_url
    elif args.vlm in {"openai", "openai_responses", "responses"}:
        if args.openai_model:
            kwargs["model"] = args.openai_model
        if args.openai_base_url:
            kwargs["base_url"] = args.openai_base_url
        kwargs["image_detail"] = args.openai_image_detail
    return create_vlm_provider(args.vlm, **kwargs)


def _first_action(understanding: Dict[str, Any]) -> str:
    intent = understanding.get("composition_intent", {}) if isinstance(understanding.get("composition_intent"), dict) else {}
    actions = intent.get("suggested_actions", [])
    if isinstance(actions, list) and actions:
        return str(actions[0])
    if isinstance(actions, str):
        return actions
    return "unknown"


def _initial_issue(understanding: Dict[str, Any]) -> str:
    intent = understanding.get("composition_intent", {}) if isinstance(understanding.get("composition_intent"), dict) else {}
    issue = intent.get("initial_issue", "")
    return str(issue or "unknown")


def _gaic_supervision(rec: Dict[str, Any]) -> Dict[str, Any]:
    sup = dict(rec.get("gaic_supervision", {}) or {})
    sup.setdefault("source", "GAICD")
    sup.setdefault("candidate_scores", "human_mos")
    sup.setdefault("best_crop_from", "highest_mos_candidate")
    sup.setdefault("num_candidates", len(rec.get("candidates", []) or []))
    return sup


def _propagate_action(candidates: List[Dict[str, Any]], action: str, issue: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cand in candidates:
        item = dict(cand)
        if item.get("action", "unknown") == "unknown":
            item["action"] = action
        if item.get("issue", "unknown") in {"unknown", "gaic_mos_supervision"}:
            item["issue"] = issue
        out.append(item)
    return out


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_done_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done: Set[str] = set()
    for rec in _iter_jsonl(path):
        sample_id = rec.get("sample_id")
        if sample_id:
            done.add(str(sample_id))
    return done


def _unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


if __name__ == "__main__":
    main()
