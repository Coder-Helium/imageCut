#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rigcrop.io import append_jsonl, iter_jsonl, write_json  # noqa: E402
from rigcrop.schema import audit_records, build_rig_targets, compact_rig_record  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile raw VLM middle-state JSONL into compact RIG training JSONL.")
    parser.add_argument("--input-jsonl", required=True, help="Raw DACC/RIG JSONL with VLM middle-state fields.")
    parser.add_argument("--out-jsonl", required=True, help="Compact training JSONL output.")
    parser.add_argument("--summary-json", default="", help="Optional summary path. Defaults to OUT.summary.json.")
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--rebuild-rig-targets", action="store_true", help="Rebuild rig_targets even if they already exist.")
    parser.add_argument("--keep-raw-middle-state", action="store_true", help="Keep raw composition_middle_state/vlm_understanding in output.")
    parser.add_argument("--keep-node-text", action="store_true", help="Keep node name/category/description text inside rig_targets.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out_jsonl)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists. Use --overwrite.")
    if args.overwrite and out_path.exists():
        out_path.unlink()

    processed = 0
    input_bytes = 0
    output_bytes = 0
    audit_sample = []
    examples = []
    for rec in iter_jsonl(args.input_jsonl):
        processed += 1
        input_bytes += _json_bytes(rec)
        work = dict(rec)
        if args.rebuild_rig_targets or not isinstance(work.get("rig_targets"), dict):
            work["rig_targets"] = build_rig_targets(work, max_nodes=args.max_nodes)
        compact = compact_rig_record(
            work,
            max_nodes=args.max_nodes,
            build_if_missing=False,
            keep_raw_middle_state=args.keep_raw_middle_state,
            keep_node_text=args.keep_node_text,
        )
        output_bytes += _json_bytes(compact)
        if len(audit_sample) < 500:
            audit_sample.append(work)
        append_jsonl(out_path, compact)
        if len(examples) < 5:
            examples.append(
                {
                    "sample_id": compact.get("sample_id", ""),
                    "input_bytes": _json_bytes(rec),
                    "output_bytes": _json_bytes(compact),
                    "has_raw_middle_state": "composition_middle_state" in compact or "vlm_understanding" in compact,
                    "node_fields": sorted((compact.get("rig_targets", {}).get("nodes", [{}]) or [{}])[0].keys()),
                }
            )
        if args.progress_interval > 0 and (processed == 1 or processed % args.progress_interval == 0):
            print(
                f"[middle-compile] {processed} input_mb={input_bytes / 1024**2:.2f} "
                f"output_mb={output_bytes / 1024**2:.2f}",
                flush=True,
            )
        if args.max_records > 0 and processed >= args.max_records:
            break

    summary: Dict[str, Any] = {
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "out_jsonl": str(out_path.resolve()),
        "processed": processed,
        "max_nodes": int(args.max_nodes),
        "keep_raw_middle_state": bool(args.keep_raw_middle_state),
        "keep_node_text": bool(args.keep_node_text),
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "size_reduction_ratio": 0.0 if input_bytes <= 0 else 1.0 - (output_bytes / input_bytes),
        "examples": examples,
        "audit_first_records": audit_records(audit_sample),
    }
    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_suffix(out_path.suffix + ".summary.json")
    write_json(summary_path, summary)
    print(f"[middle-compile] done processed={processed} out={out_path} summary={summary_path}", flush=True)


def _json_bytes(value: Dict[str, Any]) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


if __name__ == "__main__":
    main()
