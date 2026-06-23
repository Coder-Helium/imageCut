#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rigcrop.io import append_jsonl, iter_jsonl, write_json  # noqa: E402
from rigcrop.schema import audit_records, build_rig_targets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RIG-Crop graph targets from current DACC/Qwen JSONL.")
    parser.add_argument("--input-jsonl", required=True, help="DACC-style JSONL enriched by Qwen/VLM.")
    parser.add_argument("--out-jsonl", required=True, help="Output JSONL with added rig_targets.")
    parser.add_argument("--summary-json", default="", help="Optional summary path. Defaults to OUT.summary.json.")
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out_jsonl)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists. Use --overwrite.")
    if args.overwrite and out_path.exists():
        out_path.unlink()

    processed = 0
    audit_sample = []
    for rec in iter_jsonl(args.input_jsonl):
        processed += 1
        out = dict(rec)
        out["rig_targets"] = build_rig_targets(rec, max_nodes=args.max_nodes)
        append_jsonl(out_path, out)
        if len(audit_sample) < 500:
            audit_sample.append(out)
        if args.progress_interval > 0 and (processed == 1 or processed % args.progress_interval == 0):
            flags = out["rig_targets"]["graph_quality_flags"]
            print(
                f"[rig-targets] {processed} sample_id={out.get('sample_id')} "
                f"nodes={flags['valid_node_count']} boxed={flags['boxed_node_count']} relations={flags['relation_count']}",
                flush=True,
            )
        if args.max_records > 0 and processed >= args.max_records:
            break

    summary = {
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "out_jsonl": str(out_path.resolve()),
        "processed": processed,
        "max_nodes": args.max_nodes,
        "audit_first_records": audit_records(audit_sample),
    }
    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_suffix(out_path.suffix + ".summary.json")
    write_json(summary_path, summary)
    print(f"[rig-targets] done processed={processed} out={out_path} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
