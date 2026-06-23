#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rigcrop.io import load_jsonl, write_json  # noqa: E402
from rigcrop.schema import audit_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DACC/Qwen JSONL fields required by RIG-Crop.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    records = load_jsonl(args.jsonl, max_records=args.max_records)
    report = {
        "jsonl": str(Path(args.jsonl).resolve()),
        "max_records": args.max_records,
        "audit": audit_records(records),
    }
    if args.out_json:
        write_json(args.out_json, report)
    else:
        import json

        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
