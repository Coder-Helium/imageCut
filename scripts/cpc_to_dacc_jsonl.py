#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpc_utils import (  # noqa: E402
    CpcImageRecord,
    load_cpc_records,
    load_pairwise_file,
    split_records,
    write_cpc_splits,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CPC/Good View Hunting annotations into DACC-style JSONL.")
    parser.add_argument("--cpc-root", required=True, help="CPCDataset root after extracting CPCDataset.tar.gz.")
    parser.add_argument("--out-dir", default="data/cpc_dacc/metadata")
    parser.add_argument("--annotation-file", default="", help="Path to image_crop.json or CollectedAnnotationsRaw directory; auto-detected if omitted.")
    parser.add_argument("--image-dir", default="", help="Image directory; defaults to scanning --cpc-root recursively.")
    parser.add_argument("--pairwise-file", default="", help="Optional raw pairwise file. If omitted, pairs are derived from CPC view scores.")
    parser.add_argument(
        "--coord-mode",
        default="auto",
        choices=["auto", "image_xyxy", "xyxy", "image_xywh", "xywh", "normalized_xyxy", "norm_xyxy", "normalized_xywh", "norm_xywh"],
    )
    parser.add_argument("--no-clip-boxes", action="store_true")
    parser.add_argument("--min-pair-score-gap", type=float, default=0.02, help="Minimum normalized score gap for score-derived pairs.")
    parser.add_argument("--max-pairs-per-image", type=int, default=0, help="0 keeps all score-derived pairs.")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=0, help="Debug/load limit; 0 means all.")
    args = parser.parse_args()

    records, load_summary = load_cpc_records(
        args.cpc_root,
        annotation_file=args.annotation_file,
        image_dir=args.image_dir,
        coord_mode=args.coord_mode,
        min_pair_score_gap=args.min_pair_score_gap,
        max_pairs_per_image=args.max_pairs_per_image,
        seed=args.seed,
        clip_boxes=not args.no_clip_boxes,
        max_records=args.max_records,
    )

    pairwise_summary: Dict[str, Any] = {"source": "score_derived"}
    if args.pairwise_file:
        records, pairwise_summary = _override_pairwise_from_file(records, args.pairwise_file)

    splits = split_records(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_records=0,
    )
    split_summary = write_cpc_splits(args.out_dir, splits, args.cpc_root)

    summary = {
        "cpc_root": str(Path(args.cpc_root).resolve()),
        "out_dir": str(Path(args.out_dir).resolve()),
        "annotation_file": args.annotation_file,
        "image_dir": args.image_dir,
        "pairwise_file": args.pairwise_file,
        "coord_mode": args.coord_mode,
        "clip_boxes": not args.no_clip_boxes,
        "min_pair_score_gap": args.min_pair_score_gap,
        "max_pairs_per_image": args.max_pairs_per_image,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "seed": args.seed,
        "max_records": args.max_records,
        "load_summary": load_summary,
        "pairwise_summary": pairwise_summary,
        "splits": split_summary,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _override_pairwise_from_file(records: List[CpcImageRecord], pairwise_file: str) -> tuple[List[CpcImageRecord], Dict[str, Any]]:
    sample_to_candidates: Dict[str, List[str]] = {}
    alias_to_sample: Dict[str, str] = {}
    for record in records:
        ids = [view.view_id for view in record.views]
        sample_to_candidates[record.sample_id] = ids
        sample_to_candidates[record.image_name] = ids
        alias_to_sample[record.sample_id] = record.sample_id
        alias_to_sample[record.image_name] = record.sample_id
        alias_to_sample[Path(record.image_name).stem] = record.sample_id

    raw_map = load_pairwise_file(pairwise_file, sample_to_candidates)
    pref_by_sample: Dict[str, Any] = {}
    for key, prefs in raw_map.items():
        sample_id = alias_to_sample.get(key, alias_to_sample.get(Path(key).stem, key))
        pref_by_sample.setdefault(sample_id, []).extend(prefs)

    out: List[CpcImageRecord] = []
    replaced = 0
    total_prefs = 0
    for record in records:
        prefs = pref_by_sample.get(record.sample_id)
        if prefs:
            out.append(replace(record, preferences=prefs))
            replaced += 1
            total_prefs += len(prefs)
        else:
            out.append(record)
            total_prefs += len(record.preferences)
    return out, {
        "source": "pairwise_file",
        "pairwise_file": str(Path(pairwise_file).resolve()),
        "records_replaced": replaced,
        "total_pairwise_preferences": total_prefs,
    }


if __name__ == "__main__":
    main()
