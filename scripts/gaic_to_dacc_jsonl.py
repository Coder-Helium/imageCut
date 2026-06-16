#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaic_utils import (  # noqa: E402
    annotation_path_for,
    build_gaic_candidates,
    dump_jsonl,
    iter_split_images,
    load_gaic_annotations,
    read_image_size,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GAICD annotations into DACC-style JSONL.")
    parser.add_argument("--gaic-root", required=True, help="GAICD root with images/{train,test} and annotations.")
    parser.add_argument("--out-dir", default="data/gaic_dacc/metadata")
    parser.add_argument("--splits", default="train,test", help="Comma-separated GAICD splits.")
    parser.add_argument(
        "--coord-mode",
        default="auto",
        choices=["auto", "gaic_yxyx", "image_xyxy", "square1024_xyxy", "gaic", "yxyx", "image", "xyxy", "square1024"],
    )
    parser.add_argument("--annotation-size", type=int, default=1024)
    parser.add_argument("--no-clip-boxes", action="store_true", help="Keep converted boxes unclipped.")
    parser.add_argument("--max-records", type=int, default=0, help="Debug limit per split; 0 means all.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "gaic_root": str(Path(args.gaic_root).resolve()),
        "out_dir": str(out_dir.resolve()),
        "coord_mode": args.coord_mode,
        "annotation_size": args.annotation_size,
        "clip_boxes": not args.no_clip_boxes,
        "splits": {},
    }

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        records: List[Dict[str, Any]] = []
        images = iter_split_images(args.gaic_root, split)
        if args.max_records > 0:
            images = images[: args.max_records]
        resolved_modes: Dict[str, int] = {}
        skipped = 0
        total_candidates = 0

        for image_path in images:
            ann_path = annotation_path_for(args.gaic_root, image_path)
            if not ann_path.exists():
                skipped += 1
                continue
            image_w, image_h = read_image_size(image_path)
            annotations, resolved_mode = load_gaic_annotations(
                ann_path,
                image_w,
                image_h,
                coord_mode=args.coord_mode,
                annotation_size=args.annotation_size,
                clip=not args.no_clip_boxes,
            )
            resolved_modes[resolved_mode] = resolved_modes.get(resolved_mode, 0) + 1
            if not annotations:
                skipped += 1
                continue

            candidates = build_gaic_candidates(annotations, image_w, image_h)
            total_candidates += len(candidates)
            best = candidates[0]
            sample_id = image_path.stem
            rel_path = image_path.relative_to(Path(args.gaic_root)).as_posix()
            records.append(
                {
                    "sample_id": sample_id,
                    "image_path": str(image_path.resolve()),
                    "rel_path": rel_path,
                    "image_width": image_w,
                    "image_height": image_h,
                    "target_aspect_ratio": "free",
                    "caption": "",
                    "source_caption": "",
                    "semantic_type": "unknown",
                    "semantic_info": {},
                    "vlm_understanding": {},
                    "composition_middle_state": {},
                    "detections": [],
                    "crop_state_graph": {},
                    "masks": {},
                    "candidates": candidates,
                    "best_crop": best["box"],
                    "best_score": best["scores"]["mos"],
                    "best_action": "unknown",
                    "main_issue": "gaic_mos_supervision",
                    "gaic_supervision": {
                        "source": "GAICD",
                        "split": split,
                        "annotation_path": str(ann_path.resolve()),
                        "candidate_scores": "human_mos",
                        "best_crop_from": "highest_mos_candidate",
                        "num_candidates": len(candidates),
                        "coord_mode": resolved_mode,
                        "annotation_size": args.annotation_size,
                        "boxes_clipped_to_image": not args.no_clip_boxes,
                    },
                    "quality_flags": {
                        "has_valid_subject_mask": False,
                        "has_valid_key_object_mask": False,
                        "has_enough_candidates": len(candidates) >= 16,
                        "score_gap_top1_top2": _score_gap(candidates),
                        "needs_manual_review": False,
                        "has_vlm_middle_state": False,
                    },
                }
            )

        out_path = out_dir / f"{split}.jsonl"
        dump_jsonl(out_path, records)
        summary["splits"][split] = {
            "records": len(records),
            "skipped_images": skipped,
            "candidates": total_candidates,
            "resolved_coord_modes": resolved_modes,
            "jsonl": str(out_path.resolve()),
        }
        print(f"[{split}] records={len(records)} candidates={total_candidates} skipped={skipped}")

    write_json(out_dir / "summary.json", summary)
    print(f"Wrote summary: {out_dir / 'summary.json'}")


def _score_gap(candidates: List[Dict[str, Any]]) -> float:
    if len(candidates) < 2:
        return 0.0
    return float(candidates[0]["scores"]["mos"]) - float(candidates[1]["scores"]["mos"])


if __name__ == "__main__":
    main()
