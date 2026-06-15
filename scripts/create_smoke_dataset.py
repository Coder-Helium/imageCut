#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/dacc_smoke_data")
    parser.add_argument("--num-images", type=int, default=8)
    args = parser.parse_args()

    out = Path(args.out_dir)
    image_dir = out / "images"
    meta_dir = out / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    jsonl = meta_dir / "all.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for i in range(args.num_images):
            w, h = 960, 720
            img = np.full((h, w, 3), 245, dtype=np.uint8)
            shift = (i % 5 - 2) * 45
            subj = [360 + shift, 90, 580 + shift, 640]
            obj = [555 + shift, 310, 690 + shift, 430]
            cv2.rectangle(img, (subj[0], subj[1]), (subj[2], subj[3]), (70, 130, 220), -1)
            cv2.circle(img, ((subj[0] + subj[2]) // 2, subj[1] - 25), 58, (90, 155, 235), -1)
            cv2.rectangle(img, (obj[0], obj[1]), (obj[2], obj[3]), (35, 35, 35), -1)
            cv2.rectangle(img, (25, 80), (155, 670), (190, 190, 190), -1)
            image_path = image_dir / f"smoke_{i:03d}.jpg"
            cv2.imwrite(str(image_path), img)

            # Candidates are intentionally simple but schema-compatible.
            candidates = [
                make_candidate([240 + shift, 0, 816 + shift, 720], 4.7, 1, "direction_rule", "preserve_relation", "subject_object_relation_should_be_preserved"),
                make_candidate([300 + shift, 40, 760 + shift, 615], 4.2, 2, "mask_protection", "keep_full_body", "preserve_box"),
                make_candidate([0, 0, 960, 720], 3.6, 3, "fallback", "fallback_full_image", "fallback_full_image"),
                make_candidate([410 + shift, 150, 610 + shift, 520], 1.8, 4, "negative_synthetic", "bad_crop", "subject_cut_or_too_tight"),
            ]
            rec = {
                "sample_id": f"smoke_{i:03d}__4x5",
                "image_path": str(image_path.resolve()),
                "rel_path": image_path.name,
                "image_width": w,
                "image_height": h,
                "target_aspect_ratio": "4:5",
                "caption": "A person holding a camera with empty space on the left.",
                "semantic_type": "person_holding_object",
                "crop_state_graph": {
                    "subject": {"bbox": subj, "center_norm": [0.5, 0.5], "area_ratio": 0.17},
                    "key_objects": [{"name": "camera", "bbox": obj, "relation": "held_by_subject"}],
                    "union_regions": {
                        "preserve_box": [subj[0], subj[1], obj[2], subj[3]],
                        "relation_box": [subj[0], subj[1], obj[2], subj[3]],
                    },
                    "issues": [{"issue_type": "subject_object_relation_should_be_preserved", "severity": 0.7}],
                    "suggested_actions": ["preserve_relation", "place_subject_center"],
                },
                "masks": {},
                "candidates": candidates,
                "best_crop": candidates[0]["box"],
                "best_score": 4.7,
                "best_action": "preserve_relation",
                "main_issue": "subject_object_relation_should_be_preserved",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"jsonl": str(jsonl), "images": str(image_dir), "num_images": args.num_images}, indent=2))


def make_candidate(box, score, rank, source, action, issue):
    return {
        "candidate_id": f"cand_{rank:04d}",
        "box": box,
        "source": source,
        "action": action,
        "issue": issue,
        "reason": "smoke candidate",
        "features": {
            "subject_coverage": 1.0 if score > 3.0 else 0.65,
            "relation_coverage": 1.0 if score > 3.0 else 0.55,
        },
        "scores": {"final_score": score},
        "rank": rank,
        "quality_label": "excellent" if score >= 4.5 else "fair",
    }


if __name__ == "__main__":
    main()

