#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny DACC/Qwen-style dataset for RIG-Crop smoke tests.")
    parser.add_argument("--out-dir", default="RIGCrop/runs/smoke_data")
    parser.add_argument("--num-train", type=int, default=4)
    parser.add_argument("--num-val", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.out_dir)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    train = [_record(image_dir, idx, "train") for idx in range(args.num_train)]
    val = [_record(image_dir, idx, "val") for idx in range(args.num_val)]
    _dump(out / "train_qwen.jsonl", train)
    _dump(out / "val_qwen.jsonl", val)
    print(json.dumps({"train": len(train), "val": len(val), "out_dir": str(out.resolve())}, indent=2))


def _record(image_dir: Path, idx: int, split: str) -> dict:
    h, w = 240, 320
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (40 + idx * 20, 80, 120)
    cx = 90 + (idx % 3) * 40
    cv2.rectangle(img, (cx, 65), (cx + 90, 180), (220, 220, 80), -1)
    cv2.circle(img, (cx + 130, 120), 28, (70, 210, 120), -1)
    cv2.rectangle(img, (260, 20), (310, 80), (40, 40, 220), -1)
    image_path = image_dir / f"{split}_{idx:03d}.jpg"
    cv2.imwrite(str(image_path), img)
    candidates = [
        _cand("cpc_000", [0, 0, w, h], 3.4),
        _cand("cpc_001", [max(0, cx - 25), 35, min(w, cx + 175), 205], 4.8),
        _cand("cpc_002", [max(0, cx - 10), 55, min(w, cx + 120), 195], 4.2),
        _cand("cpc_003", [180, 0, w, 120], 2.0),
    ]
    return {
        "sample_id": image_path.stem,
        "image_path": str(image_path.resolve()),
        "rel_path": f"images/{image_path.name}",
        "image_width": w,
        "image_height": h,
        "target_aspect_ratio": "free",
        "caption": "person with ball",
        "source_caption": "",
        "semantic_type": "person_holding_object",
        "semantic_info": {},
        "vlm_understanding": {"source": "qwen_dashscope"},
        "composition_middle_state": {
            "source": "qwen_dashscope",
            "caption": "person with ball",
            "semantic_type": "person_holding_object",
            "main_subject": {
                "name": "person",
                "category": "person",
                "description": "central person",
                "importance": 1.0,
                "bbox_norm": [cx / w, 65 / h, (cx + 90) / w, 180 / h],
            },
            "key_objects": [
                {
                    "name": "ball",
                    "category": "object",
                    "description": "green ball",
                    "relation_to_subject": "held object",
                    "importance": 0.8,
                    "bbox_norm": [(cx + 102) / w, 92 / h, (cx + 158) / w, 148 / h],
                }
            ],
            "important_background": [],
            "distractors": [
                {
                    "name": "blue block",
                    "category": "distractor",
                    "importance": 0.4,
                    "location": "upper right",
                    "bbox_norm": [260 / w, 20 / h, 310 / w, 80 / h],
                }
            ],
            "composition_intent": {
                "preserve": ["person", "ball"],
                "optional_preserve": [],
                "avoid_cutting": ["person", "ball"],
                "leave_space_direction": "none",
                "preferred_subject_position": "center",
                "initial_issue": "distractor",
                "suggested_actions": ["preserve_relation", "remove_distractor", "place_subject_center"],
            },
        },
        "candidates": candidates,
        "pairwise_preferences": [
            {"winner": "cpc_001", "loser": "cpc_003", "weight": 1.0, "source": "smoke"},
            {"winner": "cpc_001", "loser": "cpc_000", "weight": 0.6, "source": "smoke"},
            {"winner": "cpc_002", "loser": "cpc_003", "weight": 0.7, "source": "smoke"},
        ],
        "best_crop": candidates[1]["box"],
        "best_score": 4.8,
        "cpc_supervision": {"source": "CPC", "candidate_scores": "pairwise_preference"},
        "quality_flags": {"has_vlm_middle_state": True},
    }


def _cand(cid: str, box: list[int], score: float) -> dict:
    return {"candidate_id": cid, "box": box, "box_format": "xyxy", "source": "smoke", "scores": {"final_score": score}}


def _dump(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
