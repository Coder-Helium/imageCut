from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from .candidates import generate_candidates
from .crop_state import build_crop_state_graph
from .detectors import Detector, create_detector
from .geometry import aspect_name_to_safe
from .io_utils import append_jsonl, caption_for_image, ensure_dir, iter_images, load_captions, read_image_bgr, safe_stem, write_json
from .scoring import AestheticScorer, create_aesthetic_scorer, score_candidates
from .segmenters import Segmenter, create_segmenter
from .semantic import route_caption
from .rule_adapter import generate_existing_rule_candidates
from .visualization import draw_sample_visualization
from .vlm import VLMProvider, create_vlm_provider


@dataclass
class BuilderConfig:
    image_root: str
    out_dir: str
    captions: Optional[str] = None
    target_aspects: List[str] = field(default_factory=lambda: ["original"])
    max_images: Optional[int] = None
    max_candidates: int = 80
    detector: str = "none"
    yolo_model: str = ""
    yolo_conf: float = 0.15
    segmenter: str = "bbox"
    sam_checkpoint: str = ""
    sam_model_type: str = "vit_h"
    vlm: str = "heuristic"
    vlm_precomputed: str = ""
    qwen_model: str = ""
    qwen_base_url: str = ""
    local_qwen_model: str = ""
    local_qwen_device_map: str = "auto"
    local_qwen_dtype: str = "float16"
    local_qwen_attn: str = "sdpa"
    local_qwen_max_new_tokens: int = 768
    local_qwen_min_pixels: int = 262144
    local_qwen_max_pixels: int = 1048576
    openai_model: str = ""
    openai_base_url: str = ""
    openai_image_detail: str = "auto"
    aesthetic: str = "none"
    aesthetic_model: str = ""
    caption_rule_root: str = "caption-rule-co"
    use_existing_rules: bool = True
    save_crops: bool = True
    save_visualizations: bool = True


class DatasetBuilder:
    def __init__(self, config: BuilderConfig):
        self.config = config
        self.image_root = Path(config.image_root).resolve()
        self.out_dir = Path(config.out_dir).resolve()
        self.metadata_dir = ensure_dir(self.out_dir / "metadata")
        self.mask_root = ensure_dir(self.out_dir / "masks")
        self.crop_root = ensure_dir(self.out_dir / "crops")
        self.vis_root = ensure_dir(self.out_dir / "visualizations")
        self.records_path = self.metadata_dir / "all.jsonl"
        self.failed_path = self.metadata_dir / "failed.jsonl"
        self.summary_path = self.out_dir / "reports" / "summary.json"
        self.csv_path = self.out_dir / "reports" / "summary.csv"
        ensure_dir(self.out_dir / "reports")

        self.captions = load_captions(config.captions)
        vlm_kwargs: Dict[str, Any] = {}
        if config.vlm.lower() in {"qwen", "qwen_dashscope", "dashscope"}:
            vlm_kwargs = {
                "model": config.qwen_model or None,
                "base_url": config.qwen_base_url or None,
            }
        elif config.vlm.lower() in {"local_qwen", "qwen_local", "local_qwen_transformers"}:
            vlm_kwargs = {
                "model_path": config.local_qwen_model,
                "device_map": config.local_qwen_device_map,
                "dtype": config.local_qwen_dtype,
                "attn_implementation": config.local_qwen_attn,
                "max_new_tokens": config.local_qwen_max_new_tokens,
                "min_pixels": config.local_qwen_min_pixels,
                "max_pixels": config.local_qwen_max_pixels,
            }
        elif config.vlm.lower() in {"openai", "openai_responses", "responses"}:
            vlm_kwargs = {
                "model": config.openai_model or None,
                "base_url": config.openai_base_url or None,
                "image_detail": config.openai_image_detail,
            }
        self.vlm = create_vlm_provider(
            config.vlm,
            precomputed_path=config.vlm_precomputed or None,
            **vlm_kwargs,
        )
        self.detector = create_detector(config.detector, model_path=config.yolo_model or None, conf=config.yolo_conf)
        self.segmenter = create_segmenter(config.segmenter, sam_checkpoint=config.sam_checkpoint, sam_model_type=config.sam_model_type)
        self.aesthetic = create_aesthetic_scorer(config.aesthetic, model_path=config.aesthetic_model, project_root=config.caption_rule_root)

    def run(self) -> Dict[str, Any]:
        for path in [self.records_path, self.failed_path]:
            if path.exists():
                path.unlink()

        image_paths = list(iter_images(self.image_root))
        if self.config.max_images is not None:
            image_paths = image_paths[: self.config.max_images]

        started = time.time()
        ok = 0
        failed = 0
        semantic_counts: Dict[str, int] = {}
        aspect_counts: Dict[str, int] = {}

        for idx, image_path in enumerate(image_paths, start=1):
            print(f"[{idx}/{len(image_paths)}] {image_path.relative_to(self.image_root)}")
            try:
                records = self.process_image(image_path)
                for rec in records:
                    append_jsonl(self.records_path, rec)
                    ok += 1
                    semantic_counts[rec.get("semantic_type", "unknown")] = semantic_counts.get(rec.get("semantic_type", "unknown"), 0) + 1
                    aspect_counts[rec.get("target_aspect_ratio", "unknown")] = aspect_counts.get(rec.get("target_aspect_ratio", "unknown"), 0) + 1
            except Exception as exc:
                failed += 1
                append_jsonl(
                    self.failed_path,
                    {
                        "image_path": str(image_path),
                        "status": "error",
                        "reason": repr(exc),
                    },
                )
                print(f"  ERROR: {repr(exc)}")

        summary = {
            "status": "done",
            "image_root": str(self.image_root),
            "out_dir": str(self.out_dir),
            "num_images": len(image_paths),
            "num_samples": ok,
            "num_failed_images": failed,
            "semantic_counts": semantic_counts,
            "aspect_counts": aspect_counts,
            "elapsed_sec": round(time.time() - started, 3),
            "records_path": str(self.records_path),
            "failed_path": str(self.failed_path),
        }
        write_json(self.summary_path, summary)
        self._write_summary_csv(summary)
        return summary

    def process_image(self, image_path: Path) -> List[Dict[str, Any]]:
        img = read_image_bgr(image_path)
        h, w = img.shape[:2]
        rel = image_path.relative_to(self.image_root).as_posix()
        caption = caption_for_image(self.captions, image_path, self.image_root)
        semantic_info = route_caption(caption, self.config.caption_rule_root)
        vlm_understanding = self.vlm.understand(str(image_path), caption, semantic_info)
        output_caption = str(vlm_understanding.get("caption") or caption or "")
        semantic_type = vlm_understanding.get("semantic_type") or semantic_info.get("semantic_type", "unknown")
        detections = self.detector.detect(str(image_path), vlm_understanding, w, h)

        sample_stem = safe_stem(rel)
        masks = self.segmenter.segment(
            img,
            str(image_path),
            detections,
            vlm_understanding,
            str(ensure_dir(self.mask_root / sample_stem)),
        )
        graph = build_crop_state_graph(w, h, vlm_understanding, detections, masks)

        records: List[Dict[str, Any]] = []
        for aspect in self.config.target_aspects:
            aspect_safe = aspect_name_to_safe(aspect)
            candidates = generate_candidates(
                image_w=w,
                image_h=h,
                target_aspect=aspect,
                crop_state_graph=graph,
                max_candidates=self.config.max_candidates,
            )
            if self.config.use_existing_rules:
                existing = generate_existing_rule_candidates(
                    image_w=w,
                    image_h=h,
                    target_aspect=aspect,
                    detections=detections,
                    semantic_info={**semantic_info, "semantic_type": semantic_type},
                    caption_rule_root=self.config.caption_rule_root,
                    max_num=12,
                )
                # Keep existing hand-written rules near the front before scoring.
                candidates = existing + candidates
            candidates = score_candidates(img, candidates, masks, graph, self.aesthetic)
            best = candidates[0] if candidates else None
            sample_id = f"{sample_stem}__{aspect_safe}"

            if self.config.save_crops and best is not None:
                self._save_top_crops(img, candidates[:8], sample_id)
            if self.config.save_visualizations:
                draw_sample_visualization(
                    img,
                    candidates,
                    masks,
                    str(self.vis_root / f"{sample_id}.jpg"),
                    title=f"{semantic_type} | {aspect}",
                )

            score_gap = None
            if len(candidates) >= 2:
                score_gap = candidates[0].scores.get("final_score", 0) - candidates[1].scores.get("final_score", 0)
            record = {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "rel_path": rel,
                "image_width": w,
                "image_height": h,
                "target_aspect_ratio": aspect,
                "caption": output_caption,
                "source_caption": caption,
                "semantic_type": semantic_type,
                "semantic_info": semantic_info,
                "vlm_understanding": vlm_understanding,
                "detections": [d.to_json() for d in detections],
                "crop_state_graph": graph,
                "masks": {k: [m.to_json() for m in v] for k, v in masks.items()},
                "candidates": [c.to_json() for c in candidates],
                "best_crop": best.box.to_xyxy_int() if best else None,
                "best_score": best.scores.get("final_score") if best else None,
                "best_action": best.action if best else None,
                "main_issue": best.issue if best else None,
                "quality_flags": {
                    "has_valid_subject_mask": bool(masks.get("preserve_masks")),
                    "has_valid_key_object_mask": bool(masks.get("relation_masks")),
                    "has_enough_candidates": len(candidates) >= 8,
                    "score_gap_top1_top2": score_gap,
                    "needs_manual_review": bool(score_gap is not None and score_gap < 0.08),
                },
            }
            records.append(record)
        return records

    def _save_top_crops(self, image_bgr, candidates, sample_id: str) -> None:
        from .io_utils import crop_image

        out_dir = ensure_dir(self.crop_root / sample_id)
        for cand in candidates:
            crop = crop_image(image_bgr, cand.box)
            cv2.imwrite(str(out_dir / f"rank_{cand.rank:02d}_{cand.scores.get('final_score', 0):.2f}.jpg"), crop)

    def _write_summary_csv(self, summary: Dict[str, Any]) -> None:
        ensure_dir(self.csv_path.parent)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for key, value in summary.items():
                writer.writerow([key, value])
