from __future__ import annotations

import argparse
import json

from .pipeline import BuilderConfig, DatasetBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build direction-aware composition cropping dataset.")
    parser.add_argument("--image-root", required=True, help="Input image directory.")
    parser.add_argument("--out-dir", required=True, help="Output dataset directory.")
    parser.add_argument("--captions", default="", help="Optional JSON caption mapping: {rel_path: caption}.")
    parser.add_argument("--target-aspects", default="original", help="Comma separated aspects, e.g. original,1:1,4:5,16:9.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=80)

    parser.add_argument(
        "--vlm",
        default="heuristic",
        choices=[
            "heuristic",
            "precomputed",
            "qwen",
            "qwen_dashscope",
            "local_qwen",
            "qwen_local",
            "local_qwen_transformers",
            "openai",
            "openai_responses",
            "responses",
        ],
        help="VLM teacher provider.",
    )
    parser.add_argument("--vlm-precomputed", default="", help="Precomputed VLM JSON/JSONL path.")
    parser.add_argument("--qwen-model", default="", help="Qwen model name. Defaults to env QWEN_VL_MODEL or provider default.")
    parser.add_argument("--qwen-base-url", default="", help="OpenAI-compatible Qwen base URL.")
    parser.add_argument("--local-qwen-model", default="", help="Local Qwen3-VL model path or HF cache model directory.")
    parser.add_argument("--local-qwen-device-map", default="auto")
    parser.add_argument("--local-qwen-dtype", default="float16", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--local-qwen-attn", default="sdpa")
    parser.add_argument("--local-qwen-max-new-tokens", type=int, default=768)
    parser.add_argument("--local-qwen-min-pixels", type=int, default=262144)
    parser.add_argument("--local-qwen-max-pixels", type=int, default=1048576)
    parser.add_argument("--openai-model", default="", help="OpenAI vision-capable model. Defaults to env OPENAI_VLM_MODEL or provider default.")
    parser.add_argument("--openai-base-url", default="", help="OpenAI API base URL. Defaults to env OPENAI_BASE_URL or https://api.openai.com/v1.")
    parser.add_argument("--openai-image-detail", default="auto", choices=["auto", "low", "high"], help="Responses API image detail level.")

    parser.add_argument("--detector", default="none", choices=["none", "vlm", "yolo", "yolo_world"], help="Object detector.")
    parser.add_argument("--yolo-model", default="", help="YOLO model path. For --detector yolo_world, defaults to yolov8s-world.pt.")
    parser.add_argument("--yolo-conf", type=float, default=0.15)

    parser.add_argument("--segmenter", default="bbox", choices=["bbox", "sam"], help="Segmentation backend.")
    parser.add_argument("--sam-checkpoint", default="", help="SAM checkpoint path if --segmenter sam.")
    parser.add_argument("--sam-model-type", default="vit_h", help="SAM model type.")

    parser.add_argument("--aesthetic", default="none", choices=["none", "torcheat"], help="Aesthetic scorer.")
    parser.add_argument("--aesthetic-model", default="", help="TorchEAT model path if --aesthetic torcheat.")
    parser.add_argument("--caption-rule-root", default="caption-rule-co", help="Existing caption-rule-co root.")
    parser.add_argument("--no-existing-rules", action="store_true", help="Do not add candidates from existing caption-rule-co semantic rules.")
    parser.add_argument("--no-crops", action="store_true", help="Do not save top crop images.")
    parser.add_argument("--no-vis", action="store_true", help="Do not save visualizations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BuilderConfig(
        image_root=args.image_root,
        out_dir=args.out_dir,
        captions=args.captions or None,
        target_aspects=[x.strip() for x in args.target_aspects.split(",") if x.strip()],
        max_images=args.max_images,
        max_candidates=args.max_candidates,
        detector=args.detector,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        segmenter=args.segmenter,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        vlm=args.vlm,
        vlm_precomputed=args.vlm_precomputed,
        qwen_model=args.qwen_model,
        qwen_base_url=args.qwen_base_url,
        local_qwen_model=args.local_qwen_model,
        local_qwen_device_map=args.local_qwen_device_map,
        local_qwen_dtype=args.local_qwen_dtype,
        local_qwen_attn=args.local_qwen_attn,
        local_qwen_max_new_tokens=args.local_qwen_max_new_tokens,
        local_qwen_min_pixels=args.local_qwen_min_pixels,
        local_qwen_max_pixels=args.local_qwen_max_pixels,
        openai_model=args.openai_model,
        openai_base_url=args.openai_base_url,
        openai_image_detail=args.openai_image_detail,
        aesthetic=args.aesthetic,
        aesthetic_model=args.aesthetic_model,
        caption_rule_root=args.caption_rule_root,
        use_existing_rules=not args.no_existing_rules,
        save_crops=not args.no_crops,
        save_visualizations=not args.no_vis,
    )
    summary = DatasetBuilder(cfg).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
