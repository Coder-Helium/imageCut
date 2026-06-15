#!/usr/bin/env bash
set -euo pipefail

IMAGE_ROOT="${IMAGE_ROOT:-caption-rule-co/test}"
CAPTIONS="${CAPTIONS:-}"
OUT_DIR="${OUT_DIR:-runs/dacc_dataset_qwen_pilot}"
MAX_IMAGES="${MAX_IMAGES:-20}"
TARGET_ASPECTS="${TARGET_ASPECTS:-original,4:5}"
QWEN_MODEL="${QWEN_MODEL:-qwen3-vl-flash}"

if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${QWEN_API_KEY:-}" ]]; then
  echo "[ERROR] Please set DASHSCOPE_API_KEY or QWEN_API_KEY before running Qwen VLM data building." >&2
  echo "Example:" >&2
  echo "  export DASHSCOPE_API_KEY='your_key_here'" >&2
  exit 1
fi

cmd=(
  python -m composition_dataset_builder.cli
  --image-root "$IMAGE_ROOT"
  --out-dir "$OUT_DIR"
  --target-aspects "$TARGET_ASPECTS"
  --max-images "$MAX_IMAGES"
  --vlm qwen
  --qwen-model "$QWEN_MODEL"
  --detector vlm
  --segmenter bbox
  --aesthetic none
)

if [[ -n "$CAPTIONS" ]]; then
  cmd+=(--captions "$CAPTIONS")
fi

"${cmd[@]}"

echo
echo "Done. Check:"
echo "  $OUT_DIR/metadata/all.jsonl"
echo "  $OUT_DIR/visualizations/"
echo "  $OUT_DIR/reports/summary.json"
