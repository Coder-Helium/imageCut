#!/usr/bin/env bash
set -euo pipefail

IMAGE_ROOT="${IMAGE_ROOT:-caption-rule-co/test}"
CAPTIONS="${CAPTIONS:-caption-rule-co/gemini_captions.json}"
OUT_DIR="${OUT_DIR:-runs/dacc_dataset_openai_pilot}"
MAX_IMAGES="${MAX_IMAGES:-20}"
TARGET_ASPECTS="${TARGET_ASPECTS:-original,4:5}"
OPENAI_VLM_MODEL="${OPENAI_VLM_MODEL:-gpt-4.1-mini}"
OPENAI_IMAGE_DETAIL="${OPENAI_IMAGE_DETAIL:-auto}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] Please set OPENAI_API_KEY before running OpenAI Responses VLM data building." >&2
  echo "Example:" >&2
  echo "  export OPENAI_API_KEY='your_key_here'" >&2
  exit 1
fi

python -m composition_dataset_builder.cli \
  --image-root "$IMAGE_ROOT" \
  --captions "$CAPTIONS" \
  --out-dir "$OUT_DIR" \
  --target-aspects "$TARGET_ASPECTS" \
  --max-images "$MAX_IMAGES" \
  --vlm openai \
  --openai-model "$OPENAI_VLM_MODEL" \
  --openai-image-detail "$OPENAI_IMAGE_DETAIL" \
  --detector vlm \
  --segmenter bbox \
  --aesthetic none

echo
echo "Done. Check:"
echo "  $OUT_DIR/metadata/all.jsonl"
echo "  $OUT_DIR/visualizations/"
echo "  $OUT_DIR/reports/summary.json"
