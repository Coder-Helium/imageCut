#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:-/home/mx/workspace/imageCut/models--Qwen--Qwen3-VL-30B-A3B-Instruct}"
INPUT_JSONL="${INPUT_JSONL:-/home/mx/workspace/imageCut/data/metadata/inputFile/train.jsonl}"
OUT_JSONL="${OUT_JSONL:-/home/mx/workspace/imageCut/data/metadata/outputFile/train_local_qwen.jsonl}"
VIS_DIR="${VIS_DIR:-/home/mx/workspace/imageCut/data/visualization/vis_train_local_qwen}"
LOG_PATH="${LOG_PATH:-/home/mx/workspace/imageCut/data/logs/enrich_train_local_qwen.log}"
MAX_RECORDS="${MAX_RECORDS:-0}"
MODE="${MODE:-overwrite}"
ATTN="${ATTN:-sdpa}"
DTYPE="${DTYPE:-float16}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"

mkdir -p "$(dirname "$OUT_JSONL")" "$VIS_DIR" "$(dirname "$LOG_PATH")"

ARGS=(
  scripts/enrich_gaic_with_vlm_semantics.py
  --input-jsonl "$INPUT_JSONL"
  --out-jsonl "$OUT_JSONL"
  --vlm local_qwen
  --local-qwen-model "$MODEL_PATH"
  --local-qwen-device-map auto
  --local-qwen-dtype "$DTYPE"
  --local-qwen-attn "$ATTN"
  --local-qwen-max-new-tokens 768
  --local-qwen-min-pixels 262144
  --local-qwen-max-pixels "$MAX_PIXELS"
  --visualize
  --vis-dir "$VIS_DIR"
  --vis-topk 5
)

if [[ "$MAX_RECORDS" != "0" ]]; then
  ARGS+=(--max-records "$MAX_RECORDS")
fi

if [[ "$MODE" == "resume" ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--overwrite)
fi

nohup python -u "${ARGS[@]}" > "$LOG_PATH" 2>&1 &
PID=$!
echo "$PID" > "${LOG_PATH}.pid"
echo "Started local Qwen enrichment: PID=$PID"
echo "Log: $LOG_PATH"
echo "Tail with: tail -f $LOG_PATH"
