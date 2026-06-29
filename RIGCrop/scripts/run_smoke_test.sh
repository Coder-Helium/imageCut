#!/usr/bin/env bash
set -euo pipefail

SMOKE_DIR="${SMOKE_DIR:-RIGCrop/runs/smoke_data}"
SMOKE_RUN="${SMOKE_RUN:-RIGCrop/runs/smoke_rig_crop}"
PYTHON="${PYTHON:-python}"

rm -rf "${SMOKE_DIR}" "${SMOKE_RUN}"

"${PYTHON}" RIGCrop/scripts/create_smoke_data.py \
  --out-dir "${SMOKE_DIR}" \
  --num-train 4 \
  --num-val 2

"${PYTHON}" RIGCrop/scripts/audit_middle_state_schema.py \
  --jsonl "${SMOKE_DIR}/train_qwen.jsonl" \
  --out-json "${SMOKE_DIR}/audit_train.json"

"${PYTHON}" RIGCrop/scripts/build_middle_state_targets.py \
  --input-jsonl "${SMOKE_DIR}/train_qwen.jsonl" \
  --out-jsonl "${SMOKE_DIR}/train_rig.jsonl" \
  --max-nodes 8 \
  --progress-interval 1 \
  --overwrite

"${PYTHON}" RIGCrop/scripts/build_middle_state_targets.py \
  --input-jsonl "${SMOKE_DIR}/val_qwen.jsonl" \
  --out-jsonl "${SMOKE_DIR}/val_rig.jsonl" \
  --max-nodes 8 \
  --progress-interval 1 \
  --overwrite

"${PYTHON}" RIGCrop/scripts/train_rig_crop.py \
  --config RIGCrop/configs/rig_crop_cpc_smoke.yaml

"${PYTHON}" RIGCrop/scripts/eval_rig_crop.py \
  --jsonl "${SMOKE_DIR}/val_rig.jsonl" \
  --checkpoint "${SMOKE_RUN}/best.pt" \
  --config RIGCrop/configs/rig_crop_cpc_smoke.yaml \
  --batch-size 2 \
  --image-size 128 \
  --crop-size 96

"${PYTHON}" RIGCrop/scripts/predict_rig_crop.py \
  --image "${SMOKE_DIR}/images/val_000.jpg" \
  --checkpoint "${SMOKE_RUN}/best.pt" \
  --config RIGCrop/configs/rig_crop_cpc_smoke.yaml \
  --out-json "${SMOKE_RUN}/predict_val_000.json" \
  --out-vis "${SMOKE_RUN}/predict_val_000.jpg" \
  --topk 3 \
  --image-size 128 \
  --crop-size 96 \
  --batch-size 8

echo "[smoke] OK"
