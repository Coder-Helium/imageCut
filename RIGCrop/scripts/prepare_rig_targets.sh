#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-data/cpc_semantic_qwen/metadata}"
OUT_DIR="${2:-data/cpc_rig/metadata}"
MAX_NODES="${MAX_NODES:-8}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-200}"

mkdir -p "${OUT_DIR}"

for split in train val; do
  in_jsonl="${INPUT_DIR}/${split}.jsonl"
  out_jsonl="${OUT_DIR}/${split}.jsonl"
  if [[ ! -f "${in_jsonl}" ]]; then
    echo "[prepare-rig] skip missing ${in_jsonl}" >&2
    continue
  fi
  python -u RIGCrop/scripts/audit_middle_state_schema.py \
    --jsonl "${in_jsonl}" \
    --out-json "${OUT_DIR}/${split}.audit.json"
  python -u RIGCrop/scripts/build_middle_state_targets.py \
    --input-jsonl "${in_jsonl}" \
    --out-jsonl "${out_jsonl}" \
    --max-nodes "${MAX_NODES}" \
    --progress-interval "${PROGRESS_INTERVAL}" \
    --overwrite
done
