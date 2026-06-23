#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-RIGCrop/configs/rig_crop_cpc_joint.yaml}"
NPROC="${NPROC:-4}"
LOG_DIR="${LOG_DIR:-RIGCrop/logs}"

mkdir -p "${LOG_DIR}"
log_file="${LOG_DIR}/rig_crop_train_$(date +%Y%m%d_%H%M%S).log"

echo "[run-server] config=${CONFIG}"
echo "[run-server] nproc=${NPROC}"
echo "[run-server] log=${log_file}"

nohup bash -lc "set -euo pipefail; torchrun --standalone --nproc_per_node=${NPROC} RIGCrop/scripts/train_rig_crop.py --config ${CONFIG}" \
  > "${log_file}" 2>&1 &

echo "[run-server] pid=$!"
echo "[run-server] tail -f ${log_file}"
