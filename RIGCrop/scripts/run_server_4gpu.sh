#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-RIGCrop/configs/rig_crop_cpc_joint.yaml}"
NPROC="${NPROC:-4}"
LOG_DIR="${LOG_DIR:-RIGCrop/logs}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

mkdir -p "${LOG_DIR}"
log_file="${LOG_DIR}/rig_crop_train_$(date +%Y%m%d_%H%M%S).log"

echo "[run-server] config=${CONFIG}"
echo "[run-server] nproc=${NPROC}"
echo "[run-server] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[run-server] log=${log_file}"

nohup bash -lc "set -euo pipefail; torchrun --nproc_per_node=${NPROC} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} RIGCrop/scripts/train_rig_crop.py --config ${CONFIG}" \
  > "${log_file}" 2>&1 &

echo "[run-server] pid=$!"
echo "[run-server] tail -f ${log_file}"
