#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_NAME="${ENV_NAME:-llm_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CREATE_CONDA="${CREATE_CONDA:-1}"
TORCH_CUDA="${TORCH_CUDA:-cu126}"
INSTALL_DACC="${INSTALL_DACC:-1}"
INSTALL_COMPOSITION="${INSTALL_COMPOSITION:-1}"
INSTALL_LOCAL_QWEN="${INSTALL_LOCAL_QWEN:-0}"
INSTALL_ULTRALYTICS="${INSTALL_ULTRALYTICS:-0}"
INSTALL_DINOV3_REPO="${INSTALL_DINOV3_REPO:-1}"
DINOV3_REPO_DIR="${DINOV3_REPO_DIR:-${HOME}/dinov3}"
RUN_SMOKE="${RUN_SMOKE:-0}"

log() {
  printf '[setup-rigformer] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '[setup-rigformer] missing command: %s\n' "$1" >&2
    return 1
  fi
}

pip_install() {
  (cd "${ROOT_DIR}" && python -m pip install "$@")
}

activate_or_create_conda() {
  if [[ "${CREATE_CONDA}" != "1" ]]; then
    log "CREATE_CONDA=0, using current Python: $(command -v python)"
    return
  fi
  require_cmd conda
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    log "Using existing conda env: ${ENV_NAME}"
  else
    log "Creating conda env: ${ENV_NAME} python=${PYTHON_VERSION}"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
  conda activate "${ENV_NAME}"
}

pip_install_filtered_requirements() {
  local req="$1"
  if [[ ! -f "${req}" ]]; then
    log "Skip missing requirements: ${req}"
    return
  fi
  local tmp
  tmp="$(mktemp)"
  grep -vE '^(torch|torchvision)([<>=~! ]|$)' "${req}" > "${tmp}" || true
  if [[ -s "${tmp}" ]]; then
    log "Installing ${req} without torch/torchvision"
    pip_install -r "${tmp}"
  fi
  rm -f "${tmp}"
}

install_torch() {
  if [[ "${TORCH_CUDA}" == "skip" ]]; then
    log "TORCH_CUDA=skip, not installing torch"
    return
  fi
  if [[ "${TORCH_CUDA}" == "cpu" ]]; then
    log "Installing PyTorch CPU wheels"
    pip_install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6" "torchvision>=0.21"
    return
  fi
  log "Installing PyTorch CUDA wheels: ${TORCH_CUDA}"
  pip_install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" "torch>=2.6" "torchvision>=0.21"
}

install_dinov3_repo() {
  if [[ "${INSTALL_DINOV3_REPO}" != "1" ]]; then
    log "INSTALL_DINOV3_REPO=0, skip DINOv3 repo clone"
    return
  fi
  require_cmd git
  if [[ -d "${DINOV3_REPO_DIR}/.git" ]]; then
    log "DINOv3 repo already exists: ${DINOV3_REPO_DIR}"
    git -C "${DINOV3_REPO_DIR}" pull --ff-only || log "DINOv3 pull skipped/failed; keeping existing checkout"
  elif [[ -e "${DINOV3_REPO_DIR}" ]]; then
    log "DINOv3 path exists but is not a git repo: ${DINOV3_REPO_DIR}"
    log "Keeping it untouched. Set DINOV3_REPO_DIR to another path if needed."
  else
    log "Cloning DINOv3 repo to ${DINOV3_REPO_DIR}"
    git clone https://github.com/facebookresearch/dinov3.git "${DINOV3_REPO_DIR}"
  fi
}

verify_environment() {
  log "Verifying Python imports"
  PYTHONPATH="${ROOT_DIR}/RIGCrop:${ROOT_DIR}:${PYTHONPATH:-}" python - <<'PY'
import os
import sys

import cv2
import matplotlib
import numpy
import scipy
import timm
import torch
import torchmetrics
import transformers
import yaml

from rigcrop.model import RIGCropModel

print("python =", sys.executable)
print("torch =", torch.__version__)
print("torch_cuda =", torch.version.cuda)
print("cuda_available =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("devices =", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("cv2 =", cv2.__version__)
print("numpy =", numpy.__version__)
print("transformers =", transformers.__version__)
print("timm =", timm.__version__)
print("matplotlib =", matplotlib.__version__)
print("PYTHONPATH includes rigcrop OK")
PY
}

run_smoke_if_requested() {
  if [[ "${RUN_SMOKE}" != "1" ]]; then
    return
  fi
  log "Running RIGCrop smoke test"
  (cd "${ROOT_DIR}" && bash RIGCrop/scripts/run_smoke_test.sh)
}

main() {
  cd "${ROOT_DIR}"
  log "repo=${ROOT_DIR}"
  activate_or_create_conda
  log "python=$(command -v python)"
  pip_install -U pip wheel "setuptools<82"
  install_torch
  pip_install_filtered_requirements "${ROOT_DIR}/requirements-rigformer.txt"
  if [[ "${INSTALL_DACC}" == "1" ]]; then
    pip_install_filtered_requirements "${ROOT_DIR}/requirements-dacc.txt"
  fi
  if [[ "${INSTALL_COMPOSITION}" == "1" ]]; then
    if [[ "${INSTALL_ULTRALYTICS}" == "1" ]]; then
      pip_install_filtered_requirements "${ROOT_DIR}/requirements-composition-builder.txt"
    else
      log "Installing composition builder deps without ultralytics"
      tmp="$(mktemp)"
      grep -vE '^(torch|torchvision|ultralytics)([<>=~! ]|$)' "${ROOT_DIR}/requirements-composition-builder.txt" > "${tmp}" || true
      if [[ -s "${tmp}" ]]; then
        pip_install -r "${tmp}"
      fi
      rm -f "${tmp}"
    fi
  fi
  if [[ "${INSTALL_LOCAL_QWEN}" == "1" ]]; then
    pip_install_filtered_requirements "${ROOT_DIR}/requirements-local-qwen.txt"
  fi
  pip_install -U "pillow>=10" "requests>=2.31" "tqdm>=4.66"
  install_dinov3_repo
  verify_environment
  run_smoke_if_requested
  log "Done."
  log "Activate later with: conda activate ${ENV_NAME}"
  log "For local DINOv3 .pth configs, use repo: ${DINOV3_REPO_DIR}"
}

main "$@"
