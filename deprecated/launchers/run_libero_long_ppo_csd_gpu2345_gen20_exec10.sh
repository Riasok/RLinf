#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/minjaeoh/RLinf"
LOG_DIR="${ROOT}/logs/20260827-libero_long_ppo_csd_b1_gpu2345_gen20_exec10"

export CUDA_VISIBLE_DEVICES="2,3,4,5"
export REPO_PATH="${ROOT}"
export EMBODIED_PATH="${ROOT}/examples/embodiment"
export PYTHONPATH="${ROOT}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rlinf-libero-long-csd-gpu2345}"
export TMPDIR="${TMPDIR:-/tmp/rlinf-libero-long-csd-tmp}"

set -a
. /data/minjaeoh/.env
set +a
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}" "${LOG_DIR}"

cd "${ROOT}"
exec "${ROOT}/.venv/bin/python" examples/embodiment/train_embodied_agent.py \
  --config-path "${ROOT}/examples/embodiment/config" \
  --config-name libero_10_ppo_csd_pi05_gpu2345_gen20_exec10 \
  > "${LOG_DIR}/launcher.log" 2>&1
