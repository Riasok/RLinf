#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/minjaeoh/RLinf"
LOG_DIR="${ROOT}/logs/20260828-libero_long_ppo_csd_b1_gpu45_gen10_exec5"

unset CUDA_VISIBLE_DEVICES
export RLINF_FORCE_LOCAL_RAY=1
export REPO_PATH="${ROOT}"
export EMBODIED_PATH="${ROOT}/examples/embodiment"
export PYTHONPATH="${ROOT}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export RAY_TMPDIR="/tmp/rlc45"
export TMPDIR="/tmp/tc45"
unset RAY_ADDRESS

set -a
. /data/minjaeoh/.env
set +a
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}" "${LOG_DIR}"

cd "${ROOT}"
exec "${ROOT}/.venv/bin/python" examples/embodiment/train_embodied_agent.py \
  --config-path "${ROOT}/examples/embodiment/config" \
  --config-name libero_10_ppo_csd_b1_pi05_gpu45_gen10_exec5_100steps \
  > "${LOG_DIR}/launcher.log" 2>&1
