#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/minjaeoh/RLinf
LOG_DIR="${ROOT}/logs/20260905-libero_40_ppo_csdgate_padv_gpu45_gen10_exec5"
STATE_LOG="${LOG_DIR}/state.log"

mkdir -p "${LOG_DIR}" /tmp/csdg45-ray /tmp/csdg45-tmp
cd "${ROOT}"
printf '%s START PPO+CSD positive_advantage-gated on GPUs 4-5\n' "$(date '+%F %T %Z')" > "${STATE_LOG}"

unset CUDA_VISIBLE_DEVICES RAY_ADDRESS
export RLINF_FORCE_LOCAL_RAY=1
export RLINF_LOCAL_RAY_NUM_CPUS=64
export RLINF_LOCAL_RAY_OBJECT_STORE_MEMORY_BYTES=32000000000
export REPO_PATH="${ROOT}"
export EMBODIED_PATH="${ROOT}/examples/embodiment"
export PYTHONPATH="${ROOT}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_TYPE=standard
export RAY_TMPDIR=/tmp/csdg45-ray
export TMPDIR=/tmp/csdg45-tmp
set -a
. /data/minjaeoh/.env
set +a

if "${ROOT}/.venv/bin/python" examples/embodiment/train_embodied_agent.py \
  --config-path "${ROOT}/examples/embodiment/config" \
  --config-name libero_40_ppo_csdgate_padv_pi05_gpu45_gen10_exec5_100steps \
  > "${LOG_DIR}/launcher.log" 2>&1; then
  printf '%s DONE PPO+CSD gated\n' "$(date '+%F %T %Z')" >> "${STATE_LOG}"
else
  status=$?
  printf '%s ERROR PPO+CSD gated status=%s\n' "$(date '+%F %T %Z')" "${status}" >> "${STATE_LOG}"
  exit "${status}"
fi
