#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/minjaeoh/RLinf
QUEUE_ROOT="${ROOT}/logs/20260901-libero_40_ppo_vs_csd_queue"
STATE_LOG="${QUEUE_ROOT}/queue_state.log"
EVAL_SESSION=libero_long_step90_safe_gpu4567
EVAL_ROOT="${ROOT}/logs/20260901-libero-long-step90-safe-gpu4567-v2"
CSD_CONFIG=libero_40_ppo_csd_b1_pi05_gpu45_gen10_exec5_100steps
PPO_CONFIG=libero_40_ppo_pi05_gpu67_gen10_exec5_100steps

mkdir -p "${QUEUE_ROOT}"
cd "${ROOT}"

log() {
  printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${STATE_LOG}"
}

wait_for_evaluations() {
  log "WAITING for step-90 LIBERO-Long and Long-PRO evaluations."
  while tmux has-session -t "${EVAL_SESSION}" 2>/dev/null; do
    sleep 60
  done

  local expected=(
    csd_step90_libero_long
    csd_step90_libero_long_pro
    ppo_step90_libero_long
    ppo_step90_libero_long_pro
  )
  local job
  for job in "${expected[@]}"; do
    if [[ ! -f "${EVAL_ROOT}/${job}/DONE" ]]; then
      log "ERROR evaluation queue ended without ${job}/DONE; training will not start."
      return 1
    fi
  done
  log "All four evaluations completed successfully; allowing 30 seconds for GPU cleanup."
  sleep 30
}

run_training() {
  local name=$1
  local config=$2
  local ray_tmp=$3
  local job_tmp=$4
  local out="${QUEUE_ROOT}/${name}"

  mkdir -p "${out}" "${ray_tmp}" "${job_tmp}"
  log "START ${name} with config ${config}"
  (
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
    export RAY_TMPDIR="${ray_tmp}"
    export TMPDIR="${job_tmp}"
    set -a
    . /data/minjaeoh/.env
    set +a
    "${ROOT}/.venv/bin/python" examples/embodiment/train_embodied_agent.py \
      --config-path "${ROOT}/examples/embodiment/config" \
      --config-name "${config}"
  ) > "${out}/launcher.log" 2>&1

  if grep -Eq "Error executing job|Traceback|OutOfMemoryError|CUDA out of memory|Exception occurred while" "${out}/launcher.log"; then
    log "ERROR ${name}: launcher reported a failure."
    return 1
  fi
  touch "${out}/DONE"
  log "DONE ${name}"
}

wait_for_evaluations

run_training libero_40_ppo_csd_b1_gpu45 "${CSD_CONFIG}" /tmp/libero40-csd-ray /tmp/libero40-csd-tmp &
csd_pid=$!
run_training libero_40_ppo_gpu67 "${PPO_CONFIG}" /tmp/libero40-ppo-ray /tmp/libero40-ppo-tmp &
ppo_pid=$!

csd_status=0
ppo_status=0
wait "${csd_pid}" || csd_status=$?
wait "${ppo_pid}" || ppo_status=$?

if (( csd_status != 0 || ppo_status != 0 )); then
  log "ERROR training lane failed: csd=${csd_status}, ppo=${ppo_status}"
  exit 1
fi
log "Both 40-task PPO and PPO+CSD training runs finished."
