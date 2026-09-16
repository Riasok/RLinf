#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/minjaeoh/RLinf
QUEUE_ROOT=${ROOT}/logs/20260831-robot-eval-then-goal
STATE_LOG=${QUEUE_ROOT}/queue_state.log
RLINF_MODEL=${ROOT}/checkpoints/RLinf-Pi05-LIBERO-SFT
LEROBOT_MODEL=${ROOT}/checkpoints/lerobot-pi05-libero-base
CSD_CKPT=${ROOT}/logs/20260828-libero_long_ppo_csd_b1_gpu45_gen10_exec5/libero_long_ppo_csd_b1_seed42_gpu45_gen10_exec5_20260828/checkpoints/global_step_90/actor/model_state_dict/full_weights.pt
PPO_CKPT=${ROOT}/logs/20260828-libero_long_ppo_gpu67_gen10_exec5/libero_long_ppo_seed42_gpu67_gen10_exec5_20260828/checkpoints/global_step_90/actor/model_state_dict/full_weights.pt

mkdir -p "${QUEUE_ROOT}" "${QUEUE_ROOT}/standard" "${QUEUE_ROOT}/plus" "${QUEUE_ROOT}/goal"
cd "${ROOT}"

log() {
  printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${STATE_LOG}"
}

episode_steps() {
  case "$1" in
    libero_spatial) echo 240 ;;
    libero_object) echo 280 ;;
    libero_goal) echo 320 ;;
    libero_10) echo 520 ;;
    *) return 2 ;;
  esac
}

plus_rollout_steps() {
  case "$1" in
    libero_spatial) echo 1200 ;;
    libero_object) echo 1680 ;;
    libero_goal) echo 1920 ;;
    libero_10) echo 3120 ;;
    *) return 2 ;;
  esac
}

wait_for_long_training() {
  log "Using both saved step-90 Long checkpoints as final checkpoints."
  while pgrep -f 'config-name libero_10_ppo_csd_b1_pi05_gpu45_gen10_exec5_100steps|config-name libero_10_ppo_pi05_gpu67_gen10_exec5_100steps' >/dev/null; do
    sleep 60
  done
  if [[ ! -s "${CSD_CKPT}" || ! -s "${PPO_CKPT}" ]]; then
    log "ERROR: a Long run is missing its step-90 full_weights.pt; evaluations will not start."
    return 1
  fi
  if [[ ! -s "${LEROBOT_MODEL}/model.safetensors" ]]; then
    log "ERROR: LeRobot pi0.5 checkpoint is missing."
    return 1
  fi
  log "Both step-90 Long checkpoints are ready."
}

run_rlinf_eval() {
  local mode=$1 name=$2 gpu_range=$3 suite=$4 model=$5 ckpt=${6:-}
  local out=${QUEUE_ROOT}/${mode}/${name}
  local done_file=${out}/DONE
  [[ -f "${done_file}" ]] && { log "SKIP completed ${name}"; return 0; }
  mkdir -p "${out}" "/tmp/${name}-ray" "/tmp/${name}-tmp"
  local ep rollout trials libero_type
  ep=$(episode_steps "${suite}")
  if [[ "${mode}" == plus ]]; then
    rollout=$(plus_rollout_steps "${suite}")
    trials=1
    libero_type=plus
  else
    rollout=${ep}
    trials=50
    libero_type=standard
  fi
  log "START RLinf ${name}: GPUs ${gpu_range}, ${mode} ${suite}"
  (
    unset CUDA_VISIBLE_DEVICES RAY_ADDRESS
    export RLINF_FORCE_LOCAL_RAY=1
    export REPO_PATH=${ROOT}
    export EMBODIED_PATH=${ROOT}/examples/embodiment
    export PYTHONPATH=${ROOT}
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export LIBERO_TYPE=${libero_type}
    export LIBERO_SUFFIX=all
    export MAGICK_HOME=${ROOT}/.native-deps
    export LD_LIBRARY_PATH=${ROOT}/.native-deps/lib:${LD_LIBRARY_PATH:-}
    export RAY_TMPDIR=/tmp/${name}-ray
    export TMPDIR=/tmp/${name}-tmp
    export EVAL_GPU_RANGE=${gpu_range}
    export EVAL_LOG_DIR=${out}
    export EVAL_NAME=${name}
    export EVAL_SUITE=${suite}
    export EVAL_EPISODE_STEPS=${ep}
    export EVAL_ROLLOUT_STEPS=${rollout}
    export EVAL_TRIALS_PER_TASK=${trials}
    export EVAL_MODEL_PATH=${model}
    export EVAL_CKPT_PATH=${ckpt}
    "${ROOT}/.venv/bin/python" evaluations/eval_embodied_agent.py \
      --config-path "${ROOT}/evaluations/libero" \
      --config-name libero_pi05_queued_eval
  ) > "${out}/launcher.log" 2>&1
  if grep -Eq "Error executing job|Traceback|Exception occurred while|OutOfMemoryError" "${out}/launcher.log" || ! grep -q "eval/success" "${out}/launcher.log"; then
    log "ERROR: RLinf ${name} did not produce valid evaluation metrics."
    return 1
  fi
  touch "${done_file}"
  log "DONE RLinf ${name}"
  sleep 20
}

run_lerobot_eval() {
  local mode=$1 name=$2 gpu=$3 suite=$4
  local out=${QUEUE_ROOT}/${mode}/${name}
  local done_file=${out}/DONE
  [[ -f "${done_file}" ]] && { log "SKIP completed ${name}"; return 0; }
  mkdir -p "${out}" "${QUEUE_ROOT}/libero-configs"
  local ep n_episodes batch env_type
  ep=$(episode_steps "${suite}")
  if [[ "${mode}" == plus ]]; then
    n_episodes=1
    batch=1
    env_type=libero_plus
  else
    n_episodes=50
    batch=50
    env_type=libero
  fi
  log "START LeRobot ${name}: GPU ${gpu}, ${mode} ${suite}"
  (
    export CUDA_VISIBLE_DEVICES=${gpu}
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export MAGICK_HOME=${ROOT}/.native-deps
    export LD_LIBRARY_PATH=${ROOT}/.native-deps/lib:${LD_LIBRARY_PATH:-}
    export LEROBOT_LIBERO_TYPE=${mode}
    export LEROBOT_LIBERO_CONFIG_FILE=${QUEUE_ROOT}/libero-configs/${mode}-${gpu}.yaml
    "${ROOT}/.venv-lerobot/bin/python" scripts/lerobot_libero_eval_entry.py \
      --policy.path="${LEROBOT_MODEL}" \
      --policy.device=cuda \
      --policy.dtype=bfloat16 \
      --env.type=${env_type} \
      --env.task=${suite} \
      --env.episode_length=${ep} \
      --env.max_parallel_tasks=1 \
      --eval.batch_size=${batch} \
      --eval.n_episodes=${n_episodes} \
      --eval.use_async_envs=true \
      --output_dir="${out}"
  ) > "${out}/launcher.log" 2>&1
  if [[ ! -s "${out}/eval_info.json" ]]; then
    log "ERROR: LeRobot ${name} did not produce eval_info.json."
    return 1
  fi
  touch "${done_file}"
  log "DONE LeRobot ${name}"
}

standard_lane_45() {
  run_rlinf_eval standard csd_step90_standard_long 4-5 libero_10 "${RLINF_MODEL}" "${CSD_CKPT}"
  run_rlinf_eval standard rlinf_base_standard_spatial 4-5 libero_spatial "${RLINF_MODEL}"
  run_rlinf_eval standard rlinf_base_standard_goal 4-5 libero_goal "${RLINF_MODEL}"
  run_lerobot_eval standard lerobot_base_standard_object 4 libero_object
  run_lerobot_eval standard lerobot_base_standard_long 4 libero_10
}

standard_lane_67() {
  run_rlinf_eval standard ppo_step90_standard_long 6-7 libero_10 "${RLINF_MODEL}" "${PPO_CKPT}"
  run_rlinf_eval standard rlinf_base_standard_object 6-7 libero_object "${RLINF_MODEL}"
  run_rlinf_eval standard rlinf_base_standard_long 6-7 libero_10 "${RLINF_MODEL}"
  run_lerobot_eval standard lerobot_base_standard_spatial 6 libero_spatial
  run_lerobot_eval standard lerobot_base_standard_goal 6 libero_goal
}

plus_lane_45() {
  run_rlinf_eval plus csd_step90_plus_long 4-5 libero_10 "${RLINF_MODEL}" "${CSD_CKPT}"
  run_rlinf_eval plus rlinf_base_plus_object 4-5 libero_object "${RLINF_MODEL}"
  run_rlinf_eval plus rlinf_base_plus_long 4-5 libero_10 "${RLINF_MODEL}"
  run_lerobot_eval plus lerobot_base_plus_spatial 4 libero_spatial
  run_lerobot_eval plus lerobot_base_plus_goal 4 libero_goal
}

plus_lane_67() {
  run_rlinf_eval plus ppo_step90_plus_long 6-7 libero_10 "${RLINF_MODEL}" "${PPO_CKPT}"
  run_rlinf_eval plus rlinf_base_plus_spatial 6-7 libero_spatial "${RLINF_MODEL}"
  run_rlinf_eval plus rlinf_base_plus_goal 6-7 libero_goal "${RLINF_MODEL}"
  run_lerobot_eval plus lerobot_base_plus_object 6 libero_object
  run_lerobot_eval plus lerobot_base_plus_long 6 libero_10
}

run_phase() {
  local phase=$1 left=$2 right=$3
  log "BEGIN ${phase} evaluation phase"
  set +e
  "${left}" & local left_pid=$!
  "${right}" & local right_pid=$!
  wait "${left_pid}"; local left_status=$?
  wait "${right_pid}"; local right_status=$?
  set -e
  if (( left_status != 0 || right_status != 0 )); then
    log "ERROR: ${phase} phase failed (lane45=${left_status}, lane67=${right_status}); Goal training is blocked."
    return 1
  fi
  log "END ${phase} evaluation phase"
}

run_goal() {
  local name=$1 gpu_range=$2 csd_enabled=$3 csd_beta=$4
  local out=${QUEUE_ROOT}/goal/${name}
  mkdir -p "${out}" "/tmp/${name}-ray" "/tmp/${name}-tmp"
  (
    unset CUDA_VISIBLE_DEVICES RAY_ADDRESS
    export RLINF_FORCE_LOCAL_RAY=1
    export REPO_PATH=${ROOT}
    export EMBODIED_PATH=${ROOT}/examples/embodiment
    export PYTHONPATH=${ROOT}
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export RAY_TMPDIR=/tmp/${name}-ray
    export TMPDIR=/tmp/${name}-tmp
    export GOAL_GPU_RANGE=${gpu_range}
    export GOAL_LOG_DIR=${out}
    export GOAL_EXPERIMENT_NAME=${name}
    export GOAL_CSD_ENABLED=${csd_enabled}
    export GOAL_CSD_BETA=${csd_beta}
    set -a
    source /data/minjaeoh/.env
    set +a
    exec "${ROOT}/.venv/bin/python" examples/embodiment/train_embodied_agent.py \
      --config-path "${ROOT}/examples/embodiment/config" \
      --config-name libero_goal_ppo_pi05_queued_100steps
  ) > "${out}/launcher.log" 2>&1
}

wait_for_long_training
run_phase standard standard_lane_45 standard_lane_67
run_phase plus plus_lane_45 plus_lane_67
log "All 20 requested evaluations completed; launching Goal PPO and PPO+CSD."
run_goal libero_goal_ppo_csd_b1_seed42_gpu45_gen10_exec5_100steps 4-5 true 1.0 &
goal_csd_pid=$!
run_goal libero_goal_ppo_seed42_gpu67_gen10_exec5_100steps 6-7 false 0.0 &
goal_ppo_pid=$!
printf '%s\n' "${goal_csd_pid}" > "${QUEUE_ROOT}/goal/csd.pid"
printf '%s\n' "${goal_ppo_pid}" > "${QUEUE_ROOT}/goal/ppo.pid"
wait "${goal_csd_pid}"
wait "${goal_ppo_pid}"
log "Goal PPO and PPO+CSD both finished."
