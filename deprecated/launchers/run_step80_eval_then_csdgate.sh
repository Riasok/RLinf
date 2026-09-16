#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/minjaeoh/RLinf
QUEUE_ROOT=${ROOT}/logs/20260904-step80-eval-then-csdgate
STATE_LOG=${QUEUE_ROOT}/queue_state.log
MODEL=${ROOT}/checkpoints/RLinf-Pi05-LIBERO-SFT
CSD_CKPT=${ROOT}/logs/20260901-libero_40_ppo_csd_b1_gpu45_gen10_exec5/libero_40_ppo_csd_b1_seed42_gpu45_gen10_exec5_20260901/checkpoints/global_step_80/actor/model_state_dict/full_weights.pt
DIAG_CKPT=${ROOT}/logs/20260902-libero_40_ppo_csddiag_gpu67_gen10_exec5/libero_40_ppo_csddiag_seed42_gpu67_gen10_exec5_20260902/checkpoints/global_step_80/actor/model_state_dict/full_weights.pt

mkdir -p "${QUEUE_ROOT}"
cd "${ROOT}"

log() { printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${STATE_LOG}"; }

episode_steps() {
  case "$1" in
    libero_spatial) echo 240 ;; libero_object) echo 280 ;;
    libero_goal)    echo 320 ;; libero_10)     echo 520 ;;
    *) return 2 ;;
  esac
}
plus_rollout_steps() {
  case "$1" in
    libero_spatial) echo 1200 ;; libero_object) echo 1680 ;;
    libero_goal)    echo 1920 ;; libero_10)    echo 3120 ;;
    *) return 2 ;;
  esac
}

# one eval; $6/$7 override envs/epochs so a failure can be retried smaller
run_eval() {
  local mode=$1 name=$2 gpus=$3 suite=$4 ckpt=$5 envs=$6 epochs=$7
  local out=${QUEUE_ROOT}/${name}
  [[ -f "${out}/DONE" ]] && { log "SKIP ${name} (already done)"; return 0; }
  local rt=/tmp/ev${gpus//-/} jt=/tmp/ev${gpus//-/}-t
  rm -rf "${rt}"; mkdir -p "${out}" "${rt}" "${jt}"
  local ep rollout trials ltype
  ep=$(episode_steps "${suite}")
  if [[ "${mode}" == plus ]]; then
    rollout=$(plus_rollout_steps "${suite}"); trials=1; ltype=plus
  else
    rollout=${ep}; trials=50; ltype=standard
  fi
  log "START ${name}: GPUs ${gpus}, ${mode} ${suite}, ${envs} envs x ${epochs} epoch(s)"
  (
    unset CUDA_VISIBLE_DEVICES RAY_ADDRESS
    export RLINF_FORCE_LOCAL_RAY=1
    export RLINF_LOCAL_RAY_NUM_CPUS=32
    export RLINF_LOCAL_RAY_OBJECT_STORE_MEMORY_BYTES=16000000000
    export REPO_PATH=${ROOT}
    export EMBODIED_PATH=${ROOT}/examples/embodiment
    export PYTHONPATH=${ROOT}
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export LIBERO_TYPE=${ltype}
    export LIBERO_SUFFIX=all
    export LIBERO_PERTURBATION=all
    export MAGICK_HOME=${ROOT}/.native-deps
    export LD_LIBRARY_PATH=${ROOT}/.native-deps/lib:${LD_LIBRARY_PATH:-}
    export RAY_TMPDIR=${rt}
    export TMPDIR=${jt}
    export EVAL_GPU_RANGE=${gpus}
    export EVAL_LOG_DIR=${out}
    export EVAL_NAME=${name}
    export EVAL_SUITE=${suite}
    export EVAL_EPISODE_STEPS=${ep}
    export EVAL_ROLLOUT_STEPS=${rollout}
    export EVAL_TRIALS_PER_TASK=${trials}
    export EVAL_TOTAL_ENVS=${envs}
    export EVAL_ROLLOUT_EPOCH=${epochs}
    export EVAL_MODEL_PATH=${MODEL}
    export EVAL_CKPT_PATH=${ckpt}
    "${ROOT}/.venv/bin/python" evaluations/eval_embodied_agent.py \
      --config-path "${ROOT}/evaluations/libero" \
      --config-name libero_pi05_queued_eval
  ) > "${out}/launcher.log" 2>&1 || true

  if grep -Eq "Falling back to 'libero'|Error executing job|Traceback|Exception occurred while|OutOfMemoryError" "${out}/launcher.log" \
     || ! grep -q "eval/success" "${out}/launcher.log"; then
    log "FAIL ${name}"
    return 1
  fi
  grep -h "eval/success" "${out}/launcher.log" | tail -2 | tee -a "${STATE_LOG}"
  touch "${out}/DONE"; log "DONE ${name}"; sleep 20
}

# fast settings first; on failure retry once at half parallelism
run_eval_retry() {
  local mode=$1 name=$2 gpus=$3 suite=$4 ckpt=$5 envs=$6 epochs=$7
  if run_eval "$mode" "$name" "$gpus" "$suite" "$ckpt" "$envs" "$epochs"; then return 0; fi
  local half=$(( envs / 2 )); (( half < 1 )) && half=1
  local more=$(( epochs * 2 ))
  log "RETRY ${name} at ${half} envs x ${more} epochs"
  rm -f "${QUEUE_ROOT}/${name}/DONE"
  run_eval "$mode" "$name" "$gpus" "$suite" "$ckpt" "$half" "$more" || { log "GAVE UP ${name}"; return 1; }
}

lane() {
  local tag=$1 gpus=$2 ckpt=$3 rc=0
  for suite in libero_spatial libero_object libero_goal libero_10; do
    run_eval_retry standard "${tag}_step80_standard_${suite}" "${gpus}" "${suite}" "${ckpt}" 20 25 || rc=1
  done
  for suite in libero_spatial libero_object libero_goal libero_10; do
    run_eval_retry plus "${tag}_step80_plus_${suite}" "${gpus}" "${suite}" "${ckpt}" 500 1 || rc=1
  done
  return ${rc}
}

log "waiting for both step-80 trainers to be stopped"
while pgrep -f 'config-name libero_40_ppo_csd_b1_pi05_gpu45_gen10_exec5_100steps|config-name libero_40_ppo_pi05_gpu67_gen10_exec5_100steps' >/dev/null; do
  sleep 30
done
for c in "${CSD_CKPT}" "${DIAG_CKPT}"; do
  [[ -s "$c" ]] || { log "ERROR missing checkpoint ${c}"; exit 1; }
done
log "trainers stopped; both step-80 checkpoints present. Starting 16 evaluations."

set +e
lane csd  4-5 "${CSD_CKPT}" & csd_pid=$!
lane diag 6-7 "${DIAG_CKPT}" & diag_pid=$!
wait "${csd_pid}"; csd_rc=$?
wait "${diag_pid}"; diag_rc=$?
set -e
log "evaluation phase finished (csd lane rc=${csd_rc}, diag lane rc=${diag_rc})"
if (( csd_rc != 0 || diag_rc != 0 )); then
  log "NOTE: some evaluations failed; launching training anyway as requested. Failures are listed above."
fi

log "launching PPO+CSD positive_advantage-gated training on GPUs 4-7"
nohup "${ROOT}/run_libero_40_ppo_csdgate_padv_gpu4567.sh" >> "${STATE_LOG}" 2>&1 &
log "training launched (pid $!)"
