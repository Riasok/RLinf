#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/minjaeoh/RLinf
QUEUE_ROOT=${ROOT}/logs/20260904-step80-eval-ns3-then-csdgate
STATE_LOG=${QUEUE_ROOT}/queue_state.log
MODEL=${ROOT}/checkpoints/RLinf-Pi05-LIBERO-SFT
CSD_CKPT=${ROOT}/logs/20260901-libero_40_ppo_csd_b1_gpu45_gen10_exec5/libero_40_ppo_csd_b1_seed42_gpu45_gen10_exec5_20260901/checkpoints/global_step_80/actor/model_state_dict/full_weights.pt
DIAG_CKPT=${ROOT}/logs/20260902-libero_40_ppo_csddiag_gpu67_gen10_exec5/libero_40_ppo_csddiag_seed42_gpu67_gen10_exec5_20260902/checkpoints/global_step_80/actor/model_state_dict/full_weights.pt
NS=3   # denoising steps: match training (was 5 in the queued-eval config)

mkdir -p "${QUEUE_ROOT}"; cd "${ROOT}"
log() { printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${STATE_LOG}"; }

episode_steps() { case "$1" in
  libero_spatial) echo 240;; libero_object) echo 280;;
  libero_goal) echo 320;;    libero_10) echo 520;; *) return 2;; esac; }
plus_rollout_steps() { case "$1" in
  libero_spatial) echo 1200;; libero_object) echo 1680;;
  libero_goal) echo 1920;;    libero_10) echo 3120;; *) return 2;; esac; }

run_eval() {
  local mode=$1 name=$2 gpus=$3 suite=$4 ckpt=$5 envs=$6 epochs=$7
  local out=${QUEUE_ROOT}/${name}
  [[ -f "${out}/DONE" ]] && { log "SKIP ${name}"; return 0; }
  local rt=/tmp/ev${gpus//-/} jt=/tmp/ev${gpus//-/}-t
  rm -rf "${rt}"; mkdir -p "${out}" "${rt}" "${jt}"
  local ep rollout trials ltype
  ep=$(episode_steps "${suite}")
  if [[ "${mode}" == plus ]]; then
    rollout=$(plus_rollout_steps "${suite}"); trials=1; ltype=plus
  else
    rollout=${ep}; trials=50; ltype=standard
  fi
  log "START ${name}: GPUs ${gpus}, ${mode} ${suite}, ${envs}x${epochs}, num_steps=${NS}"
  (
    unset CUDA_VISIBLE_DEVICES RAY_ADDRESS
    export RLINF_FORCE_LOCAL_RAY=1
    export RLINF_LOCAL_RAY_NUM_CPUS=32
    export RLINF_LOCAL_RAY_OBJECT_STORE_MEMORY_BYTES=16000000000
    export REPO_PATH=${ROOT} EMBODIED_PATH=${ROOT}/examples/embodiment PYTHONPATH=${ROOT}
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export LIBERO_TYPE=${ltype} LIBERO_SUFFIX=all LIBERO_PERTURBATION=all
    export MAGICK_HOME=${ROOT}/.native-deps
    export LD_LIBRARY_PATH=${ROOT}/.native-deps/lib:${LD_LIBRARY_PATH:-}
    export RAY_TMPDIR=${rt} TMPDIR=${jt}
    export EVAL_GPU_RANGE=${gpus} EVAL_LOG_DIR=${out} EVAL_NAME=${name} EVAL_SUITE=${suite}
    export EVAL_EPISODE_STEPS=${ep} EVAL_ROLLOUT_STEPS=${rollout}
    export EVAL_TRIALS_PER_TASK=${trials} EVAL_TOTAL_ENVS=${envs} EVAL_ROLLOUT_EPOCH=${epochs}
    export EVAL_MODEL_PATH=${MODEL} EVAL_CKPT_PATH=${ckpt}
    "${ROOT}/.venv/bin/python" evaluations/eval_embodied_agent.py \
      --config-path "${ROOT}/evaluations/libero" \
      --config-name libero_pi05_queued_eval \
      rollout.model.num_steps=${NS} rollout.model.openpi.num_steps=${NS}
  ) > "${out}/launcher.log" 2>&1 || true
  if grep -Eq "Falling back to 'libero'|Error executing job|Traceback|Exception occurred while|OutOfMemoryError" "${out}/launcher.log" \
     || ! grep -q "eval/success" "${out}/launcher.log"; then
    log "FAIL ${name}"; return 1
  fi
  grep -h "eval/success" "${out}/launcher.log" | tail -1 | tee -a "${STATE_LOG}"
  touch "${out}/DONE"; log "DONE ${name}"; sleep 20
}

run_eval_retry() {
  local mode=$1 name=$2 gpus=$3 suite=$4 ckpt=$5 envs=$6 epochs=$7
  run_eval "$@" && return 0
  local half=$(( envs/2 )); (( half<1 )) && half=1
  log "RETRY ${name} at ${half} envs x $(( epochs*2 )) epochs"
  rm -f "${QUEUE_ROOT}/${name}/DONE"
  run_eval "$mode" "$name" "$gpus" "$suite" "$ckpt" "$half" "$(( epochs*2 ))" || { log "GAVE UP ${name}"; return 1; }
}

# lane A = GPUs 4-5 : base(spatial,object) then CSD ckpt
lane_a() {
  local rc=0
  for s in libero_spatial libero_object; do
    run_eval_retry standard "base_ns3_standard_${s}" 4-5 "${s}" "" 20 25 || rc=1
  done
  for s in libero_spatial libero_object libero_goal libero_10; do
    run_eval_retry standard "csd_step80_ns3_standard_${s}" 4-5 "${s}" "${CSD_CKPT}" 20 25 || rc=1
  done
  for s in libero_spatial libero_object libero_goal libero_10; do
    run_eval_retry plus "csd_step80_ns3_plus_${s}" 4-5 "${s}" "${CSD_CKPT}" 500 1 || rc=1
  done
  return ${rc}
}
# lane B = GPUs 6-7 : base(goal,long) then PPO/diag ckpt
lane_b() {
  local rc=0
  for s in libero_goal libero_10; do
    run_eval_retry standard "base_ns3_standard_${s}" 6-7 "${s}" "" 20 25 || rc=1
  done
  for s in libero_spatial libero_object libero_goal libero_10; do
    run_eval_retry standard "diag_step80_ns3_standard_${s}" 6-7 "${s}" "${DIAG_CKPT}" 20 25 || rc=1
  done
  for s in libero_spatial libero_object libero_goal libero_10; do
    run_eval_retry plus "diag_step80_ns3_plus_${s}" 6-7 "${s}" "${DIAG_CKPT}" 500 1 || rc=1
  done
  return ${rc}
}

for c in "${CSD_CKPT}" "${DIAG_CKPT}"; do
  [[ -s "$c" ]] || { log "ERROR missing checkpoint ${c}"; exit 1; }
done
log "starting: num_steps=${NS} (matches training). 4 base + 8 standard + 8 plus = 20 evals."

set +e
lane_a & a_pid=$!
lane_b & b_pid=$!
wait "${a_pid}"; a_rc=$?
wait "${b_pid}"; b_rc=$?
set -e
log "evaluation finished (laneA rc=${a_rc}, laneB rc=${b_rc})"
(( a_rc != 0 || b_rc != 0 )) && log "NOTE: some evals failed; launching training anyway as requested."

log "launching PPO+CSD positive_advantage-gated training on GPUs 4-7"
nohup "${ROOT}/run_libero_40_ppo_csdgate_padv_gpu4567.sh" >> "${STATE_LOG}" 2>&1 &
log "training launched (pid $!)"
