#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/minjaeoh/RLinf
OUT_ROOT="${ROOT}/logs/20260901-libero-long-step90-pro-gpu67"
MODEL="${ROOT}/checkpoints/RLinf-Pi05-LIBERO-SFT"
CSD_CKPT="${ROOT}/logs/20260828-libero_long_ppo_csd_b1_gpu45_gen10_exec5/libero_long_ppo_csd_b1_seed42_gpu45_gen10_exec5_20260828/checkpoints/global_step_90/actor/model_state_dict/full_weights.pt"
PPO_CKPT="${ROOT}/logs/20260828-libero_long_ppo_gpu67_gen10_exec5/libero_long_ppo_seed42_gpu67_gen10_exec5_20260828/checkpoints/global_step_90/actor/model_state_dict/full_weights.pt"
STATE_LOG="${OUT_ROOT}/state.log"

mkdir -p "${OUT_ROOT}"
cd "${ROOT}"

log() {
  printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${STATE_LOG}"
}

run_pro_eval() {
  local name=$1
  local ckpt=$2
  local out="${OUT_ROOT}/${name}"
  local ray_tmp="/tmp/${name}-ray"
  local job_tmp="/tmp/${name}-tmp"

  mkdir -p "${out}" "${ray_tmp}" "${job_tmp}"
  log "START ${name}: physical GPUs 6-7, LIBERO-PRO libero_10, perturbations=all"
  (
    unset RAY_ADDRESS
    export CUDA_VISIBLE_DEVICES=6,7
    export RLINF_FORCE_LOCAL_RAY=1
    export REPO_PATH="${ROOT}"
    export EMBODIED_PATH="${ROOT}/examples/embodiment"
    export PYTHONPATH="${ROOT}"
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export LIBERO_TYPE=pro
    export LIBERO_PERTURBATION=all
    export RAY_TMPDIR="${ray_tmp}"
    export TMPDIR="${job_tmp}"
    export EVAL_GPU_RANGE=0-1
    export EVAL_LOG_DIR="${out}"
    export EVAL_NAME="${name}"
    export EVAL_SUITE=libero_10
    export EVAL_EPISODE_STEPS=520
    export EVAL_ROLLOUT_STEPS=520
    export EVAL_TRIALS_PER_TASK=50
    export EVAL_MODEL_PATH="${MODEL}"
    export EVAL_CKPT_PATH="${ckpt}"
    "${ROOT}/.venv/bin/python" evaluations/eval_embodied_agent.py --config-path "${ROOT}/evaluations/libero" --config-name libero_pi05_queued_eval
  ) > "${out}/launcher.log" 2>&1

  if grep -Eq "Falling back to 'libero'|Error executing job|Traceback|Exception occurred while|OutOfMemoryError" "${out}/launcher.log"; then
    log "ERROR ${name}: launcher reported a failure or standard-LIBERO fallback."
    return 1
  fi
  if ! grep -q "eval/success" "${out}/launcher.log"; then
    log "ERROR ${name}: no evaluation success metric was produced."
    return 1
  fi
  touch "${out}/DONE"
  log "DONE ${name}"
  sleep 20
}

for checkpoint in "${CSD_CKPT}" "${PPO_CKPT}"; do
  if [[ ! -s "${checkpoint}" ]]; then
    log "ERROR missing checkpoint: ${checkpoint}"
    exit 1
  fi
done

run_pro_eval csd_step90_libero_long_pro "${CSD_CKPT}"
run_pro_eval ppo_step90_libero_long_pro "${PPO_CKPT}"
log "All step-90 LIBERO-Long-PRO evaluations finished."
