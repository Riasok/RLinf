#!/usr/bin/env bash
# Wait for both live runs to finish step 80 (+ checkpoint), stop them, launch the gated 4-GPU run.
set -uo pipefail

ROOT=/data/minjaeoh/RLinf
SW="${ROOT}/logs/stop_at_step80.log"
CSD_DIR="${ROOT}/logs/20260901-libero_40_ppo_csd_b1_gpu45_gen10_exec5"
DIAG_DIR="${ROOT}/logs/20260902-libero_40_ppo_csddiag_gpu67_gen10_exec5"
CSD_CKPT="${CSD_DIR}/libero_40_ppo_csd_b1_seed42_gpu45_gen10_exec5_20260901/checkpoints/global_step_80"
DIAG_CKPT="${DIAG_DIR}/libero_40_ppo_csddiag_seed42_gpu67_gen10_exec5_20260902/checkpoints/global_step_80"
TRAIN_PIDS=(1152383 1527649)
SHELL_PIDS=(917246 1152373 1527644)

log() { printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >> "${SW}"; }

# a checkpoint counts as complete when it exists, is >=17GiB, and stops growing
ckpt_ready() {
  local d=$1 s1 s2
  [[ -d "$d" ]] || return 1
  s1=$(du -sk "$d" 2>/dev/null | cut -f1); [[ -n "$s1" && "$s1" -ge 17825792 ]] || return 1
  sleep 45
  s2=$(du -sk "$d" 2>/dev/null | cut -f1)
  [[ "$s1" == "$s2" ]]
}

step_logged() { grep -q "Global Step:   80/100" "$1/metrics.log" 2>/dev/null; }

log "watcher started; waiting for step 80 on both runs"
while :; do
  if step_logged "${CSD_DIR}" && step_logged "${DIAG_DIR}" \
     && ckpt_ready "${CSD_CKPT}" && ckpt_ready "${DIAG_CKPT}"; then
    log "both runs reached step 80 with complete checkpoints"
    break
  fi
  sleep 60
done

log "stopping trainers: ${SHELL_PIDS[*]} ${TRAIN_PIDS[*]}"
kill -TERM "${SHELL_PIDS[@]}" "${TRAIN_PIDS[@]}" 2>/dev/null
sleep 90

# Only ever touch PIDs owned by this user that still hold memory on GPUs 4-7.
for round in 1 2 3; do
  mapfile -t stuck < <(
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null |
    while IFS=, read -r uuid pid; do
      idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep -F "${uuid// /}" | cut -d, -f1 | tr -d ' ')
      [[ "$idx" =~ ^[4-7]$ ]] || continue
      [[ "$(ps -o user= -p "${pid// /}" 2>/dev/null)" == "minjaeoh" ]] || continue
      echo "${pid// /}"
    done
  )
  (( ${#stuck[@]} == 0 )) && break
  log "round ${round}: killing leftover GPU4-7 pids: ${stuck[*]}"
  kill -KILL "${stuck[@]}" 2>/dev/null
  sleep 30
done

free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n '5,8p')
log "GPU 4-7 state after stop:"; printf '%s\n' "$free" >> "${SW}"

log "STOP-ONLY mode: not launching anything; eval queue will be started separately"
