#!/usr/bin/env bash
# Waits for the ns5 run to finish step 100 and exit, then launches the continuation.
# This watcher NEVER kills anything -- it only launches after the trainer is already gone.
set -uo pipefail
ROOT=/data/minjaeoh/RLinf
CKPT=$ROOT/logs/20260915-spatial-ppo-ns5-gpu03/train/pi05_ppo_libero_spatial_ns5_seed42/checkpoints/global_step_100
W=$ROOT/logs/20260915-spatial-ppo-ns5-gpu03/continue_watch.log
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >> "$W"; }

ckpt_ready(){
  [[ -d "$CKPT/actor/dcp_checkpoint" ]] || return 1
  local a b
  a=$(du -sk "$CKPT" 2>/dev/null | cut -f1); [[ -n "$a" && "$a" -ge 17825792 ]] || return 1
  sleep 45; b=$(du -sk "$CKPT" 2>/dev/null | cut -f1); [[ "$a" == "$b" ]]
}

log "watcher armed: waiting for step-100 checkpoint AND trainer exit (kills nothing)"
while :; do
  if ! pgrep -f "libero_spatial_ppo_pi05_ns5_20260915" >/dev/null 2>&1; then
    if ckpt_ready; then log "trainer exited, step-100 checkpoint complete -> launching continuation"; break; fi
    log "trainer gone but no complete step-100 checkpoint; NOT launching. Manual check needed."
    exit 1
  fi
  sleep 120
done
nohup "$ROOT/continue_spatial_ppo_ns5_to120.sh" >> "$W" 2>&1 &
log "continuation launched (pid $!)"
