#!/usr/bin/env bash
# LIBERO-Spatial plain PPO, pi_RL/pi0.5 faithful, denoise 5 (previous run used 3).
set -Eeuo pipefail
ROOT=/data/minjaeoh/RLinf
LOG=$ROOT/logs/20260915-spatial-ppo-ns5-gpu03
mkdir -p "$LOG/train" /tmp/sp-ns5-ray /tmp/sp-ns5-tmp
cd "$ROOT"
printf '%s START spatial PPO ns5 on GPUs 0-3\n' "$(date '+%F %T %Z')" > "$LOG/state.log"

unset CUDA_VISIBLE_DEVICES RAY_ADDRESS
export RLINF_FORCE_LOCAL_RAY=1
export RLINF_LOCAL_RAY_NUM_CPUS=64
export RLINF_LOCAL_RAY_OBJECT_STORE_MEMORY_BYTES=32000000000
export REPO_PATH=$ROOT EMBODIED_PATH=$ROOT/examples/embodiment PYTHONPATH=$ROOT
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_TYPE=standard
export RAY_TMPDIR=/tmp/sp-ns5-ray TMPDIR=/tmp/sp-ns5-tmp
set -a; . /data/minjaeoh/.env; set +a

if "$ROOT/.venv/bin/python" examples/embodiment/train_embodied_agent.py \
  --config-path "$ROOT/examples/embodiment/config" \
  --config-name libero_spatial_ppo_pi05_ns5_20260915 \
  > "$LOG/train/launcher.log" 2>&1; then
  printf '%s DONE\n' "$(date '+%F %T %Z')" >> "$LOG/state.log"
else
  s=$?; printf '%s ERROR status=%s\n' "$(date '+%F %T %Z')" "$s" >> "$LOG/state.log"; exit "$s"
fi
