#!/usr/bin/env bash
# Resume the positive_advantage-gated PPO+CSD run from global_step_10.
#
# The original run (started 2026-09-05 02:58) reached step 18 but was killed by an
# external SIGTERM at 17:52. save_interval=10, so step 10 is the last checkpoint;
# steps 11-18 are redone. Optimizer state comes from dcp_checkpoint.
#
# NOTE: this introduces a run boundary -- rollout RNG and LR-schedule state restart.
# Report as PPO-CSD-gated(padv) [0-10] + [10-100], not a clean single run.
set -Eeuo pipefail

ROOT=/data/minjaeoh/RLinf
GPUS="${GPUS:-4-5}"
CKPT="$ROOT/logs/20260905-libero_40_ppo_csdgate_padv_gpu45_gen10_exec5/libero_40_ppo_csdgate_padv_seed42_gpu45_gen10_exec5_20260905/checkpoints/global_step_10"
NEWLOG="$ROOT/logs/20260905-libero_40_ppo_csdgate_padv_gpu45_resume10"
EXPNAME="libero_40_ppo_csdgate_padv_seed42_gpu45_resume10_20260905"
STATE_LOG="$NEWLOG/state.log"

[ -d "$CKPT/actor/dcp_checkpoint" ] || { echo "FATAL: checkpoint incomplete: $CKPT"; exit 1; }
mkdir -p "$NEWLOG" /tmp/csdg45r-ray /tmp/csdg45r-tmp
cd "$ROOT"
printf '%s START resume padv-gated CSD from step 10 on GPUs %s\n' "$(date '+%F %T %Z')" "$GPUS" > "$STATE_LOG"

unset CUDA_VISIBLE_DEVICES RAY_ADDRESS
export RLINF_FORCE_LOCAL_RAY=1
export RLINF_LOCAL_RAY_NUM_CPUS=64
export RLINF_LOCAL_RAY_OBJECT_STORE_MEMORY_BYTES=32000000000
export REPO_PATH="$ROOT" EMBODIED_PATH="$ROOT/examples/embodiment" PYTHONPATH="$ROOT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_TYPE=standard
export RAY_TMPDIR=/tmp/csdg45r-ray TMPDIR=/tmp/csdg45r-tmp
set -a; . /data/minjaeoh/.env; set +a

if "$ROOT/.venv/bin/python" examples/embodiment/train_embodied_agent.py \
  --config-path "$ROOT/examples/embodiment/config" \
  --config-name libero_40_ppo_csdgate_padv_pi05_gpu45_gen10_exec5_100steps \
  "runner.resume_dir=$CKPT" \
  "runner.logger.log_path=$NEWLOG" \
  "runner.logger.experiment_name=$EXPNAME" \
  > "$NEWLOG/launcher.log" 2>&1; then
  printf '%s DONE resume padv-gated CSD\n' "$(date '+%F %T %Z')" >> "$STATE_LOG"
else
  s=$?; printf '%s ERROR resume padv-gated CSD status=%s\n' "$(date '+%F %T %Z')" "$s" >> "$STATE_LOG"; exit "$s"
fi
