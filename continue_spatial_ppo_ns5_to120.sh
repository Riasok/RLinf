#!/usr/bin/env bash
# Continue the Spatial PPO ns5 run from global_step_100 to 120.
# LR is constant (no scheduler), optimizer state restored from dcp_checkpoint,
# so the only discontinuity is the rollout RNG stream.
set -Eeuo pipefail
ROOT=/data/minjaeoh/RLinf
CKPT=$ROOT/logs/20260915-spatial-ppo-ns5-gpu03/train/pi05_ppo_libero_spatial_ns5_seed42/checkpoints/global_step_100
NEW=$ROOT/logs/20260915-spatial-ppo-ns5-gpu03-to120
mkdir -p "$NEW" /tmp/sp120-ray /tmp/sp120-tmp
cd "$ROOT"
[ -d "$CKPT/actor/dcp_checkpoint" ] || { echo "FATAL: no step-100 ckpt at $CKPT"; exit 1; }
printf '%s START continuation 100->120\n' "$(date '+%F %T %Z')" > "$NEW/state.log"

unset CUDA_VISIBLE_DEVICES RAY_ADDRESS
export RLINF_FORCE_LOCAL_RAY=1
export RLINF_LOCAL_RAY_NUM_CPUS=64
export RLINF_LOCAL_RAY_OBJECT_STORE_MEMORY_BYTES=32000000000
export REPO_PATH=$ROOT EMBODIED_PATH=$ROOT/examples/embodiment PYTHONPATH=$ROOT
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_TYPE=standard
export RAY_TMPDIR=/tmp/sp120-ray TMPDIR=/tmp/sp120-tmp
set -a; . /data/minjaeoh/.env; set +a

if "$ROOT/.venv/bin/python" examples/embodiment/train_embodied_agent.py \
  --config-path "$ROOT/examples/embodiment/config" \
  --config-name libero_spatial_ppo_pi05_ns5_20260915 \
  "runner.resume_dir=$CKPT" \
  "runner.max_steps=120" \
  "runner.logger.log_path=$NEW/train" \
  "runner.logger.experiment_name=pi05_ppo_libero_spatial_ns5_seed42_to120" \
  > "$NEW/train_launcher.log" 2>&1; then
  printf '%s DONE 120\n' "$(date '+%F %T %Z')" >> "$NEW/state.log"
else
  s=$?; printf '%s ERROR status=%s\n' "$(date '+%F %T %Z')" "$s" >> "$NEW/state.log"; exit "$s"
fi
