#!/usr/bin/env bash
# Resume the beta=1.0 PPO+CSD Spatial arm from global_step_60.
#
# The original run reached step 66 but save_interval=10, so step 60 is the last
# checkpoint; steps 61-66 are redone. Optimizer state comes from dcp_checkpoint.
#
# NOTE: this resume introduces a run boundary in the beta=1.0 arm -- rollout RNG and
# LR-schedule state restart. Same caveat that applies to the fragmented beta=0.5 arm.
# Report it as PPO-CSD(b=1.0) [0-60] + [60-100] rather than a clean single run.
#
# Edit GPUS below if 2-3 are still in use when you resume.
set -euo pipefail

GPUS="${GPUS:-2-3}"
CUDA_DEVS="${CUDA_DEVS:-2,3}"

ROOT=/data/minjaeoh/RLinf
CKPT="$ROOT/logs/20260825-libero_spatial_ppo_csd_b1_0_gpu23_100steps/libero_spatial_ppo_csd_b1_0_seed42_gpu23_100steps_20260825/checkpoints/global_step_60"
STAMP=$(date +%Y%m%d)
NEWLOG="$ROOT/logs/${STAMP}-libero_spatial_ppo_csd_b1_0_resume60"
EXPNAME="libero_spatial_ppo_csd_b1_0_seed42_resume60_${STAMP}"

[ -d "$CKPT" ] || { echo "FATAL: checkpoint not found: $CKPT"; exit 1; }

export CUDA_VISIBLE_DEVICES="$CUDA_DEVS"
export REPO_PATH="$ROOT"
export EMBODIED_PATH="$ROOT/examples/embodiment"
export PYTHONPATH="$ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/r23b1r}"
export TMPDIR="${TMPDIR:-/tmp/t23b1r}"
set -a; . /data/minjaeoh/.env; set +a
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$NEWLOG"

cd "$ROOT"
exec "$ROOT/.venv/bin/python" examples/embodiment/train_embodied_agent.py \
  --config-path "$ROOT/examples/embodiment/config" \
  --config-name libero_spatial_ppo_csd_pi05_gpu23_100steps \
  "~cluster.component_placement" \
  "+cluster.component_placement={actor:${GPUS},env:${GPUS},rollout:${GPUS}}" \
  "algorithm.csd_beta=1.0" \
  "runner.resume_dir=$CKPT" \
  "runner.logger.log_path=$NEWLOG" \
  "runner.logger.experiment_name=$EXPNAME" \
  > "$NEWLOG/launcher.log" 2>&1
