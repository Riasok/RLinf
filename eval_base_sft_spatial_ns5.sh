#!/usr/bin/env bash
# Base few-shot SFT (RLinf-Pi05-LIBERO-SFT), LIBERO-Spatial, all 10 tasks,
# 50 trials/task = 500 episodes, 5-step denoising, noise_level 0.3 (canonical eval).
set -Eeuo pipefail
ROOT=/data/minjaeoh/RLinf
OUT=$ROOT/logs/20260915-base-sft-spatial-ns5
mkdir -p "$OUT" /tmp/bsft-ray /tmp/bsft-tmp
rm -rf /tmp/bsft-ray; mkdir -p /tmp/bsft-ray
cd "$ROOT"
printf '%s START base SFT spatial ns5\n' "$(date '+%F %T %Z')" > "$OUT/state.log"

unset CUDA_VISIBLE_DEVICES RAY_ADDRESS
export RLINF_FORCE_LOCAL_RAY=1
export RLINF_LOCAL_RAY_NUM_CPUS=48
export RLINF_LOCAL_RAY_OBJECT_STORE_MEMORY_BYTES=24000000000
export REPO_PATH=$ROOT EMBODIED_PATH=$ROOT/examples/embodiment PYTHONPATH=$ROOT
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_TYPE=standard
export RAY_TMPDIR=/tmp/bsft-ray TMPDIR=/tmp/bsft-tmp
export EVAL_GPU_RANGE=0-3
export EVAL_LOG_DIR=$OUT EVAL_NAME=base_sft_spatial_ns5
export EVAL_SUITE=libero_spatial
export EVAL_EPISODE_STEPS=240 EVAL_ROLLOUT_STEPS=240
export EVAL_TRIALS_PER_TASK=50 EVAL_TOTAL_ENVS=20 EVAL_ROLLOUT_EPOCH=25
export EVAL_MODEL_PATH=$ROOT/checkpoints/RLinf-Pi05-LIBERO-SFT
export EVAL_CKPT_PATH=""
set -a; . /data/minjaeoh/.env; set +a

if "$ROOT/.venv/bin/python" evaluations/eval_embodied_agent.py \
  --config-path "$ROOT/evaluations/libero" \
  --config-name libero_pi05_queued_eval \
  rollout.model.num_steps=5 rollout.model.openpi.num_steps=5 \
  > "$OUT/launcher.log" 2>&1; then
  printf '%s DONE\n' "$(date '+%F %T %Z')" >> "$OUT/state.log"
  grep -h "eval/success" "$OUT/launcher.log" | tail -1 >> "$OUT/state.log"
else
  s=$?; printf '%s ERROR status=%s\n' "$(date '+%F %T %Z')" "$s" >> "$OUT/state.log"; exit "$s"
fi
