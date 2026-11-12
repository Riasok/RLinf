#!/usr/bin/env bash
# usage: run_eval_ckpt.sh <name> <base_model_pt> <out_dir> [rl_full_weights_pt] [gpu_range=6-7]
# LIBERO-Spatial, 10 tasks x 50 trials = 500 episodes, ODE sampling (mode=eval), horizon 240.
set -Eeuo pipefail
NAME=${1:?name}; MODEL=${2:?model .pt}; OUT=${3:?out dir}; CKPT=${4:-}; GPUS=${5:-6-7}; CHUNK=4; NS=5
ROOT=/data/minjaeoh/RLinf
mkdir -p "$OUT" /data/minjaeoh/raytmp/vlanext-eval-ray /data/minjaeoh/raytmp/vlanext-eval-tmp
cd "$ROOT"
printf '%s START eval %s model=%s chunk=%s ns=%s gpus=%s\n' "$(date '+%F %T %Z')" "$NAME" "$MODEL" "$CHUNK" "$NS" "$GPUS" >> "$OUT/state.log"
unset CUDA_VISIBLE_DEVICES RAY_ADDRESS
export RLINF_FORCE_LOCAL_RAY=1 RLINF_LOCAL_RAY_NUM_CPUS=48 RLINF_LOCAL_RAY_OBJECT_STORE_MEMORY_BYTES=24000000000
export REPO_PATH=$ROOT EMBODIED_PATH=$ROOT/examples/embodiment PYTHONPATH=$ROOT:$ROOT/third_party/VLANeXt
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_TYPE=standard
export RAY_TMPDIR=/data/minjaeoh/raytmp/vlanext-eval-ray TMPDIR=/data/minjaeoh/raytmp/vlanext-eval-tmp
export HF_HOME=/data/minjaeoh/.cache/huggingface TOKENIZERS_PARALLELISM=false
export EVAL_GPU_RANGE=$GPUS EVAL_LOG_DIR=$OUT EVAL_NAME=$NAME EVAL_SUITE=libero_spatial
export EVAL_EPISODE_STEPS=240 EVAL_ROLLOUT_STEPS=240
export EVAL_TRIALS_PER_TASK=50 EVAL_TOTAL_ENVS=20 EVAL_ROLLOUT_EPOCH=25
export EVAL_MODEL_PATH=$MODEL EVAL_CKPT_PATH="$CKPT" EVAL_ACTION_CHUNK=$CHUNK EVAL_NUM_STEPS=$NS
set -a; . /data/minjaeoh/.env; set +a
if "$ROOT/.venv-vlanext-rl/bin/python" evaluations/eval_embodied_agent.py \
  --config-path "$ROOT/evaluations/libero" --config-name libero_vlanext_queued_eval \
  > "$OUT/launcher.log" 2>&1; then
  printf '%s DONE %s\n' "$(date '+%F %T %Z')" "$NAME" >> "$OUT/state.log"
  grep -h "eval/success" "$OUT/launcher.log" | tail -2 >> "$OUT/state.log"
else
  s=$?; printf '%s ERROR %s status=%s\n' "$(date '+%F %T %Z')" "$NAME" "$s" >> "$OUT/state.log"; exit "$s"
fi
