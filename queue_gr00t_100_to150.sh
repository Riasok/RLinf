#!/usr/bin/env bash
# Wait for the active 100-step GR00T runs, then resume each independently to 150.
set -Eeuo pipefail

ROOT=${ROOT:-/data/minjaeoh/RLinf}
ENV_FILE=${ENV_FILE:-/data/minjaeoh/.env}

SPATIAL_RUN=$ROOT/logs/20260926-gr00t-n15-spatial-ppo-bf16-100steps-parallel
SPATIAL_EXP=spatial_ppo_gr00t_n15_bf16_100steps_20260926
SPATIAL_CKPT=$SPATIAL_RUN/train/$SPATIAL_EXP/checkpoints/global_step_100
SPATIAL_OUT=$ROOT/logs/20260927-gr00t-n15-spatial-resume100-to150

GOAL_RUN=$ROOT/logs/20260926-gr00t-n15-goal-ppo-fp32head-100steps-parallel
GOAL_EXP=goal_ppo_gr00t_n15_fp32head_100steps_20260926
GOAL_CKPT=$GOAL_RUN/train/$GOAL_EXP/checkpoints/global_step_100
GOAL_OUT=$ROOT/logs/20260927-gr00t-n15-goal-resume100-to150

wait_for_step_100() {
  local label=$1 state_file=$2 checkpoint=$3 queue_state=$4
  while true; do
    local last_line
    last_line=$(tail -n 1 "$state_file" 2>/dev/null || true)
    case "$last_line" in
      *" DONE")
        if [[ -f "$checkpoint/actor/dcp_checkpoint/.metadata" && -f "$checkpoint/actor/model_state_dict/full_weights.pt" ]]; then
          printf '%s %s step 100 complete; continuation starting\n' "$(date '+%F %T %Z')" "$label" >> "$queue_state"
          return 0
        fi
        printf '%s ERROR: %s finished without a complete step-100 checkpoint\n' "$(date '+%F %T %Z')" "$label" >> "$queue_state"
        return 1
        ;;
      *" ERROR status="*)
        printf '%s ERROR: %s source run failed: %s\n' "$(date '+%F %T %Z')" "$label" "$last_line" >> "$queue_state"
        return 1
        ;;
      *) sleep 300 ;;
    esac
  done
}

continue_spatial() (
  mkdir -p "$SPATIAL_OUT"
  local state=$SPATIAL_OUT/state.log
  printf '%s QUEUED spatial step 100 -> 150\n' "$(date '+%F %T %Z')" > "$state"
  wait_for_step_100 spatial "$SPATIAL_RUN/state.log" "$SPATIAL_CKPT" "$state"
  local ray_tmp work_tmp
  ray_tmp=$(mktemp -d /data/minjaeoh/rs.XXXXXX)
  work_tmp=$(mktemp -d /data/minjaeoh/ts.XXXXXX)
  cd "$ROOT"
  unset RAY_ADDRESS
  set -a; source "$ENV_FILE"; set +a
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export REPO_PATH="$ROOT" EMBODIED_PATH="$ROOT/examples/embodiment" PYTHONPATH="$ROOT"
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl RAY_TMPDIR="$ray_tmp" TMPDIR="$work_tmp"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RLINF_FORCE_LOCAL_RAY=1
  printf '%s START spatial continuation 100 -> 150\n' "$(date '+%F %T %Z')" >> "$state"
  if "$ROOT/.venv-gr00t-n1d5/bin/python" examples/embodiment/train_embodied_agent.py \
    --config-path "$ROOT/examples/embodiment/config" \
    --config-name libero_spatial_ppo_gr00t_n15_bf16_150steps \
    runner.resume_dir="$SPATIAL_CKPT" runner.max_steps=150 runner.save_interval=10 \
    runner.logger.log_path="$SPATIAL_OUT/train" \
    runner.logger.experiment_name=spatial_ppo_gr00t_n15_bf16_resume100_to150 \
    > "$SPATIAL_OUT/launch.log" 2>&1; then
    printf '%s DONE spatial at step 150\n' "$(date '+%F %T %Z')" >> "$state"
  else
    local status=$?
    printf '%s ERROR status=%s\n' "$(date '+%F %T %Z')" "$status" >> "$state"
    exit "$status"
  fi
)

continue_goal() (
  mkdir -p "$GOAL_OUT"
  local state=$GOAL_OUT/state.log
  printf '%s QUEUED goal step 100 -> 150\n' "$(date '+%F %T %Z')" > "$state"
  wait_for_step_100 goal "$GOAL_RUN/state.log" "$GOAL_CKPT" "$state"
  local ray_tmp work_tmp
  ray_tmp=$(mktemp -d /data/minjaeoh/rg.XXXXXX)
  work_tmp=$(mktemp -d /data/minjaeoh/tg.XXXXXX)
  cd "$ROOT"
  unset RAY_ADDRESS CUDA_VISIBLE_DEVICES
  set -a; source "$ENV_FILE"; set +a
  unset CUDA_VISIBLE_DEVICES
  export REPO_PATH="$ROOT" EMBODIED_PATH="$ROOT/examples/embodiment" PYTHONPATH="$ROOT"
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl RAY_TMPDIR="$ray_tmp" TMPDIR="$work_tmp"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RLINF_FORCE_LOCAL_RAY=1
  printf '%s START goal continuation 100 -> 150\n' "$(date '+%F %T %Z')" >> "$state"
  if "$ROOT/.venv-gr00t-n1d5/bin/python" examples/embodiment/train_embodied_agent.py \
    --config-path "$ROOT/examples/embodiment/config" \
    --config-name libero_goal_ppo_gr00t_n15_fp32head_150steps \
    runner.resume_dir="$GOAL_CKPT" runner.max_steps=150 runner.save_interval=10 \
    runner.logger.log_path="$GOAL_OUT/train" \
    runner.logger.experiment_name=goal_ppo_gr00t_n15_fp32head_resume100_to150 \
    > "$GOAL_OUT/launch.log" 2>&1; then
    printf '%s DONE goal at step 150\n' "$(date '+%F %T %Z')" >> "$state"
  else
    local status=$?
    printf '%s ERROR status=%s\n' "$(date '+%F %T %Z')" "$status" >> "$state"
    exit "$status"
  fi
)

continue_spatial & spatial_pid=$!
continue_goal & goal_pid=$!
set +e
wait "$spatial_pid"; spatial_status=$?
wait "$goal_pid"; goal_status=$?
set -e
((spatial_status == 0 && goal_status == 0))
