#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/data/minjaeoh/RLinf}
ENV_FILE=${ENV_FILE:-/data/minjaeoh/.env}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs}

SPATIAL_OUT=$LOG_ROOT/${RUN_ID}-gr00t-n15-spatial-bf16-100steps
GOAL_OUT=$LOG_ROOT/${RUN_ID}-gr00t-n15-goal-fp32head-100steps

if [[ -e "$SPATIAL_OUT" || -e "$GOAL_OUT" ]]; then
  echo "Refusing to overwrite an existing run directory for RUN_ID=$RUN_ID" >&2
  exit 1
fi

mkdir -p "$SPATIAL_OUT" "$GOAL_OUT"
SPATIAL_RAY_TMP=$(mktemp -d /data/minjaeoh/rs.XXXXXX)
SPATIAL_TMP=$(mktemp -d /data/minjaeoh/ts.XXXXXX)
GOAL_RAY_TMP=$(mktemp -d /data/minjaeoh/rg.XXXXXX)
GOAL_TMP=$(mktemp -d /data/minjaeoh/tg.XXXXXX)

run_spatial() (
  cd "$ROOT"
  unset RAY_ADDRESS
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export REPO_PATH="$ROOT"
  export EMBODIED_PATH="$ROOT/examples/embodiment"
  export PYTHONPATH="$ROOT"
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export RAY_TMPDIR="$SPATIAL_RAY_TMP"
  export TMPDIR="$SPATIAL_TMP"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export RLINF_FORCE_LOCAL_RAY=1

  printf '%s START spatial GR00T N1.5 bf16 on GPUs 0-3\n' "$(date '+%F %T %Z')" > "$SPATIAL_OUT/state.log"
  if "$ROOT/.venv-gr00t-n1d5/bin/python" examples/embodiment/train_embodied_agent.py \
    --config-path "$ROOT/examples/embodiment/config" \
    --config-name libero_spatial_ppo_gr00t_n15_bf16_100steps \
    runner.logger.log_path="$SPATIAL_OUT/train" \
    runner.logger.experiment_name="spatial_ppo_gr00t_n15_bf16_100steps_$RUN_ID"; then
    printf '%s DONE\n' "$(date '+%F %T %Z')" >> "$SPATIAL_OUT/state.log"
  else
    status=$?
    printf '%s ERROR status=%s\n' "$(date '+%F %T %Z')" "$status" >> "$SPATIAL_OUT/state.log"
    exit "$status"
  fi
)

run_goal() (
  cd "$ROOT"
  unset RAY_ADDRESS CUDA_VISIBLE_DEVICES
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  # Goal placement uses physical ranks 4-7. Do not remap them through CUDA_VISIBLE_DEVICES.
  unset CUDA_VISIBLE_DEVICES
  export REPO_PATH="$ROOT"
  export EMBODIED_PATH="$ROOT/examples/embodiment"
  export PYTHONPATH="$ROOT"
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export RAY_TMPDIR="$GOAL_RAY_TMP"
  export TMPDIR="$GOAL_TMP"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export RLINF_FORCE_LOCAL_RAY=1

  printf '%s START goal GR00T N1.5 fp32 action head on GPUs 4-7\n' "$(date '+%F %T %Z')" > "$GOAL_OUT/state.log"
  if "$ROOT/.venv-gr00t-n1d5/bin/python" examples/embodiment/train_embodied_agent.py \
    --config-path "$ROOT/examples/embodiment/config" \
    --config-name libero_goal_ppo_gr00t_n15_fp32head_100steps_20260926 \
    runner.logger.log_path="$GOAL_OUT/train" \
    runner.logger.experiment_name="goal_ppo_gr00t_n15_fp32head_100steps_$RUN_ID"; then
    printf '%s DONE\n' "$(date '+%F %T %Z')" >> "$GOAL_OUT/state.log"
  else
    status=$?
    printf '%s ERROR status=%s\n' "$(date '+%F %T %Z')" "$status" >> "$GOAL_OUT/state.log"
    exit "$status"
  fi
)

run_spatial > "$SPATIAL_OUT/launch.log" 2>&1 &
spatial_pid=$!
run_goal > "$GOAL_OUT/launch.log" 2>&1 &
goal_pid=$!

printf 'spatial_pid=%s log=%s\ngoal_pid=%s log=%s\n' \
  "$spatial_pid" "$SPATIAL_OUT/launch.log" "$goal_pid" "$GOAL_OUT/launch.log"

set +e
wait "$spatial_pid"
spatial_status=$?
wait "$goal_pid"
