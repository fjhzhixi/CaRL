#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export WORK_DIR
export SCENARIO_RUNNER_ROOT="${SCENARIO_RUNNER_ROOT:-${WORK_DIR}/custom_leaderboard/scenario_runner}"
export LEADERBOARD_ROOT="${LEADERBOARD_ROOT:-${WORK_DIR}/custom_leaderboard/leaderboard}"

CARLA_ROOT="${CARLA_ROOT:-}"
if [[ -z "${CARLA_ROOT}" ]]; then
  echo "CARLA_ROOT is not set."
  echo "Example: CARLA_ROOT=/path/to/CARLA_0_9_15 bash ${SCRIPT_DIR}/train_single_gpu_multi_env_no_camera.sh"
  exit 1
fi

TRAIN_CONFIG="${TRAIN_CONFIG:-${SCRIPT_DIR}/configs/no_camera_train_default.json}"
if [[ ! -f "${TRAIN_CONFIG}" ]]; then
  echo "TRAIN_CONFIG does not exist: ${TRAIN_CONFIG}"
  exit 1
fi

export PYTHONPATH="${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla"

append_if_set() {
  local var_name="$1"
  local flag_name="$2"
  if [[ -n "${!var_name:-}" ]]; then
    CMD_ARGS+=("${flag_name}" "${!var_name}")
  fi
}

append_list_if_set() {
  local var_name="$1"
  local flag_name="$2"
  if [[ -n "${!var_name:-}" ]]; then
    local values=()
    read -r -a values <<< "${!var_name}"
    CMD_ARGS+=("${flag_name}" "${values[@]}")
  fi
}

GPU_ID="${GPU_ID:-0}"

CMD_ARGS=(
  --config_file "${TRAIN_CONFIG}"
  --train_cpp 0
  --ml_cloud 0
  --num_nodes 1
  --node_id 0
  --rdzv_addr 127.0.0.1
  --rdzv_port 0
  --git_root "${WORK_DIR}"
  --carla_root "${CARLA_ROOT}"
  --gpu_ids "${GPU_ID}"
)

append_if_set "EXP_NAME" "--exp_name"
append_if_set "SEED" "--seed"
append_if_set "START_PORT" "--start_port"
append_if_set "LOGDIR" "--logdir"
append_if_set "DEBUG_FLAG" "--debug"
append_if_set "NUM_ENVS_PER_GPU" "--num_envs_per_gpu"
append_if_set "TOTAL_TIMESTEPS" "--total_timesteps"
append_if_set "UPDATE_EPOCHS" "--update_epochs"
append_if_set "ROUTE_REPETITIONS" "--route_repetitions"
append_if_set "ROUTES_FOLDER" "--routes_folder"
append_if_set "ROLLOUT_STEPS_PER_ENV" "--rollout_steps_per_env"
append_if_set "MINIBATCHES_PER_UPDATE" "--minibatches_per_update"
append_if_set "REWARD_TYPE" "--reward_type"
append_if_set "USE_DD_PPO_PREEMPT" "--use_dd_ppo_preempt"
append_if_set "USE_BEV_INPUT" "--use_bev_input"
append_if_set "FRAME_RATE" "--frame_rate"
append_list_if_set "TRAIN_TOWNS" "--train_towns"

cd "${SCRIPT_DIR}"

echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "WORK_DIR=${WORK_DIR}"
echo "CARLA_ROOT=${CARLA_ROOT}"
echo "GPU_ID=${GPU_ID}"

python -u train_parallel.py "${CMD_ARGS[@]}"
