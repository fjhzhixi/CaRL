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
  echo "Example: CARLA_ROOT=/path/to/CARLA_0_9_15 bash ${SCRIPT_DIR}/train_multi_gpu_multi_env_no_camera.sh"
  exit 1
fi

MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"
MLP_WORKER_GPU="${MLP_WORKER_GPU:-1}"
MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-127.0.0.1}"
MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-29500}"

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

GPU_IDS_ARR=()
for ((gpu_id = 0; gpu_id < MLP_WORKER_GPU; gpu_id++)); do
  GPU_IDS_ARR+=("${gpu_id}")
done

CMD_ARGS=(
  --config_file "${TRAIN_CONFIG}"
  --train_cpp 0
  --ml_cloud 0
  --num_nodes "${MLP_WORKER_NUM}"
  --node_id "${MLP_ROLE_INDEX}"
  --rdzv_addr "${MLP_WORKER_0_HOST}"
  --rdzv_port "${MLP_WORKER_0_PORT}"
  --git_root "${WORK_DIR}"
  --carla_root "${CARLA_ROOT}"
  --gpu_ids "${GPU_IDS_ARR[@]}"
)

append_if_set "EXP_NAME" "--exp_name"
append_if_set "SEED" "--seed"
append_if_set "START_PORT" "--start_port"
append_if_set "LOGDIR" "--logdir"
append_if_set "DEBUG_FLAG" "--debug"
append_if_set "NUM_ENVS_PER_GPU" "--num_envs_per_gpu"
append_if_set "NUM_ENVS_PER_NODE" "--num_envs_per_node"
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
echo "NUM_NODES=${MLP_WORKER_NUM}"
echo "GPUS_PER_NODE=${MLP_WORKER_GPU}"
echo "NODE_RANK=${MLP_ROLE_INDEX}"
echo "MASTER_ADDR=${MLP_WORKER_0_HOST}"
echo "MASTER_PORT=${MLP_WORKER_0_PORT}"
echo "GPU_IDS=${GPU_IDS_ARR[*]}"

python -u train_parallel.py "${CMD_ARGS[@]}"
