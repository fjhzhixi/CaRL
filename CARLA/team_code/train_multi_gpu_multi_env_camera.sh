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
  echo "Example: CARLA_ROOT=/path/to/CARLA_0_9_15 bash ${SCRIPT_DIR}/train_multi_gpu_multi_env_camera.sh"
  exit 1
fi

for required_var in MLP_WORKER_NUM MLP_WORKER_GPU MLP_ROLE_INDEX MLP_WORKER_0_HOST MLP_WORKER_0_PORT; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "${required_var} is not set."
    exit 1
  fi
done

export PYTHONPATH="${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla"

NUM_NODES="${MLP_WORKER_NUM}"
GPUS_PER_NODE="${MLP_WORKER_GPU}"
NODE_RANK="${MLP_ROLE_INDEX}"
MASTER_ADDR="${MLP_WORKER_0_HOST}"
MASTER_PORT="${MLP_WORKER_0_PORT}"

NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU:-1}"
NUM_ENVS_PER_NODE=$((GPUS_PER_NODE * NUM_ENVS_PER_GPU))
SEED="${SEED:-0}"
START_PORT="${START_PORT:-20000}"
EXP_NAME="${EXP_NAME:-DD_PPO_camera_multi_gpu}"
LOGDIR="${LOGDIR:-${WORK_DIR}/results}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
ROLLOUT_STEPS_PER_ENV="${ROLLOUT_STEPS_PER_ENV:-32}"
MINIBATCHES_PER_UPDATE="${MINIBATCHES_PER_UPDATE:-2}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-3}"
ROUTE_REPETITIONS="${ROUTE_REPETITIONS:-20}"
ROUTES_FOLDER="${ROUTES_FOLDER:-1000_meters_old_scenarios_01}"
DEBUG_FLAG="${DEBUG_FLAG:-1}"
CAMERA_MODE="${CAMERA_MODE:-front}"
CAMERA_WIDTH="${CAMERA_WIDTH:-400}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-225}"
FRAME_RATE="${FRAME_RATE:-10.0}"

if (( NUM_ENVS_PER_GPU < 1 )); then
  echo "NUM_ENVS_PER_GPU must be >= 1"
  exit 1
fi

if (( MINIBATCHES_PER_UPDATE < 1 )); then
  echo "MINIBATCHES_PER_UPDATE must be >= 1"
  exit 1
fi

TOTAL_ENVS_GLOBAL=$((NUM_NODES * NUM_ENVS_PER_NODE))
TOTAL_BATCH_SIZE=$((TOTAL_ENVS_GLOBAL * ROLLOUT_STEPS_PER_ENV))
if (( TOTAL_BATCH_SIZE % MINIBATCHES_PER_UPDATE != 0 )); then
  echo "TOTAL_BATCH_SIZE (${TOTAL_BATCH_SIZE}) must be divisible by MINIBATCHES_PER_UPDATE (${MINIBATCHES_PER_UPDATE})"
  exit 1
fi
TOTAL_MINIBATCH_SIZE=$((TOTAL_BATCH_SIZE / MINIBATCHES_PER_UPDATE))

GPU_IDS_ARR=()
for ((gpu_id = 0; gpu_id < GPUS_PER_NODE; gpu_id++)); do
  GPU_IDS_ARR+=("${gpu_id}")
done

# Default to a route folder with many routes, then distribute towns round-robin per node.
AVAILABLE_TOWNS=(1 2 3 4 5 6 7 10)
if [[ -n "${TRAIN_TOWNS:-}" ]]; then
  read -r -a TRAIN_TOWNS_ARR <<< "${TRAIN_TOWNS}"
else
  TRAIN_TOWNS_ARR=()
  for ((env_idx = 0; env_idx < NUM_ENVS_PER_NODE; env_idx++)); do
    town_idx=$(((NODE_RANK * NUM_ENVS_PER_NODE + env_idx) % ${#AVAILABLE_TOWNS[@]}))
    TRAIN_TOWNS_ARR+=("${AVAILABLE_TOWNS[town_idx]}")
  done
fi

if (( ${#TRAIN_TOWNS_ARR[@]} != NUM_ENVS_PER_NODE )); then
  echo "TRAIN_TOWNS must contain exactly NUM_ENVS_PER_NODE entries (${NUM_ENVS_PER_NODE})."
  exit 1
fi

cd "${SCRIPT_DIR}"

echo "WORK_DIR=${WORK_DIR}"
echo "CARLA_ROOT=${CARLA_ROOT}"
echo "NUM_NODES=${NUM_NODES}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NUM_ENVS_PER_GPU=${NUM_ENVS_PER_GPU}"
echo "NUM_ENVS_PER_NODE=${NUM_ENVS_PER_NODE}"
echo "TOTAL_ENVS_GLOBAL=${TOTAL_ENVS_GLOBAL}"
echo "TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE}"
echo "TOTAL_MINIBATCH_SIZE=${TOTAL_MINIBATCH_SIZE}"
echo "GPU_IDS=${GPU_IDS_ARR[*]}"
echo "TRAIN_TOWNS=${TRAIN_TOWNS_ARR[*]}"

python -u train_parallel.py \
  --train_cpp 0 \
  --ml_cloud 0 \
  --num_nodes "${NUM_NODES}" \
  --node_id "${NODE_RANK}" \
  --rdzv_addr "${MASTER_ADDR}" \
  --rdzv_port "${MASTER_PORT}" \
  --git_root "${WORK_DIR}" \
  --carla_root "${CARLA_ROOT}" \
  --exp_name "${EXP_NAME}" \
  --seed "${SEED}" \
  --start_port "${START_PORT}" \
  --gpu_ids "${GPU_IDS_ARR[@]}" \
  --num_envs_per_gpu "${NUM_ENVS_PER_GPU}" \
  --num_envs_per_node "${NUM_ENVS_PER_NODE}" \
  --train_towns "${TRAIN_TOWNS_ARR[@]}" \
  --routes_folder "${ROUTES_FOLDER}" \
  --route_repetitions "${ROUTE_REPETITIONS}" \
  --use_dd_ppo_preempt False \
  --total_batch_size "${TOTAL_BATCH_SIZE}" \
  --total_minibatch_size "${TOTAL_MINIBATCH_SIZE}" \
  --update_epochs "${UPDATE_EPOCHS}" \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --reward_type simple_reward \
  --debug "${DEBUG_FLAG}" \
  --debug_type save \
  --track 0 \
  --frame_rate "${FRAME_RATE}" \
  --use_camera True \
  --camera_mode "${CAMERA_MODE}" \
  --camera_width "${CAMERA_WIDTH}" \
  --camera_height "${CAMERA_HEIGHT}" \
  --logdir "${LOGDIR}"
