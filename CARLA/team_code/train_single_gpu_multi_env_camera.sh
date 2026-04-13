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
  echo "Example: CARLA_ROOT=/path/to/CARLA_0_9_15 bash ${SCRIPT_DIR}/train_single_gpu_multi_env_camera.sh"
  exit 1
fi

export PYTHONPATH="${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla"

NUM_ENVS="${NUM_ENVS:-2}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
START_PORT="${START_PORT:-20000}"
EXP_NAME="${EXP_NAME:-DD_PPO_debug_camera_multi_env}"
LOGDIR="${LOGDIR:-${WORK_DIR}/results}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
ROLLOUT_STEPS_PER_ENV="${ROLLOUT_STEPS_PER_ENV:-32}"
MINIBATCHES_PER_UPDATE="${MINIBATCHES_PER_UPDATE:-2}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-3}"
ROUTE_REPETITIONS="${ROUTE_REPETITIONS:-100}"
ROUTES_FOLDER="${ROUTES_FOLDER:-debug_routes_with_scenarios}"
DEBUG_FLAG="${DEBUG_FLAG:-1}"

if (( NUM_ENVS < 1 )); then
  echo "NUM_ENVS must be >= 1"
  exit 1
fi

if (( MINIBATCHES_PER_UPDATE < 1 )); then
  echo "MINIBATCHES_PER_UPDATE must be >= 1"
  exit 1
fi

TOTAL_BATCH_SIZE=$((NUM_ENVS * ROLLOUT_STEPS_PER_ENV))
if (( TOTAL_BATCH_SIZE % MINIBATCHES_PER_UPDATE != 0 )); then
  echo "TOTAL_BATCH_SIZE (${TOTAL_BATCH_SIZE}) must be divisible by MINIBATCHES_PER_UPDATE (${MINIBATCHES_PER_UPDATE})"
  exit 1
fi
TOTAL_MINIBATCH_SIZE=$((TOTAL_BATCH_SIZE / MINIBATCHES_PER_UPDATE))

# The debug route folder only has one route per town, so we choose distinct towns by default.
AVAILABLE_TOWNS=(3 1 2 4 5 6 7 10 12 13 15)
if (( NUM_ENVS > ${#AVAILABLE_TOWNS[@]} )); then
  echo "NUM_ENVS=${NUM_ENVS} is too large for ROUTES_FOLDER=${ROUTES_FOLDER}."
  echo "Either lower NUM_ENVS or override ROUTES_FOLDER / TRAIN_TOWNS."
  exit 1
fi

if [[ -n "${TRAIN_TOWNS:-}" ]]; then
  read -r -a TRAIN_TOWNS_ARR <<< "${TRAIN_TOWNS}"
else
  TRAIN_TOWNS_ARR=("${AVAILABLE_TOWNS[@]:0:${NUM_ENVS}}")
fi

if (( ${#TRAIN_TOWNS_ARR[@]} != NUM_ENVS )); then
  echo "TRAIN_TOWNS must contain exactly NUM_ENVS entries."
  echo "TRAIN_TOWNS='3 1 2 4' NUM_ENVS=4 bash ${SCRIPT_DIR}/train_single_gpu_multi_env_camera.sh"
  exit 1
fi

cd "${SCRIPT_DIR}"

echo "WORK_DIR=${WORK_DIR}"
echo "CARLA_ROOT=${CARLA_ROOT}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "GPU_ID=${GPU_ID}"
echo "EXP_NAME=${EXP_NAME}"
echo "TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE}"
echo "TOTAL_MINIBATCH_SIZE=${TOTAL_MINIBATCH_SIZE}"
echo "TRAIN_TOWNS=${TRAIN_TOWNS_ARR[*]}"

python -u train_parallel.py \
  --train_cpp 0 \
  --ml_cloud 0 \
  --num_nodes 1 \
  --node_id 0 \
  --rdzv_addr 127.0.0.1 \
  --rdzv_port 0 \
  --git_root "${WORK_DIR}" \
  --carla_root "${CARLA_ROOT}" \
  --exp_name "${EXP_NAME}" \
  --seed "${SEED}" \
  --start_port "${START_PORT}" \
  --gpu_ids "${GPU_ID}" \
  --num_envs_per_gpu "${NUM_ENVS}" \
  --num_envs_per_node "${NUM_ENVS}" \
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
  --frame_rate 10.0 \
  --use_camera True \
  --camera_mode front \
  --camera_width 400 \
  --camera_height 225 \
  --logdir "${LOGDIR}"
