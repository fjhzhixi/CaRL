#!/bin/bash

export git_root=$1
export num_envs=$2
export num_nodes=$3
export node_rank=$4
export rdzv_addr=$5
export rdzv_port=$6

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8 # TODO tune
export MASTER_ADDR=${rdzv_addr}
export MASTER_PORT=${rdzv_port}
#export NCCL_BLOCKING_WAIT=1 # Experimental for debugging.
#export CUDA_LAUNCH_BLOCKING=1
torchrun --start-method spawn --nnodes=${num_nodes} --nproc_per_node=${num_envs} --node_rank=${node_rank} --master_addr=${rdzv_addr} --master_port=${rdzv_port} --max_restarts=0 ${git_root}/team_code/dd_ppo.py "${@:7}"