python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --max_restarts=0 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 ${WORK_DIR}/team_code/dd_ppo.py \
    --num_envs_per_gpu 1 --use_dd_ppo_preempt False \
    --exp_name DD_PPO_debug_camera --tcp_store_port 7000 \
    --logdir ${WORK_DIR}/results/debug_camera/ \
    --total_batch_size 32 --total_minibatch_size 16 --update_epochs 3 \
    --total_timesteps 10000000 --reward_type simple_reward \
    --debug 1 --debug_type save --ports 5555 \
    --use_camera True --camera_mode front --camera_width 400 --camera_height 225
