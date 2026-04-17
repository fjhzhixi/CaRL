python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
    --routes ${WORK_DIR}/custom_leaderboard/leaderboard/data/debug_routes_with_scenarios/route_Town03_00.xml.gz \
    --agent ${WORK_DIR}/team_code/env_agent.py --resume 0 --checkpoint ${WORK_DIR}/results/debug_camera/config.json \
    --track MAP --port 2000 --traffic-manager-port 8000 --agent-config ${WORK_DIR}/results/debug_camera/DD_PPO_debug_camera \
    --gym_port 5555 --debug 0 --repetitions 100 --frame_rate 10.0 \
    --no_rendering_mode False --timeout 900 --skip_next_route False --runtime_timeout 900 \
    "$@"
