'''
Script that statrs n carla servers, n carla leaderboard clients and a PPO training with them.
'''

import subprocess
import time
import sys
import shlex
import psutil
import argparse
import os
import re
import socket
import json
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

from rl_config import GlobalConfig

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)


def strtobool(v):
  return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')


def get_unknown_arg_value(argv, flag, default=None):
  for idx, token in enumerate(argv):
    if token == flag:
      if idx + 1 < len(argv) and not argv[idx + 1].startswith('--'):
        return argv[idx + 1]
      return True
  return default


def cli_flag_present(argv, flag):
  return flag in argv


def strip_flag_with_value(argv, flag):
  stripped = []
  idx = 0
  while idx < len(argv):
    token = argv[idx]
    if token == flag:
      idx += 1
      if idx < len(argv) and not argv[idx].startswith('--'):
        idx += 1
      continue
    stripped.append(token)
    idx += 1
  return stripped


def load_grouped_config(config_file, multi_gpu_mode):
  if config_file is None:
    return {}

  with open(config_file, 'rt', encoding='utf-8') as f:
    config = json.load(f)

  flat = {}
  for section_name, section_values in config.items():
    if section_name == 'distributed':
      continue
    if isinstance(section_values, dict):
      flat.update(section_values)

  distributed = config.get('distributed', {})
  if multi_gpu_mode and 'multi_gpu_default_routes_folder' in distributed:
    flat['routes_folder'] = distributed['multi_gpu_default_routes_folder']

  return flat


def build_cli_args_from_config(config_values, explicit_cli_flags):
  cli_args = []
  for key, value in config_values.items():
    flag = f'--{key}'
    if flag in explicit_cli_flags:
      continue
    if value is None:
      continue

    cli_args.append(flag)
    if isinstance(value, bool):
      cli_args.append('True' if value else 'False')
    elif isinstance(value, (list, tuple)):
      if len(value) == 0:
        cli_args.pop()
        continue
      cli_args.extend(str(v) for v in value)
    else:
      cli_args.append(str(value))

  return cli_args


def write_bootstrap_config(exp_folder, unknown, config_overrides):
  """Write a minimal config.json so env_agent.sensors() sees camera settings before ZMQ config sync."""
  config = GlobalConfig()

  def get_effective_value(flag, default):
    cli_value = get_unknown_arg_value(unknown, flag, None)
    if cli_value is not None:
      return cli_value
    return config_overrides.get(flag[2:], default)

  camera_fields = {
      'use_camera': bool(strtobool(get_effective_value('--use_camera', config.use_camera))),
      'camera_mode': get_effective_value('--camera_mode', config.camera_mode),
      'camera_width': int(get_effective_value('--camera_width', config.camera_width)),
      'camera_height': int(get_effective_value('--camera_height', config.camera_height)),
      'camera_fov': int(get_effective_value('--camera_fov', config.camera_fov)),
      'camera_features_dim': int(get_effective_value('--camera_features_dim', config.camera_features_dim)),
      'camera_encoder': get_effective_value('--camera_encoder', config.camera_encoder),
  }
  config.initialize(**camera_fields)
  with open(os.path.join(exp_folder, 'config.json'), 'wt', encoding='utf-8') as f:
    f.write(jsonpickle.encode(config))


def get_carla_gpu_assignments(num_envs_per_node, gpu_ids, num_envs_per_gpu):
  if num_envs_per_gpu <= 0:
    raise ValueError('--num_envs_per_gpu must be positive.')
  if len(gpu_ids) == 0:
    raise ValueError('--gpu_ids must contain at least one GPU id.')

  return [gpu_ids[(env_idx // num_envs_per_gpu) % len(gpu_ids)] for env_idx in range(num_envs_per_node)]


def print_effective_config(summary):
  print('Effective training config:')
  print(json.dumps(summary, indent=2, sort_keys=True))


def expand_town_ids(train_towns, target_length):
  if target_length <= 0:
    return tuple()
  if len(train_towns) == 0:
    raise ValueError('train_towns must contain at least one town id.')

  normalized_towns = tuple(int(town_id) for town_id in train_towns)
  return tuple(normalized_towns[idx % len(normalized_towns)] for idx in range(target_length))


def describe_returncode(returncode):
  if returncode is None:
    return 'still running'
  if returncode == 0:
    return 'exited normally (code 0)'
  if returncode < 0:
    return f'terminated by signal {-returncode}'
  return f'exited with code {returncode}'


def resolve_logdir(git_root, unknown, config_overrides):
  cli_value = get_unknown_arg_value(unknown, '--logdir', None)
  logdir_value = cli_value if cli_value is not None else config_overrides.get('logdir', None)

  if logdir_value is None:
    return os.path.join(git_root, 'results')

  logdir_value = os.path.expanduser(str(logdir_value))
  if os.path.isabs(logdir_value):
    return os.path.abspath(logdir_value)
  return os.path.abspath(os.path.join(git_root, logdir_value))


def next_free_port(port=1024, max_port=65535):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  while port <= max_port:
    try:
      sock.bind(('', port))
      sock.close()
      return port
    except OSError:
      port += 1
  raise IOError('no free ports')


def kill(proc_pid):
  if psutil.pid_exists(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
      try:
        proc.kill()
      except psutil.NoSuchProcess:  # Catch the error caused by the process no longer existing
        pass  # Ignore it
    try:
      process.kill()
    except psutil.NoSuchProcess:  # Catch the error caused by the process no longer existing
      pass  # Ignore it


def kill_all_carla_servers(ports):
  # Need a failsafe way to find and kill all carla servers. We do so by port.
  for proc in psutil.process_iter():
    # check whether the process name matches
    try:
      proc_connections = proc.connections(kind='all')
    except (PermissionError, psutil.AccessDenied, psutil.NoSuchProcess):  # Avoid sudo processes
      proc_connections = None

    if proc_connections is not None:
      for conns in proc_connections:
        if not isinstance(conns.laddr, str):  # Avoid unix paths
          if conns.laddr.port in ports:
            try:
              proc.kill()
            except psutil.NoSuchProcess:  # Catch the error caused by the process no longer existing
              pass  # Ignore it


def cleanup(carla_procs, leaderboard_procs, train_proc, c_ports):
  kill_all_carla_servers(c_ports)

  for carla_proc in carla_procs:
    kill(carla_proc.pid)

  for leaderboard_proc in leaderboard_procs:
    kill(leaderboard_proc.pid)

  if train_proc is not None:
    kill(train_proc.pid)


if __name__ == '__main__':
  try:
    training = True
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--exp_name', type=str, default='PPO_000', help='the name of this experiment')
    parser.add_argument('--config_file',
                        type=str,
                        default=None,
                        help='Path to a grouped JSON config file. CLI arguments override config values.')
    parser.add_argument('--git_root',
                        type=str,
                        default=r'/home/jaeger/ordnung/internal/CaRL/CARLA',
                        help='root folder of 2_carla')
    parser.add_argument('--gpu_ids',
                        nargs='+',
                        default=0,
                        type=int,
                        help='GPUs to run the training on. Numer of ids must be equal to --num_envs'
                        'Training runs on the first gpu')
    parser.add_argument('--carla_root',
                        default=r'/home/jaeger/ordnung/internal/carla_9_15',
                        type=str,
                        help='Path to the .sif file containing carla 0.9.15')
    parser.add_argument('--start_port',
                        default=1024,
                        type=int,
                        help='Lowest port to use. Increase the number if you want to run multiple versions of this '
                        'script on the same machine.')
    parser.add_argument('--train_towns',
                        nargs='+',
                        default=(1, 2, 3, 4, 5, 6),
                        type=int,
                        help='Towns the CARLA servers train on. Numer of ids must be equal to --num_envs')
    parser.add_argument('--routes_folder',
                        default=r'roach_preprocessed_routes',
                        type=str,
                        help='Folder in custom_leaderboard/leaderboard/data/ that contains the routes')
    parser.add_argument('--num_envs_per_gpu',
                        default=8,
                        type=int,
                        help='Number of environments per GPU. Only used with dd_ppo.')
    parser.add_argument('--seed', type=int, default=0, help='seed of the experiment')

    parser.add_argument('--num_envs_per_node',
                        default=1,
                        type=int,
                        help='Total number of environments to train with.'
                        'on this machine.')
    parser.add_argument('--num_nodes', default=1, type=int, help='Number of machines to train on.')
    parser.add_argument('--node_id', default=0, type=int, help='Id of the node that this file is running on.')
    parser.add_argument('--rdzv_addr', default='localhost', type=str, help='IP for torchrun to sync gradients over')
    parser.add_argument('--rdzv_port', default=0, type=int, help='port for torchrun to sync gradients over')
    parser.add_argument('--ml_cloud', default=0, type=int, help='Whether the script is run on the ML cloud.')
    parser.add_argument('--use_traj_sync_ppo',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        nargs='?',
                        const=True,
                        help='if True Run each env in a separate process.')
    parser.add_argument('--train_cpp',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        nargs='?',
                        const=True,
                        help='whether to train with the c++ training code.')
    parser.add_argument('--PYTORCH_KERNEL_CACHE_PATH',
                        type=str,
                        default='~/.cache',
                        help='path to a cache folder for libtorch (used only in C++)')
    parser.add_argument('--ppo_cpp_install_path',
                        type=str,
                        default='~/ppo.cpp/install',
                        help='path to where the ppo.cpp executable is installed')
    parser.add_argument('--cpp_singularity_file_path',
                        type=str,
                        default='/mnt/bernhard/code/ppo.cpp/tools/ppo_cpp.sif',
                        help='path to the singularity .sif file for c++')
    parser.add_argument('--cpp_system_lib_path_1',
                        type=str,
                        default='/usr/lib/x86_64-linux-gnu',
                        help='path that contains libcudart.so.11.0')
    parser.add_argument('--cpp_system_lib_path_2',
                        type=str,
                        default='/usr/local/cuda/lib64',
                        help='path that contains ?')
    parser.add_argument('--route_repetitions',
                        type=int,
                        default=10,
                        help='How often to repeat training routes. needs to be high enough so they do not run out, '
                        'but low enough to save RAM.')
    parser.add_argument('--debug',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        nargs='?',
                        const=True,
                        help='exits after each crash when debugging.')
    parser.add_argument('--carla_singularity',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        nargs='?',
                        const=True,
                        help='whether to run CARLA from a singularity path')
    parser.add_argument('--carla_singularity_path',
                        type=str,
                        default='/mnt/lustre/work/geiger/bjaeger25/ad_planning/2_carla/team_code_roach/custom_carla_container.sif',
                        help='/path/to/custom_carla_container.sif')

    initial_args, initial_unknown = parser.parse_known_args()
    config_overrides = load_grouped_config(initial_args.config_file, initial_args.num_nodes > 1)

    known_dests = {action.dest for action in parser._actions}
    parser.set_defaults(**{k: v for k, v in config_overrides.items() if k in known_dests})

    args, unknown = parser.parse_known_args()

    if not cli_flag_present(sys.argv[1:], '--num_envs_per_node') and not args.use_traj_sync_ppo:
      args.num_envs_per_node = len(args.gpu_ids) * args.num_envs_per_gpu

    explicit_cli_flags = {token for token in sys.argv[1:] if token.startswith('--')}
    use_camera = bool(strtobool(get_unknown_arg_value(
        unknown, '--use_camera', config_overrides.get('use_camera', False))))
    if 'CONDA_PREFIX' in os.environ:
      lib_prefix = os.environ['CONDA_PREFIX']       # conda 环境
    elif sys.prefix != sys.base_prefix:
      lib_prefix = sys.prefix                       # venv 环境
    else:
      lib_prefix = None                             # 系统 Python，不注入
    ld_lib_prefix = f'LD_LIBRARY_PATH={lib_prefix}/lib:$LD_LIBRARY_PATH ' if lib_prefix else ''
    git_root = os.path.abspath(os.path.expanduser(args.git_root))
    raw_logdir = resolve_logdir(git_root, unknown, config_overrides)
    logdir = os.path.join(raw_logdir, args.exp_name)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(raw_logdir, 'logs'), exist_ok=True)
    write_bootstrap_config(logdir, unknown, config_overrides)
    route_root_folder = os.path.join(git_root, fr'custom_leaderboard/leaderboard/data/{args.routes_folder}')
    num_routes_per_town = 32
    route_start_id = args.num_envs_per_node * args.node_id
    route_end_id = min(num_routes_per_town, route_start_id + args.num_envs_per_node)
    if route_end_id - route_start_id < args.num_envs_per_node:
      raise ValueError(f'Not enough route files in {args.routes_folder} for node {args.node_id}: '
                       f'need {args.num_envs_per_node} routes per town, but only '
                       f'{num_routes_per_town - route_start_id} remain.')
    id_to_townfile_mapping = {
        1: [
            os.path.join(route_root_folder, f'route_Town01_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        2: [
            os.path.join(route_root_folder, f'route_Town02_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        3: [
            os.path.join(route_root_folder, f'route_Town03_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        4: [
            os.path.join(route_root_folder, f'route_Town04_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        5: [
            os.path.join(route_root_folder, f'route_Town05_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        6: [
            os.path.join(route_root_folder, f'route_Town06_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        7: [
            os.path.join(route_root_folder, f'route_Town07_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        10: [
            os.path.join(route_root_folder, f'route_Town10HD_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        12: [
            os.path.join(route_root_folder, f'route_Town12_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        13: [
            os.path.join(route_root_folder, f'route_Town13_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
        15: [
            os.path.join(route_root_folder, f'route_Town15_{i:02d}.xml.gz')
            for i in range(route_start_id, route_end_id)
        ],
    }
    configured_train_towns = tuple(args.train_towns)
    args.train_towns = expand_town_ids(configured_train_towns, args.num_envs_per_node)
    print(f'Configured train_towns={configured_train_towns}')
    print(f'Expanded train_towns={args.train_towns}')
    route_files = []
    for town_id in args.train_towns:
      if town_id not in id_to_townfile_mapping:
        raise ValueError(f'Town id {town_id} is not supported by routes folder {args.routes_folder}.')
      route_files.append(id_to_townfile_mapping[town_id].pop(0))

    # CARLA has a bug where it spams Error messages to stderr freezing the entire codebase, including restarts
    # To prevent that we redirect the std err output of CARLA servers to null.
    blackhole = open(os.devnull, 'w', encoding='utf-8')  # pylint: disable=locally-disabled, consider-using-with
    server_outs = []
    server_errs = []
    client_outs = []
    client_errs = []
    for i in range(args.num_envs_per_node):
      if args.debug:
        server_outs.append(open(f"{raw_logdir}/logs/server_out_{i:03d}.txt", 'w', encoding='utf-8'))
        server_errs.append(open(f"{raw_logdir}/logs/server_err_{i:03d}.txt", 'w', encoding='utf-8'))
        client_outs.append(open(f"{raw_logdir}/logs/client_out_{i:03d}.txt", 'w', encoding='utf-8'))
        client_errs.append(open(f"{raw_logdir}/logs/client_err_{i:03d}.txt", 'w', encoding='utf-8'))
      else:
        server_outs.append(blackhole)
        server_errs.append(blackhole)
        client_outs.append(blackhole)
        client_errs.append(blackhole)
    client_ports = []
    current_port = args.start_port + 5000 * args.node_id
    skip_next_route = 'False'

    while training:
      if args.debug:
        training = False # Do not restart after a crash when debugging.
      train_process = None
      rl_ports = []
      traffic_manager_ports = []
      sensor_ports = []
      client_ports = []
      carla_primary_ports = []
      if current_port > 60000:
        current_port = args.start_port
      for i in range(args.num_envs_per_node):
        current_port = next_free_port(current_port)
        rl_ports.append(current_port)
        current_port += 3
        current_port = next_free_port(current_port)
        traffic_manager_ports.append(current_port)
        current_port += 3
        current_port = next_free_port(current_port)
        sensor_ports.append(current_port)
        current_port += 3
        current_port = next_free_port(current_port)
        client_ports.append(current_port)
        current_port += 3
        current_port = next_free_port(current_port)
        carla_primary_ports.append(current_port)
        current_port += 3

        if args.num_nodes > 1:
          # Multinode training assume we only run one training job within a node, so we can pick a port
          # Port needs to be consistent across nodes
          tcp_store_port = 7000
        else:
          # Single node training might have this script running multiple times. Find a free local port
          tcp_store_port = next_free_port(7000)

      carla_processes = []
      leaderboard_processes = []

      num_threads_per_server = 4 if use_camera else 2

      carla_rendering_flag = '' if use_camera else '-nullrhi '
      carla_threading_flag = '' if use_camera else '-nothreading'
      leaderboard_no_rendering_mode = 'False' if use_camera else 'True'
      local_server_wait_s = 5.0 if use_camera else 0.02
      local_client_wait_s = 0.5 if use_camera else 0.02
      cloud_server_wait_s = 7.0 if use_camera else 7.0
      cloud_client_wait_s = 0.5 if use_camera else 0.2
      carla_gpu_assignments = get_carla_gpu_assignments(args.num_envs_per_node, args.gpu_ids, args.num_envs_per_gpu)
      if use_camera:
        print('Camera observations enabled: starting CARLA without -nullrhi and without -nothreading')
      print(f'CARLA_GPU_ASSIGNMENTS={carla_gpu_assignments}')

      if args.ml_cloud:
        for i in range(args.num_envs_per_node):
          carla_gpu_id = carla_gpu_assignments[i]
          print(f'Start server {i} on GPU {carla_gpu_id}')
          # The -nullrhi option prevents CARLA from using the GPU at all.
          # Camera observations require rendering, so we disable -nullrhi when use_camera=True.
          if args.carla_singularity:
            carla_processes.append(
                subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
                  f'singularity exec --nv --bind {args.carla_root}:{args.carla_root},{raw_logdir}:{raw_logdir} {args.carla_singularity_path} '
                    f'bash {args.carla_root}/CarlaUE4.sh -carla-rpc-port={client_ports[i]} -nosound {carla_rendering_flag}'
                    f'-carla-primary-port={carla_primary_ports[i]} -carla-streaming-port={sensor_ports[i]} '
                    f'-RenderOffScreen -graphicsadapter={carla_gpu_id} -RPCThreads={num_threads_per_server} -StreamingThreads={num_threads_per_server} -SecondaryThreads={num_threads_per_server} {carla_threading_flag}',
                    shell=True, stdout=server_outs[i], stderr=server_errs[i]))
          else:
            carla_processes.append(
                subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
                  f'{ld_lib_prefix}'
                    f'bash {args.carla_root}/CarlaUE4.sh -carla-rpc-port={client_ports[i]} -nosound {carla_rendering_flag}'
                    f'-carla-primary-port={carla_primary_ports[i]} -carla-streaming-port={sensor_ports[i]} '
                    f'-RenderOffScreen -graphicsadapter={carla_gpu_id} -RPCThreads={num_threads_per_server} -StreamingThreads={num_threads_per_server} -SecondaryThreads={num_threads_per_server} {carla_threading_flag}',
                    shell=True, stdout=server_outs[i], stderr=server_errs[i]))
          time.sleep(cloud_server_wait_s)

        for i in range(args.num_envs_per_node):
          print(f'Start client {i}')
          leaderboard_processes.append(
              subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
                f'{ld_lib_prefix}'
                  f'bash start_leaderboard.sh {git_root} {route_files[i]} {logdir} '
                  f'{i} {client_ports[i]} {traffic_manager_ports[i]} {rl_ports[i]} {args.seed} {skip_next_route} '
                  f'{args.route_repetitions} {leaderboard_no_rendering_mode}',
                  shell=True, stdout=client_outs[i], stderr=client_errs[i]))
          time.sleep(cloud_client_wait_s)
      else:
        for i in range(args.num_envs_per_node):
          carla_gpu_id = carla_gpu_assignments[i]
          print(f'Start server {i} on GPU {carla_gpu_id}')
          # The -nullrhi option prevents CARLA from using the GPU at all.
          # Camera observations require rendering, so we disable -nullrhi when use_camera=True.

          if args.carla_singularity:
            carla_processes.append(
                subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
                  f'singularity exec --nv --bind {args.carla_root}:{args.carla_root},{raw_logdir}:{raw_logdir} {args.carla_singularity_path} '
                    f'bash {args.carla_root}/CarlaUE4.sh -carla-rpc-port={client_ports[i]} -nosound {carla_rendering_flag}'
                    f'-carla-primary-port={carla_primary_ports[i]} -carla-streaming-port={sensor_ports[i]} '
                    f'-RenderOffScreen -graphicsadapter={carla_gpu_id} -RPCThreads={num_threads_per_server} -StreamingThreads={num_threads_per_server} -SecondaryThreads={num_threads_per_server} {carla_threading_flag}',
                    shell=True, stdout=server_outs[i], stderr=server_errs[i]))
          else:
            carla_processes.append(
                subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
                  f'{ld_lib_prefix}'
                    f'bash {args.carla_root}/CarlaUE4.sh -carla-rpc-port={client_ports[i]} -nosound {carla_rendering_flag}'
                    f'-carla-primary-port={carla_primary_ports[i]} -carla-streaming-port={sensor_ports[i]} '
                    f'-RenderOffScreen -graphicsadapter={carla_gpu_id} -RPCThreads={num_threads_per_server} -StreamingThreads={num_threads_per_server} -SecondaryThreads={num_threads_per_server} {carla_threading_flag}',
                    shell=True, stdout=server_outs[i], stderr=server_errs[i]))
          time.sleep(local_server_wait_s)
          print(f'Start client {i}')
          leaderboard_processes.append(
              subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
                  f'bash start_leaderboard.sh {git_root} {route_files[i]} {logdir} '
                  f'{i} {client_ports[i]} {traffic_manager_ports[i]} {rl_ports[i]} {args.seed} {skip_next_route} '
                  f'{args.route_repetitions} {leaderboard_no_rendering_mode}',
                  shell=True, stdout=client_outs[i], stderr=client_errs[i]))
          time.sleep(local_client_wait_s)

      skip_next_route = 'False'  # After one route (potentially) was skipped we reset the variable

      config_cli_values = dict(config_overrides)
      rollout_steps_per_env = config_cli_values.pop('rollout_steps_per_env', None)
      minibatches_per_update = config_cli_values.pop('minibatches_per_update', None)
      config_cli_values.pop('multi_gpu_default_routes_folder', None)

      total_envs_global = args.num_nodes * args.num_envs_per_node
      if '--total_batch_size' not in explicit_cli_flags and rollout_steps_per_env is not None:
        config_cli_values['total_batch_size'] = total_envs_global * int(rollout_steps_per_env)
      if ('--total_minibatch_size' not in explicit_cli_flags and
          minibatches_per_update is not None and
          'total_batch_size' in config_cli_values):
        total_batch_size = int(config_cli_values['total_batch_size'])
        minibatches_per_update = int(minibatches_per_update)
        if minibatches_per_update <= 0 or total_batch_size % minibatches_per_update != 0:
          raise ValueError('Configured rollout_steps_per_env and minibatches_per_update yield an invalid minibatch size')
        config_cli_values['total_minibatch_size'] = total_batch_size // minibatches_per_update

      base_cmdline_args = strip_flag_with_value(sys.argv[1:], '--config_file')
      config_cmdline_args = build_cli_args_from_config(config_cli_values, explicit_cli_flags)
      cmdline = ' '.join(map(shlex.quote, base_cmdline_args + config_cmdline_args))
      str_ports = ' '.join(str(x) for x in rl_ports)
      cpp_str_ports = ' '.join('--ports ' + str(x) for x in rl_ports)

      # Find latest model file in case training resumes.
      load_file = None
      largest_step = 0
      if os.path.exists(logdir):
        for file in os.listdir(logdir):
          if file.startswith('model_latest_') and file.endswith('.pth'):
            full_path = os.path.join(logdir, file)
            if os.path.getsize(full_path) > 0:
              numbers_in_string = re.findall(r'\d+', file)
              if len(numbers_in_string) > 0:
                start_step = int(numbers_in_string[0])  # That step was already finished.
                if start_step > largest_step:
                  largest_step = start_step
                  load_file = os.path.join(logdir, file)

      if args.use_traj_sync_ppo:
        num_processes = args.num_envs_per_node
        num_envs_per_proc = 1
        print(f'Num processes : {num_processes}')
      else:
        num_processes = args.num_envs_per_node // args.num_envs_per_gpu
        num_envs_per_proc = args.num_envs_per_gpu

      def get_effective_forwarded_value(flag, default=None):
        cli_value = get_unknown_arg_value(unknown, flag, None)
        if cli_value is not None:
          return cli_value
        return config_cli_values.get(flag[2:], default)

      effective_summary = {
          'runtime': {
              'config_file': args.config_file,
              'exp_name': args.exp_name,
              'git_root': git_root,
              'carla_root': args.carla_root,
              'logdir': logdir,
              'seed': args.seed,
              'debug': args.debug,
              'resume_from': load_file,
          },
          'distributed': {
              'gpu_ids': list(args.gpu_ids),
              'num_nodes': args.num_nodes,
              'node_id': args.node_id,
              'rdzv_addr': args.rdzv_addr,
              'rdzv_port': args.rdzv_port,
              'num_processes': num_processes,
          },
          'parallel': {
              'num_envs_per_gpu': args.num_envs_per_gpu,
              'num_envs_per_node': args.num_envs_per_node,
              'num_envs_per_proc': num_envs_per_proc,
              'total_envs_global': total_envs_global,
              'start_port': args.start_port,
              'train_towns': list(args.train_towns),
              'routes_folder': args.routes_folder,
              'route_repetitions': args.route_repetitions,
              'rollout_steps_per_env': get_effective_forwarded_value('--rollout_steps_per_env'),
              'minibatches_per_update': get_effective_forwarded_value('--minibatches_per_update'),
          },
          'ppo': {
              'total_timesteps': get_effective_forwarded_value('--total_timesteps'),
              'total_batch_size': get_effective_forwarded_value('--total_batch_size'),
              'total_minibatch_size': get_effective_forwarded_value('--total_minibatch_size'),
              'update_epochs': get_effective_forwarded_value('--update_epochs'),
              'reward_type': get_effective_forwarded_value('--reward_type'),
              'use_dd_ppo_preempt': get_effective_forwarded_value('--use_dd_ppo_preempt'),
          },
          'camera': {
              'use_camera': use_camera,
              'camera_mode': get_effective_forwarded_value('--camera_mode'),
              'camera_width': get_effective_forwarded_value('--camera_width'),
              'camera_height': get_effective_forwarded_value('--camera_height'),
              'frame_rate': get_effective_forwarded_value('--frame_rate'),
              'leaderboard_no_rendering_mode': leaderboard_no_rendering_mode,
              'carla_rendering_enabled': use_camera,
              'carla_threading_enabled': use_camera,
              'num_threads_per_server': num_threads_per_server,
          },
      }
      print_effective_config(effective_summary)

      if args.debug:
        train_out = open(f"{raw_logdir}/logs/train_out.txt", 'w', encoding='utf-8')
        train_err = open(f"{raw_logdir}/logs/train_err.txt", 'w', encoding='utf-8')
      else:
        train_out = sys.stdout
        train_err = sys.stderr

      if args.train_cpp:
        cpp_str_gpu_ids = ' '.join('--gpu_ids ' + str(x) for x in args.gpu_ids)
        num_envs = args.num_envs_per_node * args.num_nodes
        unknown_str = ' '.join(str(x) for x in unknown)
        #   --num_envs_per_proc {num_envs_per_proc} {cmdline}
        train_process = subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
            f'bash start_learner_ac_ppo.sh {git_root} {num_processes} {args.num_nodes} {args.rdzv_addr} '
            f'{args.rdzv_port} {args.PYTORCH_KERNEL_CACHE_PATH} {args.ppo_cpp_install_path} {raw_logdir} '
            f'{args.cpp_singularity_file_path} {args.cpp_system_lib_path_1} {args.cpp_system_lib_path_2} '
            f'{cpp_str_ports} --load_file {load_file} --num_envs {num_envs} --exp_name {args.exp_name} '
            f'--tcp_store_port {tcp_store_port} {cpp_str_gpu_ids} {unknown_str}',
            shell=True, stdout=train_out, stderr=train_err)
      else:
        train_process = subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
            f'bash start_learner_dd_ppo.sh {git_root} {num_processes} {args.num_nodes} {args.rdzv_addr} '
            f'{args.rdzv_port} {cmdline} --ports {str_ports} --logdir {raw_logdir} --load_file {load_file} '
            f'--num_envs_per_proc {num_envs_per_proc} --tcp_store_port {tcp_store_port}',
            shell=True, stdout=train_out, stderr=train_err)

      time.sleep(1)

      all_processes_running = True
      ended_leaderboard = []
      ended_carla = []
      for idx, _ in enumerate(carla_processes):
        ended_leaderboard.append(idx)
        ended_carla.append(idx)
      while all_processes_running:
        time.sleep(30)
        if train_process.poll() is not None:
          all_processes_running = False
          print(f'Train process ended: {describe_returncode(train_process.returncode)}')
        for idx, carla_process in enumerate(carla_processes):
          if carla_process.poll() is not None:
            all_processes_running = False
            print(f'Carla server {idx} ended: {describe_returncode(carla_process.returncode)}')
            skip_next_route = 'True'
        for idx, leaderboard_process in enumerate(leaderboard_processes):
          if leaderboard_process.poll() is not None:
            all_processes_running = False
            print(f'Leaderboard process {idx} ended: {describe_returncode(leaderboard_process.returncode)}')
            skip_next_route = 'True'

      for i in range(360):
        for idx, carla_process in enumerate(carla_processes):
          if carla_process.poll() is not None:
            if idx in ended_carla:
              ended_carla.remove(idx)
              print(f"Server {idx} terminated: {describe_returncode(carla_process.returncode)}")
        for idx, leaderboard_process in enumerate(leaderboard_processes):
          if leaderboard_process.poll() is not None:
            if idx in ended_leaderboard:
              ended_leaderboard.remove(idx)
              print(f"Leaderboard {idx} terminated: {describe_returncode(leaderboard_process.returncode)}")
            time.sleep(1)

      for idx in ended_leaderboard:
        print(f"Leaderboard {idx} is hanging and did not terminate")

      print('Process finished:', train_process.returncode)
      if train_process.returncode == 0:
        print('Training finished succesfully')
        training = False

      cleanup(carla_processes, leaderboard_processes, train_process, client_ports)
      time.sleep(10)
      del carla_processes
      del leaderboard_processes
      del train_process

    blackhole.close()
    print('Finished cleanup')

  # Useful for debugging if the script cleans up before it shuts down.
  except KeyboardInterrupt:
    cleanup(carla_processes, leaderboard_processes, train_process, client_ports)
    blackhole.close()
    sys.exit(-1)
