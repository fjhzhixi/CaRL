# Environment Allocation During Training

This document summarizes how `CaRL/CARLA` assigns training environments to GPUs, towns, and route files when launched through `team_code/train_parallel.py`.

## Overview

For each node, the training launcher builds:

- one CARLA server per environment,
- one leaderboard client per environment,
- one DDP training process per GPU.

In the standard DDP setup, each GPU owns `num_envs_per_gpu` environments.

## Environment Count Per Node

If `--num_envs_per_node` is not passed explicitly, `train_parallel.py` computes:

```python
num_envs_per_node = len(gpu_ids) * num_envs_per_gpu
```

Example:

- `gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]`
- `num_envs_per_gpu = 2`

Then:

- `num_envs_per_node = 16`
- the node starts 16 CARLA environments,
- the node starts 8 DDP ranks,
- each rank manages 2 environments.

## GPU Assignment

CARLA servers are assigned to GPUs in contiguous blocks of `num_envs_per_gpu`.

The mapping is:

```python
gpu_ids[env_idx // num_envs_per_gpu]
```

Example with 8 GPUs and 2 environments per GPU:

- GPU 0: env 0, env 1
- GPU 1: env 2, env 3
- GPU 2: env 4, env 5
- GPU 3: env 6, env 7
- GPU 4: env 8, env 9
- GPU 5: env 10, env 11
- GPU 6: env 12, env 13
- GPU 7: env 14, env 15

The DDP training side follows the same grouping:

- one rank per GPU,
- each rank uses `num_envs_per_proc = num_envs_per_gpu`.

## Town Assignment

The config file provides `train_towns` as a list of available town ids.

Example:

```json
"train_towns": [1, 2, 3, 4, 5, 6, 7, 10]
```

At runtime, `train_parallel.py` expands that list to length `num_envs_per_node` by cycling through it:

```python
expanded_train_towns[idx] = configured_train_towns[idx % len(configured_train_towns)]
```

So the config only needs to list unique available town ids once.

Example with `num_envs_per_node = 16`:

- env 0 -> Town01
- env 1 -> Town02
- env 2 -> Town03
- env 3 -> Town04
- env 4 -> Town05
- env 5 -> Town06
- env 6 -> Town07
- env 7 -> Town10HD
- env 8 -> Town01
- env 9 -> Town02
- env 10 -> Town03
- env 11 -> Town04
- env 12 -> Town05
- env 13 -> Town06
- env 14 -> Town07
- env 15 -> Town10HD

## Route File Assignment

For each town, the launcher prepares a list of route files:

- `route_Town01_00.xml.gz`
- `route_Town01_01.xml.gz`
- ...

The route index range is selected per node:

```python
route_start_id = num_envs_per_node * node_id
route_end_id = route_start_id + num_envs_per_node
```

This means each node consumes a disjoint contiguous slice of route ids for every town.

For single-node training with:

- `node_id = 0`
- `num_envs_per_node = 16`

the node uses route ids `00..15` from each town's route pool.

When assigning route files to environments, the launcher walks through the expanded town list and pops the next available route from that town.

Example:

- env 0 -> Town01 -> `route_Town01_00.xml.gz`
- env 1 -> Town02 -> `route_Town02_00.xml.gz`
- env 2 -> Town03 -> `route_Town03_00.xml.gz`
- env 3 -> Town04 -> `route_Town04_00.xml.gz`
- env 4 -> Town05 -> `route_Town05_00.xml.gz`
- env 5 -> Town06 -> `route_Town06_00.xml.gz`
- env 6 -> Town07 -> `route_Town07_00.xml.gz`
- env 7 -> Town10HD -> `route_Town10HD_00.xml.gz`
- env 8 -> Town01 -> `route_Town01_01.xml.gz`
- env 9 -> Town02 -> `route_Town02_01.xml.gz`
- env 10 -> Town03 -> `route_Town03_01.xml.gz`
- env 11 -> Town04 -> `route_Town04_01.xml.gz`
- env 12 -> Town05 -> `route_Town05_01.xml.gz`
- env 13 -> Town06 -> `route_Town06_01.xml.gz`
- env 14 -> Town07 -> `route_Town07_01.xml.gz`
- env 15 -> Town10HD -> `route_Town10HD_01.xml.gz`

## Practical Summary

With 8 GPUs and `num_envs_per_gpu = 2`:

- total environments on the node: 16
- environments per GPU: 2
- DDP ranks: 8
- towns: assigned by cycling through the configured `train_towns`
- routes: assigned sequentially within each town, using the node's route-id slice

This gives a stable mapping from:

- environment index
- GPU
- DDP rank
- town
- route file

which is useful when debugging CARLA crashes, leaderboard exits, and route coverage.
