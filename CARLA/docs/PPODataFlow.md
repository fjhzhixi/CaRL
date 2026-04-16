# PPO Data Production And Consumption

This document summarizes how PPO samples are produced and consumed in `CaRL/CARLA`, using `team_code/debug_ppo.sh` and `team_code/debug_leaderboard.sh` as the entry points.

## Entry Points

### PPO process

`team_code/debug_ppo.sh` starts distributed PPO training with a single process and a single environment:

```bash
python -m torch.distributed.run ... ${WORK_DIR}/team_code/dd_ppo.py \
    --num_envs_per_gpu 1 \
    --total_batch_size 32 \
    --ports 5555
```

This process is responsible for:

- creating the gym wrapper environment,
- sending the runtime config to the leaderboard-side agent,
- running policy inference,
- collecting rollout data into PPO buffers,
- consuming the collected samples during PPO optimization.

### Leaderboard process

`team_code/debug_leaderboard.sh` starts the CARLA leaderboard evaluator:

```bash
python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
    --agent ${WORK_DIR}/team_code/env_agent.py \
    --gym_port 5555 \
    ...
```

This process is responsible for:

- running the CARLA simulation,
- calling the agent `run_step()` every simulator tick,
- generating observations and rewards,
- sending environment data back to PPO,
- receiving the next action from PPO.

## Main Components

### `dd_ppo.py`

`team_code/dd_ppo.py` is the top-level PPO training loop.

It does four relevant things for data flow:

1. Sends the effective config over ZeroMQ IPC to the environment-side process.
2. Creates the gym wrapper (`env_gym.py`).
3. Repeatedly calls `agent.forward(...)` and `env.step(...)` to collect samples.
4. Flattens collected samples into PPO training tensors and consumes them during optimization.

### `env_gym.py`

`team_code/env_gym.py` is the gymnasium-facing wrapper.

Its role is purely communication:

- `reset()` binds to the IPC endpoint and waits for the first observation.
- `step(action)` sends the action to the environment-side agent and waits for the next `(obs, reward, done, info)` tuple.

This file does not create observations by itself. It only bridges PPO and the leaderboard-side agent.

### `env_agent.py`

`team_code/env_agent.py` is the environment-side producer of training samples.

Inside `run_step()` it:

1. advances agent-side logic for the current CARLA tick,
2. checks `action_repeat`,
3. preprocesses the observation,
4. computes reward / termination / truncation,
5. sends the generated transition payload to PPO,
6. blocks until it receives the next action.

This file is the main producer of PPO training data.

## Communication Topology

The communication is based on local ZeroMQ IPC sockets.

There are two channels per environment port:

### Config channel

- endpoint suffix: `.conf_lock`
- producer: `dd_ppo.py`
- consumer: `env_agent.py`

Purpose:

- send the resolved `GlobalConfig` from PPO to the leaderboard-side agent before rollout collection starts.

### Rollout channel

- endpoint suffix: `.lock`
- producer of observations/rewards: `env_agent.py`
- consumer of observations/rewards: `env_gym.py`
- producer of actions: `env_gym.py`
- consumer of actions: `env_agent.py`

Purpose:

- exchange rollout data and actions for every decision step.

The IPC folder defaults to `team_code/comm_files`, but can be overridden with the `CARL_COMM_FOLDER` environment variable.

## What Counts As One Sample

Inside `dd_ppo.py`, one sample is one PPO transition collected in the rollout loop:

$$
(s_t, a_t, r_t, s_{t+1}, done_t)
$$

Operationally, one sample corresponds to one iteration of the data collection loop in `dd_ppo.py` where the code:

1. stores the current `next_obs` into rollout buffers,
2. computes an action with `agent.forward(...)`,
3. calls `env.step(...)`,
4. receives the next observation and reward,
5. stores reward, done flag, logprob, value, and action.

## What A PPO Update Sample Contains

In this implementation, the sample used for PPO updates is richer than a minimal transition tuple.

At a conceptual level, one training sample contains:

$$
(obs_t, a_t, \log \pi_{old}(a_t|obs_t), A_t, R_t, V_{old}(obs_t), done_t)
$$

and additionally stores the old policy distribution parameters used for KL-related computations.

### Observation fields

The observation is a dictionary with three parts:

- `bev_semantics`
- `measurements`
- `value_measurements`

These are stored during rollout collection and later flattened into `b_obs` before optimization.

### Action field

- `action`

This is the action sampled or selected by the old policy during rollout collection. It is later flattened into `b_actions`.

### Old policy likelihood field

- `old_logprob`

This is the log-probability of the sampled action under the old policy. It is later flattened into `b_logprobs` and is required to compute the PPO importance ratio.

### Return and advantage fields

- `return`
- `advantage`

These are computed after rollout collection using bootstrap value estimation and GAE or discounted returns. They are later flattened into `b_returns` and `b_advantages`.

### Done field

- `done`

This marks whether the sample is followed by a terminal or truncated transition boundary. It is later flattened into `b_dones`.

### Old value estimate

- `old_value`

This is the value prediction produced during rollout collection and stored in `b_values`. It is used in value-related training logic and logging.

### Old policy distribution parameters

- `old_mu`
- `old_sigma`

These are stored as `b_old_mus` and `b_old_sigmas`. They are used later when computing KL divergence between the old and new action distributions.

### Optional exploration field

- `exploration_suggest`

This field is only relevant when `use_exploration_suggest=True`. It is reconstructed and flattened into `b_exploration_suggests`.

## Minimal Versus Full PPO Sample

If the question is specifically about the minimum information needed to update the policy, the core fields are:

- `obs`
- `action`
- `old_logprob`
- `advantage`

These are sufficient to form the PPO clipped policy objective.

However, the full implementation-level sample in this repository also includes:

- `return`
- `done`
- `old_value`
- `old_mu`
- `old_sigma`
- optional exploration targets

because the training step updates both the policy and value functions and also logs or regularizes against the previous distribution.

## Production Path Of A Sample

The sample production chain is:

1. `dd_ppo.py` calls `env.step(action)`.
2. `env_gym.py` sends the action over ZeroMQ.
3. `env_agent.py` receives the action on the leaderboard side.
4. The next time `env_agent.py::run_step()` reaches a decision tick, it computes a new observation and reward.
5. `env_agent.py` sends `(observation, reward, termination, truncation, info)` back to PPO.
6. `env_gym.py` returns the result to `dd_ppo.py`.
7. `dd_ppo.py` stores the transition into PPO rollout buffers.

## Consumption Path Of A Sample

After data collection for one PPO update is complete, `dd_ppo.py` consumes the samples in these stages:

1. bootstrap the last value,
2. compute returns / advantages,
3. reshape rollout tensors into flattened training tensors,
4. optionally move rollout tensors from CPU to GPU,
5. iterate minibatches and run PPO updates.

The main consumed tensors are:

- `b_obs`
- `b_actions`
- `b_logprobs`
- `b_returns`
- `b_advantages`
- `b_values`
- `b_old_mus`
- `b_old_sigmas`

## Important Detail: `action_repeat`

`env_agent.py` only produces a new training sample on decision ticks:

```python
if self.step % self.config.action_repeat != 0:
    return self.last_control
```

This means:

- one PPO sample is not always equal to one raw CARLA simulator tick,
- if `action_repeat > 1`, the same control is reused for intermediate ticks,
- PPO only receives one new sample per decision tick.

With the default config in `rl_config.py`, `action_repeat = 1`, so one decision step normally corresponds to one sample.

## Batch Size In The Debug Setup

With the provided debug script:

- `world_size = 1`
- `num_envs_per_proc = 1`
- `total_batch_size = 32`

So one PPO update collects about 32 samples before starting optimization.

## How `total_batch_size` And `total_minibatch_size` Are Determined

There are two places involved:

- `team_code/train_parallel.py`
- `team_code/dd_ppo.py`

### Source-of-truth parameters

In normal training runs started through `train_parallel.py`, the more fundamental rollout parameters are:

- `num_nodes`
- `num_envs_per_node`
- `rollout_steps_per_env`
- `minibatches_per_update`

From these values, `train_parallel.py` derives the effective PPO batch settings before launching `dd_ppo.py`.

### Override logic in `train_parallel.py`

If `--total_batch_size` is **not** explicitly provided on the command line and `rollout_steps_per_env` is present, then:

$$
total\_batch\_size = total\_envs\_global \times rollout\_steps\_per\_env
$$

where:

$$
total\_envs\_global = num\_nodes \times num\_envs\_per\_node
$$

If `--total_minibatch_size` is **not** explicitly provided on the command line and `minibatches_per_update` is present, then:

$$
total\_minibatch\_size = \frac{total\_batch\_size}{minibatches\_per\_update}
$$

This means that values written in grouped config files can be replaced by the rollout-derived values during launch.

### Important consequence

When using grouped JSON configs such as `camera_train_default.json` or `camera_train_transfuser.json`:

- editing `total_batch_size` alone may not change the effective batch size,
- editing `total_minibatch_size` alone may not change the effective minibatch size,
- the final values may still be recomputed from `rollout_steps_per_env` and `minibatches_per_update`.

In practice, if you want the launch-time effective values to be:

- `total_batch_size = 512`
- `total_minibatch_size = 16`

and you are running with:

- `num_nodes = 1`
- `num_envs_per_node = 8`

then you need rollout parameters consistent with:

$$
rollout\_steps\_per\_env = 64
$$

because:

$$
8 \times 64 = 512
$$

and:

$$
minibatches\_per\_update = 32
$$

because:

$$
\frac{512}{32} = 16
$$

### Local per-process values inside `dd_ppo.py`

After the effective global values are finalized, `dd_ppo.py` computes per-process training sizes as:

$$
local\_batch\_size = \frac{total\_batch\_size}{world\_size}
$$

$$
local\_bs\_per\_env = \frac{local\_batch\_size}{num\_envs\_per\_proc}
$$

$$
local\_minibatch\_size = \frac{total\_minibatch\_size}{world\_size}
$$

$$
num\_minibatches = \frac{local\_batch\_size}{local\_minibatch\_size}
$$

These are the values that actually determine:

- rollout buffer shapes,
- how many samples are collected before optimization,
- how many samples go into each PPO optimization minibatch on each process.

## Where Time Is Spent

The data collection loop in `dd_ppo.py` already separates major timing components:

- `t0`: total data collection time,
- `t1`: policy forward time,
- `t2`: environment step time.

Conceptually:

- `forward time` measures policy inference cost,
- `env step time` measures the cost of waiting for the next sample from the environment side,
- `data collection time` measures end-to-end collection time for the whole rollout iteration.

## How To Measure Time Per Sample

There are two practical definitions.

### 1. Environment-side time per sample

Use the accumulated `env.step(...)` timing:

$$
T_{env/sample} = \frac{\sum env\_times}{num\_collected\_steps \times num\_envs\_per\_proc}
$$

In the debug setup `num_envs_per_proc = 1`, so this simplifies to:

$$
T_{env/sample} = \frac{\sum env\_times}{num\_collected\_steps}
$$

This is the best metric if you want to know how long PPO waits for each new environment sample.

### 2. End-to-end collection time per sample

Use the total rollout collection time:

$$
T_{collect/sample} = \frac{T_{data\ collection}}{num\_collected\_steps \times num\_envs\_per\_proc}
$$

This includes:

- policy forward,
- environment stepping,
- reward/done handling,
- tensor conversion and lightweight bookkeeping.

This is the best metric if you want the overall data production cost seen by the PPO loop.

## Recommended Timing Interpretation

If the goal is to understand rollout throughput, use:

- `env time per sample` for environment-side latency,
- `collect time per sample` for end-to-end rollout cost,
- `forward time per sample` for model inference cost.

Together these three metrics tell you whether the bottleneck is:

- policy inference,
- CARLA simulation and observation generation,
- or the rest of the rollout pipeline.

## Practical Summary

The PPO data flow in `CaRL/CARLA` is a request-response pipeline:

1. PPO computes an action.
2. The gym wrapper sends it over IPC.
3. The leaderboard-side agent advances the environment and builds the next sample.
4. The sample is returned to PPO.
5. PPO stores it in rollout buffers.
6. Once enough samples are collected, PPO consumes them for optimization.

The true producer of training samples is `env_agent.py`.
The true consumer is `dd_ppo.py`.
`env_gym.py` is the transport bridge between them.

## PPO Training Flowchart

The following flowchart summarizes the current PPO training loop implemented in `team_code/dd_ppo.py`.

```mermaid
flowchart TD
    A[Initialize env, agent, optimizer, and rollout buffers] --> B[env.reset -> next_obs, next_done, next_lstm_state]
    B --> C[Enter PPO update loop]

    C --> D[Cleanup previous iteration state<br/>switch device if cpu_collect]
    D --> E[Update learning rate schedule]
    E --> F[Start data collection]

    F --> G[For each rollout step]
    G --> H[Store current next_obs and done into rollout buffers]
    H --> I[agent.forward(next_obs, lstm_state, done)<br/>-> action, logprob, value, mu, sigma, next_lstm_state]
    I --> J[env.step(action)]
    J --> K[Receive reward, termination, truncation, info, and next_obs]
    K --> L[Convert next_obs and done to tensors<br/>record reward, action, logprob, value]
    L --> M[Process final_info and episodic stats]
    M --> N{dd-ppo preempt enabled?}
    N -->|No| O{More rollout steps?}
    N -->|Yes| P{Reached early stop condition?}
    P -->|Yes| Q[Break rollout early]
    P -->|No| O
    O -->|Yes| G
    O -->|No| R[Finish data collection]
    Q --> R

    R --> S[Bootstrap value from final next_obs]
    S --> T[Compute advantages and returns<br/>using GAE or discounted returns]
    T --> U[Reconstruct optional exploration targets]
    U --> V[Flatten rollout buffers into training tensors<br/>b_obs, b_actions, b_logprobs, b_returns, b_advantages, ...]
    V --> W{cpu_collect enabled?}
    W -->|Yes| X[Move flattened tensors to training device]
    W -->|No| Y[Keep tensors on current device]
    X --> Z[Barrier and all_reduce episodic statistics]
    Y --> Z
    Z --> AA[Rank 0 logs episodic metrics<br/>and may save best model]
    AA --> AB[Prepare minibatch indices]
    AB --> AC[Finish data pre-processing]

    AC --> AD[Start PPO optimization]
    AD --> AE[For each PPO epoch]
    AE --> AF[Shuffle sample indices or environment indices]
    AF --> AG[For each minibatch]
    AG --> AH[Run forward pass on minibatch<br/>to get new logprob, entropy, new value]
    AH --> AI[Compute ratio, policy loss, value loss, entropy loss]
    AI --> AJ[Optionally add exploration loss]
    AJ --> AK[optimizer.zero_grad]
    AK --> AL[loss.backward]
    AL --> AM[Clip gradients and optimizer.step]
    AM --> AN[Record approx_kl and clipfrac]
    AN --> AO{More minibatches?}
    AO -->|Yes| AG
    AO -->|No| AP[Barrier and all_reduce approx_kl]
    AP --> AQ{KL early stop triggered?}
    AQ -->|Yes| AR[Exit epoch loop]
    AQ -->|No| AS{More epochs?}
    AS -->|Yes| AE
    AS -->|No| AT[Finish PPO optimization]
    AR --> AT

    AT --> AU[All_reduce losses and summary tensors]
    AU --> AV[Rank 0 writes TensorBoard logs]
    AV --> AW[Save latest checkpoint and cleanup old checkpoints]
    AW --> AX[Print SPS and timing statistics]
    AX --> AY{More PPO updates?}
    AY -->|Yes| C
    AY -->|No| AZ[Close env and writer<br/>save final model]
```

## Reading The Flowchart

At a high level, each PPO update in `dd_ppo.py` has three main stages:

1. Data collection: interact with CARLA and store rollout samples.
2. Data pre-processing: bootstrap values, compute returns and advantages, and flatten the rollout tensors.
3. PPO training: reuse the collected batch for multiple epochs of minibatch optimization.

This is a standard on-policy PPO pattern: collect one rollout with the current policy, freeze those samples as the "old policy" batch, then optimize on that batch for several epochs before collecting the next rollout.
