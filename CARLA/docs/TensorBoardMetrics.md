# TensorBoard Metrics During PPO Training

This document summarizes the TensorBoard items written by `team_code/dd_ppo.py` during PPO training in `CaRL/CARLA`, what each item means, and what training trends are usually considered healthy.

The main logging code lives in:

- `team_code/dd_ppo.py`
- `team_code/train_parallel.py`
- `team_code/train_single_gpu_multi_env_camera.sh`

## Scope

The standard training launcher does not write reward sub-components, collision rates, or route-specific infractions to TensorBoard by default.

The main TensorBoard outputs are:

- `hyperparameters`
- `charts/*`
- `losses/*`
- `timing/*`
- `timing_avg/*`

## X-Axis Meaning

Most scalar items use `config.global_step` as the TensorBoard x-axis.

This value is not the PPO update index. It is the total number of environment interaction steps processed across all active training processes and environments.

Example for single-GPU camera training with:

- `num_envs_per_gpu = 8`
- `rollout_steps_per_env = 512`
- `num_nodes = 1`

one full rollout contributes about:

```text
8 * 512 = 4096
```

environment steps to `global_step`.

## Logged Items

### `hyperparameters`

Text summary of the resolved training arguments used for the run.

Use this page to verify:

- experiment name,
- batch sizes,
- learning rate schedule,
- reward settings,
- camera settings,
- environment count.

## `charts/*`

### `charts/episodic_return`

Average episode return over the episodes that finished during the current rollout window.

Important detail:

- this is not computed over all currently running episodes,
- it only uses episodes that actually ended during that collection phase,
- in parallel training it is aggregated across all processes.

Interpretation:

- higher is generally better,
- short-term noise is normal,
- for progress, prefer the smoothed version below.

### `charts/windowed_avg_return`

Moving average of recent `episodic_return` values.

In this implementation it is the average of the latest 100 logged returns.

Interpretation:

- this is the main task-performance curve,
- a healthy run usually shows a gradual upward trend,
- occasional plateaus are normal,
- sudden collapse usually means policy instability, reward-scale issues, or a broken environment/data path.

### `charts/episodic_length`

Average episode length for episodes that finished during the current rollout window.

Interpretation depends on the task:

- if longer survival usually means better driving, rising values can be good,
- if the agent gets stuck without terminating, a higher value is not necessarily good.

Read this together with return, not in isolation.

### `charts/learning_rate`

Current optimizer learning rate.

This is useful for verifying that the configured schedule is behaving as expected:

- `linear`: gradual decay,
- `none`: almost flat,
- `step`, `cosine`, `cosine_restart`, `kl`: schedule-specific behavior.

### `charts/discounted_returns`

Mean of the rollout returns used as the value-learning target.

This is not the same thing as full-episode return. It is the batch mean of bootstrapped discounted return targets.

Interpretation:

- useful for checking scale,
- should be interpreted together with `value_loss` and `explained_variance`,
- by itself it is not a direct success metric.

### `charts/advantages`

Mean advantage value in the current batch.

Interpretation:

- mostly useful as a sanity signal,
- large unstable swings can suggest critic mismatch or changing reward scale,
- near-zero average is not inherently bad.

### `charts/SPS`

Samples per second, meaning approximate training throughput in environment steps per second.

Interpretation:

- higher is faster,
- sudden drops often indicate environment slowdown, rendering overhead, I/O pressure, or GPU bottlenecks,
- with camera training, this is especially useful for spotting simulation/rendering pressure.

### `charts/restart`

Binary marker indicating whether training restarted from an interruption or failure.

Typical meaning:

- `0`: normal logging step,
- `1`: a restart was recorded.

This is mainly useful for aligning breaks in metric curves with runtime events.

## `losses/*`

## Why PPO Logs Losses Instead Of Only Reward

Reward is the optimization objective at the task level, but PPO does not directly backpropagate raw episode reward.

Instead, PPO:

1. collects rewards from the environment,
2. converts them into `returns` and `advantages`,
3. constructs differentiable objectives,
4. updates the policy and value networks using those objectives.

So:

- reward-based charts show whether the agent is getting better at the task,
- loss-based charts show whether the optimization process itself is stable.

Both are necessary.

### `losses/policy_loss`

PPO policy loss computed from the clipped surrogate objective.

Important detail:

- it is normal for this value to be negative,
- it is not expected to monotonically decrease like supervised learning loss.

Why it can be negative:

- PPO is effectively maximizing a policy objective,
- the code writes it as a quantity to minimize,
- because of the sign convention, negative values are common.

Expected trend:

- noisy,
- often negative,
- may oscillate around a small negative or near-zero range,
- should not be expected to smoothly fall forever.

Healthy interpretation:

- fluctuations are normal,
- moderate oscillation is expected,
- read it together with `approx_kl`, `clipfrac`, and return curves.

Potential warning signs:

- very large unstable swings together with return collapse,
- persistent near-zero values while return does not improve and `clipfrac` stays very low,
- sustained instability together with large `approx_kl`.

### `losses/value_loss`

Loss used to train the critic/value head.

In the standard configuration this is the PPO-style clipped value loss.

Interpretation:

- lower can be better,
- but the absolute scale depends strongly on reward scale,
- use it together with `explained_variance`.

Healthy trend:

- often high and noisy early in training,
- usually becomes more controlled as the critic improves,
- occasional spikes are normal when the policy distribution shifts.

### `losses/entropy`

Average action-distribution entropy.

Interpretation:

- higher entropy means more randomness and more exploration,
- lower entropy means the policy is becoming more certain.

Healthy trend:

- usually starts higher,
- often decreases over training,
- should not collapse too early unless the task is already solved.

Potential warning signs:

- very early sharp collapse can indicate premature convergence,
- persistently very high entropy together with flat return can indicate the policy is not settling.

### `losses/exploration`

Optional exploration-specific loss term.

This is only logged when `use_exploration_suggest=True`.

If that flag is disabled, this metric will not appear.

### `losses/old_approx_kl`

Older approximate KL-style diagnostic computed from log-probability ratio terms.

This is mainly useful as a reference diagnostic rather than the primary stability signal.

### `losses/approx_kl`

Average KL divergence between old and new policy distributions during PPO optimization.

This is one of the most important PPO stability metrics.

Interpretation:

- low to moderate values usually mean conservative updates,
- large values mean the policy changed a lot in one update.

Healthy trend:

- should stay bounded and reasonably small,
- some noise is normal,
- occasional spikes can happen but should not dominate.

Potential warning signs:

- repeated large spikes,
- sustained high values with worsening return,
- strong oscillation together with unstable `policy_loss`.

### `losses/clipfrac`

Fraction of samples for which PPO clipping was active.

Interpretation:

- low values mean most samples are not hitting the clipping boundary,
- high values mean a large share of updates are being clipped.

Healthy trend:

- usually neither pinned near zero nor pinned very high all the time,
- moderate values often indicate PPO is updating but still respecting its trust-region-like bound.

Potential warning signs:

- near-zero for a long time plus flat return may suggest updates are too weak,
- persistently high values may suggest learning rate or update aggressiveness is too high.

### `losses/explained_variance`

How well the critic predictions explain the variance of return targets.

Approximate interpretation:

- `1`: very strong value prediction,
- `0`: weak predictive value,
- negative: worse than predicting a constant mean.

Healthy trend:

- often poor early,
- usually improves over time if the critic learns,
- does not need to be perfect for policy improvement.

Potential warning signs:

- persistently very low or negative values together with unstable return,
- repeated collapse after previously improving can indicate critic drift.

### `losses/latest_epoch`

The index of the final PPO epoch actually completed during the current update.

Interpretation:

- if it reaches `update_epochs - 1`, the full scheduled PPO pass ran,
- if it stops earlier, the update may have been cut short by KL-based stopping logic.

This metric is mainly useful for debugging training-control behavior.

## `timing/*`

These are per-update wall-clock timings.

### `timing/data_collection`

Time spent collecting rollout data.

### `timing/total_forward`

Accumulated policy forward-pass time during rollout collection.

### `timing/total_env`

Accumulated environment stepping time during rollout collection.

This often reflects simulator-side cost and is especially useful when debugging CARLA slowdown.

### `timing/data_preprocessing`

Time spent preparing rollout buffers for optimization.

This includes reshaping, synchronization, and device transfer related work.

### `timing/training`

Time spent running PPO optimization after data collection.

### `timing/logging`

Time spent writing logs and saving checkpoints.

## `timing_avg/*`

Running average of the corresponding timing metric across completed updates.

These are not short moving averages. They are cumulative averages over the run so far.

Use them to detect:

- slow drift in simulator throughput,
- growing I/O overhead,
- changes after enabling cameras or other sensors.

## What Trends Are Usually Healthy

For PPO, there is no single metric that should always move in one direction. The healthy pattern is a combination of task improvement and optimization stability.

### Strong primary signals

- `charts/windowed_avg_return` gradually rises over time,
- `losses/approx_kl` stays reasonably bounded,
- `losses/clipfrac` stays in a moderate range instead of saturating,
- `losses/explained_variance` improves from poor initial values,
- `charts/SPS` stays relatively stable for a fixed system setup.

### Normal behavior

- `policy_loss` is noisy and often negative,
- `value_loss` spikes occasionally,
- `episodic_return` is much noisier than `windowed_avg_return`,
- `episodic_length` can move independently from return,
- timing curves fluctuate between updates.

### Common unhealthy patterns

- return collapses while `approx_kl` spikes and `policy_loss` becomes unstable,
- return stays flat while `policy_loss` is almost zero and `clipfrac` remains tiny,
- entropy collapses too early and performance plateaus,
- explained variance stays poor for a long time while value loss remains erratic,
- SPS or `timing/total_env` degrades sharply after enabling camera rendering.

## Practical Reading Order

When reviewing a run, a good order is:

1. `charts/windowed_avg_return`
2. `losses/approx_kl`
3. `losses/clipfrac`
4. `losses/explained_variance`
5. `losses/policy_loss`
6. `charts/SPS`
7. `timing/total_env`

This order usually separates:

- task progress,
- optimizer stability,
- systems/performance bottlenecks.

## Practical Summary

The most important mental model is:

- reward charts answer: "Is the agent getting better?"
- loss charts answer: "Is PPO updating in a stable way?"
- timing charts answer: "Is the training system staying efficient?"

For this repository, the single most useful success curve is usually `charts/windowed_avg_return`, while the most useful stability companions are `losses/approx_kl`, `losses/clipfrac`, and `losses/explained_variance`.
