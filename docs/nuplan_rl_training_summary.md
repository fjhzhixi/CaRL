# nuPlan RL 训练流程总结（CaRL）

本文总结 `nuPlan` 子项目中的 RL 训练主流程，重点包括：
- 仿真环境交互方式
- 模型设置细节
- Reward 计算方式

对应实现主要位于：
- `nuPlan/carl_nuplan/planning/script/run_gym.py`
- `nuPlan/carl_nuplan/planning/gym/training.py`
- `nuPlan/carl_nuplan/planning/gym/environment/*`
- `nuPlan/carl_nuplan/planning/gym/policy/ppo/*`

---

## 1. 训练入口与整体流水线

### 1.1 入口
训练从 `run_gym.py` 进入：
- `+py_func=train` -> 调用 `run_training(cfg)`
- `+py_func=cache` -> 调用 `run_caching(cfg, worker)`

常用脚本：
- 单卡：`nuPlan/scripts/gym/train_single_gpu.sh`
- 多卡：`nuPlan/scripts/gym/train_multi_gpu.sh`

### 1.2 高层流程
每个 update（PPO 迭代）包含两阶段：
1. 数据采集（rollout）
2. 策略更新（minibatch SGD）

整体流程如下：
1. 构建向量化环境（默认 `AsyncVectorEnv`）
2. 并行环境 rollout，缓存 `obs/action/logprob/reward/done/value`
3. 基于 GAE 计算 `advantages/returns`
4. 多 epoch PPO 更新（policy loss + value loss + entropy）
5. 记录日志、保存最新/阶段性/最佳模型

---

## 2. 仿真环境交互方式

### 2.1 场景来源：默认使用缓存场景
默认配置在 `default_gym.yaml` 中指定：
- `scenario_sampler: cache_scenario_sampler`
- `cache.cache_path` 指向场景缓存目录

`CacheScenarioSampler` 的行为：
- 从 `GymScenarioCache` 读取预处理场景（如 `.gz`）
- 按 `log_names` 过滤训练 split
- 每次 reset 随机采样场景

这意味着训练默认不是直接从原始 nuPlan `.db` 实时构建场景，而是走缓存加速路径。

### 2.2 EnvironmentWrapper 的 step 语义
`EnvironmentWrapper` 是 gym 接口封装，`step(action)` 顺序如下：
1. `trajectory_builder.build_trajectory(action, ego_state, info)`
2. `simulation_wrapper.step(trajectory)` 推进一步仿真
3. `reward_builder.build_reward(...)` 计算奖励/终止标志
4. `observation_builder.build_observation(...)` 构造下一状态

返回值为标准 gym 五元组：
`observation, reward, termination, truncation, info`

### 2.3 动力学推进：OneStageController
训练时采用：
- `ActionTrajectoryBuilder`：把 2 维动作（加速度、转向）映射为车辆控制量
- `OneStageController` + `KinematicBicycleModel`：直接用动作推进 ego 车辆状态

与 nuPlan 常见两阶段轨迹跟踪控制不同，这里是“动作直接驱动动力学”的一阶段控制。

### 2.4 背景交通模式
`DefaultSimulationBuilder` 支持：
- `tracks`（默认，日志回放，最快）
- `idm_agents`
- `mixed`
- `no_tracks`

在多卡脚本中可通过：
- `simulation_builder.agent_type=$AGENT_TYPE`
覆盖。

---

## 3. 观测、动作与模型设置

### 3.1 观测空间
`DefaultObservationBuilder` 输出三路输入：
1. `bev_semantics`：语义 BEV 栅格图
2. `measurements`：10 维动力学与控制相关量（含上一步动作）
3. `value_measurements`：4 维 value 辅助量
   - `remaining_time`
   - `remaining_progress`
   - `comfort_score`
   - `ttc_score`

### 3.2 动作空间
`ActionTrajectoryBuilder` 中动作空间为连续 2 维，范围 `[-1, 1]`：
- 维度 1：纵向（加速/减速）
- 维度 2：横向（转向）

动作会经过：
- 缩放（到物理量）
- 可选 clip（jerk / yaw accel / steering rate）
- 可选禁止倒车等约束

### 3.3 PPO 模型结构
`PPOPolicy` 主要结构：
1. 特征提取器（`XtMaCNN`）
   - BEV CNN 分支 + measurements MLP 分支融合
2. Policy Head
   - 输出动作分布参数（默认 Beta 分布）
3. Value Head
   - 输入是 `features + value_measurements`

可选能力：
- `use_lstm`
- `use_rpo`
- `use_world_model_loss`（默认关闭）

### 3.4 分布与损失
默认 `distribution=beta`，策略优化采用标准 PPO 形式：
- Policy clip loss
- Value loss（可选 value clipping）
- Entropy regularization

总损失可写成：

$$
\mathcal{L} = \mathcal{L}_{policy} + c_v\,\mathcal{L}_{value} - c_e\,\mathcal{L}_{entropy} + c_x\,\mathcal{L}_{exploration}
$$

其中探索损失项仅在 `use_exploration_suggest=True` 时启用。

---

## 4. PPO 训练循环细节

### 4.1 并行与批量
- 多进程通过 `torch.distributed` 初始化
- 模型使用 `DistributedDataParallel`
- `total_batch_size` / `total_minibatch_size` 会按 `world_size` 切分为 local 大小

### 4.2 rollout 缓冲
训练中按时间步收集：
- obs（3 路）
- actions
- old logprobs
- rewards
- dones
- values
- old mu/sigma（用于 KL 监控）

### 4.3 优势估计
默认启用 GAE：
- `gamma`
- `gae_lambda`

得到 `advantages` 与 `returns` 后 flatten 成 batch，进行多轮 minibatch 更新。

### 4.4 学习率调度
支持：
- `linear`
- `kl`
- `step`
- `cosine`
- `cosine_restart`

并支持：
- `schedule_free` 优化器分支
- `target_kl` 触发的提前停止/降学习率策略

### 4.5 日志与保存
- 默认使用 TensorBoard（`SummaryWriter`）
- `track=True` 时额外启用 wandb 同步
- 每轮保存 `model_latest_xxx.pth`
- 维护 `model_best.pth`（基于窗口平均回报）
- 结束保存 `model_final.pth`

---

## 5. Reward 计算方式（核心）

`DefaultRewardBuilder` 先计算 reward components，再按聚合策略得到标量奖励。

### 5.1 Reward components

#### A. 路径完成度（主正向项）
可选：
- `human`（默认）
- `mean`
- `nuplan`

#### B. Hard constraints（会触发终止）
- `red_light`
- `collision`
- `off_road`

`collision_type` 可选：
- `all`
- `non_stationary`（默认）
- `at_fault`

#### C. Soft constraints（乘性/加权惩罚）
- `lane_distance`
- `too_fast`
- `off_route`
- `comfort`
- `ttc`

其中：
- `comfort_type` 默认 `kinematics`
- `ttc_type` 默认 `v2`

### 5.2 三种 reward 聚合策略

#### 1) regular（默认论文主线风格）
终止条件：任一 hard constraint 为真（也可配置把 comfort/ttc 作为 terminal）

奖励形式：
$$
r = \Big(\Delta progress \cdot \prod s_i \cdot \mathbb{1}_{\neg term} + p_{terminal}\Big) \cdot reward\_factor
$$

其中：
- $s_i$ 为 soft constraints
- $p_{terminal}$ 为终止惩罚（碰撞可用单独惩罚）

#### 2) survival
在 regular 基础上加入生存项：
$$
r_{raw} = (1-\alpha)\,\Delta progress\cdot\prod s_i + \alpha / N
$$

再结合终止门控与终止惩罚，最后乘 `reward_factor`。

#### 3) nuplan
若未终止，按加权求和：
$$
r = \frac{5\,progress + 5\,ttc + 4\,speed + 2\,comfort}{16}
\cdot off\_route \cdot lane\_distance \cdot reward\_factor
$$

### 5.3 训练日志中的 reward 分项
episode 结束时会在 `info` 中汇总并写入日志：
- `reward/progress`
- `reward/red_light`
- `reward/collision`
- `reward/off_road`
- `reward/lane_distance`
- `reward/too_fast`
- `reward/off_route`
- `reward/comfort`
- `reward/ttc`

---

## 6. 关键配置覆盖建议（实践）

多卡脚本里常用覆盖项：
- `reward_builder.config.route_completion_type`
- `reward_builder.config.collision_type`
- `reward_builder.config.comfort_type`
- `reward_builder.config.reward_accumulation`
- `reward_builder.config.lane_distance_type`
- `reward_builder.config.off_route_type`
- `simulation_builder.agent_type`
- `cache.cache_path`

这些参数直接决定训练的行为分布、收敛速度和最终风格。

---

## 7. 一句话结论

nuPlan 这套 CaRL 训练是“缓存场景 + 一阶段车辆动力学 + PPO（分布式）+ 可组合 reward”的设计：
- 用缓存提高吞吐
- 用 one-stage action 控制贴近端到端 RL
- 用可拆分 reward components 支持快速实验迭代
- 用 DDP + 向量环境支撑大批量训练
