# PPO 数据保存与可视化工具参考

## 一、已实现的数据保存

| 类别 | 位置 | 说明 | 状态 |
|------|------|------|------|
| 模型检查点 | `dd_ppo.py` L50-62 | `torch.save` 保存 model/optimizer state_dict，`jsonpickle` 保存 config.json | **启用** |
| TensorBoard 标量 | `dd_ppo.py` L665 | `SummaryWriter` 记录 episodic_return、losses、SPS、timing 等 ~16 种标量 | **启用** (rank 0) |
| W&B 同步 | `dd_ppo.py` L649-662 | `wandb.init(sync_tensorboard=True)` | **默认关闭**，`--track 1` 开启 |
| 纯文本训练日志 | `dd_ppo.py` L666 | `train.log` 记录每次 update 的 timing 明细 | **启用** |
| 评估 JSON | `start_leaderboard.sh` | Leaderboard 评估结果写入 `route_*.json` | **启用** |
| 结果汇总 CSV | `tools/result_parser.py` | 将评估 JSON 汇聚为 `results.csv` | 手动运行的后处理工具 |

## 二、已实现的可视化

| 类别 | 位置 | 说明 | 启用方式 |
|------|------|------|----------|
| BEV 语义渲染 | `bev_observation.py` L396-505 | `debug=True` 时渲染彩色 BEV（道路、车辆、行人、交通灯等） | `DEBUG_ENV_AGENT=1` |
| 模型可视化叠加 | `model.py` L534-660 | `visualize_model()` 生成 BEV + 动作分布 PDF 曲线 + 测量值文字叠加的复合图像 | 同上 |
| 评估 debug 视频 | `eval_agent.py` L568-583 | `cv2.VideoWriter` 输出 AVI 视频（每帧为 `visualize_model()` 输出） | `DEBUG_ENV_AGENT=1` + `SAVE_PATH` |
| 违规片段录制 | `eval_agent.py` L484-496, `env_agent.py` L558-577 | 违规发生前后 ~5 秒的 AVI 短片 | `RECORD=1` + `SAVE_PATH` |
| 训练 BEV PNG 保存 | `env_agent.py` L261 | 每步保存渲染的 BEV 为 PNG，输出到 `{SAVE_PATH}/{exp_name}/bev_observation/{step:04}.png` | `DEBUG_ENV_AGENT=1` |
| 训练 Camera PNG 保存 | `env_agent.py` L415-417 | `use_camera` 时每步保存各相机 RGB 图像为 PNG，输出到 `{SAVE_PATH}/{exp_name}/camera_{cam_id}/{step:04}.png` | `DEBUG_ENV_AGENT=1` + `use_camera=True` |
| 问题路线 XML 保存 | `env_agent.py` L581-625 | 低奖励路线保存为 gzip XML 以供回放 | `RECORD=1` + `SAVE_PATH` |
| gym RecordVideo | `dd_ppo.py` L609-610 | `gym.wrappers.RecordVideo` | `--capture_video` |

## 三、已注释掉的可视化代码

| 位置 | 内容 |
|------|------|
| `dd_ppo.py` L889-895 | matplotlib 实时显示 BEV 通道 |
| `eval_agent.py` L519-545 | 舒适度指标（加速度、jerk、yaw rate）直方图 |
| `eval_agent.py` L589-591 | 单 episode 奖励曲线图 |
| `reward/simple_reward.py` L66-71 | 舒适度指标数据收集（用于上述直方图） |

## 四、尚未实现的部分

1. **PPO rollout 原始数据不落盘** — obs、actions、rewards、returns、advantages 等 rollout buffer 仅驻留内存（`dd_ppo.py` L828-862），从未保存到磁盘。
2. **无 TensorBoard 图像/视频日志** — 仅使用 `add_scalar` 和 `add_text`，从未调用 `add_image()` 或 `add_video()`。
3. **无轨迹/路径日志** — 训练过程中不保存自车位置、航点序列。
4. **无 numpy/pickle 数据导出** — 未使用 `np.save()` / `pickle.dump()` 持久化训练采集数据。
5. **训练时无系统性的相机图像保存** — 相机观测虽在 rollout buffer 中采集，但不写入磁盘。

## 五、环境变量速查

| 环境变量 | 作用 |
|----------|------|
| `DEBUG_ENV_AGENT=1` | 启用 BEV 渲染、模型可视化叠加、训练 BEV PNG 保存、评估 debug 视频 |
| `RECORD=1` | 启用违规片段录制、问题路线 XML 保存 |
| `SAVE_PATH=<dir>` | 可视化保存根目录，实际输出到 `{SAVE_PATH}/{exp_name}/{sensor}/` |
| `SAVE_PNG=1` | 评估时将每帧保存为独立 PNG（配合 `DEBUG_ENV_AGENT=1`） |
| `--track 1` | 启用 Weights & Biases 日志同步 |
| `--capture_video` | 启用 gym RecordVideo 包装器 |
| `--visualize` | 设置 `render_mode='human'` 在屏幕上渲染 |

## 六、BEV 观测生成原理

### 数据源

BEV 观测**不依赖任何 CARLA 传感器**（无 LiDAR、无摄像头），完全基于模拟器的特权（ground-truth）API 构建。

| BEV 元素 | CARLA API 来源 | 获取方式 |
|---|---|---|
| 道路/车道/标线 | `carla_map.generate_waypoints()` + `get_topology()` | **离线预计算**，存为 `.h5`，运行时加载 |
| 车辆 | `vehicle.get_transform()` + `.bounding_box` + `.get_velocity()` + `.get_light_state()` | 每帧实时查询 |
| 行人 | `walker.get_transform()` + `.bounding_box` + `.get_velocity()` | 每帧实时查询 |
| 静态障碍物 | `actors.filter('*static*')` + `.get_transform()` + `.bounding_box` | 每帧实时查询 |
| 交通灯 | `traffic_light.state` + 停止线顶点 | 每帧实时查询 |
| 停车标志 | `criteria_stop.target_stop_sign` trigger volume | 每帧实时查询 |
| 限速标志 | `world_map.get_all_landmarks_of_type('274')` | attach 时一次性查询 |
| 路线 | 规划器提供的 waypoint 列表 | 每帧传入 |
| 自车位姿 | `self.vehicle.get_transform()` | 每帧实时查询 |

### 渲染流程

```
离线: CARLA Map API → birdview_map.py → .h5 (road, lane_marking, shoulder, ...)
                                              ↓
初始化: h5 → hd_map_array (全图像素数组)
                                              ↓
每帧:  ego pose → get_warp_transform() → 仿射矩阵 m_warp
       ├─ hd_map_array ──→ cv.warpAffine() ──→ road(ch0), lane(ch2), shoulder(ch9)
       ├─ route waypoints → cv.transform() + cv.circle() → route(ch1)
       ├─ vehicle bboxes  → cv.transform() + cv.fillConvexPoly() → vehicles(ch3)
       ├─ walker bboxes   → cv.transform() + cv.fillConvexPoly() → walkers(ch4)
       ├─ stopline vtx    → cv.line() → traffic_lights(ch5)
       ├─ stop sign vol   → cv.fillConvexPoly() → stop_signs(ch6)
       ├─ speed signs     → cv.fillConvexPoly() → speed_signs(ch7)
       └─ static bboxes   → cv.fillConvexPoly() → statics(ch8)
                                              ↓
                              final_masks[10, 256, 256] uint8
```

### 坐标变换

**自车中心（ego-centric）**：通过 `cv.getAffineTransform` 将全局地图中以自车为中心的旋转矩形区域映射到固定大小的 BEV 网格。自车位于图像底部偏上（`pixels_ev_to_bottom` 控制偏移量），前进方向朝上。

### BEV 尺寸配置

| 配置 | 新 BEV (v1.1) | 旧 BEV (Roach) |
|---|---|---|
| 网格大小 | 256×256 px | 192×192 px |
| 像素密度 | 2 px/m | 5 px/m |
| 实际覆盖 | **128m × 128m** | **38.4m × 38.4m** |
| 通道数 | 10 | 15 |

### 10 通道含义（新 BEV）

| 通道 | 内容 | 像素编码 |
|---|---|---|
| 0 | 可行驶区域 | 0/255 |
| 1 | 规划路线 | 0/255 |
| 2 | 车道线（实线255/虚线127） | 0/127/255 |
| 3 | 车辆（bbox + 速度编码 + 灯光子区域） | 0~255 多级 |
| 4 | 行人（bbox + 速度编码） | 0/127~255 |
| 5 | 交通灯停止线（绿~80/黄~170/红=255） | 多级 |
| 6 | 停车标志触发区 | 0/255 |
| 7 | 限速标志 | 0/127~255 |
| 8 | 静态障碍物 | 0/255 |
| 9 | 路肩 | 0/255 |

### TensorBoard step 粒度

TensorBoard 中 `global_step` 的单位是**全局环境帧数**（跨所有并行 worker 累计）：

```python
# dd_ppo.py L960 — 每个 rollout step 递增
config.global_step += 1 * world_size * args.num_envs_per_proc
```

- Episode 指标（`episodic_return` 等）：episode 结束时记录，间隔不固定
- Loss / training 指标：每次 PPO update 记录一次，间隔为 `local_bs_per_env × world_size × num_envs_per_proc`
