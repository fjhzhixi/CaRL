# CaRL TransFuser Encoder 流程总结

本文总结 CaRL 中 `image_encoder == "transfuser"` 时的视觉 encoder 流程，重点说明：
- 相机输入如何拼接并进入 TransFuser backbone
- TransFuser 专用 BEV/LiDAR 坐标定义
- anchors/token 的含义
- 如何和 lead 仓库训练出的 backbone checkpoint 对齐

对应实现主要位于：
- `CARLA/team_code/camera_agent/encoder.py`
- `CARLA/team_code/rl_config.py`
- `CARLA/team_code/model.py`

---

## 1. 启用条件与整体位置

CaRL 的 PPO model 输入可以包含三类信息：
1. `bev_semantics`：CaRL 原始 BEV 语义分支
2. `measurements`：车辆状态/标量观测
3. `camera_images`：RGB 相机观测

当配置满足：
- `use_camera = True`
- `image_encoder = "transfuser"`

模型会在 `ImageEncoder` factory 中选择 `TransfuserImageEncoder`。该 encoder 是一个额外的 camera branch，其输出会 flatten 后与原始 BEV branch 和 measurements branch 的特征 concat，再进入 PPO 特征 MLP。

重要边界：
- CaRL 原始 `bev_semantics` branch 的定义不变。
- TransFuser camera branch 内部单独维护一套 lead-compatible latent BEV 定义。
- TransFuser branch 不读取 CaRL 原始 `bev_semantics` 作为输入。

---

## 2. 相机输入流程

### 2.1 环境输出格式

环境侧输出的 camera observation 形状为：

```text
(B, N, H, W, 3)
```

在 `model.py` 中会转换为 PyTorch channel-first：

```text
(B, N, 3, H, W)
```

其中：
- `B`：batch size
- `N`：相机数量
- `H/W`：单个相机图像高宽

### 2.2 多相机横向拼接

`TransfuserImageEncoder._stitch_cameras()` 会把多相机图像沿宽度方向拼成一张大图：

```text
(B, N, 3, H, W) -> (B, 3, H, N * W)
```

默认对齐 lead 的 CARLA leaderboard 6-camera setup：

```text
N = 6
H = 384
W = 384

stitched RGB = (B, 3, 384, 2304)
```

6 个相机的顺序由 `camera_mode="surround"` 决定：

```text
CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT,
CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT
```

这和 lead 的 6-camera 横向拼接输入保持一致。

---

## 3. TransFuser 专用 BEV/LiDAR 坐标定义

CaRL 原始 BEV branch 使用：

```text
bev_semantics_height = 192
bev_semantics_width = 192
pixels_per_meter = 5.0
pixels_ev_to_bottom = 40
```

这套定义继续只服务于 `bev_semantics` branch、BEV observation 生成和原有 BEV encoder。

TransFuser camera branch 单独使用 lead-compatible BEV/LiDAR 定义：

```text
transfuser_pixels_per_meter = 4.0
transfuser_min_x_meter = -32
transfuser_max_x_meter = 64
transfuser_min_y_meter = -40
transfuser_max_y_meter = 40
```

由此得到：

```text
transfuser_lidar_width_meter = 96
transfuser_lidar_height_meter = 80
transfuser_lidar_width_pixel = 384
transfuser_lidar_height_pixel = 320
```

如果 `transfuser_latent_tf = True`，TransFuser 不使用真实 LiDAR raster，而是构造 2-channel latent positional grid：

```text
latent BEV = (B, 2, 320, 384)
```

两个 channel 分别是：
- top-down 方向的 0 到 1 位置编码
- left-right 方向的 0 到 1 位置编码

该定义用于匹配 lead 中 LTF / vision-only TransFuser backbone 的训练输入。

---

## 4. Anchors 与 Transformer Tokens

TransFuser backbone 会分别用 timm image encoder 和 lidar encoder 提取多尺度特征。每个 fusion stage 前，会用 adaptive average pooling 把 feature map 压到固定 token 网格。

配置中的 anchors 指的是这个 token 网格大小，不是检测里的 anchor box。

默认 6-camera 设置下：

```text
image input: 384 x 2304
latent BEV: 320 x 384
```

由于 ResNet-style backbone 约定以 32 倍下采样尺度计算 anchors：

```text
transfuser_img_vert_anchors = 384 / 32 = 12
transfuser_img_horz_anchors = 2304 / 32 = 72

transfuser_lidar_vert_anchors = 320 / 32 = 10
transfuser_lidar_horz_anchors = 384 / 32 = 12
```

因此日志中：

```text
anchors: image=12x72, lidar=10x12
```

表示每个 fusion transformer 处理：

```text
image tokens = 12 * 72 = 864
lidar tokens = 10 * 12 = 120
total tokens = 984
```

`GPT.pos_emb` 的 token 维度也必须是 984。若 anchors 与 lead checkpoint 不一致，`pos_emb` 会出现 shape mismatch，checkpoint 无法完整初始化。

---

## 5. Backbone Forward 流程

`TransfuserBackbone.forward(stitched_rgb)` 的主要流程：

1. 对 stitched RGB 做 ImageNet normalization。
2. 构造 lead-compatible latent BEV grid：

   ```text
   (B, 2, 320, 384)
   ```

3. image branch 和 lidar branch 分别通过 timm backbone 的 4 个 stage。
4. 每个 stage 后调用 `fuse_features()`：
   - image feature adaptive pool 到 `12x72`
   - BEV/LiDAR feature adaptive pool 到 `10x12`
   - BEV channel 经 `lidar_channel_to_img` 对齐到 image channel
   - image tokens 和 BEV tokens concat 后进入 GPT-style transformer
   - transformer 输出再拆回 image feature 和 BEV feature
   - 插值回原 stage 的 feature map spatial size
   - residual add 回两条分支
   ```
   image, bev
   |
   stage 0 -> fuse -> image0, bev0
   |
   stage 1 -> fuse -> image1, bev1
   |
   stage 2 -> fuse -> image2, bev2
   |
   stage 3 -> fuse -> image3, bev3   # 最深层 fused BEV feature
   |
   top_down(bev3)
   |
   final BEV feature: (B, 64, 80, 96)

   ```
5. 最终对 BEV/LiDAR branch 的 feature 做 `top_down()`：

   ```text
   output BEV feature spatial size = (80, 96)
   ```

6. `TransfuserImageEncoder` 只取 BEV feature，flatten 后输出给 CaRL PPO feature extractor。

默认 top-down 输出：

```text
(B, transfuser_bev_features_channels, 80, 96)
```

若默认 `transfuser_bev_features_channels = 64`，flatten 后 camera branch 维度为：

```text
64 * 80 * 96 = 491520
```

---

## 6. Checkpoint 对齐与诊断

当配置了 `image_encoder_ckpt`，`TransfuserBackbone` 会从 checkpoint 中加载 `backbone.*` 前缀的参数，并在加载前打印 shape diagnostics。

正常的 6-camera lead-compatible 输出应类似：

```text
[TransfuserBackbone] Shape diagnostics before checkpoint load:
  - image: (3, 384, 2304)
  - latent BEV: (2, 320, 384)
  - anchors: image=12x72, lidar=10x12
[TransfuserBackbone] Loaded pretrained weights from: ...
[TransfuserBackbone] Matched params: 594
```

这表示：
- camera stitching shape 对齐 lead 6-camera setup
- latent BEV grid 使用 lead-compatible `320x384`
- transformer token 数和 `pos_emb` shape 与 checkpoint 匹配
- checkpoint 中有大量 backbone 参数成功加载

如果看到 `transformers.*.pos_emb` 的 shape mismatch，优先检查：
- `camera_mode`
- `camera_width`
- `camera_height`
- `transfuser_pixels_per_meter`
- `transfuser_min/max_x_meter`
- `transfuser_min/max_y_meter`
- checkpoint 是否真的是对应的 lead 6-camera LTF / vision-only backbone

---

## 7. 与 CaRL 原始 BEV Branch 的关系

当前设计故意维护两套 BEV 定义：

| 分支 | 配置前缀 | 默认尺寸 | 用途 |
| --- | --- | --- | --- |
| CaRL 原始 BEV branch | `bev_semantics_*` | `192x192` | 原始语义 BEV observation 和 BEV encoder |
| TransFuser camera branch | `transfuser_*` | `320x384` | 对齐 lead backbone checkpoint 的 latent BEV branch |

不要把 TransFuser 的 `transfuser_lidar_*` 尺寸用于环境 observation space，也不要把 CaRL 的 `bev_semantics_*` 尺寸用于 TransFuser latent BEV grid。

旧的 `min_x_meter/max_x_meter/min_y_meter/max_y_meter/lidar_*` 属性目前保留为 TransFuser-only alias，主要用于兼容已有引用；新代码应优先使用 `transfuser_*` 字段，避免和 CaRL 原始 BEV branch 混淆。

