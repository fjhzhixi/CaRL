"""Visual encoders for PPO camera and BEV training."""

import math

import timm
import torch
import torch.nn.functional as F
from torch import nn


def normalize_imagenet(x):
  """Normalize input images according to ImageNet statistics."""
  x = x.clone()
  x[:, 0] = (x[:, 0] - 0.485) / 0.229
  x[:, 1] = (x[:, 1] - 0.456) / 0.224
  x[:, 2] = (x[:, 2] - 0.406) / 0.225
  return x


class CustomCnn(nn.Module):
  """A custom CNN with timm backbone extractors."""

  def __init__(self, config, n_input_channels):
    super().__init__()
    self.config = config
    self.image_encoder = timm.create_model(
        config.image_encoder,
        in_chans=n_input_channels,
        pretrained=False,
        features_only=True,
    )
    final_width = int(self.config.bev_semantics_width / self.image_encoder.feature_info.info[-1]['reduction'])
    final_height = int(self.config.bev_semantics_height / self.image_encoder.feature_info.info[-1]['reduction'])
    final_total_pixels = final_height * final_width
    self.out_channels = int(1024 / final_total_pixels)
    self.change_channel = nn.Conv2d(
        self.image_encoder.feature_info.info[-1]['num_chs'],
        self.out_channels,
        kernel_size=1,
    )

  def forward(self, x):
    x = self.image_encoder(x)
    x = x[-1]
    x = self.change_channel(x)
    x = torch.flatten(x, start_dim=1)
    return x


class SimpleCameraBackbone(nn.Module):
  """Lightweight CNN encoder for a single RGB image."""

  def __init__(self, config):
    super().__init__()
    self.cnn = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2),
        nn.ReLU(),
    )
    with torch.no_grad():
      dummy = torch.zeros(1, 3, config.camera_height, config.camera_width)
      flattened_dim = math.prod(self.cnn(dummy).shape[1:])
    self.fc = nn.Sequential(
        nn.Linear(flattened_dim, config.camera_features_dim),
        nn.ReLU(),
    )
    self.apply(self._weights_init)

  @staticmethod
  def _weights_init(module):
    if isinstance(module, nn.Conv2d):
      nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
      nn.init.constant_(module.bias, 0.1)

  def forward(self, x):
    x = self.cnn(x)
    x = torch.flatten(x, start_dim=1)
    return self.fc(x)


class CameraEncoder(nn.Module):
  """Encode a batch of multi-camera RGB observations into a fused feature."""

  def __init__(self, config):
    super().__init__()
    if getattr(config, 'camera_encoder', 'simple_cnn') != 'simple_cnn':
      raise ValueError(f"Unsupported camera_encoder: {config.camera_encoder}")

    self.config = config
    self.single_camera_encoder = SimpleCameraBackbone(config)
    self.num_cameras = config.get_num_cameras()
    self.output_dim = self.num_cameras * config.camera_features_dim

  def forward(self, camera_images):
    batch_size, num_cameras = camera_images.shape[:2]
    if num_cameras != self.num_cameras:
      raise ValueError(f'Expected {self.num_cameras} cameras, got {num_cameras}')

    camera_images = camera_images.reshape(batch_size * num_cameras, *camera_images.shape[2:])
    features = self.single_camera_encoder(camera_images)
    return features.reshape(batch_size, self.output_dim)


class SimpleBEVEncoder(nn.Module):
  """Existing BEV semantics encoder migrated out of model.py."""

  requires_camera_input = False

  def __init__(self, observation_space, config):
    super().__init__()
    self.config = config
    n_input_channels = observation_space['bev_semantics'].shape[0]
    if self.config.use_positional_encoding:
      n_input_channels += 2

    if self.config.image_encoder == 'roach':
      self.cnn = nn.Sequential(
          nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
          nn.ReLU(),
          nn.Conv2d(8, 16, kernel_size=5, stride=2),
          nn.ReLU(),
          nn.Conv2d(16, 32, kernel_size=5, stride=2),
          nn.ReLU(),
          nn.Conv2d(32, 64, kernel_size=3, stride=2),
          nn.ReLU(),
          nn.Conv2d(64, 128, kernel_size=3, stride=2),
          nn.ReLU(),
          nn.Conv2d(128, 256, kernel_size=3, stride=1),
          nn.ReLU(),
      )
    elif self.config.image_encoder == 'roach_ln':
      self.cnn = nn.Sequential(
          nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
          nn.LayerNorm((8, 94, 94)),
          nn.ReLU(),
          nn.Conv2d(8, 16, kernel_size=5, stride=2),
          nn.LayerNorm((16, 45, 45)),
          nn.ReLU(),
          nn.Conv2d(16, 32, kernel_size=5, stride=2),
          nn.LayerNorm((32, 21, 21)),
          nn.ReLU(),
          nn.Conv2d(32, 64, kernel_size=3, stride=2),
          nn.LayerNorm((64, 10, 10)),
          nn.ReLU(),
          nn.Conv2d(64, 128, kernel_size=3, stride=2),
          nn.LayerNorm((128, 4, 4)),
          nn.ReLU(),
          nn.Conv2d(128, 256, kernel_size=3, stride=1),
          nn.LayerNorm((256, 2, 2)),
          nn.ReLU(),
      )
    elif self.config.image_encoder == 'roach_ln2':
      self.cnn = nn.Sequential(
          nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
          nn.LayerNorm((8, 126, 126)),
          nn.ReLU(),
          nn.Conv2d(8, 16, kernel_size=5, stride=2),
          nn.LayerNorm((16, 61, 61)),
          nn.ReLU(),
          nn.Conv2d(16, 24, kernel_size=5, stride=2),
          nn.LayerNorm((24, 29, 29)),
          nn.ReLU(),
          nn.Conv2d(24, 32, kernel_size=5, stride=2),
          nn.LayerNorm((32, 13, 13)),
          nn.ReLU(),
          nn.Conv2d(32, 64, kernel_size=3, stride=2),
          nn.LayerNorm((64, 6, 6)),
          nn.ReLU(),
          nn.Conv2d(64, 128, kernel_size=3, stride=1),
          nn.LayerNorm((128, 4, 4)),
          nn.ReLU(),
          nn.Conv2d(128, 256, kernel_size=3, stride=1),
          nn.LayerNorm((256, 2, 2)),
          nn.ReLU(),
      )
    else:
      self.cnn = CustomCnn(config, n_input_channels)

    with torch.no_grad():
      sample_bev = torch.as_tensor(observation_space['bev_semantics'].sample()[None]).float()
      if self.config.use_positional_encoding:
        sample_bev = torch.concatenate((sample_bev, self._position_channels(sample_bev, expand=False)), dim=1)
      self.cnn_out_shape = self.cnn(sample_bev).shape
      self.n_flatten = math.prod(self.cnn_out_shape[1:])

    if self.config.image_encoder in ('roach', 'roach_ln', 'roach_ln2'):
      self.apply(self._weights_init)

  @staticmethod
  def _weights_init(module):
    if isinstance(module, nn.Conv2d):
      nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
      nn.init.constant_(module.bias, 0.1)

  def _position_channels(self, bev_semantics, expand=True):
    x = torch.linspace(-1, 1, self.config.bev_semantics_height, device=bev_semantics.device)
    y = torch.linspace(-1, 1, self.config.bev_semantics_width, device=bev_semantics.device)
    y_grid, x_grid = torch.meshgrid(x, y, indexing='ij')
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)
    x_grid = x_grid.unsqueeze(0).unsqueeze(0)
    if expand:
      y_grid = y_grid.expand(bev_semantics.shape[0], -1, -1, -1)
      x_grid = x_grid.expand(bev_semantics.shape[0], -1, -1, -1)
    return y_grid, x_grid

  def forward(self, bev_semantics, camera_images=None):
    del camera_images
    if self.config.use_positional_encoding:
      y_grid, x_grid = self._position_channels(bev_semantics)
      bev_semantics = torch.concatenate((bev_semantics, y_grid, x_grid), dim=1)
    x = self.cnn(bev_semantics)
    return torch.flatten(x, start_dim=1)


class SelfAttention(nn.Module):
  """Multi-head self-attention used by the TransFuser blocks."""

  def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
    super().__init__()
    assert n_embd % n_head == 0
    self.n_head = n_head
    self.key = nn.Linear(n_embd, n_embd)
    self.query = nn.Linear(n_embd, n_embd)
    self.value = nn.Linear(n_embd, n_embd)
    self.attn_drop = nn.Dropout(attn_pdrop)
    self.resid_drop = nn.Dropout(resid_pdrop)
    self.proj = nn.Linear(n_embd, n_embd)

  def forward(self, x):
    batch_size, seq_len, channels = x.size()
    head_dim = channels // self.n_head

    key = self.key(x).view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
    query = self.query(x).view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
    value = self.value(x).view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)

    attention = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
    attention = F.softmax(attention, dim=-1)
    attention = self.attn_drop(attention)

    y = attention @ value
    y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
    y = self.resid_drop(self.proj(y))
    return y


class Block(nn.Module):
  """Transformer block with self-attention and feed-forward layers."""

  def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
    super().__init__()
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
    self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
    self.mlp = nn.Sequential(
        nn.Linear(n_embd, block_exp * n_embd),
        nn.ReLU(True),
        nn.Linear(block_exp * n_embd, n_embd),
        nn.Dropout(resid_pdrop),
    )

  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x


class GPT(nn.Module):
  """GPT-style transformer for cross-modal feature fusion."""

  def __init__(self, n_embd, config):
    super().__init__()
    self.n_embd = n_embd
    self.config = config
    self.pos_emb = nn.Parameter(
        torch.zeros(
            1,
            config.transfuser_img_vert_anchors * config.transfuser_img_horz_anchors +
            config.transfuser_lidar_vert_anchors * config.transfuser_lidar_horz_anchors,
            n_embd,
        ))
    self.drop = nn.Dropout(config.transfuser_embd_pdrop)
    self.blocks = nn.Sequential(*[
        Block(
            n_embd,
            config.transfuser_n_head,
            config.transfuser_block_exp,
            config.transfuser_attn_pdrop,
            config.transfuser_resid_pdrop,
        ) for _ in range(config.transfuser_n_layer)
    ])
    self.ln_f = nn.LayerNorm(n_embd)
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(
          mean=self.config.transfuser_gpt_linear_layer_init_mean,
          std=self.config.transfuser_gpt_linear_layer_init_std,
      )
      if module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(self.config.transfuser_gpt_layer_norm_init_weight)

  def forward(self, image_tensor, bev_tensor):
    batch_size = bev_tensor.shape[0]
    bev_h, bev_w = bev_tensor.shape[2:4]
    img_h, img_w = image_tensor.shape[2:4]

    image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_embd)
    bev_tensor = bev_tensor.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_embd)

    token_embeddings = torch.cat((image_tensor, bev_tensor), dim=1)
    x = self.drop(self.pos_emb + token_embeddings)
    x = self.blocks(x)
    x = self.ln_f(x)

    image_tensor_out = x[:, :self.config.transfuser_img_vert_anchors * self.config.transfuser_img_horz_anchors, :]
    image_tensor_out = image_tensor_out.view(batch_size, img_h, img_w, -1).permute(0, 3, 1, 2).contiguous()

    bev_tensor_out = x[:, self.config.transfuser_img_vert_anchors * self.config.transfuser_img_horz_anchors:, :]
    bev_tensor_out = bev_tensor_out.view(batch_size, bev_h, bev_w, -1).permute(0, 3, 1, 2).contiguous()
    return image_tensor_out, bev_tensor_out


class TransfuserBackbone(nn.Module):
  """Camera-to-BEV encoder adapted from LEAD TransFuser backbone."""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.image_encoder = timm.create_model(
        config.transfuser_image_architecture,
        pretrained=False,
        features_only=True,
    )
    self.avgpool_img = nn.AdaptiveAvgPool2d(
        (config.transfuser_img_vert_anchors, config.transfuser_img_horz_anchors))

    image_start_index = 1 if len(self.image_encoder.return_layers) > 4 else 0
    self.num_image_features = self.image_encoder.feature_info.info[image_start_index + 3]['num_chs']

    lidar_in_chans = 2 if config.transfuser_latent_tf else 1
    self.lidar_encoder = timm.create_model(
        config.transfuser_lidar_architecture,
        pretrained=False,
        in_chans=lidar_in_chans,
        features_only=True,
    )
    lidar_start_index = 1 if len(self.lidar_encoder.return_layers) > 4 else 0
    self.num_lidar_features = self.lidar_encoder.feature_info.info[lidar_start_index + 3]['num_chs']
    self.lidar_channel_to_img = nn.ModuleList([
        nn.Conv2d(
            self.lidar_encoder.feature_info.info[lidar_start_index + i]['num_chs'],
            self.image_encoder.feature_info.info[image_start_index + i]['num_chs'],
            kernel_size=1,
        ) for i in range(4)
    ])
    self.img_channel_to_lidar = nn.ModuleList([
        nn.Conv2d(
            self.image_encoder.feature_info.info[image_start_index + i]['num_chs'],
            self.lidar_encoder.feature_info.info[lidar_start_index + i]['num_chs'],
            kernel_size=1,
        ) for i in range(4)
    ])
    self.avgpool_lidar = nn.AdaptiveAvgPool2d(
        (config.transfuser_lidar_vert_anchors, config.transfuser_lidar_horz_anchors))

    self.transformers = nn.ModuleList([
        GPT(
            self.image_encoder.feature_info.info[image_start_index + i]['num_chs'],
            config,
        ) for i in range(4)
    ])

    self.upsample = nn.Upsample(scale_factor=config.transfuser_bev_upsample_factor, mode='bilinear', align_corners=False)
    self.upsample2 = nn.Upsample(
        size=(
            config.bev_semantics_height // config.transfuser_bev_down_sample_factor,
            config.bev_semantics_width // config.transfuser_bev_down_sample_factor,
        ),
        mode='bilinear',
        align_corners=False,
    )
    self.up_conv5 = nn.Conv2d(config.transfuser_bev_features_channels,
                              config.transfuser_bev_features_channels,
                              kernel_size=3,
                              padding=1)
    self.up_conv4 = nn.Conv2d(config.transfuser_bev_features_channels,
                              config.transfuser_bev_features_channels,
                              kernel_size=3,
                              padding=1)
    self.c5_conv = nn.Conv2d(self.num_lidar_features, config.transfuser_bev_features_channels, kernel_size=1)

  def top_down(self, x):
    p5 = F.relu(self.c5_conv(x), inplace=True)
    p4 = F.relu(self.up_conv5(self.upsample(p5)), inplace=True)
    p3 = F.relu(self.up_conv4(self.upsample2(p4)), inplace=True)
    return p3

  def forward_layer_block(self, layers, return_layers, features):
    for name, module in layers:
      features = module(features)
      if name in return_layers:
        break
    return features

  def fuse_features(self, image_features, bev_features, layer_idx):
    image_embd_layer = self.avgpool_img(image_features)
    bev_embd_layer = self.avgpool_lidar(bev_features)
    bev_embd_layer = self.lidar_channel_to_img[layer_idx](bev_embd_layer)

    image_features_layer, bev_features_layer = self.transformers[layer_idx](image_embd_layer, bev_embd_layer)
    bev_features_layer = self.img_channel_to_lidar[layer_idx](bev_features_layer)
    image_features_layer = F.interpolate(
        image_features_layer,
        size=(image_features.shape[2], image_features.shape[3]),
        mode='bilinear',
        align_corners=False,
    )
    bev_features_layer = F.interpolate(
        bev_features_layer,
        size=(bev_features.shape[2], bev_features.shape[3]),
        mode='bilinear',
        align_corners=False,
    )
    return image_features + image_features_layer, bev_features + bev_features_layer

  def _build_bev_positional_grid(self, batch_size, device):
    if self.config.transfuser_latent_tf:
      x = torch.linspace(0, 1, self.config.bev_semantics_width, device=device)
      y = torch.linspace(0, 1, self.config.bev_semantics_height, device=device)
      y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
      bev = torch.zeros(
          (batch_size, 2, self.config.bev_semantics_height, self.config.bev_semantics_width),
          device=device,
      )
      bev[:, 0] = y_grid.unsqueeze(0)
      bev[:, 1] = x_grid.unsqueeze(0)
      return bev
    return torch.zeros(
        (batch_size, 1, self.config.bev_semantics_height, self.config.bev_semantics_width),
        device=device,
    )

  def forward(self, stitched_rgb):
    image_features = normalize_imagenet(stitched_rgb)
    bev_features = self._build_bev_positional_grid(stitched_rgb.shape[0], stitched_rgb.device)

    image_layers = iter(self.image_encoder.items())
    lidar_layers = iter(self.lidar_encoder.items())

    if len(self.image_encoder.return_layers) > 4:
      image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
    if len(self.lidar_encoder.return_layers) > 4:
      bev_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, bev_features)

    for i in range(4):
      image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
      bev_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, bev_features)
      image_features, bev_features = self.fuse_features(image_features, bev_features, i)

    return self.top_down(bev_features), image_features


class TransfuserBEVEncoder(nn.Module):
  """Camera-to-BEV encoder using the LEAD TransFuser backbone."""

  requires_camera_input = True

  def __init__(self, observation_space, config):
    super().__init__()
    del observation_space
    self.config = config
    self.backbone = TransfuserBackbone(config)
    with torch.no_grad():
      dummy = torch.zeros(
          1,
          config.get_num_cameras(),
          3,
          config.camera_height,
          config.camera_width,
      )
      bev_features = self._encode_camera(dummy)
      self.cnn_out_shape = bev_features.shape
      self.n_flatten = math.prod(self.cnn_out_shape[1:])

  def _stitch_cameras(self, camera_images):
    batch_size, num_cameras = camera_images.shape[:2]
    if num_cameras != self.config.get_num_cameras():
      raise ValueError(f'Expected {self.config.get_num_cameras()} cameras, got {num_cameras}')
    camera_images = camera_images.permute(0, 2, 3, 1, 4).contiguous()
    return camera_images.view(batch_size, 3, camera_images.shape[2], num_cameras * camera_images.shape[4])

  def _encode_camera(self, camera_images):
    stitched_rgb = self._stitch_cameras(camera_images)
    bev_features, _ = self.backbone(stitched_rgb)
    return bev_features

  def forward(self, bev_semantics, camera_images=None):
    del bev_semantics
    if camera_images is None:
      raise KeyError('camera_images required for TransfuserBEVEncoder')
    bev_features = self._encode_camera(camera_images)
    return torch.flatten(bev_features, start_dim=1)


class BEVEncoder(nn.Module):
  """Factory wrapper for the supported visual encoder implementations."""

  def __init__(self, observation_space, config):
    super().__init__()
    encoder_type = getattr(config, 'bev_encoder_type', 'simple')
    if encoder_type == 'simple':
      self.encoder = SimpleBEVEncoder(observation_space, config)
    elif encoder_type == 'transfuser':
      self.encoder = TransfuserBEVEncoder(observation_space, config)
    else:
      raise ValueError(f'Unsupported bev_encoder_type: {encoder_type}')

    self.requires_camera_input = self.encoder.requires_camera_input
    self.cnn_out_shape = self.encoder.cnn_out_shape
    self.n_flatten = self.encoder.n_flatten

  def forward(self, bev_semantics, camera_images=None):
    return self.encoder(bev_semantics, camera_images=camera_images)
