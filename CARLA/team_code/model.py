'''
Agent architecture from https://github.com/zhejz/carla-roach
'''

from typing import Dict
from copy import deepcopy
import math

import gymnasium as gym
import torch
from torch import nn
import numpy as np
import cv2

from camera_agent.encoder import BEVEncoder, CameraEncoder
from distributions import BetaDistribution, DiagGaussianDistribution, BetaUniformMixtureDistribution

class MeasurementEncoder(nn.Module):
  """Encode low-dimensional driving measurements."""

  def __init__(self, observation_space, states_neurons, config):
    super().__init__()
    self.output_dim = states_neurons[-1]
    layer_dims = [observation_space['measurements'].shape[0]] + list(states_neurons)
    layers = []
    for i in range(len(layer_dims) - 1):
      layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
      if config.use_layer_norm:
        layers.append(nn.LayerNorm(layer_dims[i + 1]))
      layers.append(nn.ReLU())
    self.encoder = nn.Sequential(*layers)

  def forward(self, measurements):
    return self.encoder(measurements)


class SensorDataEncoder(nn.Module):
  '''
  Encodes measurements plus optional BEV and RGB camera observations.
  '''

  def __init__(self, observation_space, states_neurons, config):
    super().__init__()
    self.features_dim = config.features_dim
    self.config = config
    self.use_bev_input = getattr(config, 'use_bev_input', True)
    self.use_camera = getattr(config, 'use_camera', False)
    if not self.use_bev_input and not self.use_camera:
      raise ValueError('Invalid encoder configuration: at least one of use_bev_input/use_camera must be enabled.')

    self.measurement_encoder = MeasurementEncoder(observation_space, states_neurons, config)
    self.state_feature_dim = self.measurement_encoder.output_dim
    self.bev_encoder = None
    self.cnn_out_shape = None
    self.n_flatten = 0
    if self.use_bev_input:
      self.bev_encoder = BEVEncoder(observation_space, config)
      self.cnn_out_shape = self.bev_encoder.cnn_out_shape
      self.n_flatten = self.bev_encoder.n_flatten
      if self.bev_encoder.requires_camera_input and not self.use_camera:
        raise ValueError('bev_encoder_type=transfuser requires use_camera=True')

    camera_total_features = 0
    self.camera_encoder = None
    if self.use_camera and (self.bev_encoder is None or not self.bev_encoder.requires_camera_input):
      self.camera_encoder = CameraEncoder(config)
      camera_total_features = self.camera_encoder.output_dim

    fusion_input_dim = self.n_flatten + self.state_feature_dim + camera_total_features
    if self.config.use_layer_norm:
      self.linear = nn.Sequential(nn.Linear(fusion_input_dim, 512), nn.LayerNorm(512), nn.ReLU(),
                                  nn.Linear(512, config.features_dim), nn.LayerNorm(config.features_dim), nn.ReLU())
    else:
      self.linear = nn.Sequential(nn.Linear(fusion_input_dim, 512), nn.ReLU(),
                                  nn.Linear(512, config.features_dim), nn.ReLU())

  def forward(self, bev_semantics, measurements, camera_images=None):
    latent_state = self.measurement_encoder(measurements)

    parts = [latent_state]
    if self.use_bev_input:
      parts.insert(0, self.bev_encoder(bev_semantics, camera_images=camera_images))

    if self.camera_encoder is not None and camera_images is not None:
      parts.append(self.camera_encoder(camera_images))

    x = torch.cat(parts, dim=1)
    x = self.linear(x)
    return x


class WorldModelDecoder(nn.Module):
  '''
   Decoder that predicts a next state given features
  '''

  def __init__(self, cnn_out_shape, cnn_n_flatten, states_neurons, features_dim, config):
    super().__init__()
    self.cnn_out_shape = cnn_out_shape
    self.cnn_n_flatten = cnn_n_flatten
    self.states_neurons = states_neurons
    self.features_dim = features_dim
    self.config = config

    if self.config.use_layer_norm:
      self.linear_decoder = nn.Sequential(nn.Linear(features_dim, 512), nn.LayerNorm(512), nn.ReLU(),
                                          nn.Linear(512, cnn_n_flatten + states_neurons),
                                          nn.LayerNorm(cnn_n_flatten + states_neurons), nn.ReLU())
    else:
      self.linear_decoder = nn.Sequential(nn.Linear(features_dim, 512), nn.ReLU(),
                                          nn.Linear(512, cnn_n_flatten + states_neurons), nn.ReLU())

    self.bev_semantic_decoder = nn.Sequential(
        nn.Conv2d(self.cnn_out_shape[1], 128, (1, 1)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(128, 64, (3, 3), padding=1),
        nn.ReLU(inplace=True),
        nn.Upsample(size=(self.config.bev_semantics_height // 4, self.config.bev_semantics_width // 4),
                    mode='bilinear',
                    align_corners=False),
        nn.Conv2d(64, 32, (3, 3), padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 16, (3, 3), padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, self.config.obs_num_channels + 1, kernel_size=(1, 1)),  # + 1 =  Background class
        nn.Upsample(size=(self.config.bev_semantics_height, self.config.bev_semantics_width),
                    mode='bilinear',
                    align_corners=False))

    self.measurement_decoder = nn.Linear(states_neurons, self.config.obs_num_measurements)

  def forward(self, features):
    features = self.linear_decoder(features)
    features_cnn = features[:, :self.cnn_n_flatten]
    features_measurements = features[:, self.cnn_n_flatten:]
    features_cnn = features_cnn.view(-1, self.cnn_out_shape[1], self.cnn_out_shape[2], self.cnn_out_shape[3])

    pred_semantic = self.bev_semantic_decoder(features_cnn)
    pred_measurement = self.measurement_decoder(features_measurements)

    return pred_semantic, pred_measurement


class PPOPolicy(nn.Module):
  '''
    Neural network policy designed for driving and training with the PPO algorithm.
  '''

  def __init__(self,
               observation_space: gym.spaces.Space,
               action_space: gym.spaces.Space,
               policy_head_arch=(256, 256),
               value_head_arch=(256, 256),
               states_neurons=(256, 256),
               config=None):

    super().__init__()
    self.action_space = action_space
    self.config = config

    self.features_extractor = SensorDataEncoder(observation_space, config=config, states_neurons=states_neurons)

    if self.config.use_lstm:
      self.lstm = nn.LSTM(config.features_dim, config.features_dim, num_layers=config.num_lstm_layers)
      for name, param in self.lstm.named_parameters():
        if 'bias' in name:
          nn.init.constant_(param, 0)
        elif 'weight' in name:
          nn.init.orthogonal_(param, 1.0)

    if self.config.distribution == 'beta':
      self.action_dist = BetaDistribution(int(np.prod(action_space.shape)))
    elif self.config.distribution == 'normal':
      # Hyperparameters are from roach
      self.action_dist = DiagGaussianDistribution(int(np.prod(action_space.shape)),
                                                  dist_init=self.config.normal_dist_init,
                                                  action_dependent_std=self.config.normal_dist_action_dep_std)
    elif self.config.distribution == 'beta_uni_mix':
      self.action_dist = BetaUniformMixtureDistribution(int(np.prod(action_space.shape)),
                                                        uniform_percentage_z=self.config.uniform_percentage_z)
    else:
      raise ValueError('Distribution selected that is not implemented. Options: beta, normal, beta_uni_mix')

    self.policy_head_arch = list(policy_head_arch)
    self.value_head_arch = list(value_head_arch)
    self.activation_fn = nn.ReLU

    self.action_space_low = nn.Parameter(torch.from_numpy(self.action_space.low), requires_grad=False)
    self.action_space_high = nn.Parameter(torch.from_numpy(self.action_space.high), requires_grad=False)

    self.build()

  def build(self) -> None:
    last_layer_dim_pi = self.features_extractor.features_dim
    policy_net = []
    for layer_size in self.policy_head_arch:
      policy_net.append(nn.Linear(last_layer_dim_pi, layer_size))
      if self.config.use_layer_norm and self.config.use_layer_norm_policy_head:
        policy_net.append(nn.LayerNorm(layer_size))
      policy_net.append(self.activation_fn())
      last_layer_dim_pi = layer_size

    self.policy_head = nn.Sequential(*policy_net)
    # mu->alpha/mean, sigma->beta/log_std (nn.Module, nn.Parameter)
    self.dist_mu, self.dist_sigma = self.action_dist.proba_distribution_net(last_layer_dim_pi)

    if self.config.use_temperature:
      # * 2 for a and b assuming beta distribution
      self.temperature_layer = nn.Sequential(nn.Linear(last_layer_dim_pi, self.action_dist.action_dim * 2),
                                             nn.Sigmoid())

    if self.config.use_value_measurements:
      last_layer_dim_vf = self.features_extractor.features_dim + self.config.num_value_measurements
    else:
      last_layer_dim_vf = self.features_extractor.features_dim

    value_net = []
    for layer_size in self.value_head_arch:
      value_net.append(nn.Linear(last_layer_dim_vf, layer_size))
      if self.config.use_layer_norm:
        value_net.append(nn.LayerNorm(layer_size))
      value_net.append(self.activation_fn())
      last_layer_dim_vf = layer_size

    if self.config.use_hl_gauss_value_loss:
      value_net.append(nn.Linear(last_layer_dim_vf, self.config.hl_gauss_num_classes))
    else:
      value_net.append(nn.Linear(last_layer_dim_vf, 1))
    self.value_head = nn.Sequential(*value_net)

  def get_features(self, observations) -> torch.Tensor:
    """
        :param bev_semantics: torch.Tensor (num_envs, frame_stack*channel, height, width)
        :param measurements: torch.Tensor (num_envs, state_dim)
        """
    bev_semantics = observations['bev_semantics'].to(dtype=torch.float32)  # Cast from uint8 to float32 for CNN
    measurements = observations['measurements']
    birdview = bev_semantics / 255.0
    camera_images = None
    if self.config.use_camera:
      if 'camera_images' not in observations:
        raise KeyError('camera_images missing from observations while use_camera=True')
      # (B, N, H, W, 3) uint8 -> (B, N, 3, H, W) float32 [0, 1]
      cam = observations['camera_images'].to(dtype=torch.float32) / 255.0
      camera_images = cam.permute(0, 1, 4, 2, 3)
    features = self.features_extractor(birdview, measurements, camera_images=camera_images)
    return features

  def get_action_dist_from_features(self, features: torch.Tensor, actions=None):
    latent_pi = self.policy_head(features)
    mu = self.dist_mu(latent_pi)
    sigma = self.dist_sigma(latent_pi)

    if actions is not None and self.config.use_rpo:
      # sample again to add stochasticity to the policy, Robust policy optimization https://arxiv.org/abs/2212.07536
      # Due to the requirement of the Beta distribution to have numbers > 0 we add the random number before the
      # activation function. We add the random number only to alpha which should have a similar effect of shifting the
      # mean as for the originally proposed gaussian distribution.
      z = torch.zeros(mu.shape, dtype=torch.float32, device=mu.device).uniform_(-self.config.rpo_alpha,
                                                                                self.config.rpo_alpha)
      mu = mu + z

    # We don't need an activation function for the normal distribution because std is predicted in log space.
    if self.config.distribution in ('beta', 'beta_uni_mix'):
      mu = nn.functional.softplus(mu)
      sigma = nn.functional.softplus(sigma)
      # NOTE adding the nugget to mu only makes sense with the beta distribution.
      mu = mu + self.config.beta_min_a_b_value
      sigma = sigma + self.config.beta_min_a_b_value

    if self.config.use_temperature:
      temperature = self.temperature_layer(latent_pi)
      mu_temperature = temperature[:, :self.action_dist.action_dim]
      sigma_temperature = temperature[:, self.action_dist.action_dim:self.action_dist.action_dim * 2]
      # Put them from [0,1] into range [min, 1]
      mu_temperature = (1.0 - self.config.min_temperature) * mu_temperature + self.config.min_temperature
      sigma_temperature = (1.0 - self.config.min_temperature) * sigma_temperature + self.config.min_temperature

      mu = mu / mu_temperature
      sigma = sigma / sigma_temperature

    return self.action_dist.proba_distribution(mu, sigma), mu.detach(), sigma.detach()

  def lstm_forward(self, features, lstm_state, done):
    # LSTM logic
    batch_size = lstm_state[0].shape[1]
    hidden = features.reshape((-1, batch_size, self.lstm.input_size))
    done = done.reshape((-1, batch_size))
    new_hidden = []
    for h, d in zip(hidden, done):
      h, lstm_state = self.lstm(
          h.unsqueeze(0),
          (
              (1.0 - d).view(1, -1, 1) * lstm_state[0],
              (1.0 - d).view(1, -1, 1) * lstm_state[1],
          ),
      )
      new_hidden += [h]
    new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
    return new_hidden, lstm_state

  def get_value(self, obs_dict: Dict[str, torch.Tensor], lstm_state=None, done=None):
    features = self.get_features(obs_dict)

    if self.config.use_lstm:
      features, _ = self.lstm_forward(features, lstm_state, done)

    if self.config.use_value_measurements:
      value_features = torch.cat((features, obs_dict['value_measurements']), dim=1)
    else:
      value_features = features
    values = self.value_head(value_features)
    return values

  def forward(self,
              obs_dict: Dict[str, np.ndarray],
              actions=None,
              sample_type='sample',
              exploration_suggests=None,
              lstm_state=None,
              done=None):
    '''
        actions are expected to be unscaled actions!
        '''

    features = self.get_features(obs_dict)

    if self.config.use_lstm:
      features, lstm_state = self.lstm_forward(features, lstm_state, done)

    pred_sem = pred_measure = None

    if self.config.use_value_measurements:
      value_features = torch.cat((features, obs_dict['value_measurements']), dim=1)
    else:
      value_features = features
    values = self.value_head(value_features)
    distribution, mu, sigma = self.get_action_dist_from_features(features, actions)

    if actions is None:
      actions = distribution.get_actions(sample_type)
    else:
      actions = self.scale_action(actions)

    log_prob = distribution.log_prob(actions)

    actions = self.unscale_action(actions)

    entropy = distribution.entropy().sum(1)
    exp_loss = None

    if exploration_suggests is not None:
      exp_loss = distribution.exploration_loss(exploration_suggests)

    return (actions, log_prob, entropy, values, exp_loss, mu, sigma, distribution.distribution, pred_sem, pred_measure,
            lstm_state)

  def scale_action(self, action: torch.Tensor, eps=1e-7) -> torch.Tensor:
    # input action \in [a_low, a_high]
    # output action \in [d_low+eps, d_high-eps]
    d_low, d_high = self.action_dist.low, self.action_dist.high  # scalar

    if d_low is not None and d_high is not None:
      a_low, a_high = self.action_space_low, self.action_space_high
      action = (action - a_low) / (a_high - a_low) * (d_high - d_low) + d_low
      action = torch.clamp(action, d_low + eps, d_high - eps)
    return action

  def unscale_action(self, action: torch.Tensor) -> torch.Tensor:
    # input action \in [d_low, d_high]
    # output action \in [a_low+eps, a_high-eps]
    d_low, d_high = self.action_dist.low, self.action_dist.high  # scalar

    if d_low is not None and d_high is not None:
      a_low, a_high = self.action_space_low, self.action_space_high
      action = (action - d_low) / (d_high - d_low) * (a_high - a_low) + a_low
    return action

  def visualize_model(self,
                      distribution,
                      obs_rendered,
                      measurements,
                      control,
                      value,
                      value_measurements,
                      pred_sem,
                      pred_measure,
                      upscale_factor=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    obs_rendered_upscaled = obs_rendered.repeat(upscale_factor, axis=0).repeat(upscale_factor, axis=1)
    width, height, _ = obs_rendered_upscaled.shape

    if distribution is not None:
      if self.config.distribution in ('beta', 'beta_uni_mix'):
        device = distribution.concentration1.device
        granularity = torch.arange(start=0.0, end=1.0, step=0.001 / upscale_factor).unsqueeze(1)
        granularity = torch.ones((granularity.shape[0], self.action_space.shape[0])) * granularity
        granularity = granularity.to(device)
        granularity_cpu = deepcopy(granularity).cpu()
      elif self.config.distribution == 'normal':
        device = distribution.mean.device
        granularity_cpu = torch.arange(start=0.0, end=1.0, step=0.001 / upscale_factor).unsqueeze(1)
        granularity = torch.arange(start=-1.0, end=1.0, step=0.002 / upscale_factor).unsqueeze(1)
        granularity = torch.ones((granularity.shape[0], self.action_space.shape[0])) * granularity
        granularity = granularity.to(device)

      if self.config.distribution == 'beta_uni_mix':
        uniform_pdf = torch.ones_like(granularity, device=device, requires_grad=False)
        distribution = (self.action_dist.beta_perc * distribution.log_prob(granularity).exp() +
                        self.action_dist.uniform_perc * uniform_pdf)
        distribution = distribution.cpu().numpy()
      else:
        distribution = distribution.log_prob(granularity)
        distribution = torch.exp(distribution).cpu().numpy()

      action_type = ['Steering', 'Brake | Throttle']
      action_plots = []
      plot_height = int(round(height / (self.action_space.shape[0] + 1), 0))
      actions = [control.steer, control.throttle - control.brake]
      y_max = 12.0  # Continuous PDFs can be arbitrary high. We clipp after 25.

      for i in range(self.action_space.shape[0]):
        action_plot = np.zeros((plot_height, width, 3), dtype=np.uint8)
        cv2.line(action_plot, (width // 2, 0), (width // 2, (plot_height - 1)), (0, 255, 0), thickness=2 * upscale_factor)
        cv2.line(action_plot, (0, 0), (0, (plot_height - 1)), (0, 255, 0), thickness=2 * upscale_factor)
        cv2.line(action_plot, (width - 1, 0), (width - 1, (plot_height - 1)), (0, 255, 0), thickness=2 * upscale_factor)

        # Plot actions:
        control_pixel = int(((actions[i] + 1.0) / 2.0) * (width - 1))
        cv2.line(action_plot, (control_pixel, 0), (control_pixel, (plot_height - 1)), (255, 255, 0),
                 thickness=2 * upscale_factor)

        granularity_numpy = granularity_cpu.numpy()
        xs = (granularity_numpy[:, 0] * width).astype(np.int32)
        y_pixels = distribution[:, i] / y_max * (plot_height - 1)
        clipped_pixels = np.clip(y_pixels, a_min=None, a_max=int(plot_height - 1)).astype(np.int32)
        ys = (plot_height - 1) - clipped_pixels

        points = np.stack([xs, ys], axis=1).astype(np.int32)
        action_plot = cv2.polylines(action_plot, [points],
                                    isClosed=False,
                                    color=(255, 255, 0),
                                    lineType=cv2.LINE_AA,
                                    thickness=2 * upscale_factor)

        cv2.putText(action_plot, action_type[i], (5 * upscale_factor, 10 * upscale_factor), font, 0.33 * upscale_factor,
                    (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
        action_plots.append(action_plot)

      action_plots = np.concatenate(action_plots, axis=0)
      measure_plot_height = height - action_plots.shape[0]
      measurement_plot = np.zeros((measure_plot_height, width, 3), dtype=np.uint8)
    else:
      measurement_plot = np.zeros((height, width, 3), dtype=np.uint8)

    y_point = 10 * upscale_factor
    cv2.putText(obs_rendered_upscaled, f'Last steer: {measurements[0]:.2f}', (0, y_point), font, 0.33 * upscale_factor,
                (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor
    cv2.putText(obs_rendered_upscaled, f'Last throt: {measurements[1]:.2f}', (0, y_point), font, 0.33 * upscale_factor,
                (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor
    cv2.putText(obs_rendered_upscaled, f'Last brake: {measurements[2]:.2f}', (0, y_point), font, 0.33 * upscale_factor,
                (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)

    if self.config.use_target_point:
      y_point += 15 * upscale_factor
      cv2.putText(obs_rendered_upscaled, f'TP: {measurements[8]:.1f} {measurements[9]:.1f}', (0, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1, cv2.LINE_AA)

    y_point = 10 * upscale_factor
    cv2.putText(obs_rendered_upscaled, f'Gear: {measurements[3]:.2f}', (width // 2, y_point), font,
                0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor
    cv2.putText(obs_rendered_upscaled, f'Velocity: {measurements[4]:.1f} {measurements[5]:.1f}', (width // 2, y_point),
                font, 0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    #y_point += 15 * upscale_factor
    #cv2.putText(obs_rendered_upscaled, f'X Velocity: {measurements[6]:.2f}', (width // 2, y_point), font,
    #            0.33 * upscale_factor, (255, 255, 255), 1*upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor
    cv2.putText(obs_rendered_upscaled, f'Speed lim.: {measurements[7]:.2f}', (width // 2, y_point), font,
                0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)

    y_point = 10 * upscale_factor
    cv2.putText(measurement_plot, 'Model predictions:', (130 * upscale_factor, y_point), font, 0.33 * upscale_factor,
                (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor
    cv2.putText(measurement_plot, f'Steer:{control.steer:.2f}', (130 * upscale_factor, y_point), font,
                0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor
    cv2.putText(measurement_plot, f'Throt:{control.throttle:.2f}', (130 * upscale_factor, y_point), font,
                0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor
    cv2.putText(measurement_plot, f'Brake:{control.brake:.2f}', (130 * upscale_factor, y_point), font,
                0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
    y_point += 15 * upscale_factor

    if value is not None:
      cv2.putText(measurement_plot, f'Value:{value.item():.2f}', (130 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)

    if self.config.use_value_measurements:
      y_point = 10 * upscale_factor
      cv2.putText(measurement_plot, 'Critic inputs:', (5 * upscale_factor, y_point), font, 0.33 * upscale_factor,
                  (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
      y_point += 15 * upscale_factor
      cv2.putText(measurement_plot, f'Timeout:{value_measurements[0]:.2f}', (5 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
      y_point += 15 * upscale_factor
      cv2.putText(measurement_plot, f'Blocked:{value_measurements[1]:.2f}', (5 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
      y_point += 15 * upscale_factor
      cv2.putText(measurement_plot, f'Route:{value_measurements[2]:.2f}', (5 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)

    if self.config.use_extra_control_inputs:
      y_point = 140 * upscale_factor
      cv2.putText(measurement_plot, f'wheel: {measurements[8]:.2f}', (130 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
      y_point += 15 * upscale_factor
      cv2.putText(measurement_plot, f'error: {measurements[9]:.2f}', (130 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
      y_point += 15 * upscale_factor
      cv2.putText(measurement_plot, f'deriv: {measurements[10]:.2f}', (130 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)
      y_point += 15 * upscale_factor
      cv2.putText(measurement_plot, f'integ: {measurements[11]:.2f}', (130 * upscale_factor, y_point), font,
                  0.33 * upscale_factor, (255, 255, 255), 1 * upscale_factor, cv2.LINE_AA)

    if distribution is not None:
      action_plots = np.concatenate((measurement_plot, action_plots), axis=0)
      return np.concatenate((obs_rendered_upscaled, action_plots), axis=1)

    else:
      return np.concatenate((obs_rendered_upscaled, measurement_plot), axis=1)
