"""Build leaderboard sensor specs for RGB and GT cameras."""

from camera_agent.camera_config import get_camera_ids


GT_CAMERA_TYPES = {
    'semantic_segmentation': 'sensor.camera.semantic_segmentation',
    'depth': 'sensor.camera.depth',
    'instance_segmentation': 'sensor.camera.instance_segmentation',
}

GT_CAMERA_ID_PREFIX = {
    'semantic_segmentation': 'semantics',
    'depth': 'depth',
    'instance_segmentation': 'instance',
}


def build_camera_sensors(config):
  """Create leaderboard camera sensor definitions from config."""
  if not getattr(config, 'use_camera', False):
    return []

  sensors = []
  camera_ids = get_camera_ids(config.camera_mode)
  camera_sensors = config.camera_sensors
  width = int(config.camera_width)
  height = int(config.camera_height)

  for camera_id in camera_ids:
    camera_spec = camera_sensors[camera_id]
    sensors.append({
        'type': 'sensor.camera.rgb',
        'x': camera_spec['x'],
        'y': camera_spec['y'],
        'z': camera_spec['z'],
        'roll': camera_spec['roll'],
        'pitch': camera_spec['pitch'],
        'yaw': camera_spec['yaw'],
        'width': width,
        'height': height,
        'fov': camera_spec['fov'],
        'id': camera_id,
    })

    if not getattr(config, 'use_camera_gt', False):
      continue

    for modality in getattr(config, 'camera_gt_modalities', []):
      if modality not in GT_CAMERA_TYPES:
        raise ValueError(f'Unsupported camera GT modality: {modality}')
      sensors.append({
          'type': GT_CAMERA_TYPES[modality],
          'x': camera_spec['x'],
          'y': camera_spec['y'],
          'z': camera_spec['z'],
          'roll': camera_spec['roll'],
          'pitch': camera_spec['pitch'],
          'yaw': camera_spec['yaw'],
          'width': width,
          'height': height,
          'fov': camera_spec['fov'],
          'id': f"{GT_CAMERA_ID_PREFIX[modality]}_{camera_id}",
      })

  return sensors

