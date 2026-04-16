"""Camera rig configuration helpers for PPO camera training."""

from copy import deepcopy


LEAD_SURROUND_CAMERA_RIG = {
    'CAM_FRONT_LEFT': {
        'x': 0.0,
        'y': -0.3,
        'z': 2.25,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': -57.5,
        'fov': 60.0,
    },
    'CAM_FRONT': {
        'x': 0.25,
        'y': 0.0,
        'z': 2.25,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'fov': 60.0,
    },
    'CAM_FRONT_RIGHT': {
        'x': 0.0,
        'y': 0.3,
        'z': 2.25,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 57.5,
        'fov': 60.0,
    },
    'CAM_BACK_LEFT': {
        'x': -0.30,
        'y': 0.3,
        'z': 2.25,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 122.5,
        'fov': 60.0,
    },
    'CAM_BACK': {
        'x': -0.55,
        'y': 0.0,
        'z': 2.25,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 180.0,
        'fov': 60.0,
    },
    'CAM_BACK_RIGHT': {
        'x': -0.30,
        'y': -0.3,
        'z': 2.25,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': -122.5,
        'fov': 60.0,
    },
}

LEAD_FRONT_CAMERA_RIG = {
    'CAM_FRONT': deepcopy(LEAD_SURROUND_CAMERA_RIG['CAM_FRONT']),
}

CAMERA_MODE_TO_IDS = {
    'front': ['CAM_FRONT'],
    'surround': [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ],
}


def get_camera_ids(camera_mode):
  if camera_mode not in CAMERA_MODE_TO_IDS:
    raise ValueError(f'Unsupported camera_mode: {camera_mode}')
  return list(CAMERA_MODE_TO_IDS[camera_mode])


def build_camera_sensors(camera_mode, default_fov=None):
  if camera_mode == 'front':
    sensors = deepcopy(LEAD_FRONT_CAMERA_RIG)
  elif camera_mode == 'surround':
    sensors = deepcopy(LEAD_SURROUND_CAMERA_RIG)
  else:
    raise ValueError(f'Unsupported camera_mode: {camera_mode}')

  if default_fov is not None:
    for spec in sensors.values():
      spec['fov'] = float(default_fov)
  return sensors
