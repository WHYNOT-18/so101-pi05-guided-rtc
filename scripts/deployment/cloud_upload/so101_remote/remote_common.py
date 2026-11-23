"""Small, NumPy-only observation contract for SO-101 remote inference."""
from pathlib import Path
import numpy as np


JOINT_NAMES = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
               'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']


def validate_observation(obs):
    for key in ('front', 'wrist'):
        image = np.asarray(obs[key])
        if image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ValueError(f'{key}: expected uint8 RGB (224,224,3), got {image.shape}/{image.dtype}')
    state = np.asarray(obs['state'])
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError('state: expected six finite raw joint/clipper positions')
    if not isinstance(obs['prompt'], str) or not obs['prompt'].strip():
        raise ValueError('prompt: expected a nonempty task instruction')
    return obs


def load_observation(path):
    with np.load(Path(path), allow_pickle=False) as data:
        obs = {key: data[key].copy() for key in ('front', 'wrist', 'state')}
        obs['prompt'] = str(data['prompt'].item())
    return validate_observation(obs)


def percentiles(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite timing measurements')
    return dict(count=len(values), p50_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95)),
                mean_ms=float(values.mean()), min_ms=float(values.min()),
                max_ms=float(values.max()))
