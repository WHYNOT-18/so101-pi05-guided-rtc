"""SO-101 dataset transforms using the actual OpenPI data/config APIs.

Stored and adapted actions are absolute joint targets: five degree-valued joints
and the LeRobot gripper position in its calibrated 0..100 units. No delta/radian
conversion is applied. Raw datasets are never modified.
"""
import dataclasses
from pathlib import Path

import numpy as np

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config

JOINT_NAMES = (
    'shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
    'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos',
)
ACTION_HORIZON = 50


def parse_rgb(value):
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f'Expected three-dimensional RGB image, got {image.shape}')
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError(f'Expected RGB, got {image.shape}')
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError('Floating LeRobot images must be finite RGB in [0,1]')
        image = np.rint(image * 255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError(f'Unexpected image dtype: {image.dtype}')
    return np.ascontiguousarray(image)


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    def __call__(self, data):
        state = np.asarray(data['state'], dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError(f'Invalid SO-101 state: {state.shape}')
        front, wrist = parse_rgb(data['front']), parse_rgb(data['wrist'])
        out = dict(
            state=state,
            image={'base_0_rgb': front, 'left_wrist_0_rgb': wrist,
                   'right_wrist_0_rgb': np.zeros_like(front)},
            image_mask={'base_0_rgb': np.True_, 'left_wrist_0_rgb': np.True_,
                        'right_wrist_0_rgb': np.False_},
        )
        if 'actions' in data:
            actions = np.asarray(data['actions'], dtype=np.float32)
            if actions.shape != (ACTION_HORIZON, 6) or not np.isfinite(actions).all():
                raise ValueError(f'Invalid SO-101 action chunk: {actions.shape}')
            out['actions'] = actions
        if 'prompt' in data:
            out['prompt'] = data['prompt']
        return out


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    def __call__(self, data):
        # In a future policy pipeline, OpenPI Unnormalize must run before this.
        return {'actions': np.asarray(data['actions'])[..., :6]}


@dataclasses.dataclass(frozen=True)
class SO101DataConfig(config.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            action_sequence_keys=('action',),
            prompt_from_task=True,
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                'front': 'observation.images.front',
                'wrist': 'observation.images.wrist',
                'state': 'observation.state', 'actions': 'action', 'prompt': 'prompt',
            })]),
            data_transforms=transforms.Group(inputs=[SO101Inputs()], outputs=[SO101Outputs()]),
            model_transforms=config.ModelTransformFactory()(model_config),
        )


def make_config(split, assets_base, batch_size=2):
    if split not in ['train', 'val', 'test']:
        raise ValueError(split)
    # Constructing a model config does not instantiate a model or load weights.
    return config.TrainConfig(
        name='pi05_so101_data', exp_name='dataset_check',
        model=pi0_config.Pi0Config(pi05=True, action_dim=32,
                                   action_horizon=ACTION_HORIZON, max_token_len=200),
        data=SO101DataConfig(repo_id=f'local/so101_{split}',
                             assets=config.AssetsConfig(asset_id='so101')),
        assets_base_dir=str(Path(assets_base).resolve()),
        batch_size=batch_size, num_workers=0, seed=42, wandb_enabled=False,
    )
