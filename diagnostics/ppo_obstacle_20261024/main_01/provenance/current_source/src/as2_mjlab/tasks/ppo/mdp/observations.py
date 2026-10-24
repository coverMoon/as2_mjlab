"""Privileged foot observations; no reference gait or clearance target."""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


def foot_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """World Z of the four foot sites, preserving the existing privileged signal."""
    return env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, 2]


def foot_contact_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Current continuous stance duration, one value per foot (seconds)."""
    return env.scene[sensor_name].data.current_contact_time
