"""AS2 task-specific observations."""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg


def foot_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """World-frame Z position of each foot site."""

    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]

def phase(
    env: ManagerBasedRlEnv,
    period: float,
    command_name: str,
) -> torch.Tensor:
    global_phase = (
        (env.episode_length_buf * env.step_dt) % period / period
    )

    phase_obs = torch.zeros(
        env.num_envs, 2, device=env.device
    )
    phase_obs[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase_obs[:, 1] = torch.cos(global_phase * torch.pi * 2.0)

    stand_mask = (
        torch.linalg.norm(
            env.command_manager.get_command(command_name), dim=1
        ) < 0.1
    )

    return torch.where(
        stand_mask.unsqueeze(1),
        torch.zeros_like(phase_obs),
        phase_obs,
    )
