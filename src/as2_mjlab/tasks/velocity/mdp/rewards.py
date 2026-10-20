"""AS2 velocity tracking rewards.

Reference:
unitreerobotics/unitree_rl_mjlab
src/tasks/velocity/mdp/rewards.py
"""

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    actual = asset.data.root_link_lin_vel_b

    xy_error = torch.sum(
        torch.square(command[:, :2] - actual[:, :2]),
        dim=1,
    )
    z_error = torch.square(actual[:, 2])

    return torch.exp(-(xy_error + 2.0 * z_error) / std**2)


def track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    actual = asset.data.root_link_ang_vel_b

    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(
        torch.square(actual[:, :2]),
        dim=1,
    )

    return torch.exp(-(z_error + 0.05 * xy_error) / std**2)

def feet_clearance(
    env: ManagerBasedRlEnv,
    target_height: float,
    command_name: str | None = None,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot height error weighted by horizontal foot velocity."""
    asset: Entity = env.scene[asset_cfg.name]

    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    foot_vel_xy = asset.data.site_lin_vel_w[
        :, asset_cfg.site_ids, :2
    ]

    vel_norm = torch.norm(foot_vel_xy, dim=-1)
    delta = torch.abs(foot_z - target_height)

    cost = torch.sum(delta * vel_norm, dim=1)

    if command_name is not None:
        command = env.command_manager.get_command(command_name)

        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm

        active = (total_command > command_threshold).float()
        cost = cost * active

    return cost

def feet_gait(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """Reward matching contact states with the reference gait phase."""
    sensor: ContactSensor = env.scene[sensor_name]

    is_contact = sensor.data.current_contact_time > 0

    global_phase = (
        (env.episode_length_buf * env.step_dt) / period
    ).unsqueeze(1)

    offsets = torch.as_tensor(
        offset,
        device=env.device,
        dtype=global_phase.dtype,
    ).view(1, -1)

    leg_phase = (global_phase + offsets) % 1.0
    is_stance = leg_phase < threshold

    reward = (is_stance == is_contact).float().mean(dim=1)

    command = env.command_manager.get_command(command_name)

    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm

    active = (total_command > command_threshold).float()

    return reward * active