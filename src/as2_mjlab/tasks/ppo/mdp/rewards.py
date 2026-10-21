"""Legged-gym-style body-frame rewards, independent of the Unitree baseline."""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values


def tracking_lin_vel(
    env: ManagerBasedRlEnv, tracking_sigma: float, command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    actual = env.scene[asset_cfg.name].data.root_link_lin_vel_b
    error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    return torch.exp(-error / tracking_sigma)


def tracking_ang_vel(
    env: ManagerBasedRlEnv, tracking_sigma: float, command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    actual = env.scene[asset_cfg.name].data.root_link_ang_vel_b
    return torch.exp(-torch.square(command[:, 2] - actual[:, 2]) / tracking_sigma)


def lin_vel_z(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return torch.square(env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, 2])


def ang_vel_xy(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return torch.sum(torch.square(env.scene[asset_cfg.name].data.root_link_ang_vel_b[:, :2]), dim=1)


def termination(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Match reset & ~timeout, including simultaneous failure/time-limit cases."""
    return (env.termination_manager.terminated & ~env.termination_manager.time_outs).float()


class pose:
    """根据指令放宽姿态容差，奖励关节接近默认位置；PPO 独立实现。"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        self.default_joint_pos = asset.data.default_joint_pos

        # 使用 Manager 已解析的关节顺序，初始化时只解析一次正则容差。
        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, slice):
            joint_names = asset.joint_names[joint_ids]
        else:
            joint_names = [asset.joint_names[index] for index in joint_ids]

        tolerances = []
        for name in ("std_standing", "std_walking", "std_running"):
            _, _, values = resolve_matching_names_values(
                cfg.params[name], joint_names,
            )
            tolerances.append(torch.tensor(
                values,
                device=env.device,
                dtype=asset.data.joint_pos.dtype,
            ))
        self.tolerances = torch.stack(tolerances)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        std_running: dict[str, float],
        asset_cfg: SceneEntityCfg,
        command_name: str,
        walking_threshold: float,
        running_threshold: float,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running  # 已缓存到关节顺序对应的张量。
        command = env.command_manager.get_command(command_name)
        speed = torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()

        # 0：站立；1：行走；2：跑动。分档取决于指令，包括原地转向。
        regime = (speed >= walking_threshold).long() + (speed >= running_threshold).long()
        std = self.tolerances[regime]

        actual = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
        target = self.default_joint_pos[:, asset_cfg.joint_ids]
        normalized_error = (actual - target) / std
        return torch.exp(-normalized_error.square().mean(dim=1))
