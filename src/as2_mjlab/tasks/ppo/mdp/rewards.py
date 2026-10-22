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


def feet_clearance(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_sensor_name: str,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """局部地形净空误差乘世界系足端水平速度，返回未加权的正 cost。"""
    height = env.scene[height_sensor_name].data.heights
    velocity_xy = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    speed_xy = torch.linalg.vector_norm(velocity_xy, dim=-1)
    cost = ((height - target_height).abs() * speed_xy).sum(dim=1)

    command = env.command_manager.get_command(command_name)
    command_speed = torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()
    return cost * (command_speed > command_threshold)


def thigh_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """统计最近传感器历史中受力超过阈值的大腿，每条腿最多计一次。"""
    force_history = env.scene[sensor_name].data.force_history
    assert force_history is not None
    force_magnitude = torch.linalg.vector_norm(force_history, dim=-1)
    return (force_magnitude > force_threshold).any(dim=-1).float().sum(dim=1)


def soft_landing(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """仅累加刚触地脚的三维合力，返回未加权的正 cost。"""
    sensor = env.scene[sensor_name]
    forces = sensor.data.force
    assert forces is not None
    force_magnitude = torch.linalg.vector_norm(forces, dim=-1)
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    landing_force = force_magnitude * first_contact
    cost = landing_force.sum(dim=1)

    # 与 Unitree 相同：日志统计所有新触地脚，在运动指令门控之前计算。
    landing_count = first_contact.sum().clamp(min=1)
    env.extras["log"]["Metrics/landing_force_mean"] = landing_force.sum() / landing_count
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            speed = torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()
            cost = cost * (speed > command_threshold)
    return cost


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
