"""PPO velocity sampling with terrain-dependent forward-speed groups."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.terrains import BoxFlatTerrainCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class TerrainVelocityGroupsCfg:
    """按 |vx|（m/s）划分组；概率只作用于 Rough 中的平地环境。"""

    enabled: bool = False
    low_speed_range: tuple[float, float] = (0.0, 1.0)
    high_speed_range: tuple[float, float] = (1.0, 3.0)
    high_speed_probability: float = 0.5

    def __post_init__(self):
        low_min, low_max = self.low_speed_range
        high_min, high_max = self.high_speed_range
        if not 0 <= low_min < low_max <= high_min < high_max:
            raise ValueError("Speed groups must satisfy 0 <= low_min < low_max <= high_min < high_max.")
        if not 0 <= self.high_speed_probability <= 1:
            raise ValueError("high_speed_probability must be between 0 and 1.")


class TerrainVelocityCommand(UniformVelocityCommand):
    cfg: "TerrainVelocityCommandCfg"

    def __init__(self, cfg: "TerrainVelocityCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        generator = env.scene.terrain.cfg.terrain_generator
        # 整齐布局中每列对应一种地形；不能把随机布局的列号当成地形类型。
        if not generator.curriculum:
            raise ValueError("Terrain velocity groups require terrain_generator.curriculum=True.")
        self._plane_types = torch.tensor(
            [isinstance(terrain, BoxFlatTerrainCfg) for terrain in generator.sub_terrains.values()],
            device=self.device,
            dtype=torch.bool,
        )

    def _sample_x(self, count: int, speed_range: tuple[float, float]) -> torch.Tensor | None:
        """在当前阶段 X 范围与 ±|vx| 分组的交集中均匀采样，保留前后方向。"""
        stage_min, stage_max = self.cfg.ranges.lin_vel_x
        speed_min, speed_max = speed_range
        neg_min, neg_max = max(stage_min, -speed_max), min(stage_max, -speed_min)
        pos_min, pos_max = max(stage_min, speed_min), min(stage_max, speed_max)
        neg_width = max(0.0, neg_max - neg_min)
        pos_width = max(0.0, pos_max - pos_min)
        width = neg_width + pos_width
        if width == 0:
            # 固定 X 指令仍可使用；只有边界相交不算可采样的高速区间。
            if stage_min == stage_max and speed_min <= abs(stage_min) <= speed_max:
                return torch.full((count,), stage_min, device=self.device)
            return None
        sample = torch.rand(count, device=self.device) * width
        return torch.where(sample < neg_width, neg_min + sample, pos_min + sample - neg_width)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # 沿用站立比例、heading、vy 和 yaw 的原始采样逻辑。
        super()._resample_command(env_ids)
        groups = self.cfg.terrain_velocity_groups
        low_x = self._sample_x(len(env_ids), groups.low_speed_range)
        if low_x is None:
            raise ValueError("Current lin_vel_x stage does not intersect the low-speed group.")
        self.vel_command_b[env_ids, 0] = low_x

        # 每次重采样读取当前地形类型，reset 重新分配地形后也会立即生效。
        terrain_types = self._env.scene.terrain.terrain_types[env_ids]
        is_plane = self._plane_types[terrain_types]
        use_high = is_plane & (torch.rand(len(env_ids), device=self.device) < groups.high_speed_probability)
        high_ids = env_ids[use_high]
        high_x = self._sample_x(len(high_ids), groups.high_speed_range)
        if high_x is not None:
            self.vel_command_b[high_ids, 0] = high_x
        # 若当前课程尚未开放高速区间，plane 同样使用低速组。
        forward_ids = env_ids[self.is_forward_env[env_ids]]
        self.vel_command_b[forward_ids, 0] = self.vel_command_b[forward_ids, 0].abs()
        self.vel_command_w[env_ids, 0] = self.vel_command_b[env_ids, 0]


@dataclass(kw_only=True)
class TerrainVelocityCommandCfg(UniformVelocityCommandCfg):
    terrain_velocity_groups: TerrainVelocityGroupsCfg = field(default_factory=TerrainVelocityGroupsCfg)

    def build(self, env: "ManagerBasedRlEnv") -> UniformVelocityCommand:
        # Flat 和关闭开关时直接使用原版采样器；只改变 Rough 的训练采样。
        if not self.terrain_velocity_groups.enabled or env.scene.terrain.cfg.terrain_type == "plane":
            return UniformVelocityCommand(self, env)
        return TerrainVelocityCommand(self, env)
