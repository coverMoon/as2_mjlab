"""SLR Flat / Rough 环境参数，在本文件独立定义，便于单独调参。

默认值保持展开前的 SLR 配置：环境、奖励和碰撞规则与对应 AS2 PPO 一致。
只维护 history 一组本体观测，形状 [B, 10, 45]，按旧到新排列；Actor 和
Critic 的当前帧均取 history[:, -1, :]，不提供特权观测。MjLab 原生缓冲区
负责推进和 Reset，重置后的第一帧填满历史窗口。

本文件不调用 PPO 环境工厂；奖励和指令的计算函数继续复用 PPO MDP 模块。
机器人模型及执行器的公共定义仍在 robots/as2_constants.py。
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import (
    UniformVelocityCommandCfg,
    commands_vel,
    illegal_contact,
    terrain_levels_vel,
)
from mjlab.terrains import TerrainEntityCfg
import mjlab.terrains as terrain_gen
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

from as2_mjlab.robots.as2_constants import get_as2_robot_cfg
from as2_mjlab.sensors import TerrainProjectionSensorCfg
from as2_mjlab.tasks.ppo.mdp import rewards as ppo_rewards
from as2_mjlab.tasks.ppo.mdp.commands import TerrainVelocityCommandCfg, TerrainVelocityGroupsCfg


LEGS = ("FL", "FR", "RL", "RR")
FOOT_GEOMS = tuple(f"{leg}_foot_collision" for leg in LEGS)

# Flat / Rough 训练共用指令课程；Play 始终关闭课程。
ENABLE_COMMAND_CURRICULUM = True
# 保留原课程的环境 step 边界；此处显式定义，不再读取 PPO 的训练配置。
COMMAND_CURRICULUM_STEPS_PER_ITERATION = 24


def make_slr_env_cfg() -> ManagerBasedRlEnvCfg:
    """构造共享 SLR 配置：唯一的 10×45 本体历史，默认使用平地。"""

    # ------------------------------------------------------------------
    # 传感器：保留地形扫描用于 Play 可视化；足端扫描用于离地高度奖励。
    # 扫描与接触传感器均不进入 SLR Actor / Critic 的观测。
    # ------------------------------------------------------------------
    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),  # 仅扫描地形，忽略机器人碰撞体。
        debug_vis=False,
    )
    foot_height_scan = TerrainProjectionSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=leg, entity="robot") for leg in LEGS),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(
            radius=0.04,
            num_samples=4,
            include_center=True,
        ),
        reduction="min",  # 五个采样点中最小净空，对应局部最高地面。
        query_margin=0.1,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
        viz=TerrainProjectionSensorCfg.VizCfg(
            show_rays=True,
            hit_color=(1.0, 0.0, 1.0, 0.8),
            hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
            hit_sphere_radius=0.15,
        ),
    )
    feet_contact = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=FOOT_GEOMS,
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,  # 同时跟踪连续触地时长和空中时长。
    )
    nonfoot_contact = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=r".*_collision\d*$",
            exclude=FOOT_GEOMS,
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,  # 终止判断检查短时间内的碰撞峰值。
    )

    # ------------------------------------------------------------------
    # 观测：字典顺序就是输入拼接顺序，部署时需要保持一致。
    # history 只使用本体状态；噪声在 scale 之前加入，每帧仅采样一次。
    # ------------------------------------------------------------------
    proprioceptive_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel,
            scale=0.25,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        ),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
            scale=(2.0, 2.0, 0.25),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"biased": False},
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    observations = {
        "history": ObservationGroupCfg(
            terms=proprioceptive_terms,
            enable_corruption=True,
            history_length=10,
            flatten_history_dim=False,
            concatenate_terms=True,
            concatenate_dim=-1,
        ),
    }
    # 原生 term 缓冲区为 [B, H, D_term]，沿特征维拼接得到逐帧的 [B, 10, 45]。

    # ------------------------------------------------------------------
    # 动作与指令：关节位置残差控制，速度命令使用机身坐标系。
    # ------------------------------------------------------------------
    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.25,
            use_default_offset=True,  # q_target = q_default + 0.25 * action。
        ),
    }
    commands = {
        "twist": TerrainVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.05,
            rel_heading_envs=1.0,
            rel_forward_envs=0.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            # 不启用课程时使用这些范围；课程会在 reset 时更新它们。
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 2.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
        ),
    }

    # ------------------------------------------------------------------
    # 奖励：XY / Yaw 跟踪与 Z / Roll / Pitch 惩罚分别计算。
    # RewardManager 统一乘 weight * step_dt，函数内不再重复乘 dt。
    # ------------------------------------------------------------------
    rewards = {
        "tracking_lin_vel": RewardTermCfg(
            func=ppo_rewards.tracking_lin_vel,
            weight=2.0,
            params={"tracking_sigma": 0.25, "command_name": "twist"},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func=ppo_rewards.tracking_ang_vel,
            weight=1.5,
            params={"tracking_sigma": 0.25, "command_name": "twist"},
        ),
        "lin_vel_z": RewardTermCfg(
            func=ppo_rewards.lin_vel_z,
            weight=-2.0,
        ),
        "ang_vel_xy": RewardTermCfg(
            func=ppo_rewards.ang_vel_xy,
            weight=-0.05,
        ),
        "orientation": RewardTermCfg(
            func=mdp.flat_orientation_l2,
            weight=-0.2,
        ),
        # 与 velocity 当前最终配置一致；参数独立定义，方便 SLR 单独调节。
        "pose": RewardTermCfg(
            func=ppo_rewards.pose,
            weight=0.5,
            params={
                "command_name": "twist",
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "walking_threshold": 0.1,
                "running_threshold": 1.5,
                "std_standing": {
                    ".*hip_joint": 0.05,
                    ".*thigh_joint": 0.1,
                    ".*calf_joint": 0.15,
                },
                "std_walking": {
                    ".*hip_joint": 0.15,
                    ".*thigh_joint": 0.35,
                    ".*calf_joint": 0.5,
                },
                "std_running": {
                    ".*hip_joint": 0.15,
                    ".*thigh_joint": 0.35,
                    ".*calf_joint": 0.5,
                },
            },
        ),
        "foot_clearance": RewardTermCfg(
            func=ppo_rewards.feet_clearance,
            weight=-1.0,  # 函数返回正 cost，负号和 dt 只在 RewardManager 施加。
            params={
                "target_height": 0.10,
                "height_sensor_name": "foot_height_scan",
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    site_names=LEGS,
                    preserve_order=True,  # 与传感器 frame 的四脚顺序一致。
                ),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=ppo_rewards.soft_landing,
            weight=-1e-3,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.1,
            },
        ),
        "torques": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-1e-5,
        ),
        "joint_acc": RewardTermCfg(
            func=mdp.joint_acc_l2,
            weight=-2.5e-7,
        ),
        "action_rate": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.01,
        ),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
        ),
        "termination": RewardTermCfg(
            func=ppo_rewards.termination,
            weight=-200.0,
        ),
    }

    # ------------------------------------------------------------------
    # 重置与 Domain Randomization：startup 采样物理扰动，interval 施加推力。
    # ------------------------------------------------------------------
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 6.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        ),
        "foot_friction": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
                "operation": "abs",
                "ranges": (0.3, 1.6),
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            func=dr.encoder_bias,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "base_com": EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
                "operation": "add",
                "ranges": {
                    0: (-0.05, 0.05),
                    1: (-0.05, 0.05),
                    2: (-0.05, 0.05),
                },
            },
        ),
    }

    # 超时供 PPO bootstrap 使用；翻倒和非足端碰撞属于失败终止。
    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70)},
        ),
        "illegal_contact": TerminationTermCfg(
            func=illegal_contact,
            params={
                "sensor_name": "nonfoot_ground_touch",
                "force_threshold": 10.0,
            },
        ),
    }

    # ------------------------------------------------------------------
    # 指令课程：step 对应全局 common_step_counter，不是 iteration。
    # 所有并行环境每次推进时只计一步，不乘环境数量，也不随 episode 重置。
    # 按默认 rollout 长度换算为约第 0 / 500 / 1000 / 1500 次 iteration。
    # 实际范围在 reset 时按 common_step_counter 更新，超过边界后分段切换。
    # ------------------------------------------------------------------
    num_steps_per_env = COMMAND_CURRICULUM_STEPS_PER_ITERATION
    velocity_stages = [
        {
            "step": 0,
            "lin_vel_x": (-1.0, 1.0),
            "lin_vel_y": (-0.3, 0.3),
            "ang_vel_z": (-0.5, 0.5),
        },
        {
            "step": 500 * num_steps_per_env,
            "lin_vel_x": (-1.5, 1.5),
            "lin_vel_y": (-0.5, 0.5),
            "ang_vel_z": (-1.3, 1.3),
        },
        {
            "step": 1000 * num_steps_per_env,
            "lin_vel_x": (-1.75, 1.75),
            "lin_vel_y": (-0.8, 0.8),
            "ang_vel_z": (-2.5, 2.5),
        },
        {
            "step": 1500 * num_steps_per_env,
            "lin_vel_x": (-2.0, 2.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-3.14, 3.14),
        },
        {
            "step": 3000 * num_steps_per_env,
            "lin_vel_x": (-3.0, 3.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-3.14, 3.14),
        },
    ]
    # Rough 专用：按 |vx| 分组（m/s），与当前 velocity_stages 范围取交集。
    # plane 两组混合采样，其余地形只用低速；vy / yaw 不参与分组。
    # 高速区间尚未开放时回退到低速组。Flat / Play 使用原版采样器。
    terrain_velocity_groups = TerrainVelocityGroupsCfg(
        enabled=True,
        low_speed_range=(0.0, 1.0),
        high_speed_range=(1.0, 3.0),
        high_speed_probability=0.5,
    )
    commands["twist"].terrain_velocity_groups = terrain_velocity_groups

    curriculum = {}
    if ENABLE_COMMAND_CURRICULUM:
        curriculum["command_vel"] = CurriculumTermCfg(
            func=commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": velocity_stages,
            },
        )

    # ------------------------------------------------------------------
    # 场景与仿真：0.005 s * 4 = 0.02 s，每秒执行 50 次策略推理。
    # ------------------------------------------------------------------
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={"robot": get_as2_robot_cfg()},
            terrain=TerrainEntityCfg(terrain_type="plane"),
            sensors=(terrain_scan, foot_height_scan, feet_contact, nonfoot_contact),
            num_envs=1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        rewards=rewards,
        events=events,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base_link",
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            njmax=300,
            nconmax=None,
            contact_sensor_maxmatch=64,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=50,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
        scale_rewards_by_dt=True,
    )


def _configure_play(cfg: ManagerBasedRlEnvCfg) -> None:
    """关闭训练扰动、课程和短时限，方便持续观察策略表现。"""
    cfg.episode_length_s = int(1e9)
    cfg.observations["history"].enable_corruption = False
    cfg.events.pop("push_robot")
    cfg.curriculum = {}
    cfg.commands["twist"].terrain_velocity_groups.enabled = False
    for sensor in cfg.scene.sensors:
        if sensor.name in ("terrain_scan", "foot_height_scan"):
            sensor.debug_vis = True
            sensor.viz.show_rays = sensor.name == "foot_height_scan"


def slr_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """平地：保留指令课程，不加入地形难度课程。"""
    cfg = make_slr_env_cfg()
    if play:
        _configure_play(cfg)
        twist = cfg.commands["twist"]
        twist.ranges.lin_vel_x = (-0.5, 1.0)
        twist.ranges.lin_vel_y = (-0.5, 0.5)
        twist.ranges.ang_vel_z = (-0.5, 0.5)
    return cfg


def slr_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """崎岖地形：共享 SLR 配置，追加地形难度课程和碰撞监测差异。"""
    cfg = make_slr_env_cfg()

    # Rough 地形参数在此完整列出，不引用上游 ROUGH_TERRAINS_CFG。
    # 子地形 size 是模板值，生成时统一使用 generator.size。
    generator = terrain_gen.TerrainGeneratorCfg(
        seed=None,
        curriculum=True,
        size=(8.0, 8.0),
        border_width=20.0,
        border_height=1.0,
        num_rows=10,
        num_cols=20,
        color_scheme="height",
        sub_terrains={
            "flat": terrain_gen.BoxFlatTerrainCfg(
                proportion=0.1,
                size=(10.0, 10.0),
                flat_patch_sampling=None,
            ),
            "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
                proportion=0.3,
                size=(10.0, 10.0),
                flat_patch_sampling=None,
                border_width=1.0,
                step_height_range=(0.0, 0.3),
                step_width=0.3,
                platform_width=3.0,
                holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
                proportion=0.2,
                size=(10.0, 10.0),
                flat_patch_sampling=None,
                border_width=1.0,
                step_height_range=(0.0, 0.3),
                step_width=0.3,
                platform_width=3.0,
                holes=False,
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1,
                size=(10.0, 10.0),
                flat_patch_sampling=None,
                slope_range=(0.0, 1.0),
                platform_width=2.0,
                inverted=False,
                border_width=0.25,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                base_thickness_ratio=1.0,
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1,
                size=(10.0, 10.0),
                flat_patch_sampling=None,
                slope_range=(0.0, 1.0),
                platform_width=2.0,
                inverted=True,
                border_width=0.25,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                base_thickness_ratio=1.0,
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.1,
                size=(10.0, 10.0),
                flat_patch_sampling=None,
                noise_range=(0.02, 0.1),
                noise_step=0.02,
                downsampled_scale=None,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                base_thickness_ratio=1.0,
                border_width=0.25,
                scale_with_difficulty=False,
            ),
            "wave_terrain": terrain_gen.HfWaveTerrainCfg(
                proportion=0.1,
                size=(10.0, 10.0),
                flat_patch_sampling=None,
                amplitude_range=(0.0, 0.2),
                num_waves=4,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                base_thickness_ratio=0.25,
                border_width=0.25,
            ),
        },
        difficulty_range=(0.0, 1.0),
        add_lights=True,
    )

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=5,
    )
    cfg.sim.contact_sensor_maxmatch = 500

    # 大腿接触改为奖励惩罚，小腿接触继续允许；机身等其余部位仍触发终止。
    # 仅调整监测，所有碰撞物理仍然生效。
    for sensor in cfg.scene.sensors:
        if sensor.name == "nonfoot_ground_touch":
            sensor.primary.exclude = (
                *FOOT_GEOMS,
                r".*_calf\d*_collision\d*$",
                r".*_thigh\d*_collision\d*$",
            )

    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name="thigh_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=tuple(f"{leg}_thigh" for leg in LEGS),
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="maxforce",  # 每条大腿取最强接触，不重复统计多个碰撞体。
        num_slots=1,
        history_length=4,
    ))
    cfg.rewards["thigh_contact"] = RewardTermCfg(
        func=ppo_rewards.thigh_contact,
        weight=-1.0,
        params={"sensor_name": "thigh_ground_contact", "force_threshold": 10.0},
    )

    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=terrain_levels_vel,
        params={"command_name": "twist"},
    )
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    if play:
        _configure_play(cfg)
        # 先选择地形，再将 root 重置到新地块；字典顺序决定 reset 事件顺序。
        cfg.events = {
            "randomize_terrain": EventTermCfg(
                func=mdp.randomize_terrain,
                mode="reset",
            ),
            **cfg.events,
        }
    return cfg
