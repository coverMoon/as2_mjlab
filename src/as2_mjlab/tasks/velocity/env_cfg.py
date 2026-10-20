"""Shared AS2 velocity configuration with Flat and Rough terrain variants."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
)

from .mdp import observations as as2_mdp
from .mdp import rewards as as2_rewards

from .velocity_env_cfg import make_velocity_env_cfg

from as2_mjlab.robots.as2_constants import get_as2_robot_cfg


LEGS = ("FL", "FR", "RL", "RR")
FOOT_GEOMS = tuple(f"{leg}_foot_collision" for leg in LEGS)


def _as2_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Configure the AS2 robot, MDP, and common train/play behavior."""
    cfg = make_velocity_env_cfg()

    # Robot
    cfg.scene.entities = {"robot": get_as2_robot_cfg()}
    cfg.viewer.body_name = "base_link"

    # Keep the base terrain raycast for Rough; use site heights for feet.
    sensors = [
        sensor
        for sensor in (cfg.scene.sensors or ())
        if sensor.name != "foot_height_scan"
    ]
    for sensor in sensors:
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame = ObjRef(type="body", name="base_link", entity="robot")

    # Foot contact sensor
    feet_contact = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=FOOT_GEOMS,
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    nonfoot_contact = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=r".*_collision\d*$",
            exclude=FOOT_GEOMS,
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    cfg.scene.sensors = tuple(sensors) + (
        feet_contact,
        nonfoot_contact,
    )

    # Official AS2 privileged foot height observation.
    foot_height_term = cfg.observations["critic"].terms["foot_height"]

    foot_height_term.func = as2_mdp.foot_height
    foot_height_term.params = {
        "asset_cfg": SceneEntityCfg(
            "robot",
            site_names=LEGS,
        ),
    }

    # Align observation layout with Unitree AS2 Flat.
    actor_terms = cfg.observations["actor"].terms
    critic_terms = cfg.observations["critic"].terms
    actor_terms["joint_pos"].params["biased"] = False

    # Linear velocity is only available to the critic.
    actor_terms.pop("base_lin_vel", None)

    phase_cfg = ObservationTermCfg(
        func=as2_mdp.phase,
        params={
            "period": 0.6,
            "command_name": "twist",
        },
    )

    actor_terms["phase"] = phase_cfg
    critic_terms["phase"] = ObservationTermCfg(
        func=as2_mdp.phase,
        params={
            "period": 0.6,
            "command_name": "twist",
        },
    )

    # Explicit input ordering for future deployment.
    actor_order = (
        "base_ang_vel",
        "projected_gravity",
        "command",
        "phase",
        "joint_pos",
        "joint_vel",
        "actions",
        "height_scan",
    )

    critic_order = (
        *actor_order[:-1],
        "base_lin_vel",
        "height_scan",
        "foot_height",
        "foot_air_time",
        "foot_contact",
        "foot_contact_forces",
    )

    cfg.observations["actor"].terms = {
        name: actor_terms[name] for name in actor_order
    }

    cfg.observations["critic"].terms = {
        name: critic_terms[name] for name in critic_order
    }

    # Action
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = 0.25

    # Domain randomization
    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = FOOT_GEOMS

    cfg.events["base_com"].params[
        "asset_cfg"
    ].body_names = ("base_link",)

    # Robot-specific reward parameters
    pose = cfg.rewards["pose"].params
    pose["std_standing"] = {
        ".*hip_joint": 0.05,
        ".*thigh_joint": 0.1,
        ".*calf_joint": 0.15,
    }
    pose["std_walking"] = {
        ".*hip_joint": 0.15,
        ".*thigh_joint": 0.35,
        ".*calf_joint": 0.5,
    }
    pose["std_running"] = dict(pose["std_walking"])

    cfg.rewards["upright"].params[
        "asset_cfg"
    ].body_names = ("base_link",)

    cfg.rewards["body_ang_vel"].params[
        "asset_cfg"
    ].body_names = ("base_link",)

    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params[
            "asset_cfg"
        ].site_names = LEGS

    # --------------------------------------------------
    # Unitree AS2 baseline: velocity tracking rewards
    # --------------------------------------------------

    cfg.rewards["track_linear_velocity"].func = (
        as2_rewards.track_linear_velocity
    )
    cfg.rewards["track_linear_velocity"].weight = 1.0

    cfg.rewards["track_angular_velocity"].func = (
        as2_rewards.track_angular_velocity
    )
    cfg.rewards["track_angular_velocity"].weight = 1.0

    # --------------------------------------------------
    # Unitree AS2 baseline: body orientation penalty
    # --------------------------------------------------

    cfg.rewards.pop("upright", None)

    cfg.rewards["body_orientation_l2"] = RewardTermCfg(
        func=envs_mdp.flat_orientation_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # --------------------------------------------------
    # Official AS2 foot clearance
    # --------------------------------------------------

    cfg.rewards["foot_clearance"].func = as2_rewards.feet_clearance
    cfg.rewards["foot_clearance"].weight = -1.0

    cfg.rewards["foot_clearance"].params = {
        "target_height": 0.10,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg(
            "robot",
            site_names=LEGS,
        ),
    }

    # --------------------------------------------------
    # Official AS2 gait reward
    # --------------------------------------------------

    cfg.rewards["foot_gait"] = RewardTermCfg(
        func=as2_rewards.feet_gait,
        weight=0.5,
        params={
            "period": 0.6,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "threshold": 0.56,
            "command_threshold": 0.1,
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
        },
    )

    # --------------------------------------------------
    # Unitree AS2 baseline: remaining reward alignment
    # --------------------------------------------------

    # Pose
    cfg.rewards["pose"].params["walking_threshold"] = 0.1
    cfg.rewards["pose"].params["running_threshold"] = 1.5

    # Body angular velocity and momentum
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.025

    # Termination penalty
    cfg.rewards["is_terminated"] = RewardTermCfg(
        func=envs_mdp.is_terminated,
        weight=-200.0,
    )

    # Joint acceleration
    cfg.rewards["joint_acc_l2"] = RewardTermCfg(
        func=envs_mdp.joint_acc_l2,
        weight=-2.5e-7,
    )

    # Joint position limits
    cfg.rewards["joint_pos_limits"] = cfg.rewards.pop(
        "dof_pos_limits"
    )
    cfg.rewards["joint_pos_limits"].weight = -10.0

    # Action smoothness
    cfg.rewards["action_rate_l2"].weight = -0.05

    # Foot slip
    cfg.rewards["foot_slip"].weight = -0.25
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.1

    # Soft landing
    cfg.rewards["soft_landing"].weight = -1e-3
    cfg.rewards["soft_landing"].params["command_threshold"] = 0.1

    # Stand still
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=as2_rewards.stand_still,
        weight=-1.0,
        params={
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(".*",),
            ),
        },
    )

    # Not part of the official AS2 reward set.
    cfg.rewards.pop("air_time", None)

    # MjLab's swing height term depends on the removed
    # TerrainHeightSensor and is absent from Unitree baseline.
    cfg.rewards.pop("foot_swing_height", None)

    # Match Unitree AS2: orientation and illegal contact handle termination.
    cfg.terminations.pop("out_of_terrain_bounds", None)

    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=velocity_mdp.illegal_contact,
        params={
            "sensor_name": "nonfoot_ground_touch",
            "force_threshold": 10.0,
        },
    )

    if play:
        # Effectively disable episode timeout.
        cfg.episode_length_s = int(1e9)

        # Disable observation corruption and push events.
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        # Disable curriculum learning.
        cfg.curriculum = {}

    return cfg


def as2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Preserve the existing Flat observation/checkpoint and training contract."""
    cfg = _as2_velocity_env_cfg(play=play)
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    cfg.sim.njmax = 300
    cfg.sim.nconmax = None
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64

    cfg.scene.sensors = tuple(
        sensor for sensor in cfg.scene.sensors or ()
        if sensor.name != "terrain_scan"
    )
    for group in ("actor", "critic"):
        cfg.observations[group].terms.pop("height_scan", None)
    cfg.curriculum.pop("terrain_levels", None)

    # Keep the user's existing Flat velocity curriculum; Rough uses the
    # official stages from the local make_velocity_env_cfg() template.
    if not play:
        cfg.curriculum["command_vel"].params["velocity_stages"] = [
            {
                "step": 0,
                "lin_vel_x": (-0.5, 1.0),
                "lin_vel_y": (-0.3, 0.3),
                "ang_vel_z": (-0.5, 0.5),
            },
            {
                "step": 500 * 24,
                "lin_vel_x": (-2.0, 3.0),
                "lin_vel_y": (-1.0, 1.0),
                "ang_vel_z": (-3.14, 3.14),
            },
        ]

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
        twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
        twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

    return cfg


def as2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """AS2 Rough with MjLab 1.6 terrain curriculum and 187 height rays."""
    cfg = _as2_velocity_env_cfg(play=play)
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    assert cfg.scene.terrain is not None
    assert cfg.scene.terrain.terrain_generator is not None
    # The base factory uses a shallow dataclass replacement. Isolate nested
    # terrain configs from the global preset and other task/play instances.
    generator = deepcopy(cfg.scene.terrain.terrain_generator)
    cfg.scene.terrain.terrain_generator = generator
    generator.curriculum = not play

    if play:
        # In 1.6 randomize_terrain only changes origins. Run it before the
        # root reset so the robot is placed on the newly selected patch.
        cfg.events = {
            "randomize_terrain": EventTermCfg(
                func=envs_mdp.randomize_terrain,
                mode="reset",
                params={},
            ),
            **cfg.events,
        }
        generator.num_rows = 5
        generator.num_cols = 5
        generator.border_width = 10.0

    return cfg
