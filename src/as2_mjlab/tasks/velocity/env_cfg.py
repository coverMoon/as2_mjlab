"""AS2 flat velocity tracking environment."""

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg
)

from .mdp import observations as as2_mdp
from .mdp import rewards as as2_rewards

from .velocity_env_cfg import make_velocity_env_cfg

from as2_mjlab.robots.as2_constants import get_as2_robot_cfg


LEGS = ("FR", "FL", "RR", "RL")
FOOT_GEOMS = tuple(f"{leg}_foot_collision" for leg in LEGS)


def as2_flat_env_cfg(play: bool = False):
    cfg = make_velocity_env_cfg()

    # Robot
    cfg.scene.entities = {"robot": get_as2_robot_cfg()}
    cfg.viewer.body_name = "base_link"

    # Flat terrain
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    # Flat baseline uses foot site positions directly.
    # Terrain raycast and foot height sensors are not required.
    sensors = [
        sensor
        for sensor in (cfg.scene.sensors or ())
        if sensor.name not in ("terrain_scan", "foot_height_scan")
    ]

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
    cfg.scene.sensors = tuple(sensors) + (feet_contact,)

    # Remove terrain height observations on flat ground.
    for group in ("actor", "critic"):
        cfg.observations[group].terms.pop("height_scan", None)

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
    )

    critic_order = (
        *actor_order,
        "base_lin_vel",
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

    # Flat terrain does not require terrain boundary checks.
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}

    return cfg