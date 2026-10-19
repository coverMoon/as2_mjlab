"""AS2 flat velocity tracking environment."""

from mjlab.managers.observation_manager import ObservationGroupCfg

from .mdp import observations as as2_mdp

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
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

    # Configure foot height sensor.
    sensors = []
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            continue

        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=leg, entity="robot")
                for leg in LEGS
            )
            sensor.pattern = RingPatternCfg.single_ring(
                radius=0.04,
                num_samples=4,
            )

        sensors.append(sensor)

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

    # Flat terrain does not require terrain boundary checks.
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}

    return cfg