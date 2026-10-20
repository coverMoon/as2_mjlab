"""CPU-only configuration regression checks; no viewer or GPU required.

Run: uv run python -m unittest discover -s tests
"""

import unittest

from mjlab.tasks.registry import list_tasks
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

from as2_mjlab.tasks.velocity.env_cfg import as2_flat_env_cfg, as2_rough_env_cfg


ACTOR_ORDER = (
    "base_ang_vel", "projected_gravity", "command", "phase",
    "joint_pos", "joint_vel", "actions",
)
CRITIC_ORDER = ACTOR_ORDER + (
    "base_lin_vel", "foot_height", "foot_air_time", "foot_contact",
    "foot_contact_forces",
)
REWARD_WEIGHTS = {
    "track_linear_velocity": 1.0, "track_angular_velocity": 1.0,
    "pose": 1.0, "body_ang_vel": -0.05, "angular_momentum": -0.025,
    "action_rate_l2": -0.05, "foot_clearance": -1.0, "foot_slip": -0.25,
    "soft_landing": -1e-3, "body_orientation_l2": -1.0, "foot_gait": 0.5,
    "is_terminated": -200.0, "joint_acc_l2": -2.5e-7,
    "joint_pos_limits": -10.0, "stand_still": -1.0,
}


class VelocityConfigTest(unittest.TestCase):
    def test_registration(self):
        self.assertTrue({"as2-flat", "as2-rough"}.issubset(list_tasks()))

    def test_flat_contract(self):
        cfg = as2_flat_env_cfg()
        self.assertEqual(tuple(cfg.observations["actor"].terms), ACTOR_ORDER)
        self.assertEqual(tuple(cfg.observations["critic"].terms), CRITIC_ORDER)
        self.assertEqual({k: v.weight for k, v in cfg.rewards.items()}, REWARD_WEIGHTS)
        self.assertEqual(cfg.scene.terrain.terrain_type, "plane")
        self.assertIsNone(cfg.scene.terrain.terrain_generator)
        self.assertEqual({s.name for s in cfg.scene.sensors},
                         {"feet_ground_contact", "nonfoot_ground_touch"})
        self.assertEqual((cfg.sim.njmax, cfg.sim.nconmax), (300, None))
        self.assertEqual(cfg.sim.mujoco.ccd_iterations, 50)
        self.assertEqual(cfg.sim.contact_sensor_maxmatch, 64)
        self.assertEqual(set(cfg.curriculum), {"command_vel"})
        self.assertEqual(cfg.curriculum["command_vel"].params["velocity_stages"][1]["step"], 500 * 24)

    def test_rough_shared_contract(self):
        flat, rough = as2_flat_env_cfg(), as2_rough_env_cfg()
        self.assertEqual(flat.rewards, rough.rewards)
        self.assertEqual(flat.actions, rough.actions)
        self.assertEqual(flat.events, rough.events)
        self.assertEqual(flat.terminations, rough.terminations)
        self.assertEqual(rough.sim.mujoco.timestep * rough.decimation, 0.02)
        self.assertEqual(rough.episode_length_s, 20.0)
        self.assertEqual(rough.sim.mujoco.ccd_iterations, 500)
        self.assertEqual(rough.sim.contact_sensor_maxmatch, 500)
        self.assertEqual(rough.scene.terrain.terrain_type, "generator")
        self.assertTrue(rough.scene.terrain.terrain_generator.curriculum)
        self.assertEqual(set(rough.curriculum), {"command_vel", "terrain_levels"})
        self.assertEqual(rough.curriculum["command_vel"].params["velocity_stages"][1]["step"], 5000 * 24)
        self.assertEqual(tuple(rough.observations["actor"].terms), ACTOR_ORDER + ("height_scan",))
        self.assertEqual(tuple(rough.observations["critic"].terms),
                         ACTOR_ORDER + ("base_lin_vel", "height_scan") + CRITIC_ORDER[8:])
        scan = next(s for s in rough.scene.sensors if s.name == "terrain_scan")
        self.assertEqual(scan.frame.name, "base_link")
        offsets, _ = scan.pattern.generate_rays(None, "cpu")
        self.assertEqual(offsets.shape, (187, 3))
        self.assertNotIn("foot_height_scan", {s.name for s in rough.scene.sensors})

    def test_play_and_config_isolation(self):
        train = as2_rough_env_cfg()
        play = as2_rough_env_cfg(play=True)
        generator = play.scene.terrain.terrain_generator
        self.assertEqual((generator.num_rows, generator.num_cols, generator.border_width), (5, 5, 10.0))
        self.assertFalse(generator.curriculum)
        self.assertTrue(train.scene.terrain.terrain_generator.curriculum)
        self.assertEqual(play.curriculum, {})
        self.assertIn("randomize_terrain", play.events)
        reset_order = [name for name, event in play.events.items() if event.mode == "reset"]
        self.assertLess(reset_order.index("randomize_terrain"), reset_order.index("reset_base"))
        self.assertNotIn("push_robot", play.events)
        self.assertFalse(play.observations["actor"].enable_corruption)
        self.assertEqual(play.episode_length_s, 1_000_000_000)
        self.assertEqual(play.commands["twist"].ranges.ang_vel_z, (-1.0, 1.0))
        flat_play = as2_flat_env_cfg(play=True)
        self.assertEqual(flat_play.commands["twist"].ranges.ang_vel_z, (-0.5, 0.5))
        self.assertNotIn("randomize_terrain", flat_play.events)
        generator.sub_terrains["flat"].proportion = 0.9
        self.assertEqual(train.scene.terrain.terrain_generator.sub_terrains["flat"].proportion, 0.2)
        self.assertEqual(ROUGH_TERRAINS_CFG.sub_terrains["flat"].proportion, 0.2)


if __name__ == "__main__":
    unittest.main()
