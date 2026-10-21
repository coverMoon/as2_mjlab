"""Check terrain-relative clearance and its sensor configuration."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from mjlab.sensor import TerrainHeightSensor

from as2_mjlab.tasks.velocity.env_cfg import as2_flat_env_cfg, as2_rough_env_cfg, LEGS
from as2_mjlab.tasks.velocity.mdp.rewards import feet_clearance
from as2_mjlab.tasks.velocity.sensors import TerrainProjectionSensorCfg


class TerrainRelativeClearanceTest(unittest.TestCase):
    def make_env(self, heights, command):
        sensor = Mock(spec=TerrainHeightSensor)
        sensor.data.heights = torch.tensor([heights])
        robot = SimpleNamespace(data=SimpleNamespace(
            site_pos_w=torch.full((1, 4, 3), 0.4),
            site_lin_vel_w=torch.tensor([[[1., 0., 0.], [0., 2., 0.],
                                          [0., 0., 0.], [1., 0., 0.]]]),
        ))
        return SimpleNamespace(
            scene={'robot': robot, 'foot_height_scan': sensor},
            command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([command])),
        )

    def cost(self, env):
        return feet_clearance(env, 0.1, 'foot_height_scan', command_name='twist',
                              command_threshold=0.1,
                              asset_cfg=SimpleNamespace(name='robot', site_ids=slice(None)))

    def test_high_world_position_with_correct_clearance_has_zero_cost(self):
        env = self.make_env([0.1, 0.1, 0.1, 0.1], [1., 0., 0.])
        torch.testing.assert_close(self.cost(env), torch.zeros(1))

    def test_speed_weighting_and_command_gate(self):
        env = self.make_env([0.2, 0.05, 0.1, 0.1], [1., 0., 0.])
        torch.testing.assert_close(self.cost(env), torch.tensor([0.2]))
        env.command_manager.get_command = lambda name: torch.zeros(1, 3)
        torch.testing.assert_close(self.cost(env), torch.zeros(1))

    def test_sensors_and_observation_contract(self):
        for factory in (as2_flat_env_cfg, as2_rough_env_cfg):
            for play in (False, True):
                cfg = factory(play=play)
                sensor = next(s for s in cfg.scene.sensors if s.name == 'foot_height_scan')
                self.assertIsInstance(sensor, TerrainProjectionSensorCfg)
                self.assertEqual(tuple(frame.name for frame in sensor.frame), LEGS)
                self.assertEqual(sensor.reduction, 'min')
                offsets, _ = sensor.pattern.generate_rays(None, 'cpu')
                self.assertEqual(offsets.shape, (5, 3))
                torch.testing.assert_close(offsets[:, 2], torch.zeros(5))
                self.assertEqual(sensor.ray_alignment, 'yaw')
                self.assertEqual(sensor.query_margin, 0.1)
                if not play:
                    self.assertFalse(sensor.debug_vis)
                self.assertEqual(cfg.rewards['foot_clearance'].params['height_sensor_name'], sensor.name)
                self.assertNotIn('foot_height_scan', str(cfg.observations))
                self.assertEqual(cfg.observations['critic'].terms['foot_height'].func.__name__, 'foot_height')


if __name__ == '__main__':
    unittest.main()
