"""Validate termination geom matching against the actual AS2 entity."""

import unittest

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor

from as2_mjlab.robots.as2_constants import get_as2_robot_cfg
from as2_mjlab.tasks.velocity.env_cfg import as2_flat_env_cfg, as2_rough_env_cfg


class RoughContactTerminationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = Entity(get_as2_robot_cfg())

    def test_calf_excluded_only_on_rough(self):
        for play in (False, True):
            with self.subTest(play=play):
                flat, rough = as2_flat_env_cfg(play), as2_rough_env_cfg(play)
                for cfg, expect_calves in ((flat, True), (rough, False)):
                    sensor_cfg = next(s for s in cfg.scene.sensors if s.name == 'nonfoot_ground_touch')
                    sensor = ContactSensor(sensor_cfg)
                    names = sensor._resolve_primary_names({'robot': self.robot}, sensor_cfg.primary)
                    calves = [name for name in names if '_calf' in name]
                    self.assertEqual(len(calves), 12 if expect_calves else 0)
                    self.assertIn('base1_collision', names)
                    self.assertIn('FR_thigh3_collision', names)
                    self.assertNotIn('FR_foot_collision', names)
                    self.assertIn('illegal_contact', cfg.terminations)
                self.assertEqual(flat.rewards, rough.rewards)
                self.assertEqual(flat.scene.entities, rough.scene.entities)


if __name__ == '__main__':
    unittest.main()
