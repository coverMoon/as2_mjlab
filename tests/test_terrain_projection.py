"""Scene-height bounds, real stair-edge queries, and clearance-only display."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import torch
from mjlab.sensor import ObjRef, RingPatternCfg

from as2_mjlab.tasks.velocity.sensors import (
    TerrainProjectionSensor, TerrainProjectionSensorCfg, terrain_height_bounds,
)


class TerrainProjectionTest(unittest.TestCase):
    def test_step_above_foot_is_hit_without_biasing_clearance(self):
        model = mujoco.MjModel.from_xml_string('''
            <mujoco><worldbody>
              <geom type="plane" size="2 2 0.1"/>
              <geom type="box" size="0.5 0.5 0.15" pos="0.5 0 0.15"/>
            </worldbody></mujoco>
        ''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        low, high = terrain_height_bounds(model, (0,))
        self.assertAlmostEqual(low, 0.)
        self.assertAlmostEqual(high, 0.3)
        offsets, directions = RingPatternCfg.single_ring(radius=0.04, num_samples=4).generate_rays(None, 'cpu')
        site = np.array([-0.02, 0., 0.1])
        clearances = []
        for offset, direction in zip(offsets.numpy(), directions.numpy(), strict=True):
            origin = site + offset
            origin[2] = high + 0.1
            distance = mujoco.mj_ray(model, data, origin, direction.astype(np.float64),
                                     None, True, -1, np.zeros(1, dtype=np.int32))
            self.assertGreaterEqual(distance, 0.)
            clearances.append(site[2] - (origin[2] - distance))
        self.assertAlmostEqual(clearances[0], 0.1)
        self.assertAlmostEqual(clearances[1], -0.2)

    def test_bounds_include_hfield_and_exclude_robot_and_other_groups(self):
        model = mujoco.MjModel.from_xml_string('''
            <mujoco><asset>
              <hfield name="h" nrow="2" ncol="2" size="1 1 .5 .1" elevation="0 1 0 1"/>
            </asset><worldbody>
              <geom type="hfield" hfield="h" pos="0 0 2"/>
              <geom type="box" group="1" size="1 1 1" pos="0 0 50"/>
              <body pos="0 0 100"><freejoint/><geom type="sphere" size="1"/></body>
            </worldbody></mujoco>
        ''')
        low, high = terrain_height_bounds(model, (0,))
        self.assertAlmostEqual(low, 1.9)
        self.assertAlmostEqual(high, 2.5)

    def test_bounds_transform_rotated_boxes(self):
        model = mujoco.MjModel.from_xml_string('''
            <mujoco><compiler angle="degree"/><worldbody>
              <geom type="box" size=".5 .2 .1" pos="0 0 2" euler="45 0 0"/>
            </worldbody></mujoco>
        ''')
        low, high = terrain_height_bounds(model, (0,))
        self.assertAlmostEqual(low, 2 - 0.3 / np.sqrt(2))
        self.assertAlmostEqual(high, 2 + 0.3 / np.sqrt(2))

    def test_visualization_never_draws_from_query_plane(self):
        cfg = TerrainProjectionSensorCfg(name='feet', frame=ObjRef(type='site', name='FL'), debug_vis=True)
        cfg.viz.show_rays = True
        sensor = TerrainProjectionSensor(cfg)
        sensor._num_frames, sensor._num_rays_per_frame = 1, 3
        sensor._cached_world_origins = torch.tensor([[[0., 0., 2.], [.04, 0., 2.], [-.04, 0., 2.]]])
        sensor._cached_frame_pos = torch.tensor([[[0., 0., .1]]])
        data = SimpleNamespace(
            hit_pos_w=torch.tensor([[[0., 0., 0.], [.04, 0., .3], [-.04, 0., 2.]]]),
            distances=torch.tensor([[2., 1.7, -1.]]),
            normals_w=torch.tensor([[[0., 0., 1.]]]).repeat(1,3,1),
        )
        viz = Mock(meansize=0.1)
        viz.get_env_indices.return_value = [0]
        with patch.object(TerrainProjectionSensor, 'data', property(lambda self: data)):
            sensor.debug_vis(viz)
        self.assertEqual(viz.add_arrow.call_count, 2)
        for call in viz.add_arrow.call_args_list:
            self.assertAlmostEqual(float(call.kwargs['start'][2]), 0.1)
        self.assertEqual(viz.add_arrow.call_args_list[1].kwargs['color'], (1., 0.3, 0., 0.9))
        self.assertTrue(viz.add_sphere.call_args_list[-1].kwargs['label'].endswith('_miss'))
        self.assertAlmostEqual(float(viz.add_sphere.call_args_list[-1].kwargs['center'][2]), 0.1)
