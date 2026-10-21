"""Check Play logging without running physics or opening a viewer."""

from contextlib import redirect_stdout
from io import StringIO
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from as2_mjlab.tasks.velocity.env_cfg import as2_flat_env_cfg, as2_rough_env_cfg
from as2_mjlab.tasks.velocity.mdp.events import log_play_terminations


class PlayTerminationLoggingTest(unittest.TestCase):
    def test_switch_only_enables_play_logging(self):
        for factory in (as2_flat_env_cfg, as2_rough_env_cfg):
            for enabled in (False, True):
                with patch('as2_mjlab.tasks.velocity.env_cfg.LOG_PLAY_TERMINATIONS', enabled):
                    self.assertNotIn('log_play_terminations', factory().events)
                    cfg = factory(play=True)
                    self.assertEqual('log_play_terminations' in cfg.events, enabled)
                    if enabled:
                        self.assertEqual(cfg.events['log_play_terminations'].mode, 'step')

    def test_history_peak_identifies_geom_and_reason(self):
        sensor = SimpleNamespace(
            data=SimpleNamespace(
                force=torch.zeros(1, 2, 3),
                force_history=torch.tensor([[[[0., 0., 0.], [12., 0., 0.]],
                                             [[0., 0., 0.], [0., 0., 0.]]]]),
            ),
            primary_names=['FL_calf3_collision', 'base_collision'],
            cfg=SimpleNamespace(num_slots=1),
        )
        dones = {'illegal_contact': torch.tensor([True]), 'fell_over': torch.tensor([False])}
        env = SimpleNamespace(
            termination_manager=SimpleNamespace(
                dones=torch.tensor([True]), active_terms=list(dones),
                get_term=lambda name: dones[name],
                get_term_cfg=lambda name: SimpleNamespace(params={
                    'sensor_name': 'nonfoot_ground_touch', 'force_threshold': 10.0,
                }),
            ),
            scene={'nonfoot_ground_touch': sensor},
            command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[1., 0., 0.]])),
            episode_length_buf=torch.tensor([50]), step_dt=0.02,
        )
        out = StringIO()
        with redirect_stdout(out):
            log_play_terminations(env, None)
        record = json.loads(out.getvalue().removeprefix('[PLAY_TERMINATION] '))
        self.assertEqual(record['reasons'], ['illegal_contact'])
        self.assertEqual(record['episode_time_s'], 1.0)
        self.assertEqual(record['nonfoot_contacts'], [{
            'geom': 'FL_calf3_collision', 'peak_force_N': 12.0, 'current_force_N': 0.0,
        }])

    def test_silent_without_termination(self):
        env = SimpleNamespace(termination_manager=SimpleNamespace(dones=torch.tensor([False])))
        out = StringIO()
        with redirect_stdout(out):
            log_play_terminations(env, None)
        self.assertEqual(out.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
