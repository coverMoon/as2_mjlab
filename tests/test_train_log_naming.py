"""Exercise upstream log naming without starting training or using CUDA."""

from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

import mjlab.tasks  # noqa: F401
from mjlab.scripts.train import TrainConfig

from as2_mjlab.scripts.train import launch_training


class TrainLogNamingTest(TestCase):
    def test_task_default_and_explicit_override(self):
        for task, run_name, suffix in (
            ('as2-flat', '', 'as2-flat'),
            ('as2-rough', '', 'as2-rough'),
            ('as2-flat', 'smoke', 'smoke'),
            ('as2-rough', 'rough_first', 'rough_first'),
        ):
            with self.subTest(task=task, run_name=run_name):
                cfg = TrainConfig.from_task(task)
                cfg = replace(cfg, agent=replace(cfg.agent, run_name=run_name))
                with patch('mjlab.scripts.train.datetime') as clock, \
                     patch('mjlab.scripts.train.select_gpus', return_value=(None, 0)), \
                     patch('mjlab.scripts.train.run_train') as run, \
                     patch.dict('os.environ'):
                    clock.now.return_value.strftime.return_value = '2026-10-20_16-26-28'
                    launch_training(task, cfg)
                self.assertEqual(run.call_args.args[2].name, f'2026-10-20_16-26-28_{suffix}')
                self.assertEqual(cfg.agent.run_name, run_name)

    def test_resume_source_unchanged(self):
        cfg = TrainConfig.from_task('as2-flat')
        cfg = replace(cfg, agent=replace(cfg.agent, resume=True,
                                        load_run='2026-10-20_18-31-32',
                                        load_checkpoint='model_5000.pt'))
        with patch('mjlab.scripts.train.launch_training') as launch:
            launch_training('as2-flat', cfg)
        forwarded = launch.call_args.args[1]
        self.assertEqual(forwarded.agent.run_name, 'as2-flat')
        self.assertTrue(forwarded.agent.resume)
        self.assertEqual(forwarded.agent.load_run, cfg.agent.load_run)
        self.assertEqual(forwarded.agent.load_checkpoint, cfg.agent.load_checkpoint)
