"""Check export validation and default destination without starting physics."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import onnx
from onnx import TensorProto, helper

from as2_mjlab.scripts.export_onnx import ExportConfig, export_checkpoint


@dataclass
class Agent:
    clip_actions: float | None = None


class ExportOnnxTest(unittest.TestCase):
    def test_missing_checkpoint_fails_before_environment_creation(self):
        with TemporaryDirectory() as directory, \
             patch('as2_mjlab.scripts.export_onnx.ManagerBasedRlEnv') as env:
            with self.assertRaises(FileNotFoundError):
                export_checkpoint('as2-flat', ExportConfig(Path(directory) / 'missing.pt'))
            env.assert_not_called()

    def test_export_destination_metadata_and_cleanup(self):
        module = 'as2_mjlab.scripts.export_onnx'
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model_123.pt'
            checkpoint.write_bytes(b'checkpoint remains unchanged')
            env, runner = Mock(), Mock()
            cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=16), events={})

            def write_model(path, name):
                graph = helper.make_graph(
                    [helper.make_node('Identity', ['obs'], ['actions'])], 'test',
                    [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, 12])],
                    [helper.make_tensor_value_info('actions', TensorProto.FLOAT, [1, 12])],
                )
                onnx.save(helper.make_model(graph), str(Path(path) / name))

            runner.export_policy_to_onnx.side_effect = write_model
            with patch(f'{module}.load_env_cfg', return_value=cfg), \
                 patch(f'{module}.load_rl_cfg', return_value=Agent()), \
                 patch(f'{module}.ManagerBasedRlEnv', return_value=env), \
                 patch(f'{module}.RslRlVecEnvWrapper'), \
                 patch(f'{module}.load_runner_cls', return_value=Mock(return_value=runner)), \
                 patch(f'{module}.get_base_metadata', return_value={'joint_names': ['test_joint']}):
                output = export_checkpoint('as2-flat', ExportConfig(checkpoint, device='cpu'))
            self.assertEqual(output, checkpoint.with_suffix('.onnx'))
            self.assertEqual(checkpoint.read_bytes(), b'checkpoint remains unchanged')
            runner.load.assert_called_once_with(str(checkpoint), load_cfg={'actor': True},
                                                strict=True, map_location='cpu')
            env.close.assert_called_once()
            metadata = {m.key: m.value for m in onnx.load(str(output)).metadata_props}
            self.assertEqual(metadata['checkpoint_file'], checkpoint.name)
            self.assertEqual(metadata['task_id'], 'as2-flat')
            self.assertIn('joint_names', metadata)
            self.assertFalse(list(Path(directory).glob('.onnx-export-*')))

    def test_refuses_checkpoint_as_output(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pt'
            checkpoint.touch()
            with self.assertRaisesRegex(ValueError, 'overwrite the checkpoint'):
                export_checkpoint('as2-flat', ExportConfig(checkpoint, output_file=checkpoint))
