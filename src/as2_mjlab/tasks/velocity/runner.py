"""AS2 runner that persists the effective deployment contract in checkpoints."""

from copy import deepcopy
from pathlib import Path

import yaml
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class AS2OnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep imports lazy: task registration and manifest semantics share modules.
        from as2_mjlab.policy_manifest import SCHEMA_VERSION, build_contract, source_revision

        self.policy_task_id = ("as2-flat-unitree" if self.env.unwrapped.cfg.scene.terrain.terrain_type == "plane"
                               else "as2-rough-unitree")
        self.policy_manifest_snapshot = {
            "schema_version": SCHEMA_VERSION,
            "contract": build_contract(self.env.unwrapped, self.alg.get_policy(), self.env.clip_actions),
            "source": {**source_revision(), "task_id": self.policy_task_id},
        }

    def save(self, path: str, infos=None):
        from as2_mjlab.policy_manifest import create_manifest, manifest_path

        infos = {**(infos or {}), "policy_manifest": deepcopy(self.policy_manifest_snapshot)}
        super().save(path, infos)
        _, _, onnx_path = self._get_export_paths(path)
        try:
            manifest = create_manifest(
                self.env.unwrapped, self.alg.get_policy(), onnx_path,
                Path(path), self.policy_task_id, infos, self.env.clip_actions,
            )
            destination = manifest_path(onnx_path)
            temporary = destination.with_suffix(".yaml.tmp")
            temporary.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
            temporary.replace(destination)
        except Exception as error:
            print(f"[WARN] Policy manifest export failed (training continues): {error}")
