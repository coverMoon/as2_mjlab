"""Export a selected checkpoint's normalized actor policy without a viewer."""

from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import copy2
import json
import sys
from tempfile import TemporaryDirectory

import mjlab
import onnx
import torch
import tyro
import yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

from as2_mjlab.policy_manifest import create_manifest, manifest_path
from as2_mjlab.tasks.slr.algorithm import SLRPolicy
from as2_mjlab.tasks.slr.models import SLRModelCfg
from as2_mjlab.tasks.slr.rl_cfg import SLRRunnerCfg
from as2_mjlab.tasks.slr.exporter import verify_slr_onnx


@dataclass
class ExportConfig:
    checkpoint_file: Path
    """The .pt checkpoint to load (actor and observation normalization only)."""
    output_file: Path | None = None
    """Defaults to the checkpoint path with its suffix replaced by .onnx."""
    device: str | None = None
    """Defaults to cuda:0 when CUDA is available, otherwise cpu."""


def _publish_bundle(temporary: Path, output: Path) -> None:
    """Publish a validated pair, restoring existing files on a write failure."""
    pairs = [(temporary, output), (manifest_path(temporary), manifest_path(output))]
    backups = []
    for index, (_, destination) in enumerate(pairs):
        backup = temporary.parent / f"previous-{index}"
        if destination.exists():
            copy2(destination, backup)
            backups.append((destination, backup))
        else:
            backups.append((destination, None))
    try:
        for source, destination in pairs:
            source.replace(destination)
    except OSError:
        for destination, backup in backups:
            if backup is not None:
                backup.replace(destination)
            else:
                destination.unlink(missing_ok=True)
        raise


def export_checkpoint(task_id: str, cfg: ExportConfig) -> Path:
    checkpoint = cfg.checkpoint_file.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output = (cfg.output_file or checkpoint.with_suffix(".onnx")).expanduser().resolve()
    if output == checkpoint:
        raise ValueError("Output must not overwrite the checkpoint")
    if output.suffix.lower() != ".onnx":
        raise ValueError("Output file must have the .onnx suffix")

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.events.pop("log_play_terminations", None)
    agent_cfg = load_rl_cfg(task_id)
    if isinstance(agent_cfg, SLRRunnerCfg):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        agent_cfg.algorithm.model_cfg = SLRModelCfg(**saved["slr_model_cfg"])
        normalize = any(key.startswith("obs_normalizer.") for key in saved["actor_state_dict"])
        agent_cfg.actor.obs_normalization = normalize
        agent_cfg.critic.obs_normalization = normalize
    env = ManagerBasedRlEnv(env_cfg, device=device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        infos = runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        manifest = None

        # Validate the complete model before replacing an existing export.
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".onnx-export-", dir=output.parent) as temp_dir:
            temporary = Path(temp_dir) / output.name
            runner.export_policy_to_onnx(temp_dir, output.name)
            if isinstance(policy, SLRPolicy):
                names = env.observation_manager.active_terms["history"]
                joint_action = env.action_manager.get_term("joint_pos")
                metadata = {
                    "run_path": checkpoint.parent.name,
                    "policy_type": "slr",
                    "policy_dt": env.step_dt,
                    "joint_names": list(env.scene["robot"].joint_names),
                    "action_scale": joint_action._scale[0].cpu().tolist()
                    if isinstance(joint_action._scale, torch.Tensor) else joint_action._scale,
                    "action_offset": joint_action._offset[0].cpu().tolist()
                    if isinstance(joint_action._offset, torch.Tensor) else joint_action._offset,
                    "slr_input_contract": json.dumps({
                        "shape": [1, policy.networks.cfg.history_length, policy.networks.cfg.observation_dim],
                        "layout": "batch,time,feature", "time_order": "oldest_to_newest",
                        "current_frame": -1, "reset": "repeat_first_frame",
                        "history_buffer": "external", "normalization": "inside_onnx",
                        "terms": [{"name": name,
                                   "scale": env.observation_manager.get_term_cfg("history", name).scale,
                                   "clip": env.observation_manager.get_term_cfg("history", name).clip}
                                  for name in names],
                        "output": "deterministic_action_mean_before_joint_position_scaling",
                    }, default=lambda value: value.detach().cpu().flatten().tolist()),
                }
            else:
                metadata = get_base_metadata(env, checkpoint.parent.name)
            metadata.update(task_id=task_id, checkpoint_file=checkpoint.name)
            attach_metadata_to_onnx(str(temporary), metadata)
            onnx.checker.check_model(str(temporary))
            if isinstance(policy, SLRPolicy):
                verification = verify_slr_onnx(
                    policy, onnx.load(str(temporary)), wrapped.get_observations()["history"],
                )
                attach_metadata_to_onnx(str(temporary), {"verification": json.dumps(verification)})
                temporary.replace(output)
                print(f"SLR ONNX verification: {verification}")
            else:
                manifest = create_manifest(env, policy, temporary,
                                           checkpoint, task_id, infos, agent_cfg.clip_actions)
                temporary_manifest = manifest_path(temporary)
                temporary_manifest.write_text(
                    yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8",
                )
                _publish_bundle(temporary, output)
    finally:
        env.close()

    print(f"Exported ONNX policy: {output}")
    if manifest is not None:
        print(f"Exported policy manifest: {manifest_path(output)}")
        if not manifest["provenance"]["training_contract_verified"]:
            print("[INFO] Checkpoint has no stored deployment contract; manifest describes the export environment.")
    return output


def main() -> None:
    maybe_print_top_level_help("export")
    import mjlab.tasks  # noqa: F401 -- populate task registry and plugins.

    task_id, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        ExportConfig,
        args=remaining_args,
        prog=f"{sys.argv[0]} {task_id}",
        config=mjlab.TYRO_FLAGS,
    )
    export_checkpoint(task_id, cfg)


if __name__ == "__main__":
    main()
