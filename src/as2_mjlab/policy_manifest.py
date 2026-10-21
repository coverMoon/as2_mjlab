"""Versioned deployment contracts for the current AS2 proprioceptive MLP policies.

Only supported observation functions are translated to deployment semantics.
Unknown functions, history and delays fail explicitly instead of guessing meaning
from a term's name or tensor dimension. Model hashes are intentionally omitted.
"""

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import argparse
import math
import subprocess

import mujoco
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch
from tensordict import TensorDict
import yaml

from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization

from as2_mjlab.tasks.velocity.mdp.observations import phase


SCHEMA_VERSION = 1


def source_revision() -> dict:
    """Identify this checkout without claiming it is the checkpoint's source."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    versions = {}
    for package in ("as2-mjlab", "mjlab", "rsl-rl-lib", "torch", "onnx"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {"git_commit": commit, "git_dirty": dirty, "versions": versions}


def _vector(value, dim: int) -> list[float]:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim == 2:
        tensor = tensor[0]
    values = torch.broadcast_to(tensor, (dim,)).tolist()
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Contract values must be finite")
    return values


def _joints(env, cfg: SceneEntityCfg) -> tuple[list[str], list[float], list[float]]:
    asset = env.scene[cfg.name]
    ids = list(range(asset.num_joints))[cfg.joint_ids] if isinstance(cfg.joint_ids, slice) else list(cfg.joint_ids)
    names = [asset.joint_names[index] for index in ids]
    return (names, asset.data.default_joint_pos[0, ids].cpu().tolist(),
            asset.data.default_joint_vel[0, ids].cpu().tolist())


def _observation_semantics(env, cfg) -> dict:
    params = dict(cfg.params)
    func = cfg.func
    if func is mdp.base_ang_vel:
        asset_cfg = params.pop("asset_cfg", SceneEntityCfg("robot"))
        if params:
            raise ValueError("Unsupported root angular velocity parameters")
        return {"kind": "base_angular_velocity", "source": "root_link_ang_vel_b",
                "entity": asset_cfg.name, "frame": env.scene[asset_cfg.name].root_body.name,
                "units": "rad/s", "components": ["x", "y", "z"]}
    if func is mdp.builtin_sensor:
        if set(params) != {"sensor_name"}:
            raise ValueError("Unsupported builtin sensor parameters")
        model = env.sim.mj_model
        sensor = model.sensor(params["sensor_name"])
        if int(sensor.type[0]) != int(mujoco.mjtSensor.mjSENS_GYRO):
            raise ValueError("Manifest v1 supports gyro builtin observations only")
        site = model.site(int(sensor.objid[0]))
        return {
            "kind": "imu_angular_velocity", "source": params["sensor_name"],
            "frame": "sensor_site", "units": "rad/s", "components": ["x", "y", "z"],
            "site": site.name, "parent_body": model.body(int(site.bodyid[0])).name,
            "site_position_in_parent_m": site.pos.tolist(),
            "site_quaternion_in_parent_wxyz": site.quat.tolist(),
        }
    if func is mdp.projected_gravity:
        asset_cfg = params.pop("asset_cfg", SceneEntityCfg("robot"))
        if params:
            raise ValueError("Unsupported projected gravity parameters")
        return {"kind": "projected_gravity", "source": "root_link_orientation",
                "entity": asset_cfg.name, "frame": env.scene[asset_cfg.name].root_body.name,
                "units": "unit_vector", "world_vector": [0.0, 0.0, -1.0],
                "formula": "inverse_rotate(root_quaternion_wxyz, world_vector)"}
    if func is mdp.generated_commands:
        if set(params) != {"command_name"}:
            raise ValueError("Unsupported command parameters")
        command = env.command_manager.get_command(params["command_name"])
        command_term = env.command_manager.get_term(params["command_name"])
        if type(command_term) is not UniformVelocityCommand or command_term.cfg.entity_name != "robot" or command.shape[-1] != 3:
            raise ValueError("Manifest v1 requires a three-component velocity command")
        return {"kind": "velocity_command", "source": "active_velocity_command",
                "command_name": params["command_name"], "frame": "robot_base",
                "components": ["lin_vel_x", "lin_vel_y", "ang_vel_z"],
                "units": ["m/s", "m/s", "rad/s"]}
    if func is phase:
        if set(params) != {"period", "command_name"} or params["period"] <= 0:
            raise ValueError("Unsupported gait phase parameters")
        return {"kind": "gait_phase_sin_cos", "source": "episode_step_and_command",
                "units": "dimensionless", "period_s": float(params["period"]),
                "command_name": params["command_name"],
                "standing_test": {"norm": "l2_xyz_command", "operator": "<", "threshold": 0.1},
                "formula": "phi = ((episode_step * policy_dt) % period_s) / period_s",
                "components": ["sin(2*pi*phi)", "cos(2*pi*phi)"],
                "standing_value": [0.0, 0.0], "reset_episode_step": 0}
    if func in (mdp.joint_pos_rel, mdp.joint_vel_rel):
        asset_cfg = params.pop("asset_cfg", SceneEntityCfg("robot"))
        biased = params.pop("biased", False) if func is mdp.joint_pos_rel else False
        if params or biased:
            raise ValueError("Manifest v1 requires unbiased relative joint observations")
        names, positions, velocities = _joints(env, asset_cfg)
        return {"kind": "joint_position_relative" if func is mdp.joint_pos_rel else "joint_velocity_relative",
                "source": "joint_encoder", "entity": asset_cfg.name, "joint_names": names,
                "units": "rad" if func is mdp.joint_pos_rel else "rad/s",
                "reference": positions if func is mdp.joint_pos_rel else velocities,
                "formula": "measured - reference", "encoder_bias": "none"}
    if func is mdp.last_action:
        if params.get("action_name") is not None or set(params) - {"action_name"}:
            raise ValueError("Manifest v1 requires the complete previous action vector")
        return {"kind": "previous_action", "source": "action_manager.action",
                "units": "dimensionless", "reset_value": 0.0,
                "update": "after_policy_inference_and_wrapper_clip_before_next_observation",
                "representation": "before_action_scale_offset_and_target_clip"}
    raise ValueError(f"Unsupported observation function: {func.__module__}.{func.__qualname__}")


def build_contract(env, policy, clip_actions: float | None = None) -> dict:
    """Read the effective managers and resolved action targets, never regex order."""
    if type(policy) is not MLPModel or list(policy.obs_groups) != ["actor"]:
        raise ValueError("Manifest v1 supports MLPModel with the single actor group")
    manager = env.observation_manager
    group = manager.cfg["actor"]
    if not group.concatenate_terms or group.concatenate_dim != -1:
        raise ValueError("Manifest v1 requires flat actor term concatenation")
    terms, offset = [], 0
    for name, shape in zip(manager.active_terms["actor"], manager.group_obs_term_dim["actor"], strict=True):
        cfg = manager.get_term_cfg("actor", name)
        if len(shape) != 1 or cfg.history_length != 0 or cfg.delay_max_lag != 0:
            raise ValueError(f"Manifest v1 does not support history or delay: {name}")
        dim = int(shape[0])
        terms.append({"name": name, "offset": offset, "dim": dim,
                      **_observation_semantics(env, cfg),
                      "preprocess": {"clip": list(cfg.clip) if cfg.clip is not None else None,
                                     "scale": _vector(1.0 if cfg.scale is None else cfg.scale, dim),
                                     "history_length": 0, "delay_steps": 0,
                                     "noise_at_deployment": None}})
        offset += dim
    if offset != policy.obs_dim:
        raise ValueError("Actor manager and policy input dimensions disagree")
    if env.action_manager.active_terms != ["joint_pos"]:
        raise ValueError("Manifest v1 requires one joint_pos action term")
    action = env.action_manager.get_term("joint_pos")
    if type(action) is not JointPositionAction or action.cfg.entity_name != "robot":
        raise ValueError("Manifest v1 requires robot joint position targets")
    dim = action.action_dim
    names = list(action.target_names)
    asset = env.scene["robot"]
    if names != [asset.joint_names[index] for index in action.target_ids.tolist()]:
        raise ValueError("Action names and resolved joint IDs disagree")
    if clip_actions is not None and (not math.isfinite(clip_actions) or clip_actions <= 0):
        raise ValueError("Action wrapper clip must be positive and finite")
    normalizer = policy.obs_normalizer
    if type(normalizer) is EmpiricalNormalization:
        normalization = {"kind": "empirical", "location": "inside_onnx",
                         "formula": "(x - mean) / (std + epsilon)",
                         "epsilon": normalizer.eps, "external_normalization": False}
    elif type(normalizer) is torch.nn.Identity:
        normalization = {"kind": "identity", "location": "inside_onnx", "external_normalization": False}
    else:
        raise ValueError("Unsupported actor normalizer")
    model = env.sim.mj_model
    stiffness, damping = [], []
    for name in names:
        joint_id = model.joint(f"robot/{name}").id
        ctrl_ids = [i for i in range(model.nu)
                    if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                    and model.actuator_trnid[i, 0] == joint_id]
        if len(ctrl_ids) != 1:
            raise ValueError(f"Expected one position actuator for {name}")
        ctrl_id = ctrl_ids[0]
        stiffness.append(float(model.actuator_gainprm[ctrl_id, 0]))
        damping.append(float(-model.actuator_biasprm[ctrl_id, 2]))
    return {
        "robot": "as2", "policy_dt": float(env.step_dt),
        "clock": {"reset_episode_step": 0, "increment": "after_action_before_next_observation",
                  "reset_previous_action": 0.0},
        "observation": {"group": "actor", "dim": offset, "layout": "term_major",
                        "pipeline": ["compute", "clip", "scale", "concatenate", "onnx_normalization"],
                        "ordered_terms": terms, "normalization": normalization},
        "action": {"kind": "joint_position_residual", "dim": dim, "entity": "robot",
                   "joint_names": names, "default_joint_pos": asset.data.default_joint_pos[0, action.target_ids].cpu().tolist(),
                   "default_joint_vel": asset.data.default_joint_vel[0, action.target_ids].cpu().tolist(),
                   "scale": _vector(action.scale, dim), "offset": _vector(action.offset, dim),
                   "wrapper_clip": [-clip_actions, clip_actions] if clip_actions is not None else None,
                   "target_clip": action._clip[0].cpu().tolist() if action.cfg.clip is not None else None,
                   "formula": "q_target = clip_target(offset + scale * clip_wrapper(onnx_output))",
                   "output": "deterministic_policy_action_before_external_processing",
                   "stiffness": stiffness, "damping": damping,
                   "training_encoder_bias": "subtracted_from_simulated_target_only"},
        "requirements": {"missing_observation": "reject", "nonfinite_observation": "reject",
                         "external_terrain_sensing": False},
    }


def contract_difference(expected, actual, path: str = "contract") -> str | None:
    """Return the first meaningful mismatch, including equal-size reordered terms."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return f"{path}: fields differ"
        for key in expected:
            difference = contract_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path}: lengths differ"
        for i, (left, right) in enumerate(zip(expected, actual, strict=True)):
            difference = contract_difference(left, right, f"{path}[{i}]")
            if difference:
                return difference
    elif expected != actual:
        return f"{path}: expected {expected!r}, got {actual!r}"
    return None


def model_tensors(model: onnx.ModelProto) -> dict:
    def describe(value):
        tensor = value.type.tensor_type
        dtype = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.DOUBLE: "float64"}.get(
            tensor.elem_type, onnx.TensorProto.DataType.Name(tensor.elem_type).lower(),
        )
        return {"name": value.name, "dtype": dtype,
                "shape": [d.dim_value if d.HasField("dim_value") else d.dim_param or None
                          for d in tensor.shape.dim]}
    return {"inputs": [describe(value) for value in model.graph.input],
            "outputs": [describe(value) for value in model.graph.output]}


def verify_onnx(policy, model: onnx.ModelProto, sample: torch.Tensor) -> dict:
    """Compare the full normalized actor against an independent ONNX evaluator."""
    policy = deepcopy(policy).cpu().eval()
    evaluator = ReferenceEvaluator(model)
    generator = torch.Generator().manual_seed(20261021)
    samples = [sample[:1].detach().cpu().float(), torch.zeros(1, policy.obs_dim),
               torch.randn(1, policy.obs_dim, generator=generator),
               torch.randn(1, policy.obs_dim, generator=generator) * 3]
    max_error = 0.0
    with torch.inference_mode():
        for values in samples:
            expected = policy(TensorDict({"actor": values}, batch_size=[1])).numpy()
            actual = evaluator.run(None, {model.graph.input[0].name: values.numpy()})[0]
            if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                raise ValueError("Policy verification produced nonfinite output")
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    return {"backend": "onnx.reference.ReferenceEvaluator", "samples": len(samples),
            "rtol": 1e-4, "atol": 1e-5, "max_abs_error": max_error,
            "includes_observation_normalization": True}


def create_manifest(env, policy, model_path: Path, checkpoint: Path, task_id: str,
                    infos: dict | None, clip_actions: float | None = None) -> dict:
    contract = build_contract(env, policy, clip_actions)
    recorded = (infos or {}).get("policy_manifest")
    training_source = None
    if recorded is not None:
        if recorded.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint policy contract version")
        difference = contract_difference(recorded["contract"], contract)
        if difference:
            raise ValueError(f"Checkpoint deployment contract mismatch: {difference}")
        training_source = recorded["source"]
    model = onnx.load(str(model_path))
    manifest = {"schema_version": SCHEMA_VERSION, "task_id": task_id, **contract,
                "model": {"file": model_path.name, **model_tensors(model)},
                "provenance": {"checkpoint_file": checkpoint.name,
                               "contract_source": "checkpoint" if recorded is not None else "export_environment",
                               "training_contract_verified": recorded is not None,
                               "training": training_source, "export": source_revision()}}
    validate_manifest(manifest, model)
    manifest["verification"] = verify_onnx(policy, model, env.observation_manager.compute()["actor"])
    return manifest


def validate_manifest(manifest: dict, model: onnx.ModelProto, expected_contract: dict | None = None) -> None:
    """Check deployable v1 structure and optionally a consumer's exact contract."""
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("robot") != "as2":
        raise ValueError("Unsupported policy manifest version or robot")
    onnx.checker.check_model(model)
    tensors = model_tensors(model)
    for key in ("inputs", "outputs"):
        if manifest["model"][key] != tensors[key]:
            raise ValueError(f"Manifest {key} do not match ONNX")
    obs, action = manifest["observation"], manifest["action"]
    for direction, dim in (("inputs", obs["dim"]), ("outputs", action["dim"])):
        values = tensors[direction]
        if len(values) != 1 or values[0]["shape"] != [1, dim] or values[0]["dtype"] != "float32":
            raise ValueError(f"Manifest v1 requires one float32 {direction} tensor [1, {dim}]")
    cursor, names = 0, set()
    kinds = {"imu_angular_velocity", "base_angular_velocity", "projected_gravity", "velocity_command", "gait_phase_sin_cos",
             "joint_position_relative", "joint_velocity_relative", "previous_action"}
    for term in obs["ordered_terms"]:
        if term["name"] in names or term["kind"] not in kinds:
            raise ValueError("Duplicate observation name or unsupported observation kind")
        names.add(term["name"])
        if term["offset"] != cursor or term["dim"] <= 0:
            raise ValueError("Observation offsets must form contiguous ordered slices")
        if len(term["preprocess"]["scale"]) != term["dim"]:
            raise ValueError("Observation scale length mismatch")
        processing = term["preprocess"]
        if processing["history_length"] != 0 or processing["delay_steps"] != 0 or processing["noise_at_deployment"] is not None:
            raise ValueError("Manifest v1 does not support deployment noise, history or delay")
        if not all(math.isfinite(value) for value in processing["scale"]):
            raise ValueError("Observation scales must be finite")
        expected_dim = {"imu_angular_velocity": 3, "base_angular_velocity": 3, "projected_gravity": 3,
                        "velocity_command": 3, "gait_phase_sin_cos": 2,
                        "previous_action": action["dim"]}.get(term["kind"])
        if expected_dim is not None and term["dim"] != expected_dim:
            raise ValueError(f"Observation semantic dimension mismatch: {term['name']}")
        if "joint_names" in term and (len(term["joint_names"]) != term["dim"] or len(term["reference"]) != term["dim"]):
            raise ValueError("Observation joint mapping length mismatch")
        cursor += term["dim"]
    if cursor != obs["dim"] or not math.isfinite(manifest["policy_dt"]) or manifest["policy_dt"] <= 0:
        raise ValueError("Invalid observation dimension or policy period")
    if action["kind"] != "joint_position_residual" or len(set(action["joint_names"])) != action["dim"]:
        raise ValueError("Invalid action kind or duplicate action joints")
    for key in ("joint_names", "default_joint_pos", "default_joint_vel", "scale", "offset", "stiffness", "damping"):
        if len(action[key]) != action["dim"]:
            raise ValueError(f"Action {key} length mismatch")
        if key != "joint_names" and not all(math.isfinite(value) for value in action[key]):
            raise ValueError(f"Action {key} values must be finite")
    normalization = obs["normalization"]
    if normalization["kind"] not in ("identity", "empirical") or normalization["external_normalization"] or normalization["location"] != "inside_onnx":
        raise ValueError("Unsupported observation normalization")
    if normalization["kind"] == "empirical" and (not math.isfinite(normalization["epsilon"]) or normalization["epsilon"] <= 0):
        raise ValueError("Invalid normalization epsilon")
    if obs["layout"] != "term_major" or obs["group"] != "actor":
        raise ValueError("Unsupported observation layout")
    if manifest["requirements"] != {"missing_observation": "reject", "nonfinite_observation": "reject", "external_terrain_sensing": False}:
        raise ValueError("Unsupported observation requirements")
    provenance = manifest["provenance"]
    recorded = provenance["contract_source"] == "checkpoint"
    if provenance["contract_source"] not in ("checkpoint", "export_environment") or provenance["training_contract_verified"] is not recorded:
        raise ValueError("Inconsistent training contract provenance")
    if recorded != (provenance["training"] is not None):
        raise ValueError("Inconsistent training source provenance")
    if expected_contract is not None:
        difference = contract_difference(expected_contract, {key: manifest[key] for key in expected_contract})
        if difference:
            raise ValueError(f"Consumer contract mismatch: {difference}")


def manifest_path(model_path: Path) -> Path:
    return model_path.with_suffix(".manifest.yaml")


def validate_bundle(path: Path, require_training_contract: bool = False) -> dict:
    with path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    filename = manifest["model"]["file"]
    if Path(filename).name != filename:
        raise ValueError("Model must be a file beside its manifest")
    if require_training_contract and not manifest["provenance"]["training_contract_verified"]:
        raise ValueError("This checkpoint has no recorded training contract")
    validate_manifest(manifest, onnx.load(str(path.parent / filename)))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a policy manifest and adjacent ONNX model")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-training-contract", action="store_true")
    args = parser.parse_args()
    manifest = validate_bundle(args.manifest, args.require_training_contract)
    print(f"Valid policy manifest: {args.manifest} ({manifest['observation']['dim']} inputs, {manifest['action']['dim']} actions)")


if __name__ == "__main__":
    main()
