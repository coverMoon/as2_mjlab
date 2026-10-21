"""Optional diagnostics for Play, captured before automatic episode reset."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def log_play_terminations(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    max_envs: int = 4,
) -> None:
    """Print termination causes and nonfoot force peaks before sensors reset.

    Force history is the same window used by illegal_contact. A brief impact
    can trigger termination even when the current contact force is smaller.
    Limit output to max_envs terminated environments per control step.
    """
    del env_ids  # Step events receive None; select terminating environments.
    manager = env.termination_manager
    done_ids = manager.dones.nonzero(as_tuple=False).flatten()
    if done_ids.numel() == 0:
        return

    contact_cfg = manager.get_term_cfg("illegal_contact")
    sensor = env.scene[contact_cfg.params["sensor_name"]]
    threshold = contact_cfg.params.get("force_threshold", 10.0)
    data = sensor.data
    if data.force_history is not None:
        peaks = torch.linalg.vector_norm(data.force_history, dim=-1).amax(dim=-1)
    else:
        peaks = torch.linalg.vector_norm(data.force, dim=-1)
    current = torch.linalg.vector_norm(data.force, dim=-1)
    names = sensor.primary_names
    command = env.command_manager.get_command("twist")

    for idx in done_ids[:max_envs].tolist():
        reasons = [name for name in manager.active_terms if manager.get_term(name)[idx].item()]
        contacts = []
        for slot in (peaks[idx] > threshold).nonzero(as_tuple=False).flatten().tolist():
            contacts.append({
                "geom": names[slot // sensor.cfg.num_slots],
                "peak_force_N": round(peaks[idx, slot].item(), 3),
                "current_force_N": round(current[idx, slot].item(), 3),
            })
        record = {
            "env": idx,
            "episode_time_s": round(env.episode_length_buf[idx].item() * env.step_dt, 3),
            "reasons": reasons,
            "command": command[idx].tolist(),
            "force_threshold_N": threshold,
            "nonfoot_contacts": contacts,
        }
        print("[PLAY_TERMINATION] " + json.dumps(record), flush=True)

    if done_ids.numel() > max_envs:
        print(f"[PLAY_TERMINATION] {done_ids.numel() - max_envs} additional environments omitted", flush=True)
