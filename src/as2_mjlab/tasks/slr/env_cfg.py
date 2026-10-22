"""SLR environments with one native, non-privileged observation history.

Reuse the PPO environment factories without changing their returned baseline
configs. Only the observation groups are adapted: ``obs["history"]`` has shape
``[B, 10, 45]``, oldest to newest. Both Actor and Critic must use the same
``history[:, -1, :]`` as their current observation, and Encoder uses the full
history. No separate current/critic group is computed or noise sampled twice.

MjLab owns all history buffers and resets. A newly reset environment's first
frame fills its entire window; other environments retain their own histories.
Sensors used by rewards/terrain remain available but are not policy inputs.
No SLR training tasks are registered here.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from as2_mjlab.tasks.ppo.env_cfg import ppo_flat_env_cfg, ppo_rough_env_cfg


def _with_slr_history(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Keep PPO Actor terms/order/scales/noise, discard privileged Critic terms."""
    cfg.observations = {
        "history": replace(
            cfg.observations["actor"],
            history_length=10,
            flatten_history_dim=False,
            concatenate_terms=True,
            concatenate_dim=-1,
        ),
    }
    # Native term buffers are [B, H, D_term]. Concatenating D_term preserves
    # [B, H, 45]; flattening each term first would instead be term-major.
    return cfg


def slr_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """PPO Flat physics/rewards/curricula with one proprioceptive history group."""
    return _with_slr_history(ppo_flat_env_cfg(play=play))


def slr_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """PPO Rough terrain/contact rules with the same non-privileged history."""
    return _with_slr_history(ppo_rough_env_cfg(play=play))
