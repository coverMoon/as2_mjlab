"""SLR building blocks and AS2 Flat/Rough task registration."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .losses import SLR_TRIPLET_LOSS_COEF, slr_triplet_loss
from .models import (
    SLRActor,
    SLRCritic,
    SLREncoder,
    SLRModelCfg,
    SLRNetworks,
    SLRTransitionModel,
)
from .storage import SLRRolloutStorage, sample_negative_indices
from .algorithm import SLRPPO
from .env_cfg import slr_flat_env_cfg, slr_rough_env_cfg
from .rl_cfg import slr_runner_cfg


register_mjlab_task(
    task_id="as2-flat-slr",
    env_cfg=slr_flat_env_cfg(),
    play_env_cfg=slr_flat_env_cfg(play=True),
    rl_cfg=slr_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)
register_mjlab_task(
    task_id="as2-rough-slr",
    env_cfg=slr_rough_env_cfg(),
    play_env_cfg=slr_rough_env_cfg(play=True),
    rl_cfg=slr_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)

__all__ = [
    "SLR_TRIPLET_LOSS_COEF",
    "SLRActor",
    "SLRCritic",
    "SLREncoder",
    "SLRModelCfg",
    "SLRNetworks",
    "SLRPPO",
    "SLRRolloutStorage",
    "SLRTransitionModel",
    "slr_triplet_loss",
    "sample_negative_indices",
]
