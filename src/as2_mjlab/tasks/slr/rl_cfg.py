"""Typed SLR configuration for MjLab/Tyro and the standard on-policy runner."""

from dataclasses import asdict, dataclass, field, fields

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from as2_mjlab.tasks.ppo.rl_cfg import ppo_runner_cfg

from .losses import SLR_TRIPLET_LOSS_COEF
from .models import SLRModelCfg


@dataclass
class SLRPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "as2_mjlab.tasks.slr.algorithm:SLRPPO"
    model_cfg: SLRModelCfg = field(default_factory=SLRModelCfg)
    triplet_loss_coef: float = SLR_TRIPLET_LOSS_COEF
    triplet_margin: float = 1.0


@dataclass
class SLRRunnerCfg(RslRlOnPolicyRunnerCfg):
    algorithm: SLRPpoAlgorithmCfg = field(default_factory=SLRPpoAlgorithmCfg)


def slr_runner_cfg() -> SLRRunnerCfg:
    """Create independent configs, inheriting current AS2 PPO hyperparameters.

    SLR model sizes are authoritative in algorithm.model_cfg. No global config
    is modified, and the derived algorithm type exposes SLR fields to Tyro.
    """
    baseline = ppo_runner_cfg()
    ppo_options = asdict(baseline.algorithm)
    ppo_options.pop("class_name")
    options = {item.name: getattr(baseline, item.name) for item in fields(baseline)}
    options.update(
        algorithm=SLRPpoAlgorithmCfg(**ppo_options),
        experiment_name="as2_slr",
        obs_groups={"actor": ("history",), "critic": ("history",)},
    )
    return SLRRunnerCfg(**options)
