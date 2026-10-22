"""SLR network/loss building blocks only; importing this package registers no tasks."""

from .losses import SLR_TRIPLET_LOSS_COEF, slr_triplet_loss
from .models import (
    SLRActor,
    SLRCritic,
    SLREncoder,
    SLRModelCfg,
    SLRNetworks,
    SLRTransitionModel,
)

__all__ = [
    "SLR_TRIPLET_LOSS_COEF",
    "SLRActor",
    "SLRCritic",
    "SLREncoder",
    "SLRModelCfg",
    "SLRNetworks",
    "SLRTransitionModel",
    "slr_triplet_loss",
]
