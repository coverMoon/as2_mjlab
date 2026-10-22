"""Official-code SLR latent loss; sampling, rollout pairing and masks are external.

Reference: https://github.com/11chens/SLR-master/blob/master/algorithm/ppo.py
This differs from the joint hinge in paper Equation (5).
"""

import torch


SLR_TRIPLET_LOSS_COEF = 1e-3
"""Official coefficient for adding the unweighted latent loss to the PPO loss."""


def slr_triplet_loss(
    z_next: torch.Tensor,
    z_pred: torch.Tensor,
    z_negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Official code: mean(||next-pred||²) + mean(relu(m - ||next-neg||²)).

    Inputs have the same [B, latent_dim] shape. Squared L2 distances are summed
    over latent dimensions. The negative latent is detached, while next/pred
    retain gradients. Return the unweighted scalar; the future training caller
    multiplies it by SLR_TRIPLET_LOSS_COEF once when assembling the total loss.
    """
    if z_next.ndim != 2 or z_next.shape != z_pred.shape or z_next.shape != z_negative.shape:
        raise ValueError("Expected three latent tensors with the same [B, latent_dim] shape")
    positive_distance = (z_next - z_pred).square().sum(dim=-1)
    negative_distance = (z_next - z_negative.detach()).square().sum(dim=-1)
    return positive_distance.mean() + torch.relu(margin - negative_distance).mean()
