"""Pure SLR latent loss; sampling, rollout pairing and masks are caller concerns."""

import torch


def slr_triplet_loss(
    z_next: torch.Tensor,
    z_pred: torch.Tensor,
    z_negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Equation (5): mean(relu(||next-pred||² - ||next-negative||² + m)).

    Inputs have the same [B, latent_dim] shape. Squared L2 distances are summed
    over latent dimensions, then the hinge is averaged over the batch. All
    three inputs retain their gradient paths; no normalization or detach.
    """
    if z_next.ndim != 2 or z_next.shape != z_pred.shape or z_next.shape != z_negative.shape:
        raise ValueError("Expected three latent tensors with the same [B, latent_dim] shape")
    positive_distance = (z_next - z_pred).square().sum(dim=-1)
    negative_distance = (z_next - z_negative).square().sum(dim=-1)
    return torch.relu(positive_distance - negative_distance + margin).mean()
