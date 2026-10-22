"""Standalone SLR networks; no environment or training-loop integration.

Default dimensions prioritize the official SLR implementation:
https://github.com/11chens/SLR-master/blob/master/modules/actor_critic_slr.py
and configs/go2_config.py. Unlike paper Table A3, the Encoder has hidden
layers (256, 128, 64) and Transition has (32,). Hidden-layer ELU is the AS2
adaptation matching the PPO baseline and also agrees with official code.
All output layers are linear. History assembly and normalization belong to
the caller; history is supplied in a consistent temporal order [B, H, O].
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SLRModelCfg:
    """Network dimensions, independent of PPO environment/optimizer settings."""

    observation_dim: int = 45
    history_length: int = 10
    latent_dim: int = 20
    action_dim: int = 12
    encoder_hidden_dims: tuple[int, ...] = (256, 128, 64)
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    transition_hidden_dims: tuple[int, ...] = (32,)


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(input_dim, hidden_dim), nn.ELU()))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class SLREncoder(nn.Module):
    """Encode [B, history_length, observation_dim] into [B, latent_dim]."""

    def __init__(self, cfg: SLRModelCfg = SLRModelCfg()):
        super().__init__()
        self.cfg = cfg
        self.mlp = _mlp(
            cfg.history_length * cfg.observation_dim,
            cfg.encoder_hidden_dims,
            cfg.latent_dim,
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (
            self.cfg.history_length, self.cfg.observation_dim,
        ):
            raise ValueError(
                f"Expected history [B, {self.cfg.history_length}, "
                f"{self.cfg.observation_dim}], got {tuple(history.shape)}"
            )
        return self.mlp(history.flatten(start_dim=1))


class SLRActor(nn.Module):
    """Action mean from current observation and stop-gradient latent, Eq. (2)."""

    def __init__(self, cfg: SLRModelCfg = SLRModelCfg()):
        super().__init__()
        self.mlp = _mlp(
            cfg.observation_dim + cfg.latent_dim, cfg.actor_hidden_dims, cfg.action_dim,
        )

    def forward(self, observation: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """[B, O], [B, Z] -> [B, A]; no distribution or action clipping."""
        return self.mlp(torch.cat((latent.detach(), observation), dim=-1))


class SLRCritic(nn.Module):
    """State value with gradients through the latent to the Encoder, Eq. (3)."""

    def __init__(self, cfg: SLRModelCfg = SLRModelCfg()):
        super().__init__()
        self.mlp = _mlp(cfg.observation_dim + cfg.latent_dim, cfg.critic_hidden_dims, 1)

    def forward(self, observation: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """[B, O], [B, Z] -> [B, 1]; latent remains attached."""
        return self.mlp(torch.cat((latent, observation), dim=-1))


class SLRTransitionModel(nn.Module):
    """Predict the next latent from current latent and action, Eq. (4)."""

    def __init__(self, cfg: SLRModelCfg = SLRModelCfg()):
        super().__init__()
        self.mlp = _mlp(
            cfg.latent_dim + cfg.action_dim, cfg.transition_hidden_dims, cfg.latent_dim,
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """[B, Z], [B, A] -> [B, Z]; neither input is detached."""
        return self.mlp(torch.cat((latent, action), dim=-1))


class SLRNetworks(nn.Module):
    """Own the single shared Encoder and three heads, each registered once.

    Actor/Critic heads accept latent tensors instead of storing Encoder aliases.
    This keeps one Encoder entry in state_dict and one set of its parameters.
    """

    def __init__(self, cfg: SLRModelCfg = SLRModelCfg()):
        super().__init__()
        self.cfg = cfg
        self.encoder = SLREncoder(cfg)
        self.actor = SLRActor(cfg)
        self.critic = SLRCritic(cfg)
        self.transition = SLRTransitionModel(cfg)

    def act(self, observation: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Current [B, O] and history [B, H, O] -> action mean [B, A]."""
        return self.actor(observation, self.encoder(history))

    def evaluate(self, observation: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Current [B, O] and history [B, H, O] -> value [B, 1]."""
        return self.critic(observation, self.encoder(history))

    def forward(
        self, observation: torch.Tensor, history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode once and return (action_mean [B, A], value [B, 1])."""
        latent = self.encoder(history)
        return self.actor(observation, latent), self.critic(observation, latent)
