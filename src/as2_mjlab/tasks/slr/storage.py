"""Feedforward SLR rollout storage on RSL-RL 5.4.2's PPO storage.

Collection contract: capture observations (H_t) and the raw sampled action
before env.step(), then set next_history from the returned H_{t+1}. MjLab's
wrapper done is terminated | truncated. Under auto-reset, done rows contain
the *next episode's* history, so they are zeroed here and excluded from latent
learning, including the negative pool. PPO rewards/dones remain untouched;
time-out reward bootstrapping and GAE stay in the upstream PPO algorithm.

Store every step's next history explicitly, including the last rollout step.
clear() only resets the inherited write cursor, not environment histories:
the caller carries the last returned observation into the next iteration.
"""

from collections.abc import Generator

import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


def sample_negative_indices(
    valid_transition_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (anchor_indices, negative_indices) into one original minibatch.

    Randomly permute only valid rows, as in official SLR's minibatch shuffle.
    Fixed points (self-matches) are allowed; a singleton always matches itself.
    No valid transitions returns two empty index tensors. Mask must be [M]
    bool. Keeping this sampler separate allows later sampling comparisons.
    """
    if valid_transition_mask.ndim != 1 or valid_transition_mask.dtype != torch.bool:
        raise ValueError("Expected a boolean valid-transition mask of shape [M]")
    anchor_indices = valid_transition_mask.nonzero(as_tuple=False).squeeze(-1)
    permutation = torch.randperm(
        anchor_indices.numel(), device=anchor_indices.device, generator=generator,
    )
    return anchor_indices, anchor_indices[permutation]


class SLRRolloutStorage(RolloutStorage):
    """Reuse all PPO fields; allocate only next_history in addition.

    The history shape is inferred from obs[history_key], normally [N, 10, 45].
    Models and env must agree on this shape. No second current-observation or
    action buffer is created, and validity is derived directly from dones.
    """

    class Transition(RolloutStorage.Transition):
        def __init__(self) -> None:
            super().__init__()
            self.next_history: torch.Tensor | None = None

    class Batch(RolloutStorage.Batch):
        def __init__(
            self, *, next_history: torch.Tensor, history_key: str = "history", **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self.next_history = next_history
            self.history_key = history_key

        @property
        def history(self) -> torch.Tensor:
            """[M, H, O], the same tensor as observations[history_key]."""
            return self.observations[self.history_key]

        @property
        def valid_transition_mask(self) -> torch.Tensor:
            """[M] bool; select these rows before latent encoding/loss."""
            return ~self.dones.squeeze(-1).bool()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
        *,
        history_key: str = "history",
    ) -> None:
        history = obs[history_key]
        if history.ndim != 3 or history.shape[0] != num_envs:
            raise ValueError("Expected observation history of shape [num_envs, H, O]")
        super().__init__("rl", num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.history_key = history_key
        self.next_history = torch.zeros_like(self.history)

    @property
    def history(self) -> torch.Tensor:
        """[T, N, H, O], aliases the upstream observation storage."""
        return self.observations[self.history_key]

    @property
    def valid_transition_mask(self) -> torch.Tensor:
        """[T, N] bool, derived without allocating another persistent buffer.

        Unwritten rows (including all rows just after clear()) are invalid.
        Minibatches require a full rollout, so old/unwritten rows cannot leak.
        """
        valid = ~self.dones.squeeze(-1).bool()
        valid[self.step:] = False
        return valid

    @torch.no_grad()
    def add_transition(self, transition: Transition) -> None:
        """Copy the aligned PPO transition and its post-step history snapshot."""
        if transition.next_history is None or transition.next_history.shape != self.history.shape[1:]:
            raise ValueError("next_history must have shape [num_envs, H, O]")
        index = self.step
        super().add_transition(transition)
        self.next_history[index].copy_(transition.next_history)
        # A reset observation is not a terminal-state successor. Zeroing is only
        # a placeholder; the mask must still be applied before the latent loss.
        self.next_history[index].masked_fill_(self.dones[index].bool().unsqueeze(-1), 0.0)

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8,
    ) -> Generator[Batch, None, None]:
        """Select every PPO/SLR field with exactly the same flattened indices.

        Match upstream 5.4.2: time-major flattening (index=t*N+env), one random
        permutation reused across epochs, equal minibatches, and any final
        remainder rows omitted. No separate SLR shuffle is performed.
        """
        if self.step != self.num_transitions_per_env:
            raise RuntimeError("Fill the rollout before generating minibatches")
        batch_size = self.num_envs * self.num_transitions_per_env
        if not 1 <= num_mini_batches <= batch_size or num_epochs < 1:
            raise ValueError("Require 1 <= num_mini_batches <= rollout size and num_epochs >= 1")
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
        observations = self.observations.flatten(0, 1)
        next_history = self.next_history.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        dones = self.dones.flatten(0, 1)
        values = self.values.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        log_prob = self.actions_log_prob.flatten(0, 1)
        distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)
        for _ in range(num_epochs):
            for start in range(0, indices.numel(), mini_batch_size):
                selected = indices[start:start + mini_batch_size]
                yield self.Batch(
                    observations=observations[selected],
                    actions=actions[selected],
                    dones=dones[selected],
                    values=values[selected],
                    advantages=advantages[selected],
                    returns=returns[selected],
                    old_actions_log_prob=log_prob[selected],
                    old_distribution_params=tuple(p[selected] for p in distribution_params),
                    next_history=next_history[selected],
                    history_key=self.history_key,
                )
