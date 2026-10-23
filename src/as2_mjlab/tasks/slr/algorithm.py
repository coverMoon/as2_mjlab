"""Single-device SLR PPO for the RSL-RL 5.4.2 OnPolicyRunner interface.

Reuse PPO's optimizer construction, GAE, timeout bootstrap and model modes.
Override construction/collection for the shared history model and paired storage,
and update() to add the official SLR objective and clip the unique parameter set.
RND, symmetry extensions, distributed updates and compilation are not integrated.
No environment or runner is registered by this module.
"""

from copy import deepcopy
from dataclasses import asdict
import weakref

import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.modules import EmpiricalNormalization, GaussianDistribution

from .losses import SLR_TRIPLET_LOSS_COEF, slr_triplet_loss
from .models import SLRModelCfg, SLRNetworks
from .storage import SLRRolloutStorage, sample_negative_indices


class _FeedForwardModel(nn.Module):
    is_recurrent = False

    def get_hidden_state(self):
        return None

    def reset(self, dones=None, hidden_state=None) -> None:
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        pass


class SLRPolicy(_FeedForwardModel):
    """Own all SLR networks once; inference maps raw history to action means.

    Both heads use one 45-feature normalizer, broadcast over history frames.
    Current observations are sliced *after* normalization, never normalized twice.
    running_normalizer accumulates only newly observed last frames, not repeated
    history slots. Its statistics are copied into the forward normalizer only at
    rollout boundaries, so old/new PPO probabilities use identical preprocessing.
    Both sets of statistics are buffers in the checkpoint; forward never updates
    either, and eval_mode also disables statistics collection.
    """

    def __init__(
        self, networks: SLRNetworks, obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.networks = networks
        self.obs_normalization = obs_normalization
        self.obs_normalizer = (
            EmpiricalNormalization(networks.cfg.observation_dim)
            if obs_normalization else nn.Identity()
        )
        self.running_normalizer = deepcopy(self.obs_normalizer)
        options = deepcopy(distribution_cfg) if distribution_cfg is not None else {}
        if options.pop("class_name", "GaussianDistribution") != "GaussianDistribution":
            raise ValueError("SLR requires GaussianDistribution")
        self.distribution = GaussianDistribution(networks.cfg.action_dim, **options)

    def normalize_history(self, history: torch.Tensor) -> torch.Tensor:
        return self.obs_normalizer(history)

    def forward(self, obs: TensorDict, stochastic_output: bool = False) -> torch.Tensor:
        history = self.normalize_history(obs["history"])
        mean = self.networks.act(history[:, -1, :], history)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def value(self, obs: TensorDict) -> torch.Tensor:
        history = self.normalize_history(obs["history"])
        return self.networks.evaluate(history[:, -1, :], history)

    def update_distribution(self, obs: TensorDict) -> None:
        """Evaluate the current mean without resampling rollout actions."""
        self.distribution.update(self(obs))

    def get_output_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    @torch.no_grad()
    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.running_normalizer.update(obs["history"][:, -1, :])

    @torch.no_grad()
    def begin_rollout(self, obs: TensorDict) -> None:
        if self.obs_normalization and self.training:
            if self.running_normalizer.count.item() == 0:
                self.update_normalization(obs)
            self.obs_normalizer.load_state_dict(self.running_normalizer.state_dict())


class _CriticView(_FeedForwardModel):
    """Parameter-free access to the sole policy owner's value path.

    The weak reference is deliberately not a registered child module. PPO's
    actor+critic parameter chain thus includes each of the four networks once.
    """

    def __init__(self, policy: SLRPolicy) -> None:
        super().__init__()
        self._policy = weakref.ref(policy)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        return self._policy().value(obs)


class SLRPPO(PPO):
    def __init__(
        self,
        networks: SLRNetworks,
        storage: SLRRolloutStorage,
        *,
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        triplet_loss_coef: float = SLR_TRIPLET_LOSS_COEF,
        triplet_margin: float = 1.0,
        **ppo_kwargs,
    ) -> None:
        for name in ("rnd_cfg", "symmetry_cfg", "multi_gpu_cfg"):
            if ppo_kwargs.get(name) is not None:
                raise ValueError(f"SLR does not yet integrate {name}")
        policy = SLRPolicy(networks, obs_normalization, distribution_cfg)
        super().__init__(policy, _CriticView(policy), storage, **ppo_kwargs)
        self.transition = storage.Transition()
        self.triplet_loss_coef = triplet_loss_coef
        self.triplet_margin = triplet_margin
        self._slr_metrics: dict[str, float] = {}

    @property
    def networks(self) -> SLRNetworks:
        return self._raw_actor.networks

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.storage.step == 0:
            self._raw_actor.begin_rollout(obs)
        return super().act(obs)

    @torch.no_grad()
    def process_env_step(self, obs, rewards, dones, extras) -> None:
        self.transition.next_history = obs["history"]
        # Parent collects next-frame running statistics once, bootstraps timeout
        # rewards, records the transition and clears its temporary container.
        super().process_env_step(obs, rewards, dones, extras)

    def ppo_loss_terms(self, batch: SLRRolloutStorage.Batch) -> dict[str, torch.Tensor]:
        """RSL-RL clipped objectives evaluated on the *stored* sampled actions."""
        self.actor.update_distribution(batch.observations)
        log_prob = self.actor.get_output_log_prob(batch.actions)
        values = self.critic(batch.observations)
        advantages = batch.advantages.squeeze(-1)
        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze(-1))
        surrogate = torch.maximum(
            -advantages * ratio,
            -advantages * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param),
        ).mean()
        value_errors = (values - batch.returns).square()
        if self.use_clipped_value_loss:
            clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
            value_errors = torch.maximum(value_errors, (clipped - batch.returns).square())
        with torch.no_grad():
            kl = self.actor.get_kl_divergence(
                batch.old_distribution_params, self.actor.output_distribution_params,
            ).mean()
        return {
            "surrogate": surrogate, "value": value_errors.mean(),
            "entropy": self.actor.output_entropy.mean(), "kl": kl, "ratio": ratio,
        }

    def compute_slr_loss(self, batch: SLRRolloutStorage.Batch) -> torch.Tensor:
        valid, negative = sample_negative_indices(batch.valid_transition_mask)
        self._slr_metrics = {
            "valid_triplet_fraction": valid.numel() / batch.history.shape[0],
            "latent_norm": 0.0, "positive_distance": 0.0, "negative_distance": 0.0,
        }
        if valid.numel() == 0:
            return batch.history.new_zeros(())
        history = self.actor.normalize_history(batch.history[valid])
        next_history = self.actor.normalize_history(batch.next_history[valid])
        z = self.networks.encoder(history)
        z_next = self.networks.encoder(next_history)
        # Sampler indices refer to original minibatch rows; map them to the
        # compact valid-only latent batch. No invalid next histories are encoded.
        permutation = torch.searchsorted(valid, negative)
        z_pred = self.networks.transition(z, batch.actions[valid])
        with torch.no_grad():
            self._slr_metrics.update(
                latent_norm=z.norm(dim=-1).mean().item(),
                positive_distance=(z_next - z_pred).square().sum(-1).mean().item(),
                negative_distance=(z_next - z_next[permutation]).square().sum(-1).mean().item(),
            )
        return slr_triplet_loss(z_next, z_pred, z_next[permutation], self.triplet_margin)

    def update(self) -> dict[str, float]:
        totals = dict.fromkeys((
            "surrogate", "value", "entropy", "slr", "slr_weighted", "total", "kl",
            "valid_triplet_fraction", "latent_norm", "positive_distance", "negative_distance",
            "encoder_grad_norm",
        ), 0.0)
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            terms = self.ppo_loss_terms(batch)
            if self.desired_kl is not None and self.schedule == "adaptive":
                kl = terms["kl"].item()
                if kl > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif 0.0 < kl < self.desired_kl / 2.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate
            slr = self.compute_slr_loss(batch)
            loss = (terms["surrogate"] + self.value_loss_coef * terms["value"]
                    - self.entropy_coef * terms["entropy"] + self.triplet_loss_coef * slr)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            encoder_grad_squares = [
                p.grad.detach().square().sum()
                for p in self.networks.encoder.parameters() if p.grad is not None
            ]
            if encoder_grad_squares:
                totals["encoder_grad_norm"] += torch.stack(encoder_grad_squares).sum().sqrt().item()
            # One global clip includes Encoder, both heads, Transition and std
            # exactly once, following official SLR's joint optimizer semantics.
            nn.utils.clip_grad_norm_(self._raw_actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            for name in ("surrogate", "value", "entropy", "kl"):
                totals[name] += terms[name].item()
            totals["slr"] += slr.item()
            totals["slr_weighted"] += self.triplet_loss_coef * slr.item()
            totals["total"] += loss.item()
            for name, value in self._slr_metrics.items():
                totals[name] += value
        count = self.num_mini_batches * self.num_learning_epochs
        self.storage.clear()
        return {name: value / count for name, value in totals.items()}

    def save(self) -> dict:
        """Parent actor_state_dict contains all networks, std and both norm states.

        critic_state_dict is empty: the Critic view does not own duplicate
        parameters. Full save/load otherwise follows the Runner's normal API.
        """
        saved = super().save()
        saved["slr_model_cfg"] = asdict(self.networks.cfg)
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        if strict and loaded_dict["slr_model_cfg"] != asdict(self.networks.cfg):
            raise ValueError("Checkpoint SLR dimensions differ from the constructed model")
        if load_cfg is not None:
            load_cfg = dict(load_cfg)
            # Actor and Critic share an owner; either request restores that
            # complete owner, including the single Encoder and normalizers.
            if load_cfg.get("critic"):
                load_cfg["actor"] = True
        return super().load(loaded_dict, load_cfg, strict)

    def compile(self, mode: str | None = None) -> None:
        if mode is not None:
            raise NotImplementedError("SLR torch.compile integration is deferred")

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> "SLRPPO":
        """Construct from a standard Runner dict, without changing PPO configs.

        Set algorithm.class_name to this class's qualified path. The Actor/
        Critic hidden sizes and normalization toggle come from their config;
        optional algorithm.model_cfg overrides the SLR dimension defaults.
        """
        if set(obs.keys()) != {"history"}:
            raise ValueError("SLR expects only the non-privileged history observation group")
        if cfg.get("clip_actions") is not None or getattr(env, "clip_actions", None) is not None:
            raise ValueError("SLR requires unclipped raw sampled actions")
        if any(cfg[name].get("activation", "elu") != "elu" for name in ("actor", "critic")):
            raise ValueError("SLRNetworks currently uses ELU hidden activations")
        normalize = cfg["actor"].get("obs_normalization", True)
        if normalize != cfg["critic"].get("obs_normalization", True):
            raise ValueError("SLR Actor and Critic must share the normalization setting")
        options = deepcopy(cfg["algorithm"])
        options.pop("class_name", None)
        if options.pop("share_cnn_encoders", False):
            raise ValueError("SLR does not use CNN encoders")
        dimensions = options.pop("model_cfg", {})
        defaults = {
            "observation_dim": obs["history"].shape[-1],
            "history_length": obs["history"].shape[-2], "action_dim": env.num_actions,
            "actor_hidden_dims": tuple(cfg["actor"].get("hidden_dims", (512, 256, 128))),
            "critic_hidden_dims": tuple(cfg["critic"].get("hidden_dims", (512, 256, 128))),
        }
        model_cfg = SLRModelCfg(**(defaults | dimensions))
        if (model_cfg.history_length, model_cfg.observation_dim) != tuple(obs["history"].shape[1:]):
            raise ValueError("SLR model dimensions do not match the observation history")
        if model_cfg.action_dim != env.num_actions:
            raise ValueError("SLR action dimension does not match the environment")
        storage = SLRRolloutStorage(
            env.num_envs, cfg["num_steps_per_env"], obs, (env.num_actions,), device,
        )
        cfg["obs_groups"] = {"actor": ["history"], "critic": ["history"]}
        # Runner/Logger expect the optional extension key even when disabled.
        cfg["algorithm"].setdefault("rnd_cfg", None)
        cfg["algorithm"].setdefault("symmetry_cfg", None)
        algorithm = SLRPPO(
            SLRNetworks(model_cfg), storage, obs_normalization=normalize,
            distribution_cfg=cfg["actor"].get("distribution_cfg"),
            device=device, multi_gpu_cfg=cfg.get("multi_gpu"), **options,
        )
        algorithm.compile(cfg.get("torch_compile_mode"))
        return algorithm
