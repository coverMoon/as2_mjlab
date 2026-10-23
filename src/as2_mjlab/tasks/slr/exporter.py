"""Stateless SLR ONNX inference: normalized history -> Encoder -> Actor mean."""

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch
from torch import nn
from tensordict import TensorDict

if TYPE_CHECKING:
    from .algorithm import SLRPolicy


class OnnxSLRPolicy(nn.Module):
    """MjlabOnPolicyRunner export interface, without training-only networks.

    Input history already contains observation term scaling/clipping. Feature
    normalization is inside the graph. The caller owns the oldest-to-newest
    history window and fills it with the first observation after each reset.
    """

    is_recurrent = False

    def __init__(self, policy: "SLRPolicy", verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.history_length = policy.networks.cfg.history_length
        self.observation_dim = policy.networks.cfg.observation_dim
        self.obs_normalizer = deepcopy(policy.obs_normalizer)
        self.encoder = deepcopy(policy.networks.encoder.mlp)
        self.actor = deepcopy(policy.networks.actor.mlp)
        self.cpu().eval()

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        history = self.obs_normalizer(history)
        latent = self.encoder(history.flatten(start_dim=1))
        return self.actor(torch.cat((latent, history[:, -1, :]), dim=-1))

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.history_length, self.observation_dim),)

    @property
    def input_names(self) -> list[str]:
        return ["history"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


def verify_slr_onnx(policy: "SLRPolicy", model: onnx.ModelProto, sample: torch.Tensor) -> dict:
    """Compare the ONNX graph with the full PyTorch policy on raw histories."""
    policy = deepcopy(policy).cpu().eval()
    evaluator = ReferenceEvaluator(model)
    shape = (1, policy.networks.cfg.history_length, policy.networks.cfg.observation_dim)
    generator = torch.Generator().manual_seed(20261023)
    random_history = torch.randn(shape, generator=generator)
    samples = [sample[:1].detach().cpu().float(), torch.zeros(shape), random_history,
               random_history * 3, random_history[:, -1:, :].expand(shape).clone()]
    max_error = 0.0
    with torch.inference_mode():
        for history in samples:
            expected = policy(TensorDict({"history": history}, batch_size=[1])).numpy()
            actual = evaluator.run(None, {"history": history.numpy()})[0]
            if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                raise ValueError("SLR ONNX verification produced nonfinite output")
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    return {"backend": "onnx.reference.ReferenceEvaluator", "samples": len(samples),
            "rtol": 1e-4, "atol": 1e-5, "max_abs_error": max_error,
            "includes_observation_normalization": True}
