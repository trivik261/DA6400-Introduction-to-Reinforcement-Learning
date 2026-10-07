"""Network modules used by SAC, DQN and PEBBLE.

The continuous actor produces a tanh-squashed Gaussian.  The discrete actor is a
softmax categorical.  Both critics are double Q-networks for stability (minimum
of two heads).  The PEBBLE reward predictor is a small MLP used in an ensemble.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int, act=nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Continuous SAC
# ---------------------------------------------------------------------------
class GaussianActor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        self.trunk = _mlp(obs_dim, hidden, hidden[-1])
        self.mu_head = nn.Linear(hidden[-1], action_dim)
        self.log_std_head = nn.Linear(hidden[-1], action_dim)

    def forward(self, obs: torch.Tensor):
        h = F.relu(self.trunk(obs))
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor):
        """Return action in [-1, 1], its log-prob and the deterministic mean."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)


class QNetwork(nn.Module):
    """Double Q-network for continuous actions: returns (q1, q2)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = _mlp(obs_dim + action_dim, hidden, 1)
        self.q2 = _mlp(obs_dim + action_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], -1)
        return self.q1(x), self.q2(x)


# ---------------------------------------------------------------------------
# Discrete SAC
# ---------------------------------------------------------------------------
class CategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden=(256, 256)):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, num_actions)

    def forward(self, obs: torch.Tensor):
        logits = self.net(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return probs, log_probs

    def sample(self, obs: torch.Tensor):
        probs, log_probs = self.forward(obs)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return action, probs, log_probs


class DiscreteQNetwork(nn.Module):
    """Outputs Q(s, a) for every discrete action.  Two heads for clipped-double-Q."""

    def __init__(self, obs_dim: int, num_actions: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = _mlp(obs_dim, hidden, num_actions)
        self.q2 = _mlp(obs_dim, hidden, num_actions)

    def forward(self, obs: torch.Tensor):
        return self.q1(obs), self.q2(obs)


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------
class DQNNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden=(256, 256)):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------------------------------------------------------
# PEBBLE reward predictor (ensemble member)
# ---------------------------------------------------------------------------
class RewardPredictor(nn.Module):
    """Maps (obs, action) -> scalar predicted reward."""

    def __init__(self, obs_dim: int, action_dim: int, hidden=(256, 256, 256)):
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, hidden, 1, act=nn.LeakyReLU)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], -1)
        # Tanh keeps the predicted reward bounded, as in PEBBLE.
        return torch.tanh(self.net(x))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)
