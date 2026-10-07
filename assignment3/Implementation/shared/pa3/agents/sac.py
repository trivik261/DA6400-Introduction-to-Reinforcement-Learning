"""SAC implementation supporting continuous *and* discrete actions.

Reference: Haarnoja et al. 2018 (continuous), Christodoulou 2019 (discrete).

Continuous-action variant
-------------------------
* squashed Gaussian policy
* clipped double-Q critic
* target critic, polyak updated with `tau`
* automated temperature: minimise  -α·(log π(a|s) + H_target).w.r.t. log α

Discrete-action variant
-----------------------
* categorical policy
* clipped double-Q critic with one head per action
* target value uses an *expectation* over actions (no sampling needed):
    V(s) = Σ_a π(a|s)·(Q(s,a) - α·log π(a|s))
* automated temperature uses the same gradient with the *expected* entropy
* target entropy defaults to `0.98·log|A|` (Christodoulou 2019)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from pa3.utils.networks import (
    CategoricalActor,
    DiscreteQNetwork,
    GaussianActor,
    QNetwork,
    soft_update,
)


@dataclass
class SACConfig:
    obs_dim: int
    action_dim: int               # continuous: dim, discrete: |A|
    discrete: bool = False
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    auto_alpha: bool = True
    init_alpha: float = 0.2
    target_entropy: float | None = None
    hidden: tuple[int, ...] = (256, 256)
    device: str = "cpu"


class SACAgent:
    def __init__(self, cfg: SACConfig):
        self.cfg = cfg
        device = torch.device(cfg.device)
        self.device = device

        if cfg.discrete:
            self.actor = CategoricalActor(cfg.obs_dim, cfg.action_dim, cfg.hidden).to(device)
            self.critic = DiscreteQNetwork(cfg.obs_dim, cfg.action_dim, cfg.hidden).to(device)
            self.target_critic = DiscreteQNetwork(cfg.obs_dim, cfg.action_dim, cfg.hidden).to(device)
            target_entropy = cfg.target_entropy
            if target_entropy is None:
                target_entropy = 0.98 * (-np.log(1.0 / cfg.action_dim))
        else:
            self.actor = GaussianActor(cfg.obs_dim, cfg.action_dim, cfg.hidden).to(device)
            self.critic = QNetwork(cfg.obs_dim, cfg.action_dim, cfg.hidden).to(device)
            self.target_critic = QNetwork(cfg.obs_dim, cfg.action_dim, cfg.hidden).to(device)
            target_entropy = cfg.target_entropy
            if target_entropy is None:
                target_entropy = -float(cfg.action_dim)
        self.target_entropy = float(target_entropy)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # Automated entropy.  Optimise log_alpha for numerical stability.
        self._log_alpha = torch.tensor(
            np.log(cfg.init_alpha), dtype=torch.float32, device=device, requires_grad=cfg.auto_alpha
        )
        if cfg.auto_alpha:
            self.alpha_opt = torch.optim.Adam([self._log_alpha], lr=cfg.alpha_lr)
        else:
            self.alpha_opt = None

    @property
    def alpha(self) -> torch.Tensor:
        return self._log_alpha.exp()

    # --------------------------- action selection -------------------------
    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.cfg.discrete:
            probs, _ = self.actor(obs_t)
            if deterministic:
                a = int(torch.argmax(probs, dim=-1).item())
            else:
                a = int(torch.distributions.Categorical(probs=probs).sample().item())
            return np.int64(a)
        else:
            action, _, mean_action = self.actor.sample(obs_t)
            if deterministic:
                return mean_action.squeeze(0).cpu().numpy()
            return action.squeeze(0).cpu().numpy()

    # --------------------------- updates ----------------------------------
    def update(self, batch) -> dict[str, float]:
        if self.cfg.discrete:
            return self._update_discrete(batch)
        return self._update_continuous(batch)

    def _update_continuous(self, batch) -> dict[str, float]:
        obs = batch.obs
        action = batch.action
        reward = batch.reward.unsqueeze(-1)
        next_obs = batch.next_obs
        not_done = (1.0 - batch.done).unsqueeze(-1)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_obs)
            tq1, tq2 = self.target_critic(next_obs, next_action)
            target_q = torch.min(tq1, tq2) - self.alpha.detach() * next_log_pi
            y = reward + self.cfg.gamma * not_done * target_q

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        new_action, log_pi, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_pi - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.alpha_opt is not None:
            alpha_loss = -(self._log_alpha * (log_pi.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
        else:
            alpha_loss = torch.zeros((), device=self.device)

        soft_update(self.target_critic, self.critic, self.cfg.tau)
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "entropy": float(-log_pi.mean().detach().cpu()),
        }

    def _update_discrete(self, batch) -> dict[str, float]:
        obs = batch.obs
        action = batch.action.long()
        reward = batch.reward.unsqueeze(-1)
        next_obs = batch.next_obs
        not_done = (1.0 - batch.done).unsqueeze(-1)

        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_obs)
            tq1, tq2 = self.target_critic(next_obs)
            tq = torch.min(tq1, tq2)
            v_next = (next_probs * (tq - self.alpha.detach() * next_log_probs)).sum(-1, keepdim=True)
            y = reward + self.cfg.gamma * not_done * v_next

        q1_all, q2_all = self.critic(obs)
        q1 = q1_all.gather(1, action.unsqueeze(-1))
        q2 = q2_all.gather(1, action.unsqueeze(-1))
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        probs, log_probs = self.actor(obs)
        q1_all, q2_all = self.critic(obs)
        q_min = torch.min(q1_all, q2_all)
        actor_loss = (probs * (self.alpha.detach() * log_probs - q_min)).sum(-1).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        entropy = -(probs * log_probs).sum(-1)  # H[π(·|s)]
        if self.alpha_opt is not None:
            alpha_loss = (self._log_alpha * (entropy.detach() - self.target_entropy)).mean()
            # Note: minimising  log_alpha · (H - H_target)  pushes alpha *up* when
            # entropy < target and *down* when entropy > target — same gradient
            # as the continuous case (with H = -log π).
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
        else:
            alpha_loss = torch.zeros((), device=self.device)

        soft_update(self.target_critic, self.critic, self.cfg.tau)
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "entropy": float(entropy.mean().detach().cpu()),
        }

    # --------------------------- ckpt -------------------------------------
    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self._log_alpha.detach().cpu(),
        }

    def full_state_dict(self) -> dict:
        """Includes optimizer state — used for resumable checkpoints."""
        d = self.state_dict()
        d["actor_opt"] = self.actor_opt.state_dict()
        d["critic_opt"] = self.critic_opt.state_dict()
        if self.alpha_opt is not None:
            d["alpha_opt"] = self.alpha_opt.state_dict()
        return d

    def load_full_state_dict(self, d: dict) -> None:
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.target_critic.load_state_dict(d["target_critic"])
        with torch.no_grad():
            self._log_alpha.copy_(d["log_alpha"].to(self._log_alpha.device))
        if "actor_opt" in d:
            self.actor_opt.load_state_dict(d["actor_opt"])
        if "critic_opt" in d:
            self.critic_opt.load_state_dict(d["critic_opt"])
        if self.alpha_opt is not None and "alpha_opt" in d:
            self.alpha_opt.load_state_dict(d["alpha_opt"])
