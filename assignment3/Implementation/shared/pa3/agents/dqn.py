"""DQN with target network and ε-greedy exploration.

Used for the LunarLander-v3 (discrete) comparison against discrete-SAC in
Section 2.2.4.  Implementation is intentionally close to the original DQN
paper (no double-DQN, no dueling): we want a fair comparison with a *standard*
DQN as referenced in PA2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from pa3.utils.networks import DQNNetwork


@dataclass
class DQNConfig:
    obs_dim: int
    num_actions: int
    gamma: float = 0.99
    lr: float = 1e-3
    target_update_every: int = 1000  # gradient steps between target sync
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    hidden: tuple[int, ...] = (256, 256)
    device: str = "cpu"


class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.q = DQNNetwork(cfg.obs_dim, cfg.num_actions, cfg.hidden).to(self.device)
        self.q_target = DQNNetwork(cfg.obs_dim, cfg.num_actions, cfg.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self._n_updates = 0

    def epsilon(self, env_step: int) -> float:
        frac = min(1.0, env_step / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, env_step: int = 0, deterministic: bool = False) -> int:
        if (not deterministic) and np.random.rand() < self.epsilon(env_step):
            return int(np.random.randint(self.cfg.num_actions))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.q(obs_t).argmax(-1).item())

    def update(self, batch) -> dict[str, float]:
        obs = batch.obs
        action = batch.action.long()
        reward = batch.reward
        next_obs = batch.next_obs
        not_done = 1.0 - batch.done

        with torch.no_grad():
            target = reward + self.cfg.gamma * not_done * self.q_target(next_obs).max(-1).values

        q_sa = self.q(obs).gather(1, action.unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self._n_updates += 1
        if self._n_updates % self.cfg.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return {"loss": float(loss.detach().cpu()), "q_mean": float(q_sa.mean().detach().cpu())}

    # --------------------------- ckpt ------------------------------------
    def full_state_dict(self) -> dict:
        return {
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "opt": self.opt.state_dict(),
            "n_updates": self._n_updates,
        }

    def load_full_state_dict(self, d: dict) -> None:
        self.q.load_state_dict(d["q"])
        self.q_target.load_state_dict(d["q_target"])
        if "opt" in d:
            self.opt.load_state_dict(d["opt"])
        self._n_updates = int(d.get("n_updates", 0))
