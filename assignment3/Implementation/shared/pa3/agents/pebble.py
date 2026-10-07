"""PEBBLE: Preference-based RL on top of SAC (Lee et al. 2021).

Building blocks (all faithful to the paper):

1. Reward ensemble  ``r̂_ψ``  (3 small MLPs by default).
2. Preference buffer  D — sampled segments with simulated-teacher labels.
3. Unsupervised pre-training: maximise *state entropy* (k-NN on obs) for the
   first `unsup_steps` env steps before any reward query / SAC reward learning.
4. Each iteration: sample preference queries (uniform sampling here; the paper
   also studies disagreement-based sampling — toggleable), label them with the
   simulated teacher (Bradley-Terry on segment ground-truth return sums), train
   the reward ensemble with a binary cross-entropy loss, **relabel the entire
   replay buffer** with the latest mean predicted reward, and continue SAC
   updates on the relabelled buffer.

The training loop lives in ``train.train_pebble``; this file exposes a single
class ``PEBBLEAgent`` that owns the SAC critic/actor *and* the reward ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pa3.agents.sac import SACAgent, SACConfig
from pa3.utils.networks import RewardPredictor


@dataclass
class PEBBLEConfig:
    obs_dim: int
    action_dim: int
    discrete: bool = False  # PEBBLE in this assignment is only used continuous
    ensemble_size: int = 3
    seg_len: int = 50
    pref_budget: int = 1000          # max queries
    queries_per_session: int = 64
    sessions_every: int = 5_000      # env steps between query sessions
    reward_train_epochs: int = 50
    reward_lr: float = 3e-4
    reward_hidden: tuple[int, ...] = (256, 256, 256)
    unsup_steps: int = 9_000         # exploration via state-entropy bonus
    knn_k: int = 5
    sac: SACConfig = field(default_factory=lambda: SACConfig(obs_dim=0, action_dim=0))


class PEBBLEAgent:
    def __init__(self, cfg: PEBBLEConfig):
        self.cfg = cfg
        self.cfg.sac.obs_dim = cfg.obs_dim
        self.cfg.sac.action_dim = cfg.action_dim
        self.cfg.sac.discrete = cfg.discrete
        self.sac = SACAgent(cfg.sac)
        self.device = self.sac.device
        self.ensemble = nn.ModuleList(
            [RewardPredictor(cfg.obs_dim, cfg.action_dim, cfg.reward_hidden) for _ in range(cfg.ensemble_size)]
        ).to(self.device)
        self.opt = torch.optim.Adam(self.ensemble.parameters(), lr=cfg.reward_lr)

        # Preference buffer: store *flat* segment data + binary label.
        self.pref_segs1: list[tuple[np.ndarray, np.ndarray]] = []
        self.pref_segs2: list[tuple[np.ndarray, np.ndarray]] = []
        self.pref_labels: list[float] = []  # 1.0 if seg2 preferred, 0.0 if seg1, 0.5 tie
        self.queries_used = 0

    # -------------------- reward prediction -------------------------------
    @torch.no_grad()
    def predict_reward(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if act_t.ndim == 1:
            act_t = act_t.unsqueeze(0)
        preds = torch.stack([m(obs_t, act_t) for m in self.ensemble], dim=0)  # E,B,1
        return preds.mean(0).squeeze(-1).cpu().numpy()

    @torch.no_grad()
    def reward_disagreement(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        preds = torch.stack([m(obs, action) for m in self.ensemble], dim=0)
        return preds.std(0)

    # -------------------- preference queries ------------------------------
    def add_pref_query(
        self,
        seg1_obs: np.ndarray,
        seg1_act: np.ndarray,
        seg2_obs: np.ndarray,
        seg2_act: np.ndarray,
        label: float,
    ) -> None:
        self.pref_segs1.append((seg1_obs.astype(np.float32), seg1_act.astype(np.float32)))
        self.pref_segs2.append((seg2_obs.astype(np.float32), seg2_act.astype(np.float32)))
        self.pref_labels.append(float(label))
        self.queries_used += 1

    def train_reward(self, epochs: int | None = None, batch_size: int = 64) -> dict[str, float]:
        if not self.pref_labels:
            return {"reward_loss": 0.0, "reward_acc": 0.0}
        epochs = epochs if epochs is not None else self.cfg.reward_train_epochs
        N = len(self.pref_labels)
        labels = torch.as_tensor(self.pref_labels, dtype=torch.float32, device=self.device)
        s1_obs = torch.as_tensor(np.stack([s[0] for s in self.pref_segs1]), device=self.device)
        s1_act = torch.as_tensor(np.stack([s[1] for s in self.pref_segs1]), device=self.device)
        s2_obs = torch.as_tensor(np.stack([s[0] for s in self.pref_segs2]), device=self.device)
        s2_act = torch.as_tensor(np.stack([s[1] for s in self.pref_segs2]), device=self.device)

        last_loss = 0.0
        last_acc = 0.0
        for _ in range(epochs):
            idx = np.random.permutation(N)
            losses, accs = [], []
            for start in range(0, N, batch_size):
                b = idx[start : start + batch_size]
                if len(b) == 0:
                    continue
                ob1 = s1_obs[b]
                ac1 = s1_act[b]
                ob2 = s2_obs[b]
                ac2 = s2_act[b]
                lab = labels[b]
                # Train each ensemble member on the same batch (simple, robust).
                loss_total = 0.0
                for m in self.ensemble:
                    r1 = m(ob1, ac1).sum(dim=1).squeeze(-1)
                    r2 = m(ob2, ac2).sum(dim=1).squeeze(-1)
                    logits = torch.stack([r1, r2], dim=1)  # B,2
                    log_probs = F.log_softmax(logits, dim=1)
                    target = torch.stack([1.0 - lab, lab], dim=1)
                    member_loss = -(target * log_probs).sum(1).mean()
                    loss_total = loss_total + member_loss
                self.opt.zero_grad(set_to_none=True)
                loss_total.backward()
                self.opt.step()
                losses.append(float(loss_total.detach().cpu()) / len(self.ensemble))
                with torch.no_grad():
                    pred_probs = log_probs.exp()  # use last member's probs
                    pred = pred_probs.argmax(1).float()
                    truth = (lab >= 0.5).float()
                    accs.append(float((pred == truth).float().mean().cpu()))
            last_loss = float(np.mean(losses)) if losses else 0.0
            last_acc = float(np.mean(accs)) if accs else 0.0
        return {"reward_loss": last_loss, "reward_acc": last_acc}

    # -------------------- intrinsic / unsupervised ------------------------
    @staticmethod
    def state_entropy_bonus(obs_buffer: np.ndarray, k: int = 5) -> np.ndarray:
        """k-NN state-entropy bonus used by PEBBLE during unsupervised pre-training.

        Computes, for each row, log(distance to k-th nearest neighbour).  Vectorised.
        """
        if obs_buffer.shape[0] <= k + 1:
            return np.zeros(obs_buffer.shape[0], dtype=np.float32)
        x = torch.as_tensor(obs_buffer, dtype=torch.float32)
        d = torch.cdist(x, x)
        d.fill_diagonal_(float("inf"))
        kth, _ = torch.topk(d, k=k, dim=1, largest=False)
        return torch.log(kth[:, -1] + 1.0).cpu().numpy()
