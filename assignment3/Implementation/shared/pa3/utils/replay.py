"""Replay buffers: a standard transition buffer and a trajectory buffer used
by PEBBLE for sampling preference segments.

The transition buffer pre-allocates fixed-size NumPy arrays for speed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_shape: tuple[int, ...],
        action_dtype: np.dtype,
        device: str | torch.device = "cpu",
    ):
        self.capacity = capacity
        self.device = torch.device(device)
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros((capacity, *action_shape), dtype=action_dtype)
        self._reward = np.zeros((capacity,), dtype=np.float32)
        self._done = np.zeros((capacity,), dtype=np.float32)
        # Episode/step bookkeeping is useful for PEBBLE (segment sampling).
        self._ep_id = np.zeros((capacity,), dtype=np.int64)
        self._t_in_ep = np.zeros((capacity,), dtype=np.int64)
        self._idx = 0
        self._size = 0
        self._cur_ep = 0
        self._cur_t = 0

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        episode_done: bool | None = None,
    ) -> None:
        i = self._idx
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._action[i] = action
        self._reward[i] = reward
        self._done[i] = float(done)
        self._ep_id[i] = self._cur_ep
        self._t_in_ep[i] = self._cur_t
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        if episode_done is None:
            episode_done = bool(done)
        if episode_done:
            self._cur_ep += 1
            self._cur_t = 0
        else:
            self._cur_t += 1

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self._size, size=batch_size)
        return Batch(
            obs=torch.as_tensor(self._obs[idx], device=self.device),
            action=torch.as_tensor(self._action[idx], device=self.device),
            reward=torch.as_tensor(self._reward[idx], device=self.device),
            next_obs=torch.as_tensor(self._next_obs[idx], device=self.device),
            done=torch.as_tensor(self._done[idx], device=self.device),
        )

    def state_dict(self) -> dict:
        return {
            "obs": self._obs[: self._size].copy(),
            "next_obs": self._next_obs[: self._size].copy(),
            "action": self._action[: self._size].copy(),
            "reward": self._reward[: self._size].copy(),
            "done": self._done[: self._size].copy(),
            "ep_id": self._ep_id[: self._size].copy(),
            "t_in_ep": self._t_in_ep[: self._size].copy(),
            "idx": int(self._idx),
            "size": int(self._size),
            "cur_ep": int(self._cur_ep),
            "cur_t": int(self._cur_t),
            "capacity": int(self.capacity),
        }

    def load_state_dict(self, d: dict) -> None:
        n = int(d["size"])
        self._obs[:n] = d["obs"]
        self._next_obs[:n] = d["next_obs"]
        self._action[:n] = d["action"]
        self._reward[:n] = d["reward"]
        self._done[:n] = d["done"]
        self._ep_id[:n] = d["ep_id"]
        self._t_in_ep[:n] = d["t_in_ep"]
        self._idx = int(d["idx"])
        self._size = n
        self._cur_ep = int(d["cur_ep"])
        self._cur_t = int(d["cur_t"])

    def relabel_rewards(self, fn) -> None:
        """Replace stored rewards in-place using `fn(obs, action) -> reward`.

        Used by PEBBLE to relabel the buffer with the latest learned reward.
        """
        if self._size == 0:
            return
        obs = self._obs[: self._size]
        act = self._action[: self._size]
        with torch.no_grad():
            r = fn(obs, act)
        self._reward[: self._size] = np.asarray(r, dtype=np.float32).reshape(-1)

    def sample_segments(self, num_segments: int, seg_len: int):
        """Return contiguous segments of length `seg_len` (used by PEBBLE).

        Segments are required to lie inside a single episode.  We sample with
        rejection: pick a starting index and only accept it if all `seg_len`
        consecutive entries share the same episode id.
        """
        out_idx = np.empty((num_segments, seg_len), dtype=np.int64)
        if self._size <= seg_len:
            return None
        max_start = self._size - seg_len
        n_filled = 0
        attempts = 0
        while n_filled < num_segments and attempts < num_segments * 50:
            start = np.random.randint(0, max_start)
            ep = self._ep_id[start]
            if np.all(self._ep_id[start : start + seg_len] == ep):
                out_idx[n_filled] = np.arange(start, start + seg_len)
                n_filled += 1
            attempts += 1
        if n_filled < num_segments:  # fall back: ignore episode boundaries
            for k in range(n_filled, num_segments):
                start = np.random.randint(0, max_start)
                out_idx[k] = np.arange(start, start + seg_len)
        obs = self._obs[out_idx]
        act = self._action[out_idx]
        rew = self._reward[out_idx]
        return obs, act, rew, out_idx
