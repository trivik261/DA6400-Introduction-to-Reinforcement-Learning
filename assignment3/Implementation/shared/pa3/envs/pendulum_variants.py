"""Multi-target Pendulum with three reward formulations.

This is the environment used in Section 2.1 of the assignment (and also for
Section 3 PEBBLE on Pendulum).  At each `reset`, a target angle is sampled
uniformly from `theta_targets` (default {0, -60, 90, 120, -150} degrees).  The
target is exposed as part of the observation so the policy can condition on it.

The three reward formulations follow the screenshot from the assignment:

* `dense`  : 1 if in target, else  -‖x_goal - x_pos‖ - ‖action‖²
* `sparse` : 1 if in target, else  0
* `survival`: -1 every step until the agent reaches the target with near-zero
              velocity (then the episode terminates).  Episodes have varying
              length but are timed out at `T = 1000`; on timeout the agent
              receives a -100 timeout penalty and the env is reset.

"In target" is defined as the (x,y) tip of the pendulum being within
`tol` (default 0.1) of the target tip.
"""
from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces


_DEFAULT_TARGETS_DEG: tuple[int, ...] = (0, -60, 90, 120, -150)


class PendulumMultiTarget(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        reward_type: str = "dense",
        theta_targets_deg: Sequence[float] = _DEFAULT_TARGETS_DEG,
        max_episode_steps: int = 1000,
        tol: float = 0.1,
        vel_tol: float = 0.5,
        timeout_penalty: float = -100.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        assert reward_type in {"dense", "sparse", "survival"}
        self.reward_type = reward_type
        self.theta_targets = np.deg2rad(np.asarray(theta_targets_deg, dtype=np.float32))
        self.max_episode_steps = max_episode_steps
        self.tol = tol
        self.vel_tol = vel_tol
        self.timeout_penalty = timeout_penalty
        self._inner = gym.make("Pendulum-v1", render_mode=render_mode)
        self.render_mode = render_mode

        # Observation: cos(th), sin(th), th_dot, target_x, target_y
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -8, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 8, 1, 1], dtype=np.float32),
        )
        self.action_space = self._inner.action_space  # Box(-2, 2, (1,))
        self._target_xy = np.zeros(2, dtype=np.float32)
        self._t = 0

    # ---------------------------- helpers ---------------------------------
    def _tip_xy(self, obs5: np.ndarray) -> np.ndarray:
        return np.array([obs5[1], -obs5[0]], dtype=np.float32)  # (sin, -cos)

    def _make_obs(self) -> np.ndarray:
        inner_obs = self._inner.unwrapped._get_obs()  # type: ignore[attr-defined]
        return np.concatenate([inner_obs.astype(np.float32), self._target_xy])

    def _in_target(self, obs5: np.ndarray) -> bool:
        tip = self._tip_xy(obs5)
        return float(np.linalg.norm(tip - self._target_xy)) < self.tol

    # ---------------------------- API -------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        _, info = self._inner.reset(seed=seed)
        idx = int(np.random.randint(len(self.theta_targets)))
        theta = float(self.theta_targets[idx])
        self._target_xy = np.array([np.sin(theta), -np.cos(theta)], dtype=np.float32)
        self._t = 0
        return self._make_obs(), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        _, _, _, _, info = self._inner.step(action)
        obs5 = self._make_obs()
        in_t = self._in_target(obs5)
        self._t += 1

        if self.reward_type == "dense":
            if in_t:
                reward = 1.0
            else:
                tip = self._tip_xy(obs5)
                reward = -float(np.linalg.norm(tip - self._target_xy)) - float(
                    np.square(action).sum()
                )
            terminated = False
            truncated = self._t >= self.max_episode_steps
        elif self.reward_type == "sparse":
            reward = 1.0 if in_t else 0.0
            terminated = False
            truncated = self._t >= self.max_episode_steps
        else:  # "survival"
            theta_dot = float(obs5[2])
            done_at_target = in_t and (abs(theta_dot) < self.vel_tol)
            if done_at_target:
                reward = -1.0
                terminated = True
                truncated = False
            elif self._t >= self.max_episode_steps:
                reward = self.timeout_penalty
                terminated = False
                truncated = True
            else:
                reward = -1.0
                terminated = False
                truncated = False
        return obs5, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self._inner.render()

    def close(self):
        return self._inner.close()
