"""LunarLander-v3 wrapper that grants a one-shot hover bonus.

* Stage 0  (`bonus = +200`) : reward shaping that *encourages* hovering inside
  the box `|x| < 0.1` and `0.4 < |y| < 0.6`.
* Stage 1  (`bonus = -100`) : the same hover event is now penalised, used in
  Section 2.2.3 of the assignment to test how SAC adapts when the reward
  function changes mid-training.

The bonus is applied at most once per episode (the first time the lander enters
the hover box) — exactly matching the wording of the assignment.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class HoverRewardWrapper(gym.Wrapper):
    def __init__(self, env, bonus: float = 200.0, x_tol: float = 0.1, y_lo: float = 0.4, y_hi: float = 0.6):
        super().__init__(env)
        self.bonus = float(bonus)
        self.x_tol = x_tol
        self.y_lo = y_lo
        self.y_hi = y_hi
        self._given = False

    def set_bonus(self, value: float) -> None:
        """Switch between +200 (stage 0) and -100 (stage 1) at runtime."""
        self.bonus = float(value)

    def reset(self, **kwargs):
        self._given = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self._given:
            x, y = float(obs[0]), float(obs[1])
            if abs(x) < self.x_tol and self.y_lo < abs(y) < self.y_hi:
                reward += self.bonus
                self._given = True
                info = dict(info or {})
                info["hover_bonus"] = self.bonus
        return obs, reward, terminated, truncated, info
