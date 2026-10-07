"""Offline evaluation utilities.

* `evaluate_policy`   — average undiscounted return over `n_episodes`.
* `evaluate_with_rewards` — for the Reacher cross-evaluation we want to know,
  for the *same* trajectories, what the return would have been under several
  candidate reward functions Ra/Rb/Rc.  The env is expected to expose a
  `compute_reward(reward_id, obs, action, next_obs, info)` callable.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np


_RC_EVAL_TIMEOUT_PENALTY = -20.0


def evaluate_policy(
    env,
    select_action: Callable[[np.ndarray], np.ndarray],
    n_episodes: int = 20,
    max_steps: int = 0,
) -> float:
    """Average undiscounted return over `n_episodes`.

    For Rc evaluation (middleground from TA): pass ``max_steps=1000``.
    Episodes are capped at ``max_steps``; the env itself already applies
    the -20 timeout penalty on the 1000th step, so we just stop the
    rollout here without adding anything extra.  Set ``max_steps=0``
    (default) for no cap.
    """
    returns: list[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        t = 0
        while not done:
            action = select_action(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += float(r)
            t += 1
            done = terminated or truncated
            if max_steps > 0 and t >= max_steps and not done:
                break
        returns.append(ep_ret)
    return float(np.mean(returns))


def evaluate_with_rewards(
    env,
    select_action: Callable[[np.ndarray], np.ndarray],
    reward_ids: Iterable[str],
    n_episodes: int = 20,
    max_steps: int = 0,
) -> dict[str, float]:
    """Roll out greedy episodes once and compute the undiscounted return under
    each reward formulation in `reward_ids`.

    The env must implement `compute_reward(reward_id, obs, action, next_obs, info)`.

    For Rc eval (middleground): pass ``max_steps=1000``.  On timeout the
    Rc return gets a -20 penalty; other formulations just use the accumulated
    value up to that point.
    """
    reward_ids = list(reward_ids)
    sums = {rid: [] for rid in reward_ids}
    is_rc = getattr(env, "reward_id", None) == "Rc"
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        partial = {rid: 0.0 for rid in reward_ids}
        t = 0
        while not done:
            action = select_action(obs)
            next_obs, _, terminated, truncated, info = env.step(action)
            for rid in reward_ids:
                partial[rid] += float(env.compute_reward(rid, obs, action, next_obs, info))
            obs = next_obs
            t += 1
            done = terminated or truncated
            if max_steps > 0 and t >= max_steps and not done:
                if is_rc:
                    partial["Rc"] = partial.get("Rc", 0.0) + _RC_EVAL_TIMEOUT_PENALTY
                break
        for rid in reward_ids:
            sums[rid].append(partial[rid])
    return {rid: float(np.mean(vals)) for rid, vals in sums.items()}
