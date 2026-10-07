"""Single factory used by every training script.

Supported `env_id`s
-------------------
* ``LunarLanderContinuous-v3``     : standard continuous-action env
* ``LunarLander-v3``               : standard discrete-action env
* ``LunarLanderHover-v3``          : LunarLanderContinuous-v3 + HoverRewardWrapper
* ``Pendulum-{dense,sparse,survival}`` : multi-target pendulum
* ``Pendulum-multi``               : alias for ``Pendulum-dense``
* ``ReacherEasy-{Ra,Rb,Rc}``       : dm_control reacher_easy with reward Ri
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from .lunarlander_hover import HoverRewardWrapper
from .pendulum_variants import PendulumMultiTarget
from .reacher_dmc import ReacherEasyEnv


def make_env(env_id: str, seed: int | None = None, render_mode: str | None = None, **kwargs: Any):
    """Factory used by the training scripts.

    Returns a *single* (non-vectorised) gym Env.  All training in this
    assignment is single-process which keeps the SAC update loop simple and
    reproducible across seeds.
    """
    if env_id == "LunarLanderContinuous-v3":
        return gym.make("LunarLander-v3", continuous=True, render_mode=render_mode, **kwargs)
    if env_id == "LunarLander-v3":
        return gym.make("LunarLander-v3", continuous=False, render_mode=render_mode, **kwargs)
    if env_id == "LunarLanderHover-v3":
        env = gym.make("LunarLander-v3", continuous=True, render_mode=render_mode, **kwargs)
        return HoverRewardWrapper(env, bonus=200.0)
    if env_id.startswith("Pendulum-"):
        suffix = env_id.split("-", 1)[1]
        if suffix == "multi":
            suffix = "dense"
        targets = kwargs.pop("theta_targets_deg", None)
        if targets is None:
            return PendulumMultiTarget(reward_type=suffix, render_mode=render_mode)
        return PendulumMultiTarget(
            reward_type=suffix, theta_targets_deg=targets, render_mode=render_mode
        )
    if env_id.startswith("ReacherEasy-"):
        rid = env_id.split("-", 1)[1]
        return ReacherEasyEnv(reward_id=rid, seed=seed, render_mode=render_mode)
    raise ValueError(f"unknown env_id={env_id!r}")
