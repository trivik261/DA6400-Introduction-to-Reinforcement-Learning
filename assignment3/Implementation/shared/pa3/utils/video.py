"""Video recording helpers.

The code uses `imageio.mimwrite` directly so that we work for both
`gymnasium` envs (which give us `env.render()` -> RGB array) and the
dm_control-derived Reacher env (where we render via the underlying physics
object).  A single `record_episode(env, policy, path)` function produces an mp4
of one greedy episode.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import imageio
import numpy as np


def record_episode(
    env,
    select_action: Callable[[np.ndarray], np.ndarray],
    path: str,
    max_steps: int = 1000,
    fps: int = 30,
    every_n: int = 1,
) -> float:
    """Roll out one greedy episode while recording frames; return total reward.

    `every_n` records only every N-th frame which keeps mp4 size manageable for
    long episodes (e.g. Reacher / LunarLander runs of 1000 steps).
    """
    frames: list[np.ndarray] = []
    obs, _ = env.reset()
    total = 0.0
    for t in range(max_steps):
        if t % every_n == 0:
            try:
                frame = env.render()
            except Exception:
                frame = None
            if frame is not None:
                frames.append(np.asarray(frame))
        action = select_action(obs)
        obs, r, terminated, truncated, _ = env.step(action)
        total += float(r)
        if terminated or truncated:
            break
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if frames:
        imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    return total
