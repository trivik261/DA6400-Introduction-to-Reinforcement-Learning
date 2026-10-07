"""dm_control `reacher_easy` exposed as a Gymnasium env with three reward
formulations.

The same env can compute the reward under **any** of the three formulations on
the fly via `compute_reward(reward_id, obs, action, next_obs, info)` — this is
the API used by the cross-evaluation in Section 2.3.

Reward formulations (matching the assignment screenshot):

* `Ra` (dense + control) :  +1 in target,  else  -‖x_goal - x_pos‖ - ‖action‖²
* `Rb` (sparse + 1)      :  +1 in target,  else  0   (this is the dm_control default)
* `Rc` (survival)        :  -1 every step until the agent reaches the target with
                            near-zero velocity, then the episode terminates.
                            On timeout (T = 1000) the arm is reset to a random
                            position (target stays) and a -20 penalty is applied.
                            The episode CONTINUES — it only terminates when the
                            goal is actually reached. Episode return & length
                            accumulate across timeouts.

The "in target" indicator comes directly from the underlying dm_control task
(`physics.finger_to_target_dist() < target_radius`).

Episode lengths follow the assignment: T = 1000 with truncation for Ra/Rb;
varying length for Rc (episodes span multiple timeout resets).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from dm_control import suite
except Exception as _e:  # noqa: BLE001
    suite = None
    _DMC_IMPORT_ERROR = _e
else:
    _DMC_IMPORT_ERROR = None


_TIMEOUT_PENALTY = -20.0


def _flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(v).ravel() for v in obs.values()]).astype(np.float32)


class ReacherEasyEnv(gym.Env):
    """`reacher_easy` from dm_control with selectable reward formulation."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    _RC_MAX_EPISODE_STEPS = 50_000

    def __init__(
        self,
        reward_id: str = "Rb",
        max_episode_steps: int = 1000,
        vel_tol: float = 0.5,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        if suite is None:
            raise ImportError(
                "dm_control is required for ReacherEasyEnv: " + repr(_DMC_IMPORT_ERROR)
            )
        assert reward_id in {"Ra", "Rb", "Rc"}
        self.reward_id = reward_id
        self.max_episode_steps = max_episode_steps
        self.vel_tol = vel_tol
        self.render_mode = render_mode
        self._dmc = suite.load(
            domain_name="reacher",
            task_name="easy",
            task_kwargs={"random": seed},
        )
        spec = self._dmc.action_spec()
        self.action_space = spaces.Box(
            low=spec.minimum.astype(np.float32),
            high=spec.maximum.astype(np.float32),
            dtype=np.float32,
        )
        ts = self._dmc.reset()
        flat = _flatten_obs(ts.observation)
        high = np.full(flat.shape, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self._t = 0
        self._t_since_reset = 0  # steps since last timeout/start within an Rc episode

    # ---------------------------- helpers ---------------------------------
    def _physics(self):
        return self._dmc.physics  # type: ignore[attr-defined]

    def _in_target(self) -> bool:
        phys = self._physics()
        try:
            return bool(phys.finger_to_target_dist() < phys.named.model.geom_size["target", 0])
        except Exception:
            return False

    def _finger_to_target(self) -> float:
        return float(self._physics().finger_to_target_dist())

    def _finger_speed(self) -> float:
        # dm_control reacher: velocity = self.physics.named.data.qvel.copy()
        try:
            return float(np.linalg.norm(self._physics().named.data.qvel))
        except Exception:
            return 0.0

    # --------------------- reward computation -----------------------------
    def _compute_reward_active(self, action: np.ndarray) -> tuple[float, bool, bool]:
        """Reward used during *training* — depends on `self.reward_id`."""
        in_t = self._in_target()
        terminated = False
        truncated = False
        if self.reward_id == "Ra":
            if in_t:
                r = 1.0
            else:
                r = -self._finger_to_target() - float(np.square(action).sum())
            truncated = self._t >= self.max_episode_steps
        elif self.reward_id == "Rb":
            r = 1.0 if in_t else 0.0
            truncated = self._t >= self.max_episode_steps
        else:  # Rc
            done_at_target = in_t and self._finger_speed() < self.vel_tol
            if done_at_target:
                r = -1.0
                terminated = True
            elif self._t >= self._RC_MAX_EPISODE_STEPS:
                r = float(_TIMEOUT_PENALTY)
                truncated = True
            elif self._t_since_reset >= self.max_episode_steps:
                r = float(_TIMEOUT_PENALTY)
                self._arm_only_reset()
                self._t_since_reset = 0
            else:
                r = -1.0
        return float(r), terminated, truncated

    def compute_reward(
        self,
        reward_id: str,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> float:
        """Recompute the reward under an arbitrary formulation.

        Used by the cross-evaluation: SAC-Ri evaluated under Rj.  We rely on
        cached "in_target" / "finger_to_target" / "finger_speed" stored in
        `info` because we cannot necessarily reconstruct those from `obs`.
        """
        info = info or {}
        in_t = bool(info.get("in_target", False))
        dist = float(info.get("finger_to_target", 0.0))
        if reward_id == "Ra":
            return 1.0 if in_t else -dist - float(np.square(action).sum())
        if reward_id == "Rb":
            return 1.0 if in_t else 0.0
        # Rc has *terminations* tied to the env, but as a per-step pseudo-reward
        # we treat each step as -1 unless the agent is in target with low speed.
        speed = float(info.get("finger_speed", 0.0))
        if in_t and speed < self.vel_tol:
            return -1.0
        return -1.0

    # ---------------------------- API -------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None and not getattr(self, "_seed_once", False):
            self._dmc = suite.load(
                domain_name="reacher",
                task_name="easy",
                task_kwargs={"random": seed},
            )
            self._seed_once = True
        ts = self._dmc.reset()
        self._t = 0
        self._t_since_reset = 0
        return _flatten_obs(ts.observation), {}

    def _arm_only_reset(self) -> None:
        """Randomise the two arm joint positions while keeping the target geom
        location intact.  Mirrors dm_control's `randomize_limited_and_rotational_joints`
        but only for joints whose name starts with `arm_`/`shoulder`/`wrist`."""
        physics = self._physics()
        with physics.reset_context():
            # Re-use dm_control's randomiser for joint positions.
            from dm_control.suite.utils import randomizers
            randomizers.randomize_limited_and_rotational_joints(
                physics, np.random.RandomState(np.random.randint(0, 2**31 - 1))
            )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ts = self._dmc.step(action)
        self._t += 1
        self._t_since_reset += 1
        reward, terminated, truncated = self._compute_reward_active(action)
        # After _compute_reward_active, the arm may have been reset (Rc timeout).
        # Re-read obs from the physics state so the agent sees the new arm position.
        obs = _flatten_obs(
            self._dmc.task.get_observation(self._physics())
            if self.reward_id == "Rc" else ts.observation
        )
        info = {
            "in_target": self._in_target(),
            "finger_to_target": self._finger_to_target(),
            "finger_speed": self._finger_speed(),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        try:
            return self._physics().render(camera_id=0, height=240, width=320)
        except Exception:
            return None

    def close(self):
        pass
