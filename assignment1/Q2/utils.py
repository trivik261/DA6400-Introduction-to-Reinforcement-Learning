"""
Shared utilities for TD-based Control in Acrobot-v1.

Environment reference:
  https://gymnasium.farama.org/environments/classic_control/acrobot/

  Observation : 6-dim continuous  [cos(θ1), sin(θ1), cos(θ2), sin(θ2), dθ1, dθ2]
  Actions     : Discrete(3)      torque ∈ {-1, 0, +1}
  Reward      : -1 per step (0 on termination when goal height reached)
  Termination : -cos(θ1) - cos(θ1 + θ2) > 1.0
  Truncation  : episode length > 500

Provides:
  - State discretisation (10 bins per dimension → 10^6 states)
  - Epsilon-greedy action selection
  - SARSA (on-policy TD control)
  - Q-Learning (off-policy TD control)
  - Greedy policy evaluation
  - Smoothing helper for plots
"""

import numpy as np
import gymnasium as gym
from collections import defaultdict

# ─── Environment constants (from assignment spec) ────────────────────────
NUM_BINS = 10        # given in question
GAMMA = 0.99         # given in question

# Observation bounds taken directly from Gymnasium docs:
#   Box([-1, -1, -1, -1, -12.566371, -28.274334],
#       [ 1,  1,  1,  1,  12.566371,  28.274334], (6,), float32)
_env_tmp = gym.make("Acrobot-v1")
OBS_LOW = _env_tmp.observation_space.low.astype(np.float64)
OBS_HIGH = _env_tmp.observation_space.high.astype(np.float64)
N_ACTIONS = _env_tmp.action_space.n      # 3
OBS_DIM = OBS_LOW.shape[0]               # 6
_env_tmp.close()

BIN_EDGES = [
    np.linspace(OBS_LOW[i], OBS_HIGH[i], NUM_BINS + 1)[1:-1]
    for i in range(OBS_DIM)
]


def make_env():
    return gym.make("Acrobot-v1")


# ─── Discretisation ──────────────────────────────────────────────────────
def discretise(obs):
    """Map a continuous 6-dim observation to a tuple of bin indices (one per dim)."""
    return tuple(int(np.digitize(obs[i], BIN_EDGES[i])) for i in range(OBS_DIM))


# ─── Epsilon-greedy policy ───────────────────────────────────────────────
def epsilon_greedy_action(Q, state, n_actions, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    return int(np.argmax(Q[state]))


# ─── SARSA ────────────────────────────────────────────────────────────────
def sarsa(env, num_episodes, alpha, gamma, eps_start, eps_min, eps_decay,
          seed, verbose=False):
    """On-policy TD control (SARSA) with epsilon-greedy exploration."""
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions))
    epsilon = eps_start
    rewards = []
    timesteps_per_ep = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1e6)))
        state = discretise(obs)
        action = epsilon_greedy_action(Q, state, n_actions, epsilon, rng)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretise(next_obs)
            next_action = epsilon_greedy_action(Q, next_state, n_actions, epsilon, rng)

            td_target = reward + gamma * Q[next_state][next_action] * (1 - int(terminated))
            Q[state][action] += alpha * (td_target - Q[state][action])

            state, action = next_state, next_action
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        timesteps_per_ep.append(steps)
        epsilon = max(eps_min, epsilon * eps_decay)

        if verbose and (ep + 1) % 500 == 0:
            avg = np.mean(rewards[-100:])
            print(f"  [SARSA]  Ep {ep+1:>5d} | Avg(100): {avg:>8.2f} | ε={epsilon:.4f}")

    return Q, rewards, timesteps_per_ep


# ─── Q-Learning ───────────────────────────────────────────────────────────
def q_learning(env, num_episodes, alpha, gamma, eps_start, eps_min, eps_decay,
               seed, verbose=False):
    """Off-policy TD control (Q-Learning) with epsilon-greedy exploration."""
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions))
    epsilon = eps_start
    rewards = []
    timesteps_per_ep = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1e6)))
        state = discretise(obs)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = epsilon_greedy_action(Q, state, n_actions, epsilon, rng)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretise(next_obs)

            td_target = reward + gamma * np.max(Q[next_state]) * (1 - int(terminated))
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        timesteps_per_ep.append(steps)
        epsilon = max(eps_min, epsilon * eps_decay)

        if verbose and (ep + 1) % 500 == 0:
            avg = np.mean(rewards[-100:])
            print(f"  [Q-Lrn] Ep {ep+1:>5d} | Avg(100): {avg:>8.2f} | ε={epsilon:.4f}")

    return Q, rewards, timesteps_per_ep


# ─── Evaluation (greedy, ε = 0) ──────────────────────────────────────────
def evaluate_policy(env, Q, num_episodes=50, seed=9999):
    """Roll out the greedy policy (no exploration) and return per-episode rewards."""
    rng = np.random.default_rng(seed)
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1e6)))
        state = discretise(obs)
        done, ep_reward = False, 0.0
        while not done:
            action = int(np.argmax(Q[state]))
            obs, reward, terminated, truncated, _ = env.step(action)
            state = discretise(obs)
            done = terminated or truncated
            ep_reward += reward
        rewards.append(ep_reward)
    return rewards


# ─── Plotting helper ─────────────────────────────────────────────────────
def smooth(data, window=100):
    """Running mean for smoother learning curves."""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')
