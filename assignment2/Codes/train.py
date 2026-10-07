"""
Training loop for DQN on MountainCar-v0 with comprehensive logging.
Supports both vanilla DQN and DQN with PER.
"""

import os
import json
import time
import numpy as np
import gymnasium as gym
import torch

from dqn_core import DQNAgent, QNetwork, ReplayBuffer
from per_buffer import PrioritizedReplayBuffer


def make_env(max_episode_steps=2000):
    """Create MountainCar-v0 with custom truncation length."""
    env = gym.make("MountainCar-v0", max_episode_steps=max_episode_steps)
    return env


def train_dqn(
    seed,
    max_timesteps=100000,
    max_episode_steps=2000,
    lr=1e-3,
    gamma=0.99,
    buffer_capacity=50000,
    batch_size=64,
    target_update_freq=500,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_steps=10000,
    hidden_dims=(128, 128),
    replay_factor=1,
    use_per=False,
    per_alpha=0.6,
    per_beta_start=0.4,
    per_beta_end=1.0,
    log_dir="results",
    tag="",
    eval_interval=1000,
    eval_episodes=10,
):
    """
    Train a DQN agent and log episode returns, losses, and eval performance.

    Returns a dict of logged data.
    """
    # Skip if this run's log already exists (resume support)
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, f"log_{tag}_seed{seed}.json")
    if os.path.exists(filepath):
        return None

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(max_episode_steps=max_episode_steps)
    eval_env = make_env(max_episode_steps=max_episode_steps)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        hidden_dims=hidden_dims,
        replay_factor=replay_factor,
    )

    if use_per:
        agent.buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_end=per_beta_end,
            beta_anneal_steps=max_timesteps,
        )

    log = {
        "seed": seed,
        "tag": tag,
        "hyperparams": {
            "lr": lr,
            "gamma": gamma,
            "buffer_capacity": buffer_capacity,
            "batch_size": batch_size,
            "target_update_freq": target_update_freq,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay_steps": epsilon_decay_steps,
            "hidden_dims": list(hidden_dims),
            "replay_factor": replay_factor,
            "max_episode_steps": max_episode_steps,
            "max_timesteps": max_timesteps,
            "use_per": use_per,
        },
        "episode_returns": [],
        "episode_lengths": [],
        "episode_end_timesteps": [],
        "eval_timesteps": [],
        "eval_mean_returns": [],
        "train_losses": [],
        "timestep_returns": [],
    }

    state, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_length = 0
    total_timesteps = 0
    episode_count = 0

    while total_timesteps < max_timesteps:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated

        if use_per:
            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.total_steps += 1
            losses = []
            for _ in range(agent.replay_factor):
                loss = _update_per(agent)
                if loss is not None:
                    losses.append(loss)
            if agent.total_steps % agent.target_update_freq == 0:
                agent.target_net.load_state_dict(agent.q_net.state_dict())
            avg_loss = np.mean(losses) if losses else None
        else:
            avg_loss = agent.step(state, action, reward, next_state, float(done))

        episode_return += reward
        episode_length += 1
        total_timesteps += 1

        if avg_loss is not None:
            log["train_losses"].append((total_timesteps, avg_loss))

        if terminated or truncated:
            log["episode_returns"].append(episode_return)
            log["episode_lengths"].append(episode_length)
            log["episode_end_timesteps"].append(total_timesteps)
            episode_count += 1

            state, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
        else:
            state = next_state

        if total_timesteps % eval_interval == 0:
            mean_ret = evaluate(agent, eval_env, eval_episodes, max_episode_steps)
            log["eval_timesteps"].append(total_timesteps)
            log["eval_mean_returns"].append(mean_ret)

    env.close()
    eval_env.close()

    os.makedirs(log_dir, exist_ok=True)
    filename = f"log_{tag}_seed{seed}.json"
    filepath = os.path.join(log_dir, filename)
    save_log(log, filepath)

    return log


def _update_per(agent):
    """Single PER update step: sample, compute loss with IS weights, update priorities."""
    if len(agent.buffer) < agent.batch_size:
        return None

    states, actions, rewards, next_states, dones, indices, weights = (
        agent.buffer.sample(agent.batch_size)
    )

    states_t = torch.FloatTensor(states).to(agent.device)
    actions_t = torch.LongTensor(actions).to(agent.device)
    rewards_t = torch.FloatTensor(rewards).to(agent.device)
    next_states_t = torch.FloatTensor(next_states).to(agent.device)
    dones_t = torch.FloatTensor(dones).to(agent.device)
    weights_t = torch.FloatTensor(weights).to(agent.device)

    q_values = agent.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = agent.target_net(next_states_t).max(dim=1)[0]
        targets = rewards_t + agent.gamma * next_q_values * (1.0 - dones_t)

    td_errors = (q_values - targets).detach().cpu().numpy()
    agent.buffer.update_priorities(indices, td_errors)

    loss = (weights_t * (q_values - targets) ** 2).mean()
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    return loss.item()


def evaluate(agent, env, num_episodes=10, max_steps=2000):
    """Evaluate the agent (greedy policy) for num_episodes."""
    returns = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            state = next_state
        returns.append(total_reward)
    return np.mean(returns)


def save_log(log, filepath):
    """Save log dict to JSON, converting numpy types."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(log, default=convert))
    with open(filepath, "w") as f:
        json.dump(serializable, f)


def load_log(filepath):
    """Load a log dict from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


def load_all_logs(log_dir, tag):
    """Load all seed logs matching a given tag from a directory."""
    logs = []
    for fname in sorted(os.listdir(log_dir)):
        if fname.startswith(f"log_{tag}_seed") and fname.endswith(".json"):
            logs.append(load_log(os.path.join(log_dir, fname)))
    return logs
