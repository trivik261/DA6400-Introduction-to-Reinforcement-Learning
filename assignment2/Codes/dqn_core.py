"""
Core DQN components: Q-Network, Replay Buffer, and DQN Agent.
Vanilla DQN with epsilon-greedy, hard target updates, uniform replay.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class QNetwork(nn.Module):
    """Feed-forward Q-network with Kaiming (He) initialization for ReLU layers."""

    def __init__(self, state_dim, action_dim, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size uniform replay buffer."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Vanilla DQN agent with epsilon-greedy, hard target updates, uniform replay."""

    def __init__(
        self,
        state_dim,
        action_dim,
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
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.replay_factor = replay_factor

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.q_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.total_steps = 0

    def get_epsilon(self):
        fraction = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.get_epsilon():
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(state_t)
        return q_vals.argmax(dim=1).item()

    def update(self):
        """Perform one gradient step on a mini-batch from the replay buffer."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1)[0]
            targets = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def step(self, state, action, reward, next_state, done):
        """Store transition and perform replay_factor updates."""
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

        losses = []
        for _ in range(self.replay_factor):
            loss = self.update()
            if loss is not None:
                losses.append(loss)

        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return np.mean(losses) if losses else None

    def save(self, path):
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]
