"""
SAC Agent implementation for LunarLander-v3 (Continuous Action Space)
Implements Soft Actor-Critic with automated temperature tuning.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
class ReplayBuffer:
    """Experience replay buffer for SAC."""
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_states[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device)
        )
class GaussianPolicy(nn.Module):
    """Gaussian policy network for continuous actions."""
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean
class QNetwork(nn.Module):
    """Q-network (critic) for SAC."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
class SACAgent:
    """Soft Actor-Critic agent with automated temperature tuning."""
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=1000000,
        batch_size=256,
        auto_entropy_tuning=True,
        manual_alpha=0.2,
        device='cuda'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.auto_entropy_tuning = auto_entropy_tuning
        self.target_entropy = -action_dim
        if auto_entropy_tuning:
            self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor([manual_alpha], device=device)
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim, device)
        self.total_steps = 0
    def select_action(self, state, evaluate=False):
        """Select action from policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    def update(self):
        """Update SAC agent."""
        if self.replay_buffer.size < self.batch_size:
            return {}
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            q1_next_target = self.q1_target(next_state, next_action)
            q2_next_target = self.q2_target(next_state, next_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * min_q_next_target
        q1_pred = self.q1(state, action)
        q2_pred = self.q2(state, action)
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        new_action, log_prob, _ = self.policy.sample(state)
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_prob - min_q_new).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        alpha_loss = None
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        self.total_steps += 1
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha.item(),
            'alpha_loss': alpha_loss.item() if alpha_loss is not None else 0.0
        }
    def _soft_update(self, source, target):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    def save(self, filepath):
        """Save agent."""
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
        }, filepath)
    def load(self, filepath):
        """Load agent."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp()
