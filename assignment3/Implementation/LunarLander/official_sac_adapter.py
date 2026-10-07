"""
Adapter to use official pytorch_sac with Gymnasium Pendulum-v1.
This file bridges the gap between the official implementation and our assignment needs.
IMPORTANT: Requires pytorch_sac repo to be cloned at ../pytorch_sac/
Clone with: git clone https://github.com/denisyarats/pytorch_sac.git
"""
import sys
import os
pytorch_sac_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pytorch_sac'))
if not os.path.exists(pytorch_sac_path):
    raise ImportError(
        f"pytorch_sac repo not found at {pytorch_sac_path}\n"
        "Please clone it with: cd D:\\RL\\A3\\Implementation && git clone https://github.com/denisyarats/pytorch_sac.git"
    )
sys.path.insert(0, pytorch_sac_path)
from agent.actor import DiagGaussianActor  
from agent.critic import DoubleQCritic
from replay_buffer import ReplayBuffer
import utils
class SACAgentForAssignment:
    """
    Wrapper that provides a simple interface to the official SAC implementation.
    Handles initialization without Hydra configs.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        action_scale=2.0,
        action_bias=0.0,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        hidden_dim=256,
        buffer_size=int(1e6),
        batch_size=256,
        auto_entropy_tuning=True,
        alpha=0.2,
        device='cpu'
    ):
        import torch
        self.device = torch.device(device)
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.batch_size = batch_size
        self.critic = DoubleQCritic(
            obs_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=2
        ).to(self.device)
        self.critic_target = DoubleQCritic(
            obs_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=2
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = DiagGaussianActor(
            obs_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=2,
            log_std_bounds=[-5, 2]
        ).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr, betas=[0.9, 0.999]
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr, betas=[0.9, 0.999]
        )
        self.auto_entropy_tuning = auto_entropy_tuning
        self.target_entropy = -action_dim * 0.5
        if auto_entropy_tuning:
            self.log_alpha = torch.tensor([-1.6], requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr * 0.1)
        else:
            self.log_alpha = torch.tensor([alpha]).log().to(self.device)
        self.gamma = gamma
        self.tau = tau
        self.action_range = [-1.0, 1.0]  # Official SAC expects [-1, 1]
        self._replay_buffer = ReplayBuffer(
            obs_shape=(state_dim,),
            action_shape=(action_dim,),
            capacity=buffer_size,
            device=self.device
        )
        self.update_step = 0
    @property
    def replay_buffer(self):
        """Wrapper to provide compatible interface."""
        class ReplayBufferWrapper:
            def __init__(self, buffer):
                self._buffer = buffer
            def add(self, obs, action, reward, next_obs, done):
                """Add transition with done_no_max handling."""
                self._buffer.add(obs, action, reward, next_obs, done, done)
            @property
            def size(self):
                return len(self._buffer)
            @property
            def idx(self):
                return self._buffer.idx
            def sample(self, batch_size):
                return self._buffer.sample(batch_size)
        return ReplayBufferWrapper(self._replay_buffer)
    @property
    def alpha(self):
        return self.log_alpha.exp()
    def select_action(self, state, evaluate=False):
        """Select action, handling scaling for Pendulum."""
        import torch
        import numpy as np
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        state = state.unsqueeze(0) if state.dim() == 1 else state
        with torch.no_grad():
            dist = self.actor(state)
            action = dist.mean if evaluate else dist.sample()
            action = action.clamp(*self.action_range)
        action = action.cpu().numpy()[0]
        action = action * self.action_scale + self.action_bias
        return action
    def update_parameters(self):
        """Update SAC parameters using official implementation logic."""
        import torch
        import torch.nn.functional as F
        if self._replay_buffer.idx < self.batch_size:
            return None, None, None, None
        obs, action, reward, next_obs, not_done, not_done_no_max = self._replay_buffer.sample(self.batch_size)
        action = (action - self.action_bias) / self.action_scale
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.sample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done_no_max * self.gamma * target_V)
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        dist = self.actor(obs)
        action_new = dist.rsample()
        log_prob = dist.log_prob(action_new).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action_new)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        alpha_loss = None
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        utils.soft_update_params(self.critic, self.critic_target, self.tau)
        self.update_step += 1
        return (
            critic_loss.item(),
            critic_loss.item(),
            actor_loss.item(),
            alpha_loss.item() if alpha_loss else None
        )
    def save(self, filename):
        """Save model."""
        import torch
        torch.save({
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor': self.actor.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
        }, filename)
    def load(self, filename):
        """Load model."""
        import torch
        checkpoint = torch.load(filename, map_location=self.device)
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        if 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            if self.auto_entropy_tuning and checkpoint['alpha_optimizer']:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
SACAgent = SACAgentForAssignment
