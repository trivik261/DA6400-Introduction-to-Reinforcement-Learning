"""
Vanilla DQN implementation for MountainCar-v0.
Implements standard DQN with hard target network updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class DQNetwork(nn.Module):
    """
    Q-network for DQN.
    
    Architecture:
    - Input: 2D state (position, velocity)
    - Hidden layers: 2 layers with 64 units each
    - Output: Q-values for 3 discrete actions
    """
    
    def __init__(self, state_dim: int = 2, action_dim: int = 3, hidden_dim: int = 64):
        """
        Initialize Q-network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of hidden units
        """
        super(DQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Kaiming initialization for ReLU layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            x: State tensor
            
        Returns:
            Q-values for each action
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    """
    Vanilla DQN agent with hard target network updates.
    
    Implements standard DQN with:
    - ε-greedy exploration with decay
    - Hard target network updates
    - Huber loss
    - Adam optimizer
    """
    
    def __init__(self, state_dim: int = 2, action_dim: int = 3, 
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 device: str = 'cpu'):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor (fixed at 0.99 for assignment)
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        
        # Main and target networks
        self.q_network = DQNetwork(state_dim, action_dim).to(device)
        self.target_network = DQNetwork(state_dim, action_dim).to(device)
        
        # Initialize target network with same weights
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        
        # Training step counter
        self.training_steps = 0
    
    def set_exploration(self, eps_start: float, eps_end: float, eps_decay: float):
        """
        Set exploration parameters.
        
        Args:
            eps_start: Starting epsilon
            eps_end: Ending epsilon
            eps_decay: Decay rate
        """
        self.epsilon_start = eps_start
        self.epsilon_end = eps_end
        self.epsilon_decay = eps_decay
        self.epsilon = eps_start
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def decay_epsilon(self):
        """Decay exploration epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_step(self, states: torch.Tensor, actions: torch.Tensor, 
                   rewards: torch.Tensor, next_states: torch.Tensor, 
                   dones: torch.Tensor) -> float:
        """
        Perform one training step.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of termination flags
            
        Returns:
            Loss value
        """
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_steps += 1
        
        return loss.item()
    
    def save_checkpoint(self, filepath: str, include_optimizer: bool = True):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            include_optimizer: Whether to include optimizer state
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str, include_optimizer: bool = True):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint
            include_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.training_steps = checkpoint.get('training_steps', 0)
        
        if include_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for a state.
        
        Args:
            state: State observation
            
        Returns:
            Q-values for all actions
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().squeeze()
    
    def get_network_info(self) -> dict:
        """
        Get network information.
        
        Returns:
            dict: Network parameters and statistics
        """
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'device': self.device
        }


if __name__ == "__main__":
    # Example usage and testing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create agent
    agent = DQNAgent(device=device)
    
    # Test network
    state = np.array([-0.5, 0.0])  # Example MountainCar state
    action = agent.select_action(state)
    q_values = agent.get_q_values(state)
    
    print(f"Selected action: {action}")
    print(f"Q-values: {q_values}")
    print(f"Network info: {agent.get_network_info()}")
    
    # Test training step with dummy data
    batch_size = 32
    states = torch.randn(batch_size, 2).to(device)
    actions = torch.randint(0, 3, (batch_size,)).to(device)
    rewards = torch.randn(batch_size).to(device)
    next_states = torch.randn(batch_size, 2).to(device)
    dones = torch.randint(0, 2, (batch_size,)).bool().to(device)
    
    loss = agent.train_step(states, actions, rewards, next_states, dones)
    print(f"Training loss: {loss:.4f}")
    
    # Test target network update
    agent.update_target_network()
    print("Target network updated")
    
    # Test checkpoint saving/loading
    agent.save_checkpoint("test_checkpoint.pth")
    print("Checkpoint saved")
    
    new_agent = DQNAgent(device=device)
    new_agent.load_checkpoint("test_checkpoint.pth")
    print("Checkpoint loaded")
