"""
Uniform replay buffer for DQN training.
Implements simple uniform sampling without prioritization.
"""

import random
from collections import deque
from typing import List, Tuple, Any
import numpy as np


class Transition:
    """A single transition in the environment."""
    
    def __init__(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
    
    def to_tuple(self) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        """Convert transition to tuple format."""
        return (self.state, self.action, self.reward, self.next_state, self.done)


class ReplayBuffer:
    """
    Uniform replay buffer for DQN.
    
    Stores transitions and samples them uniformly for training.
    Uses deque for efficient O(1) append and O(n) sampling.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
        """
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """
        Sample a batch of transitions uniformly.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of transition tuples
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough transitions in buffer. Have {len(self.buffer)}, need {batch_size}")
        
        transitions = random.sample(self.buffer, batch_size)
        return [t.to_tuple() for t in transitions]
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough transitions to sample.
        
        Args:
            batch_size: Required batch size
            
        Returns:
            bool: True if ready for sampling
        """
        return len(self.buffer) >= batch_size
    
    def clear(self) -> None:
        """Clear all transitions from buffer."""
        self.buffer.clear()
        self.position = 0
    
    def get_stats(self) -> dict:
        """
        Get buffer statistics.
        
        Returns:
            dict: Buffer statistics
        """
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity
        }


class BatchProcessor:
    """Utility class for processing sampled batches."""
    
    @staticmethod
    def process_batch(batch: List[Tuple[np.ndarray, int, float, np.ndarray, bool]], 
                     device: str = 'cpu') -> Tuple[Any, Any, Any, Any, Any]:
        """
        Process a batch of transitions for training.
        
        Args:
            batch: List of transition tuples
            device: Device to move tensors to
            
        Returns:
            Processed batch as tensors
        """
        # Unzip batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays first
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)
        
        # Convert to tensors if torch is available
        try:
            import torch
            states = torch.from_numpy(states).to(device)
            actions = torch.from_numpy(actions).to(device)
            rewards = torch.from_numpy(rewards).to(device)
            next_states = torch.from_numpy(next_states).to(device)
            dones = torch.from_numpy(dones).to(device)
        except ImportError:
            # Return numpy arrays if torch not available
            pass
        
        return states, actions, rewards, next_states, dones


if __name__ == "__main__":
    # Example usage and testing
    buffer = ReplayBuffer(capacity=1000)
    
    # Add some dummy transitions
    for i in range(100):
        state = np.random.rand(2)  # MountainCar has 2D state
        action = np.random.randint(0, 3)  # 3 discrete actions
        reward = np.random.rand()
        next_state = np.random.rand(2)
        done = np.random.rand() > 0.9
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Buffer stats: {buffer.get_stats()}")
    
    # Sample a batch
    if buffer.is_ready(batch_size=32):
        batch = buffer.sample(batch_size=32)
        states, actions, rewards, next_states, dones = BatchProcessor.process_batch(batch)
        print(f"Sampled batch shapes:")
        print(f"  States: {states.shape}")
        print(f"  Actions: {actions.shape}")
        print(f"  Rewards: {rewards.shape}")
        print(f"  Next states: {next_states.shape}")
        print(f"  Dones: {dones.shape}")
    else:
        print("Buffer not ready for sampling")
