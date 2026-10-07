"""
Training module for DQN on MountainCar-v0.
Handles episode execution, training loops, and checkpointing.
"""

import gymnasium as gym
import numpy as np
import torch
import time
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging

from config_loader import ExperimentConfig
from dqn_model import DQNAgent
from replay_buffer import ReplayBuffer, BatchProcessor


class DQNTrainer:
    """
    Trainer for DQN agent on MountainCar-v0.
    
    Handles:
    - Environment interaction
    - Training loop
    - Checkpointing
    - Logging
    """
    
    def __init__(self, config: ExperimentConfig, seed: int, experiment_dir: str):
        """
        Initialize trainer.
        
        Args:
            config: Experiment configuration
            seed: Random seed for reproducibility
            experiment_dir: Directory to save results
        """
        self.config = config
        self.seed = seed
        self.experiment_dir = experiment_dir
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Create environment
        self.env = gym.make(
            config.env_name,
            render_mode=None,
            max_episode_steps=config.max_episode_steps
        )
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        
        # Initialize agent and buffer
        self.agent = DQNAgent(
            state_dim=2,
            action_dim=3,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            device=config.device
        )
        
        self.agent.set_exploration(
            config.eps_start,
            config.eps_end,
            config.eps_decay
        )
        
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        
        # Training metrics
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.losses: List[float] = []
        self.truncation_count = 0
        self.replay_factor = config.replay_factor
        
        # Setup logging
        self.setup_logging()
        
        # Create directories
        self.create_directories()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.experiment_dir, 'logs', f'seed_{self.seed}.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Get existing logger from manager instead of creating new one
        self.logger = logging.getLogger(f'trainer_seed_{self.seed}')
        
        # Create file handler for this seed
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Configure logger
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Set formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
    
    def create_directories(self):
        """Create necessary directories for this seed."""
        seed_dir = os.path.join(self.experiment_dir, 'models', 'checkpoints', f'seed_{self.seed}')
        os.makedirs(seed_dir, exist_ok=True)
    
    def train_episode(self, episode: int) -> Tuple[float, int, bool]:
        """
        Train for one episode.
        
        Args:
            episode: Current episode number
            
        Returns:
            Tuple of (episode_return, episode_length, was_truncated)
        """
        state, _ = self.env.reset(seed=self.seed + episode)
        episode_return = 0.0
        episode_length = 0
        
        while True:
            # Select action
            action = self.agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Store transition
            done = terminated or truncated  # CRITICAL: Use both for correct DQN targets
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update metrics
            episode_return += reward
            episode_length += 1
            
            state = next_state
            
            # Train if buffer is ready
            if (self.replay_buffer.is_ready(self.config.batch_size) and 
                episode_length % self.config.update_every == 0):
                
                # Implement replay factor: multiple updates per timestep
                for _ in range(self.replay_factor):
                    batch = self.replay_buffer.sample(self.config.batch_size)
                    states, actions, rewards, next_states, dones = BatchProcessor.process_batch(
                        batch, self.config.device
                    )
                    
                    loss = self.agent.train_step(states, actions, rewards, next_states, dones)
                    self.losses.append(loss)
            
            # Check if episode ended
            if terminated or truncated:
                if truncated:
                    self.truncation_count += 1
                break
        
        # Decay exploration
        self.agent.decay_epsilon()
        
        # Update target network
        if episode % self.config.target_update_freq == 0:
            self.agent.update_target_network()
        
        return episode_return, episode_length, truncated
    
    def train(self) -> Dict:
        """
        Run complete training for all episodes.
        
        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Starting training for seed {self.seed}")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"Episodes: {self.config.num_episodes}")
        
        start_time = time.time()
        
        # Training loop
        for episode in tqdm(range(self.config.num_episodes), 
                           desc=f"Seed {self.seed}"):
            
            episode_return, episode_length, truncated = self.train_episode(episode)
            self.episode_returns.append(episode_return)
            self.episode_lengths.append(episode_length)
            
            # Log progress
            if episode % 50 == 0 or episode == self.config.num_episodes - 1:
                avg_return = np.mean(self.episode_returns[-50:]) if len(self.episode_returns) >= 50 else np.mean(self.episode_returns)
                avg_loss = np.mean(self.losses[-50:]) if len(self.losses) >= 50 else 0.0
                
                self.logger.info(
                    f"Episode {episode:4d} | "
                    f"Return: {episode_return:7.2f} | "
                    f"Avg50: {avg_return:7.2f} | "
                    f"Epsilon: {self.agent.epsilon:.3f} | "
                    f"Loss: {avg_loss:.4f}"
                )
            
            # Save checkpoint
            if (self.config.save_checkpoints and 
                episode % 50 == 0 and episode > 0):
                self.save_checkpoint(episode)
        
        # Save final results
        total_time = time.time() - start_time
        
        results = {
            'seed': self.seed,
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'truncation_count': self.truncation_count,
            'total_episodes': self.config.num_episodes,
            'truncation_rate': self.truncation_count / self.config.num_episodes,
            'final_return': self.episode_returns[-1] if self.episode_returns else 0.0,
            'best_return': max(self.episode_returns) if self.episode_returns else 0.0,
            'mean_return': np.mean(self.episode_returns) if self.episode_returns else 0.0,
            'std_return': np.std(self.episode_returns) if self.episode_returns else 0.0,
            'training_time': total_time,
            'final_epsilon': self.agent.epsilon
        }
        
        # Save final model and results
        self.save_final_results(results)
        
        self.logger.info(f"Training completed for seed {self.seed}")
        self.logger.info(f"Final return: {results['final_return']:.2f}")
        self.logger.info(f"Best return: {results['best_return']:.2f}")
        self.logger.info(f"Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
        self.logger.info(f"Truncation rate: {results['truncation_rate']:.2%}")
        self.logger.info(f"Training time: {total_time:.2f}s")
        
        return results
    
    def save_checkpoint(self, episode: int):
        """
        Save training checkpoint.
        
        Args:
            episode: Current episode number
        """
        checkpoint_dir = os.path.join(self.experiment_dir, 'models', 'checkpoints', f'seed_{self.seed}')
        checkpoint_path = os.path.join(checkpoint_dir, f'model_ep_{episode}.pth')
        returns_path = os.path.join(checkpoint_dir, f'returns_ep_{episode}.npy')
        
        # Save model
        self.agent.save_checkpoint(
            checkpoint_path, 
            include_optimizer=not self.config.fast_mode
        )
        
        # Save returns so far
        np.save(returns_path, np.array(self.episode_returns))
    
    def save_final_results(self, results: Dict):
        """
        Save final results and models.
        
        Args:
            results: Training results dictionary
        """
        # Save episode returns
        data_dir = os.path.join(self.experiment_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        returns_path = os.path.join(data_dir, f'returns_seed_{self.seed}.npy')
        np.save(returns_path, np.array(self.episode_returns))
        
        # Save seed summary
        import json
        summary_path = os.path.join(data_dir, f'seed_{self.seed}_summary.json')
        
        # Convert numpy types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        with open(summary_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save final model
        models_dir = os.path.join(self.experiment_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        final_model_path = os.path.join(models_dir, f'final_model_seed_{self.seed}.pth')
        self.agent.save_checkpoint(final_model_path, include_optimizer=not self.config.fast_mode)
    
    def close(self):
        """Clean up resources."""
        self.env.close()


if __name__ == "__main__":
    # Example usage
    from config_loader import ConfigLoader
    
    # Load config
    config = ConfigLoader.load_config("config.json")
    
    # Create temporary experiment directory
    experiment_dir = "test_experiment"
    
    # Train single seed
    trainer = DQNTrainer(config, seed=0, experiment_dir=experiment_dir)
    results = trainer.train()
    trainer.close()
    
    print("Training completed!")
    print(f"Results: {results}")
