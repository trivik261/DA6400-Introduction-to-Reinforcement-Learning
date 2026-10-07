"""
Evaluation module for DQN agents.
Handles model evaluation and performance testing.
"""

import gymnasium as gym
import numpy as np
import torch
import argparse
import os
from typing import Dict, List, Tuple
from tqdm import tqdm

from config_loader import ExperimentConfig, ConfigLoader
from dqn_model import DQNAgent


class DQNEvaluator:
    """
    Evaluator for trained DQN agents.
    
    Handles:
    - Model loading
    - Policy evaluation
    - Performance metrics
    """
    
    def __init__(self, model_path: str, config_path: str = "config.json"):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.config = ConfigLoader.load_config(config_path)
        
        # Create environment
        self.env = gym.make(
            self.config.env_name,
            render_mode=None,
            max_episode_steps=self.config.max_episode_steps
        )
        
        # Initialize agent
        self.agent = DQNAgent(
            state_dim=2,
            action_dim=3,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            device=self.config.device
        )
        
        # Load trained model
        self.load_model()
    
    def load_model(self):
        """Load trained model from checkpoint."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.agent.load_checkpoint(self.model_path, include_optimizer=False)
        print(f"Loaded model from: {self.model_path}")
        print(f"Model epsilon: {self.agent.epsilon:.4f}")
        print(f"Training steps: {self.agent.training_steps}")
    
    def evaluate_episode(self, render: bool = False) -> Tuple[float, int, bool]:
        """
        Evaluate agent for one episode.
        
        Args:
            render: Whether to render the episode
            
        Returns:
            Tuple of (episode_return, episode_length, was_truncated)
        """
        state, _ = self.env.reset()
        episode_return = 0.0
        episode_length = 0
        
        while True:
            # Select action (no exploration during evaluation)
            action = self.agent.select_action(state, training=False)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Update metrics
            episode_return += reward
            episode_length += 1
            
            state = next_state
            
            # Check if episode ended
            if terminated or truncated:
                break
        
        return episode_return, episode_length, truncated
    
    def evaluate(self, num_episodes: int = 100, render: bool = False) -> Dict:
        """
        Evaluate agent over multiple episodes.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating agent for {num_episodes} episodes...")
        
        returns = []
        lengths = []
        truncations = []
        
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            episode_return, episode_length, truncated = self.evaluate_episode(render)
            returns.append(episode_return)
            lengths.append(episode_length)
            truncations.append(truncated)
        
        # Calculate statistics
        results = {
            'num_episodes': num_episodes,
            'returns': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'median': np.median(returns),
                'all': returns
            },
            'lengths': {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'all': lengths
            },
            'truncations': {
                'count': sum(truncations),
                'rate': sum(truncations) / num_episodes
            }
        }
        
        # Print results
        print(f"\nEvaluation Results ({num_episodes} episodes):")
        print(f"  Return: {results['returns']['mean']:.2f} ± {results['returns']['std']:.2f}")
        print(f"  Range: [{results['returns']['min']:.2f}, {results['returns']['max']:.2f}]")
        print(f"  Length: {results['lengths']['mean']:.1f} ± {results['lengths']['std']:.1f}")
        print(f"  Truncation rate: {results['truncations']['rate']:.2%}")
        
        return results
    
    def evaluate_with_epsilon_decay(self, num_episodes: int = 100) -> Dict:
        """
        Evaluate agent with epsilon decay (for ablation studies).
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating agent with epsilon decay for {num_episodes} episodes...")
        
        # Reset epsilon to starting value
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = self.config.eps_start
        
        returns = []
        epsilons = []
        
        for episode in tqdm(range(num_episodes), desc="Evaluating with decay"):
            epsilons.append(self.agent.epsilon)
            
            # Evaluate one episode
            episode_return, _, _ = self.evaluate_episode()
            returns.append(episode_return)
            
            # Decay epsilon
            self.agent.decay_epsilon()
        
        # Restore original epsilon
        self.agent.epsilon = original_epsilon
        
        results = {
            'num_episodes': num_episodes,
            'returns': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'all': returns
            },
            'epsilons': epsilons
        }
        
        print(f"\nEvaluation with Epsilon Decay Results:")
        print(f"  Return: {results['returns']['mean']:.2f} ± {results['returns']['std']:.2f}")
        print(f"  Final epsilon: {epsilons[-1]:.4f}")
        
        return results
    
    def compare_models(self, model_paths: List[str], model_names: List[str], 
                      num_episodes: int = 50) -> Dict:
        """
        Compare multiple trained models.
        
        Args:
            model_paths: List of paths to model checkpoints
            model_names: List of names for the models
            num_episodes: Number of evaluation episodes per model
            
        Returns:
            Dictionary with comparison results
        """
        if len(model_paths) != len(model_names):
            raise ValueError("model_paths and model_names must have same length")
        
        results = {}
        
        for model_path, model_name in zip(model_paths, model_names):
            print(f"\nEvaluating {model_name}...")
            
            # Load model
            self.agent.load_checkpoint(model_path, include_optimizer=False)
            
            # Evaluate
            model_results = self.evaluate(num_episodes, render=False)
            results[model_name] = model_results
        
        # Print comparison
        print(f"\nModel Comparison ({num_episodes} episodes each):")
        print(f"{'Model':<20} {'Return':<15} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 70)
        
        for model_name, model_results in results.items():
            mean_return = model_results['returns']['mean']
            std_return = model_results['returns']['std']
            min_return = model_results['returns']['min']
            max_return = model_results['returns']['max']
            
            print(f"{model_name:<20} {mean_return:<15.2f} {std_return:<10.2f} "
                  f"{min_return:<10.2f} {max_return:<10.2f}")
        
        return results
    
    def close(self):
        """Clean up resources."""
        self.env.close()


def main():
    """Command line interface for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate DQN agent')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes during evaluation')
    parser.add_argument('--compare', nargs='+', default=[],
                       help='Additional models to compare')
    parser.add_argument('--names', nargs='+', default=[],
                       help='Names for comparison models')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = DQNEvaluator(args.model, args.config)
    
    if args.compare:
        # Compare multiple models
        model_paths = [args.model] + args.compare
        model_names = ['baseline'] + args.names if args.names else [f'model_{i}' for i in range(len(model_paths))]
        
        results = evaluator.compare_models(model_paths, model_names, args.episodes)
    else:
        # Single model evaluation
        results = evaluator.evaluate(args.episodes, args.render)
    
    evaluator.close()


if __name__ == "__main__":
    main()
