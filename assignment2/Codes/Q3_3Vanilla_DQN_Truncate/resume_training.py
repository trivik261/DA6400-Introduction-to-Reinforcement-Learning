#!/usr/bin/env python3
"""
Resume training from saved checkpoints.
Continue training from episode 1150 to 1500 for all seeds.
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
from collections import namedtuple, deque
import time
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_loader import ConfigLoader
from dqn_model import DQNAgent
from trainer import DQNTrainer

def resume_training(checkpoint_dir, target_episodes=1500):
    """Resume training from existing checkpoints."""
    
    # Load config
    config = ConfigLoader.load_config('config.json')
    
    print(f"Resuming training from {checkpoint_dir}")
    print(f"Target episodes: {target_episodes}")
    print(f"Number of seeds: {config.num_seeds}")
    
    all_results = {}
    
    for seed in range(config.num_seeds):
        print(f"\n{'='*60}")
        print(f"Resuming training for seed {seed}")
        print(f"{'='*60}")
        
        # Find latest checkpoint
        seed_checkpoint_dir = os.path.join(checkpoint_dir, 'models', 'checkpoints', f'seed_{seed}')
        
        if not os.path.exists(seed_checkpoint_dir):
            print(f"Warning: No checkpoint directory for seed {seed}, starting fresh")
            continue
            
        # Find latest checkpoint
        checkpoint_files = [f for f in os.listdir(seed_checkpoint_dir) if f.startswith('model_ep_') and f.endswith('.pth')]
        if not checkpoint_files:
            print(f"Warning: No checkpoints found for seed {seed}")
            continue
            
        # Get latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        checkpoint_path = os.path.join(seed_checkpoint_dir, latest_checkpoint)
        
        # Extract current episode from filename
        current_episode = int(latest_checkpoint.split('_')[2].split('.')[0])
        
        if current_episode >= target_episodes:
            print(f"Seed {seed} already trained to episode {current_episode}")
            continue
            
        print(f"Resuming from episode {current_episode} to {target_episodes}")
        
        # Create trainer for this seed
        trainer = DQNTrainer(config, seed, checkpoint_dir)
        
        # Load checkpoint
        trainer.agent.load_checkpoint(checkpoint_path, include_optimizer=True)
        
        # Load returns history
        returns_path = os.path.join(seed_checkpoint_dir, f'returns_ep_{current_episode}.npy')
        if os.path.exists(returns_path):
            returns_history = list(np.load(returns_path))
        else:
            returns_history = []
        
        print(f"Loaded checkpoint: {latest_checkpoint}")
        print(f"Returns history length: {len(returns_history)}")
        print(f"Current epsilon: {trainer.agent.epsilon:.4f}")
        
        # Continue training
        start_time = time.time()
        
        for episode in range(current_episode + 1, target_episodes + 1):
            return_val, episode_length, truncated = trainer.train_episode(episode)
            returns_history.append(return_val)
            
            # Log progress
            if episode % 50 == 0:
                avg_return = np.mean(returns_history[-50:]) if len(returns_history) >= 50 else np.mean(returns_history)
                print(f"Episode {episode:4d} | Return: {return_val:7.1f} | Avg50: {avg_return:7.1f} | "
                      f"Epsilon: {trainer.agent.epsilon:.3f} | Length: {episode_length}")
            
            # Save checkpoint
            if episode % 50 == 0:
                trainer.save_checkpoint(episode)
                np.save(os.path.join(seed_checkpoint_dir, f'returns_ep_{episode}.npy'), returns_history)
        
        # Calculate results
        end_time = time.time()
        training_time = end_time - start_time
        
        final_return = returns_history[-1]
        best_return = min(returns_history)
        mean_return = np.mean(returns_history)
        std_return = np.std(returns_history)
        truncation_rate = sum(1 for r in returns_history if r <= -1999) / len(returns_history)
        
        # Save final results
        results = {
            'final_return': final_return,
            'best_return': best_return,
            'mean_return': mean_return,
            'std_return': std_return,
            'truncation_rate': truncation_rate,
            'training_time': training_time
        }
        trainer.save_final_results(results)
        
        print(f"\nSeed {seed} completed:")
        print(f"  Final return: {final_return:.1f}")
        print(f"  Best return: {best_return:.1f}")
        print(f"  Mean return: {mean_return:.1f} ± {std_return:.1f}")
        print(f"  Truncation rate: {truncation_rate:.1%}")
        print(f"  Training time: {training_time:.1f}s")
        
        all_results[seed] = {
            'final_return': final_return,
            'best_return': best_return,
            'mean_return': mean_return,
            'std_return': std_return,
            'truncation_rate': truncation_rate,
            'training_time': training_time,
            'returns_history': returns_history
        }
        
        trainer.close()
    
    # Save aggregated results
    if all_results:
        save_results(all_results, checkpoint_dir, target_episodes)
    
    return all_results

def save_results(all_results, experiment_dir, target_episodes):
    """Save aggregated results."""
    
    # Aggregate statistics
    final_returns = [r['final_return'] for r in all_results.values()]
    best_returns = [r['best_return'] for r in all_results.values()]
    mean_returns = [r['mean_return'] for r in all_results.values()]
    truncation_rates = [r['truncation_rate'] for r in all_results.values()]
    training_times = [r['training_time'] for r in all_results.values()]
    
    aggregated = {
        'experiment_info': {
            'experiment_type': 'Q3.2_resumed',
            'num_seeds': len(all_results),
            'target_episodes': target_episodes,
            'total_runtime': sum(training_times),
            'device': 'cpu'
        },
        'final_returns': {
            'mean': np.mean(final_returns),
            'std': np.std(final_returns),
            'min': np.min(final_returns),
            'max': np.max(final_returns),
            'all_values': final_returns
        },
        'best_returns': {
            'mean': np.mean(best_returns),
            'std': np.std(best_returns),
            'min': np.min(best_returns),
            'max': np.max(best_returns),
            'all_values': best_returns
        },
        'training_times': {
            'mean': np.mean(training_times),
            'std': np.std(training_times),
            'total': sum(training_times)
        },
        'truncation_rates': {
            'mean': np.mean(truncation_rates),
            'std': np.std(truncation_rates)
        }
    }
    
    # Save results
    results_path = os.path.join(experiment_dir, 'resumed_results.json')
    with open(results_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\n{'='*60}")
    print("RESUMED TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Final return: {aggregated['final_returns']['mean']:.1f} ± {aggregated['final_returns']['std']:.1f}")
    print(f"Best return: {aggregated['best_returns']['mean']:.1f} ± {aggregated['best_returns']['std']:.1f}")
    print(f"Truncation rate: {aggregated['truncation_rates']['mean']:.1%}")
    print(f"Total runtime: {aggregated['training_times']['total']:.1f}s")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    # Resume from the latest experiment
    checkpoint_dir = "d:/College/RL/Assignment2/RL_MountainCar2/experiments/Experiment_Q3.2_20260407_183041"
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    resume_training(checkpoint_dir, target_episodes=1500)
