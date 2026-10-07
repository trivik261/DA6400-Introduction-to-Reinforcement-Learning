"""
Plotting utilities for LunarLander SAC experiments.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
def plot_learning_curve(results_path, save_path=None):
    """Plot learning curve from results.json."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    timesteps = results['eval_timesteps']
    rewards = results['eval_rewards']
    alphas = results['alpha_values']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(timesteps, rewards, linewidth=2)
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Evaluation Reward')
    ax1.set_title('SAC Learning Curve - LunarLander-v3')
    ax1.grid(True, alpha=0.3)
    ax2.plot(timesteps, alphas, linewidth=2, color='orange')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Alpha (Temperature)')
    ax2.set_title('Temperature (Alpha) Evolution')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()
def plot_q3_comparison(auto_results_path, manual_results_path, save_path=None):
    """
    Plot Q3 comparison: auto tuning vs manual alpha with reward switching.
    """
    with open(auto_results_path, 'r') as f:
        auto_results = json.load(f)
    with open(manual_results_path, 'r') as f:
        manual_results = json.load(f)
    switch_timestep = auto_results['switch_timestep']
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    ax1 = axes[0]
    ax1.plot(auto_results['eval_timesteps'], auto_results['eval_rewards'], 
             label='Auto Tuning', linewidth=2)
    ax1.plot(manual_results['eval_timesteps'], manual_results['eval_rewards'], 
             label='Manual α=0.01', linewidth=2)
    ax1.axvline(switch_timestep, color='red', linestyle='--', linewidth=2, 
                label='Reward Switch (+200 → -100)')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Evaluation Reward')
    ax1.set_title('Q3: Learning Curves with Reward Switching')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2 = axes[1]
    ax2.plot(auto_results['eval_timesteps'], auto_results['alpha_values'], 
             label='Auto Tuning', linewidth=2, color='orange')
    ax2.axhline(manual_results['config']['manual_alpha'], 
                color='blue', linestyle='--', linewidth=2, label='Manual α=0.01')
    ax2.axvline(switch_timestep, color='red', linestyle='--', linewidth=2, 
                label='Reward Switch')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Alpha (Temperature)')
    ax2.set_title('Temperature Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax3 = axes[2]
    ax3.plot(auto_results['eval_timesteps'], auto_results['hover_trigger_rates'], 
             label='Auto Tuning', linewidth=2)
    ax3.plot(manual_results['eval_timesteps'], manual_results['hover_trigger_rates'], 
             label='Manual α=0.01', linewidth=2)
    ax3.axvline(switch_timestep, color='red', linestyle='--', linewidth=2, 
                label='Reward Switch')
    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Hover Box Trigger Rate')
    ax3.set_title('Hover Box Visitation Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    plt.close()
def plot_multi_seed_results(results_dir, pattern="SAC_*.json", save_path=None):
    """
    Plot learning curves with confidence intervals across multiple seeds.
    """
    results_files = list(Path(results_dir).glob(f"*/{pattern}"))
    if not results_files:
        print(f"No results found matching pattern: {pattern}")
        return
    all_timesteps = []
    all_rewards = []
    all_alphas = []
    for results_file in results_files:
        with open(results_file, 'r') as f:
            results = json.load(f)
        all_timesteps.append(results['eval_timesteps'])
        all_rewards.append(results['eval_rewards'])
        all_alphas.append(results['alpha_values'])
    min_len = min(len(ts) for ts in all_timesteps)
    timesteps = all_timesteps[0][:min_len]
    rewards_array = np.array([r[:min_len] for r in all_rewards])
    alphas_array = np.array([a[:min_len] for a in all_alphas])
    mean_rewards = np.mean(rewards_array, axis=0)
    std_rewards = np.std(rewards_array, axis=0)
    mean_alphas = np.mean(alphas_array, axis=0)
    std_alphas = np.std(alphas_array, axis=0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(timesteps, mean_rewards, linewidth=2, label='Mean')
    ax1.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, 
                      alpha=0.3, label='±1 Std Dev')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Evaluation Reward')
    ax1.set_title(f'Learning Curve ({len(results_files)} seeds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(timesteps, mean_alphas, linewidth=2, color='orange', label='Mean')
    ax2.fill_between(timesteps, mean_alphas - std_alphas, mean_alphas + std_alphas, 
                      alpha=0.3, color='orange', label='±1 Std Dev')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Alpha (Temperature)')
    ax2.set_title('Temperature Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-seed plot to {save_path}")
    else:
        plt.show()
    plt.close()
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True, help='Path to results.json')
    parser.add_argument('--output', type=str, default=None, help='Output plot path')
    parser.add_argument('--type', type=str, default='single', 
                       choices=['single', 'q3', 'multi'], help='Plot type')
    parser.add_argument('--manual-results', type=str, default=None, 
                       help='Manual alpha results for Q3 comparison')
    args = parser.parse_args()
    if args.type == 'single':
        plot_learning_curve(args.results, args.output)
    elif args.type == 'q3':
        if not args.manual_results:
            print("Error: --manual-results required for Q3 comparison")
        else:
            plot_q3_comparison(args.results, args.manual_results, args.output)
    elif args.type == 'multi':
        plot_multi_seed_results(os.path.dirname(args.results), save_path=args.output)
