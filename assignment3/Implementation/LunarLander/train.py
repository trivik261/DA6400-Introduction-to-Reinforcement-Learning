"""
Training script for SAC on LunarLander-v3.
Supports both default environment and hover reward variants.
"""
import os
import json
import time
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
import sys
from official_sac_adapter import SACAgentForAssignment
from lunar_lander_env import make_lunar_lander_env
def evaluate_policy(agent, env, num_episodes=20):
    """Evaluate policy over multiple episodes."""
    eval_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.act(state, sample=False)  # Deterministic for evaluation
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            done = terminated or truncated
        eval_rewards.append(episode_reward)
    return {
        'mean': np.mean(eval_rewards),
        'std': np.std(eval_rewards),
        'min': np.min(eval_rewards),
        'max': np.max(eval_rewards)
    }
def train_sac(
    config_path='config.json',
    seed=0,
    hover_reward=0,
    auto_entropy_tuning=True,
    manual_alpha=0.01,
    output_dir=None,
    resume_from=None,
    verbose=True
):
    """
    Train SAC agent on LunarLander-v3.
    Args:
        config_path: Path to config file
        seed: Random seed
        hover_reward: Hover reward value (0, +200, or -100)
        auto_entropy_tuning: Use automated temperature tuning
        manual_alpha: Fixed alpha if auto_entropy_tuning=False
        output_dir: Output directory for results
        resume_from: Path to checkpoint to resume from
        verbose: Print training progress
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hover_box = config['environment_variants']['default']['hover_box']
    env = make_lunar_lander_env(hover_reward=hover_reward, hover_box=hover_box, continuous=True)
    eval_env = make_lunar_lander_env(hover_reward=hover_reward, hover_box=hover_box, continuous=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device_config = config['training']['device']
    if device_config == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_config
    agent = SACAgentForAssignment(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=1.0,  # LunarLander uses [-1, 1] directly
        action_bias=0.0,
        gamma=config['sac']['gamma'],
        tau=config['sac']['tau'],
        lr=config['sac']['learning_rate'],
        hidden_dim=config['sac']['hidden_dim'],
        buffer_size=config['sac']['buffer_size'],  # 1M for Q1/Q2 (infinite-like)
        batch_size=config['sac']['batch_size'],
        auto_entropy_tuning=auto_entropy_tuning,
        alpha=manual_alpha,
        device=device
    )
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hover_str = f"hover{hover_reward:+d}" if hover_reward != 0 else "default"
        alpha_str = "auto" if auto_entropy_tuning else f"alpha{manual_alpha}"
        output_dir = os.path.join(
            config['training']['output_dir'],
            f"SAC_{hover_str}_{alpha_str}_seed{seed}_{timestamp}"
        )
    os.makedirs(output_dir, exist_ok=True)
    run_config = {
        'seed': seed,
        'hover_reward': hover_reward,
        'auto_entropy_tuning': auto_entropy_tuning,
        'manual_alpha': manual_alpha,
        'config': config
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    total_timesteps = config['training']['total_timesteps']
    eval_freq = config['training']['eval_freq']
    random_exploration_steps = config['training']['random_exploration_steps']
    state, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_step = 0
    episode_num = 0
    eval_timesteps = []
    eval_rewards = []
    alpha_values = []
    best_eval_reward = -np.inf
    if verbose:
        print("=" * 70)
        print(f"Training SAC on LunarLander-v3")
        print(f"Seed: {seed}")
        print(f"Hover Reward: {hover_reward}")
        print(f"Auto Entropy Tuning: {auto_entropy_tuning}")
        if not auto_entropy_tuning:
            print(f"Manual Alpha: {manual_alpha}")
        print(f"Device: {device}")
        print(f"Output: {output_dir}")
        print("=" * 70)
    start_time = time.time()
    for timestep in range(1, total_timesteps + 1):
        if timestep < random_exploration_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(state, sample=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.add(state, action, reward, next_state, done, done)
        episode_reward += reward
        episode_step += 1
        state = next_state
        if timestep >= random_exploration_steps:
            update_info = agent.update()
        if done:
            if verbose and episode_num % 10 == 0:
                current_alpha = agent.log_alpha.exp().item() if agent.auto_entropy_tuning else agent.alpha
                print(f"Episode {episode_num:4d} | Steps: {episode_step:4d} | Reward: {episode_reward:8.2f} | Alpha: {current_alpha:.4f}")
            state, _ = env.reset()
            episode_reward = 0
            episode_step = 0
            episode_num += 1
        if timestep % eval_freq == 0:
            eval_result = evaluate_policy(agent, eval_env, config['training']['num_eval_episodes'])
            eval_timesteps.append(timestep)
            eval_rewards.append(eval_result['mean'])
            current_alpha = agent.log_alpha.exp().item() if agent.auto_entropy_tuning else agent.alpha
            alpha_values.append(current_alpha)
            if verbose:
                elapsed = time.time() - start_time
                print(f"\n[Eval @ {timestep:6d}] Reward: {eval_result['mean']:8.2f} ± {eval_result['std']:6.2f} | Alpha: {current_alpha:.4f} | Time: {elapsed:.1f}s\n")
            if eval_result['mean'] > best_eval_reward:
                best_eval_reward = eval_result['mean']
                agent.save(os.path.join(output_dir, 'best_model.pt'))
                if verbose:
                    print(f"[Best Model] Saved with reward: {best_eval_reward:.2f}\n")
            if config['features']['enable_checkpoints'] and timestep % config['features']['checkpoint_freq'] == 0:
                agent.save(os.path.join(output_dir, f'checkpoint_{timestep}.pt'))
    results = {
        'eval_timesteps': eval_timesteps,
        'eval_rewards': eval_rewards,
        'alpha_values': alpha_values,
        'best_eval_reward': best_eval_reward,
        'total_episodes': episode_num,
        'config': run_config
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    agent.save(os.path.join(output_dir, 'final_model.pt'))
    if verbose:
        print("=" * 70)
        print(f"Training Complete!")
        print(f"Best Eval Reward: {best_eval_reward:.2f}")
        print(f"Total Episodes: {episode_num}")
        print(f"Results saved to: {output_dir}")
        print("=" * 70)
    env.close()
    eval_env.close()
    return results
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hover-reward', type=int, default=0, choices=[0, 200, -100])
    parser.add_argument('--auto-entropy', action='store_true', default=True)
    parser.add_argument('--manual-alpha', type=float, default=0.01)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    train_sac(
        config_path=args.config,
        seed=args.seed,
        hover_reward=args.hover_reward,
        auto_entropy_tuning=args.auto_entropy,
        manual_alpha=args.manual_alpha,
        output_dir=args.output_dir
    )
