"""
Unified SAC training script for LunarLander-v3.
Handles all experiments (Q1, Q2, Q3) via config file.
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
    hover_triggers = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        hover_triggered = False
        while not done:
            action = agent.select_action(state, evaluate=True)  # Deterministic for evaluation
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            done = terminated or truncated
            if info.get('hover_reward_triggered', False):
                hover_triggered = True
        eval_rewards.append(episode_reward)
        hover_triggers.append(hover_triggered)
    return {
        'mean': np.mean(eval_rewards),
        'std': np.std(eval_rewards),
        'min': np.min(eval_rewards),
        'max': np.max(eval_rewards),
        'hover_trigger_rate': np.mean(hover_triggers) if hover_triggers else 0.0
    }
def train_sac(config_path='config.json', experiment_config=None):
    """
    Train SAC agent on LunarLander-v3.
    Args:
        config_path: Path to base config file
        experiment_config: Dict with experiment-specific overrides:
            - seed: Random seed
            - hover_reward: Hover reward value (0, +200, or -100)
            - auto_entropy_tuning: Use automated temperature tuning
            - manual_alpha: Fixed alpha if auto_entropy_tuning=False
            - buffer_size: Replay buffer size (None = use config default)
            - switch_timestep: For Q3, when to switch reward (None = no switch)
            - final_hover_reward: For Q3, reward after switch
            - output_dir: Custom output directory
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    if experiment_config is None:
        experiment_config = {}
    seed = experiment_config.get('seed', 0)
    hover_reward = experiment_config.get('hover_reward', 0)
    auto_entropy_tuning = experiment_config.get('auto_entropy_tuning', True)
    manual_alpha = experiment_config.get('manual_alpha', 0.01)
    buffer_size = experiment_config.get('buffer_size', config['sac']['buffer_size'])
    switch_timestep = experiment_config.get('switch_timestep', None)
    final_hover_reward = experiment_config.get('final_hover_reward', None)
    output_dir = experiment_config.get('output_dir', None)
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
        action_scale=1.0,
        action_bias=0.0,
        gamma=config['sac']['gamma'],
        tau=config['sac']['tau'],
        lr=config['sac']['learning_rate'],
        hidden_dim=config['sac']['hidden_dim'],
        buffer_size=buffer_size,
        batch_size=config['sac']['batch_size'],
        auto_entropy_tuning=auto_entropy_tuning,
        alpha=manual_alpha,
        device=device
    )
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hover_str = f"hover{hover_reward:+d}" if hover_reward != 0 else "default"
        alpha_str = "auto" if auto_entropy_tuning else f"alpha{manual_alpha}"
        if switch_timestep:
            hover_str = f"Q3_{hover_str}to{final_hover_reward:+d}"
        output_dir = os.path.join(
            config['training']['output_dir'],
            f"SAC_{hover_str}_{alpha_str}_seed{seed}_{timestamp}"
        )
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training_progress.log')
    def log_print(msg):
        """Print to console and log file."""
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    run_config = {
        'seed': seed,
        'hover_reward': hover_reward,
        'auto_entropy_tuning': auto_entropy_tuning,
        'manual_alpha': manual_alpha,
        'buffer_size': buffer_size,
        'switch_timestep': switch_timestep,
        'final_hover_reward': final_hover_reward,
        'base_config': config
    }
    with open(os.path.join(output_dir, 'run_config.json'), 'w') as f:
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
    hover_trigger_rates = []
    best_eval_reward = -np.inf
    reward_switched = False
    log_print("=" * 80)
    log_print(f"Training SAC on LunarLander-v3")
    log_print(f"Seed: {seed}")
    log_print(f"Hover Reward: {hover_reward}")
    if switch_timestep:
        log_print(f"Reward Switch @ {switch_timestep}: {hover_reward} → {final_hover_reward}")
    log_print(f"Auto Entropy Tuning: {auto_entropy_tuning}")
    if not auto_entropy_tuning:
        log_print(f"Manual Alpha: {manual_alpha}")
    log_print(f"Buffer Size: {buffer_size}")
    log_print(f"Device: {device}")
    log_print(f"Output: {output_dir}")
    log_print("=" * 80)
    start_time = time.time()
    for timestep in range(1, total_timesteps + 1):
        if switch_timestep and timestep == switch_timestep and not reward_switched:
            log_print("\n" + "=" * 80)
            log_print(f"[REWARD SWITCH @ {timestep}] {hover_reward} → {final_hover_reward}")
            log_print("=" * 80 + "\n")
            env.close()
            eval_env.close()
            env = make_lunar_lander_env(hover_reward=final_hover_reward, hover_box=hover_box, continuous=True)
            eval_env = make_lunar_lander_env(hover_reward=final_hover_reward, hover_box=hover_box, continuous=True)
            state, _ = env.reset()
            episode_reward = 0
            episode_step = 0
            reward_switched = True
        if timestep < random_exploration_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, evaluate=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.add(state, action, reward, next_state, done)
        episode_reward += reward
        episode_step += 1
        state = next_state
        if timestep >= random_exploration_steps:
            agent.update_parameters()
        if done:
            if episode_num % 10 == 0:
                current_alpha = agent.log_alpha.exp().item() if agent.auto_entropy_tuning else float(agent.alpha)
                phase = f"[Phase {2 if reward_switched else 1}]" if switch_timestep else ""
                log_print(f"{phase} Ep {episode_num:4d} | Steps: {episode_step:4d} | Reward: {episode_reward:8.2f} | Alpha: {current_alpha:.4f}")
            state, _ = env.reset()
            episode_reward = 0
            episode_step = 0
            episode_num += 1
        if timestep % eval_freq == 0:
            eval_result = evaluate_policy(agent, eval_env, config['training']['num_eval_episodes'])
            eval_timesteps.append(timestep)
            eval_rewards.append(eval_result['mean'])
            current_alpha = agent.log_alpha.exp().item() if agent.auto_entropy_tuning else float(agent.alpha)
            alpha_values.append(current_alpha)
            hover_trigger_rates.append(eval_result['hover_trigger_rate'])
            elapsed = time.time() - start_time
            phase = f"[Phase {2 if reward_switched else 1}]" if switch_timestep else ""
            log_print(f"\n{phase} [Eval @ {timestep:6d}] Reward: {eval_result['mean']:8.2f} ± {eval_result['std']:6.2f} | "
                  f"Alpha: {current_alpha:.4f} | Time: {elapsed:.1f}s\n")
            if eval_result['mean'] > best_eval_reward:
                best_eval_reward = eval_result['mean']
                agent.save(os.path.join(output_dir, 'best_model.pt'))
                log_print(f"[Best Model] Saved with reward: {best_eval_reward:.2f}\n")
            if config['features']['enable_checkpoints'] and timestep % config['features']['checkpoint_freq'] == 0:
                agent.save(os.path.join(output_dir, f'checkpoint_{timestep}.pt'))
    results = {
        'eval_timesteps': eval_timesteps,
        'eval_rewards': eval_rewards,
        'alpha_values': alpha_values,
        'hover_trigger_rates': hover_trigger_rates,
        'best_eval_reward': best_eval_reward,
        'total_episodes': episode_num,
        'run_config': run_config
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    agent.save(os.path.join(output_dir, 'final_model.pt'))
    log_print("=" * 80)
    log_print(f"Training Complete!")
    log_print(f"Best Eval Reward: {best_eval_reward:.2f}")
    log_print(f"Total Episodes: {episode_num}")
    log_print(f"Results saved to: {output_dir}")
    log_print("=" * 80)
    env.close()
    eval_env.close()
    return results
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--experiment', type=str, default='q1', 
                       choices=['q1', 'q2', 'q3_fixed', 'q3_auto'],
                       help='Experiment type')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        base_config = json.load(f)
    if args.experiment == 'q1' or args.experiment == 'q2':
        exp_config = {
            'seed': args.seed,
            'hover_reward': 0,
            'auto_entropy_tuning': True,
            'buffer_size': base_config['sac']['buffer_size']  # 1M
        }
    elif args.experiment == 'q3_fixed':
        exp_config = {
            'seed': args.seed,
            'hover_reward': 200,
            'auto_entropy_tuning': False,
            'manual_alpha': 0.01,
            'buffer_size': base_config['q3_specific']['buffer_size'],  # 100K
            'switch_timestep': 250000,
            'final_hover_reward': -100
        }
    elif args.experiment == 'q3_auto':
        exp_config = {
            'seed': args.seed,
            'hover_reward': 200,
            'auto_entropy_tuning': True,
            'buffer_size': base_config['q3_specific']['buffer_size'],  # 100K
            'switch_timestep': 250000,
            'final_hover_reward': -100
        }
    train_sac(args.config, exp_config)
