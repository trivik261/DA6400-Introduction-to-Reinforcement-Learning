"""
Unified SAC Training Script for Pendulum
Reads all configuration from config.json including:
- Reward type: 'default' or 'adaptive'
- SAC type: 'manual' or 'automatic'
- All hyperparameters and training settings
No parallel processing - run multiple instances via separate scripts
"""
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control.pendulum import PendulumEnv
import matplotlib.pyplot as plt
import os
import json
import argparse
from datetime import datetime
import sys
class Logger:
    """Dual logger that writes to both console and file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from checkpoint_manager import CheckpointManager
from early_stopping import EarlyStopping
from official_sac_adapter import SACAgent
class PendulumEnvUnified(gym.Wrapper):
    """
    Unified Pendulum environment supporting both default and adaptive reward types.
    """
    def __init__(self, target_angle=0.0, max_episode_steps=1000, 
                 reward_type='default', reward_scale=1.0, max_torque=2.0):
        """
        Args:
            target_angle: Target angle in degrees
            max_episode_steps: Maximum steps per episode
            reward_type: 'default' (fixed velocity penalty) or 'adaptive' (angle-dependent)
            reward_scale: Universal reward multiplier
            max_torque: Maximum control torque
        """
        env = PendulumEnv(render_mode=None)
        super().__init__(env)
        self.target_angle = np.deg2rad(target_angle)
        self.target_angle_deg = target_angle
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.reward_type = reward_type
        self.reward_scale = reward_scale
        self.max_torque = max_torque
        if max_torque != 2.0:
            self.action_space = spaces.Box(
                low=-max_torque,
                high=max_torque,
                shape=(1,),
                dtype=np.float32
            )
        self.max_speed = self.env.max_speed
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0
        return obs, info
    def step(self, action):
        if self.max_torque != 2.0:
            action = np.clip(action, -self.max_torque, self.max_torque)
            theta = self.env.state[0]
            thetadot = self.env.state[1]
            g = self.env.g
            m = self.env.m
            l = self.env.l
            dt = self.env.dt
            u = action[0]
            newthetadot = thetadot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * u) * dt
            newthetadot = np.clip(newthetadot, -self.max_speed, self.max_speed)
            newtheta = theta + newthetadot * dt
            self.env.state = np.array([newtheta, newthetadot], dtype=np.float32)
            obs = np.array([np.cos(newtheta), np.sin(newtheta), newthetadot], dtype=np.float32)
            terminated = False
        else:
            obs, _, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        theta = np.arctan2(obs[1], obs[0])
        thetadot = obs[2]
        reward = self._compute_reward(theta, thetadot, action)
        truncated = self.current_step >= self.max_episode_steps
        info = {}
        return obs, reward, terminated, truncated, info
    def _compute_reward(self, theta, thetadot, action):
        """Compute reward based on reward_type."""
        angle_diff = np.arctan2(
            np.sin(theta - self.target_angle),
            np.cos(theta - self.target_angle)
        )
        angle_cost = angle_diff ** 2
        if self.reward_type == 'adaptive':
            velocity_weight = self._get_adaptive_velocity_weight()
        else:
            velocity_weight = 0.1
        velocity_cost = velocity_weight * (thetadot ** 2)
        control_cost = 0.001 * (action[0] ** 2)
        total_cost = angle_cost + velocity_cost + control_cost
        reward = -total_cost * self.reward_scale
        return reward
    def _get_adaptive_velocity_weight(self):
        """Get velocity weight based on target angle difficulty."""
        abs_angle = abs(self.target_angle_deg)
        if 80 <= abs_angle <= 100:
            return 0.02  # Near horizontal: allow oscillation
        elif 60 <= abs_angle < 80 or 100 < abs_angle <= 120:
            return 0.05  # Moderately difficult
        else:
            return 0.1   # Easy angles
def evaluate_policy(agent, env, num_episodes=10):
    """Evaluate policy over multiple episodes."""
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / num_episodes
def train_sac(config, target_angle, alpha=None, seed=0):
    """
    Train SAC agent with configuration from config.json.
    Args:
        config: Configuration dictionary
        target_angle: Target angle in degrees
        alpha: Manual alpha value (None for automatic tuning)
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    total_timesteps = config['training']['total_timesteps']
    eval_freq = config['training']['eval_freq']
    num_eval_episodes = config['training']['num_eval_episodes']
    random_exploration_steps = config['training']['random_exploration_steps']
    reward_type = config['environment'].get('reward_type', 'default')
    reward_scale = config['environment'].get('reward_scale', 1.0)
    max_torque = config['environment'].get('max_torque', 2.0)
    max_episode_steps = config['training']['max_episode_steps']
    enable_checkpoints = config['features']['enable_checkpoints']
    checkpoint_freq = config['features']['checkpoint_freq']
    enable_early_stopping = config['features']['enable_early_stopping']
    early_stopping_patience = config['features']['early_stopping_patience']
    early_stopping_min_delta = config['features']['early_stopping_min_delta']
    device_name = config['training']['device']
    if device_name == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    sac_type_config = config['sac'].get('type', 'automatic')
    manual_alpha_config = config['sac'].get('manual_alpha', 0.2)
    if alpha is not None:
        auto_entropy_tuning = False
        sac_type = 'manual'
    elif sac_type_config == 'manual':
        alpha = manual_alpha_config
        auto_entropy_tuning = False
        sac_type = 'manual'
    else:
        auto_entropy_tuning = True
        sac_type = 'automatic'
    env = PendulumEnvUnified(
        target_angle=target_angle,
        max_episode_steps=max_episode_steps,
        reward_type=reward_type,
        reward_scale=reward_scale,
        max_torque=max_torque
    )
    env.reset(seed=seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        config['training']['output_dir'],
        f"SAC_{sac_type}_theta{target_angle}_alpha{alpha if alpha else 'auto'}_seed{seed}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training.log')
    logger = Logger(log_file)
    sys.stdout = logger
    run_config = {
        'target_angle': target_angle,
        'seed': seed,
        'sac_type': sac_type,
        'alpha': alpha,
        'reward_type': reward_type,
        'reward_scale': reward_scale,
        'max_torque': max_torque,
        'timestamp': timestamp,
        **config
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f"\n{'='*80}")
    print(f"SAC TRAINING - Target Angle: {target_angle}°")
    print(f"{'='*80}")
    print(f"SAC Type: {sac_type.upper()}")
    if sac_type == 'manual':
        print(f"Alpha: {alpha}")
    print(f"Reward Type: {reward_type.upper()}")
    print(f"Reward Scale: {reward_scale}")
    print(f"Max Torque: {max_torque} N·m")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        action_bias=0.0,
        device=device,
        gamma=config['sac']['gamma'],
        tau=config['sac']['tau'],
        lr=config['sac']['learning_rate'],
        hidden_dim=config['sac']['hidden_dim'],
        buffer_size=config['sac']['buffer_size'],
        batch_size=config['sac']['batch_size'],
        auto_entropy_tuning=auto_entropy_tuning,
        alpha=alpha if not auto_entropy_tuning else 0.2
    )
    if enable_checkpoints:
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        checkpoint_manager = CheckpointManager(
            checkpoint_dir,
            keep_best_n=config['features']['keep_best_n'],
            keep_last_n=config['features']['keep_last_n']
        )
    if enable_early_stopping:
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode='max',
            verbose=True
        )
    state, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_num = 0
    training_rewards = []
    eval_rewards = []
    eval_steps = []
    alpha_values = []
    best_eval_reward = -float('inf')
    start_time = datetime.now()
    for t in range(total_timesteps):
        if t < random_exploration_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, evaluate=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.add(state, action, reward, next_state, float(done))
        episode_reward += reward
        state = next_state
        if t >= random_exploration_steps:
            agent.update_parameters()
        if done:
            training_rewards.append(episode_reward)
            episode_num += 1
            if episode_num % 10 == 0:
                current_alpha = agent.log_alpha.exp().item() if auto_entropy_tuning else float(alpha)
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"[Episode {episode_num:4d}] Step: {t+1:6d} | Reward: {episode_reward:8.2f} | Alpha: {current_alpha:.4f} | Time: {elapsed:.1f}s")
            state, _ = env.reset()
            episode_reward = 0
        if (t + 1) % eval_freq == 0:
            eval_reward = evaluate_policy(agent, env, num_episodes=num_eval_episodes)
            eval_rewards.append(eval_reward)
            eval_steps.append(t + 1)
            current_alpha = agent.log_alpha.exp().item() if auto_entropy_tuning else float(alpha)
            alpha_values.append(current_alpha)
            is_best = eval_reward > best_eval_reward
            if is_best:
                best_eval_reward = eval_reward
            elapsed = (datetime.now() - start_time).total_seconds()
            status = "NEW BEST" if is_best else ""
            print(f"\n{'='*80}")
            print(f"[EVAL] Step {t+1}/{total_timesteps} | Reward: {eval_reward:.2f} | Best: {best_eval_reward:.2f} | {status}")
            print(f"Alpha: {current_alpha:.4f} | Time: {elapsed:.1f}s")
            print(f"{'='*80}\n")
            if enable_checkpoints and (t + 1) % checkpoint_freq == 0:
                metadata = {
                    'episode': episode_num,
                    'elapsed_time': elapsed
                }
                checkpoint_manager.save_checkpoint(
                    agent, t + 1, eval_reward, is_best=is_best, metadata=metadata
                )
            if enable_early_stopping:
                eval_num = len(eval_rewards)
                if early_stopping(eval_reward, eval_num):
                    print(f"\n{'='*80}")
                    print(f"EARLY STOPPING TRIGGERED at step {t + 1}")
                    print(f"Best reward: {best_eval_reward:.2f}")
                    print(f"{'='*80}\n")
                    break
    if enable_checkpoints:
        final_metadata = {
            'episode': episode_num,
            'elapsed_time': (datetime.now() - start_time).total_seconds()
        }
        checkpoint_manager.save_checkpoint(
            agent, t + 1, eval_rewards[-1] if eval_rewards else 0,
            is_best=False, metadata=final_metadata
        )
    total_time = (datetime.now() - start_time).total_seconds()
    results = {
        'target_angle': target_angle,
        'seed': seed,
        'sac_type': sac_type,
        'alpha': alpha,
        'reward_type': reward_type,
        'reward_scale': reward_scale,
        'max_torque': max_torque,
        'final_eval_reward': eval_rewards[-1] if eval_rewards else None,
        'best_eval_reward': best_eval_reward,
        'eval_rewards': eval_rewards,
        'eval_steps': eval_steps,
        'training_rewards': training_rewards,
        'alpha_values': alpha_values,
        'training_time': total_time,
        'total_episodes': episode_num,
        'early_stopped': early_stopping.early_stop if enable_early_stopping else False
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    plot_results(results, output_dir)
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Best Reward: {best_eval_reward:.2f}")
    print(f"Final Reward: {eval_rewards[-1] if eval_rewards else 'N/A'}")
    print(f"Training Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to: {output_dir}")
    print(f"Log file: {log_file}")
    print(f"{'='*80}\n")
    logger.close()
    sys.stdout = logger.terminal
    return results
def plot_results(results, save_dir):
    """Generate training plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(results['eval_steps'], results['eval_rewards'], 
                    linewidth=2, color='blue', marker='o', markersize=4)
    axes[0, 0].axhline(y=results['best_eval_reward'], color='red', linestyle='--',
                       label=f'Best: {results["best_eval_reward"]:.2f}')
    axes[0, 0].set_xlabel('Training Steps', fontsize=11)
    axes[0, 0].set_ylabel('Evaluation Reward', fontsize=11)
    axes[0, 0].set_title(f'Evaluation Performance ({results["sac_type"].title()}, {results["reward_type"].title()} Reward)',
                        fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    info_text = f"Configuration:\n\n"
    info_text += f"Target Angle: {results['target_angle']}°\n"
    info_text += f"SAC Type: {results['sac_type'].title()}\n"
    info_text += f"Alpha: {results['alpha']}\n"
    info_text += f"Reward Type: {results['reward_type'].title()}\n"
    info_text += f"Reward Scale: {results['reward_scale']}\n"
    info_text += f"Max Torque: {results['max_torque']} N·m\n\n"
    info_text += f"Results:\n\n"
    info_text += f"Best Reward: {results['best_eval_reward']:.2f}\n"
    info_text += f"Final Reward: {results['final_eval_reward']:.2f}\n"
    info_text += f"Training Time: {results['training_time']/60:.1f} min"
    axes[0, 1].text(0.1, 0.5, info_text, ha='left', va='center', fontsize=10,
                   transform=axes[0, 1].transAxes, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    axes[0, 1].set_title('Experiment Configuration', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    if len(results['training_rewards']) > 0:
        window = min(100, len(results['training_rewards']))
        if window > 1:
            smoothed = np.convolve(results['training_rewards'],
                                  np.ones(window)/window, mode='valid')
            axes[1, 0].plot(smoothed, linewidth=1.5, color='green', alpha=0.8)
        else:
            axes[1, 0].plot(results['training_rewards'], linewidth=1.5, color='green', alpha=0.8)
    axes[1, 0].set_xlabel('Episode', fontsize=11)
    axes[1, 0].set_ylabel('Episode Reward (Smoothed)', fontsize=11)
    axes[1, 0].set_title('Training Episode Rewards', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    if results['sac_type'] == 'automatic' and len(results['alpha_values']) > 0:
        axes[1, 1].plot(results['eval_steps'], results['alpha_values'],
                       linewidth=2, color='purple', marker='s', markersize=4)
        axes[1, 1].set_xlabel('Training Steps', fontsize=11)
        axes[1, 1].set_ylabel('Alpha (Temperature)', fontsize=11)
        axes[1, 1].set_title('Alpha Evolution (Automatic Tuning)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        if len(results['eval_rewards']) > 0:
            axes[1, 1].hist(results['eval_rewards'], bins=15, color='purple', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=results['best_eval_reward'], color='red', linestyle='--',
                              linewidth=2, label=f'Best: {results["best_eval_reward"]:.2f}')
            axes[1, 1].axvline(x=np.mean(results['eval_rewards']), color='blue', linestyle='--',
                              linewidth=2, label=f'Mean: {np.mean(results["eval_rewards"]):.2f}')
        axes[1, 1].set_xlabel('Evaluation Reward', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title('Evaluation Reward Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
def main():
    parser = argparse.ArgumentParser(description='Train SAC on Pendulum with configurable settings')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--target_angle', type=float, required=True, help='Target angle in degrees')
    parser.add_argument('--alpha', type=float, default=None, help='Manual alpha (None for automatic)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    train_sac(config, args.target_angle, args.alpha, args.seed)
if __name__ == "__main__":
    main()
