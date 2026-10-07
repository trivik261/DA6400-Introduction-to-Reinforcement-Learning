"""
Q3 Experiment: Reward Switching
Train SAC with +200 hover reward, then switch to -100 and continue training.
Compare fixed alpha vs automated temperature tuning.
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
            action = agent.act(state, sample=False)  # Deterministic for evaluation
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
        'hover_trigger_rate': np.mean(hover_triggers)
    }
def train_q3_experiment(
    seed=0,
    initial_hover_reward=200,
    final_hover_reward=-100,
    switch_timestep=250000,
    total_timesteps=500000,
    auto_entropy_tuning=True,
    manual_alpha=0.01,
    config_path='config.json'
):
    """
    Q3 Experiment: Train with initial hover reward, then switch to final hover reward.
    Args:
        seed: Random seed
        initial_hover_reward: Hover reward for first phase (e.g., +200)
        final_hover_reward: Hover reward for second phase (e.g., -100)
        switch_timestep: When to switch rewards
        total_timesteps: Total training timesteps
        auto_entropy_tuning: Use automated temperature tuning
        manual_alpha: Fixed alpha if auto_entropy_tuning=False
        config_path: Path to config file
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_config = config['training']['device']
    if device_config == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_config
    hover_box = config['environment_variants']['default']['hover_box']
    env = make_lunar_lander_env(hover_reward=initial_hover_reward, hover_box=hover_box, continuous=True)
    eval_env = make_lunar_lander_env(hover_reward=initial_hover_reward, hover_box=hover_box, continuous=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    q3_buffer_size = config['q3_specific']['buffer_size']  # 100K
    agent = SACAgentForAssignment(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=1.0,
        action_bias=0.0,
        gamma=config['sac']['gamma'],
        tau=config['sac']['tau'],
        lr=config['sac']['learning_rate'],
        hidden_dim=config['sac']['hidden_dim'],
        buffer_size=q3_buffer_size,  # 100K for Q3, not 1M
        batch_size=config['sac']['batch_size'],
        auto_entropy_tuning=auto_entropy_tuning,
        alpha=manual_alpha,
        device=device
    )
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    alpha_str = "auto" if auto_entropy_tuning else f"alpha{manual_alpha}"
    output_dir = os.path.join(
        config['training']['output_dir'],
        f"Q3_hover{initial_hover_reward:+d}to{final_hover_reward:+d}_{alpha_str}_seed{seed}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    run_config = {
        'seed': seed,
        'initial_hover_reward': initial_hover_reward,
        'final_hover_reward': final_hover_reward,
        'switch_timestep': switch_timestep,
        'total_timesteps': total_timesteps,
        'auto_entropy_tuning': auto_entropy_tuning,
        'manual_alpha': manual_alpha,
        'config': config
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
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
    best_eval_reward_phase1 = -np.inf
    best_eval_reward_phase2 = -np.inf
    reward_switched = False
    print("=" * 80)
    print(f"Q3 Experiment: Reward Switching")
    print(f"Seed: {seed}")
    print(f"Phase 1 (0-{switch_timestep}): Hover Reward = {initial_hover_reward:+d}")
    print(f"Phase 2 ({switch_timestep}-{total_timesteps}): Hover Reward = {final_hover_reward:+d}")
    print(f"Auto Entropy Tuning: {auto_entropy_tuning}")
    if not auto_entropy_tuning:
        print(f"Manual Alpha: {manual_alpha}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    start_time = time.time()
    for timestep in range(1, total_timesteps + 1):
        if timestep == switch_timestep and not reward_switched:
            print("\n" + "=" * 80)
            print(f"[REWARD SWITCH @ {timestep}] {initial_hover_reward:+d} → {final_hover_reward:+d}")
            print("=" * 80 + "\n")
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
            if episode_num % 10 == 0:
                phase = "Phase 1" if timestep < switch_timestep else "Phase 2"
                current_alpha = agent.log_alpha.exp().item() if agent.auto_entropy_tuning else agent.alpha
                print(f"[{phase}] Ep {episode_num:4d} | Steps: {episode_step:4d} | Reward: {episode_reward:8.2f} | Alpha: {current_alpha:.4f}")
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
            hover_trigger_rates.append(eval_result['hover_trigger_rate'])
            phase = "Phase 1" if timestep < switch_timestep else "Phase 2"
            elapsed = time.time() - start_time
            print(f"\n[{phase} Eval @ {timestep:6d}] Reward: {eval_result['mean']:8.2f} ± {eval_result['std']:6.2f} | "
                  f"Hover Rate: {eval_result['hover_trigger_rate']:.2%} | Alpha: {current_alpha:.4f} | Time: {elapsed:.1f}s\n")
            if timestep < switch_timestep:
                if eval_result['mean'] > best_eval_reward_phase1:
                    best_eval_reward_phase1 = eval_result['mean']
                    agent.save(os.path.join(output_dir, 'best_model_phase1.pt'))
                    print(f"[Best Phase 1] Saved with reward: {best_eval_reward_phase1:.2f}\n")
            else:
                if eval_result['mean'] > best_eval_reward_phase2:
                    best_eval_reward_phase2 = eval_result['mean']
                    agent.save(os.path.join(output_dir, 'best_model_phase2.pt'))
                    print(f"[Best Phase 2] Saved with reward: {best_eval_reward_phase2:.2f}\n")
            if timestep == switch_timestep:
                agent.save(os.path.join(output_dir, 'checkpoint_at_switch.pt'))
    results = {
        'eval_timesteps': eval_timesteps,
        'eval_rewards': eval_rewards,
        'alpha_values': alpha_values,
        'hover_trigger_rates': hover_trigger_rates,
        'switch_timestep': switch_timestep,
        'best_eval_reward_phase1': best_eval_reward_phase1,
        'best_eval_reward_phase2': best_eval_reward_phase2,
        'total_episodes': episode_num,
        'config': run_config
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    agent.save(os.path.join(output_dir, 'final_model.pt'))
    print("=" * 80)
    print(f"Q3 Experiment Complete!")
    print(f"Best Eval Reward Phase 1: {best_eval_reward_phase1:.2f}")
    print(f"Best Eval Reward Phase 2: {best_eval_reward_phase2:.2f}")
    print(f"Total Episodes: {episode_num}")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)
    env.close()
    eval_env.close()
    return results
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Q3 Reward Switching Experiment')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--initial-reward', type=int, default=200)
    parser.add_argument('--final-reward', type=int, default=-100)
    parser.add_argument('--switch-timestep', type=int, default=250000)
    parser.add_argument('--total-timesteps', type=int, default=500000)
    parser.add_argument('--auto-entropy', action='store_true', default=False)
    parser.add_argument('--manual-alpha', type=float, default=0.01)
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()
    train_q3_experiment(
        seed=args.seed,
        initial_hover_reward=args.initial_reward,
        final_hover_reward=args.final_reward,
        switch_timestep=args.switch_timestep,
        total_timesteps=args.total_timesteps,
        auto_entropy_tuning=args.auto_entropy,
        manual_alpha=args.manual_alpha,
        config_path=args.config
    )
