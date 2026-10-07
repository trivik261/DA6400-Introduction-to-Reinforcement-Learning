"""
Generate videos for LunarLander experiments from trained checkpoints
"""
import gymnasium as gym
import torch
import sys
import os
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).parent.parent / 'pytorch_sac'))
def load_checkpoint_and_generate_video(checkpoint_path, output_dir, experiment_name, num_episodes=3):
    """Load trained model and generate video"""
    print(f"\nGenerating video for: {experiment_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    env = gym.make('LunarLander-v3', continuous=True, render_mode='rgb_array')
    os.makedirs(output_dir, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env, 
        output_dir,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix=experiment_name
    )
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        env.close()
        return False
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    sys.path.insert(0, str(Path(__file__).parent))
    from sac_agent import SACAgent
    state_dim = 8  # LunarLander state dimension
    action_dim = 2  # LunarLander action dimension
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr=0.0003,
        gamma=0.99,
        tau=0.005,
        auto_entropy_tuning=True,
        device='cpu'
    )
    try:
        agent.policy.load_state_dict(checkpoint['actor'])
        agent.q1.load_state_dict(checkpoint['critic'])
        agent.q2.load_state_dict(checkpoint['critic'])  # Use same critic weights
        print(f"Successfully loaded checkpoint")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        env.close()
        return False
    total_reward = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        while not done and steps < 1000:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, _, _ = agent.policy.sample(state_tensor)
                action = action.cpu().numpy()[0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        total_reward += episode_reward
        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    avg_reward = total_reward / num_episodes
    print(f"  Average Reward: {avg_reward:.2f}")
    env.close()
    print(f"Videos saved to: {output_dir}")
    return True
def main():
    """Generate videos for experiments that need them"""
    base_path = Path(__file__).parent
    outputs_path = base_path / 'outputs'
    results_path = base_path.parent / 'LunarLander_Results'
    experiments = [
        {
            'name': 'B_Q3_FixedAlpha',
            'source_dir': outputs_path / 'Q3_Fixed',
            'dest_dir': results_path / 'B_Q3_FixedAlpha' / 'videos',
            'prefix': 'q3_fixed'
        },
        {
            'name': 'C_Q3_AutoAlpha',
            'source_dir': outputs_path / 'Q3_Auto_Output',
            'dest_dir': results_path / 'C_Q3_AutoAlpha' / 'videos',
            'prefix': 'q3_auto'
        }
    ]
    for exp in experiments:
        print(f"\n{'='*80}")
        print(f"Processing: {exp['name']}")
        print(f"{'='*80}")
        seed_dirs = sorted(list(exp['source_dir'].glob('*seed0*')))
        if not seed_dirs:
            print(f"No seed directories found in {exp['source_dir']}")
            continue
        seed_dir = seed_dirs[0]
        print(f"Using seed directory: {seed_dir.name}")
        checkpoint_path = seed_dir / 'best_model.pt'
        if not checkpoint_path.exists():
            checkpoint_path = seed_dir / 'final_model.pt'
        if not checkpoint_path.exists():
            print(f"No checkpoint found in {seed_dir}")
            continue
        success = load_checkpoint_and_generate_video(
            str(checkpoint_path),
            str(exp['dest_dir']),
            exp['prefix'],
            num_episodes=3
        )
        if success:
            print(f"[SUCCESS] Generated videos for {exp['name']}")
        else:
            print(f"[FAILED] Could not generate videos for {exp['name']}")
    print(f"\n{'='*80}")
    print("Video generation complete!")
    print(f"{'='*80}\n")
if __name__ == '__main__':
    main()
