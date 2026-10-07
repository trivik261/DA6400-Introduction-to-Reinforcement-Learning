"""
Generate videos from trained SAC checkpoints
Loads saved models and records episodes to video files
"""
import sys
import os
import torch
import gymnasium as gym
from pathlib import Path
import argparse
pytorch_sac_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pytorch_sac'))
sys.path.insert(0, pytorch_sac_path)
from official_sac_adapter import SACAgentForAssignment
def generate_video_from_checkpoint(checkpoint_path, output_video_path, num_episodes=3, max_steps=200):
    """
    Generate video from a trained checkpoint
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        output_video_path: Path to save the video (without extension)
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
    """
    print(f"\nGenerating video from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    agent = SACAgentForAssignment(
        state_dim=3,
        action_dim=1,
        action_scale=2.0,
        action_bias=0.0,
        gamma=0.99,
        tau=0.005,
        lr=0.0003,
        hidden_dim=256,
        auto_entropy_tuning=True,
        device='cpu'
    )
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    agent.critic_target.load_state_dict(checkpoint['critic_target'])
    print(f"Loaded checkpoint from step {checkpoint.get('timestep', 'unknown')}")
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(
        env, 
        output_video_path,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix="episode"
    )
    total_reward = 0
    episodes_recorded = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            if done or truncated:
                break
        total_reward += episode_reward
        episodes_recorded += 1
        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}")
    env.close()
    avg_reward = total_reward / episodes_recorded
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Videos saved to: {output_video_path}")
    return avg_reward
def main():
    parser = argparse.ArgumentParser(description='Generate videos from SAC checkpoints')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (e.g., outputs/experiment/checkpoints/best_*.pt)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for videos (default: same as checkpoint dir)')
    parser.add_argument('--num_episodes', type=int, default=3,
                       help='Number of episodes to record (default: 3)')
    parser.add_argument('--max_steps', type=int, default=200,
                       help='Maximum steps per episode (default: 200)')
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = checkpoint_path.parent.parent / 'videos'
    output_dir.mkdir(parents=True, exist_ok=True)
    avg_reward = generate_video_from_checkpoint(
        str(checkpoint_path),
        str(output_dir),
        num_episodes=args.num_episodes,
        max_steps=args.max_steps
    )
    print(f"\n✓ Video generation complete!")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Videos saved in: {output_dir}")
if __name__ == '__main__':
    main()
