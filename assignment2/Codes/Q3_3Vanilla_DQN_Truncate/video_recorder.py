"""
Video recording module for DQN agents.
Handles recording and saving of agent episodes as videos.
"""

import gymnasium as gym
import numpy as np
import torch
import argparse
import os
from typing import List, Tuple, Optional
import imageio
from tqdm import tqdm

from config_loader import ExperimentConfig, ConfigLoader
from dqn_model import DQNAgent


class VideoRecorder:
    """
    Video recorder for DQN agents.
    
    Handles:
    - Episode recording with rendering
    - Video file creation
    - Multiple episode recording
    """
    
    def __init__(self, model_path: str, config_path: str = "config.json", 
                 fps: int = 30, render_mode: str = 'rgb_array'):
        """
        Initialize video recorder.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            fps: Frames per second for video
            render_mode: Rendering mode for gymnasium
        """
        self.model_path = model_path
        self.config = ConfigLoader.load_config(config_path)
        self.fps = fps
        self.render_mode = render_mode
        
        # Create environment with rendering
        self.env = gym.make(
            self.config.env_name,
            render_mode=render_mode,
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
    
    def record_episode(self, save_path: str, show_progress: bool = True) -> Tuple[float, int, bool]:
        """
        Record a single episode and save as video.
        
        Args:
            save_path: Path to save video file
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (episode_return, episode_length, was_truncated)
        """
        frames = []
        state, _ = self.env.reset()
        episode_return = 0.0
        episode_length = 0
        
        # Add initial frame
        if self.render_mode == 'rgb_array':
            frame = self.env.render()
            frames.append(frame)
        
        while True:
            # Select action (no exploration during recording)
            action = self.agent.select_action(state, training=False)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Capture frame
            if self.render_mode == 'rgb_array':
                frame = self.env.render()
                frames.append(frame)
            
            # Update metrics
            episode_return += reward
            episode_length += 1
            
            state = next_state
            
            # Check if episode ended
            if terminated or truncated:
                # Add final frame
                if self.render_mode == 'rgb_array':
                    frame = self.env.render()
                    frames.append(frame)
                break
        
        # Save video
        if len(frames) > 0:
            # Ensure frames are numpy arrays and have correct shape
            frames = [np.array(frame) for frame in frames]
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save video
            imageio.mimsave(save_path, frames, fps=self.fps)
            print(f"Saved video to: {save_path}")
            print(f"Video length: {len(frames)} frames ({len(frames)/self.fps:.1f}s)")
        
        return episode_return, episode_length, truncated
    
    def record_multiple_episodes(self, num_episodes: int, output_dir: str, 
                                prefix: str = "episode") -> List[Tuple[float, int, bool]]:
        """
        Record multiple episodes.
        
        Args:
            num_episodes: Number of episodes to record
            output_dir: Directory to save videos
            prefix: Prefix for video filenames
            
        Returns:
            List of episode results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for episode in tqdm(range(num_episodes), desc="Recording episodes"):
            filename = f"{prefix}_{episode+1:03d}.mp4"
            save_path = os.path.join(output_dir, filename)
            
            episode_return, episode_length, truncated = self.record_episode(
                save_path, show_progress=False
            )
            
            results.append((episode_return, episode_length, truncated))
            
            print(f"Episode {episode+1}: Return={episode_return:.2f}, "
                  f"Length={episode_length}, Truncated={truncated}")
        
        # Print summary
        returns = [r[0] for r in results]
        lengths = [r[1] for r in results]
        truncations = [r[2] for r in results]
        
        print(f"\nRecording Summary ({num_episodes} episodes):")
        print(f"  Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        print(f"  Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"  Truncation rate: {sum(truncations)/num_episodes:.2%}")
        
        return results
    
    def record_best_episodes(self, num_episodes: int = 5, output_dir: str = "videos") -> None:
        """
        Record the best performing episodes.
        
        Args:
            num_episodes: Number of best episodes to record
            output_dir: Directory to save videos
        """
        print(f"Recording {num_episodes} best episodes...")
        
        # First, evaluate multiple episodes to find the best ones
        evaluation_results = []
        
        print("Evaluating episodes to find best performers...")
        for episode in tqdm(range(50), desc="Evaluating"):
            state, _ = self.env.reset()
            episode_return = 0.0
            episode_length = 0
            
            while True:
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                episode_return += reward
                episode_length += 1
                state = next_state
                
                if terminated or truncated:
                    break
            
            evaluation_results.append((episode_return, episode_length, truncated))
        
        # Sort by return and get top episodes
        evaluation_results.sort(key=lambda x: x[0], reverse=True)
        best_results = evaluation_results[:num_episodes]
        
        print(f"Found best episodes with returns: {[r[0] for r in best_results]}")
        
        # Now record the best episodes
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (target_return, target_length, target_truncated) in enumerate(best_results):
            print(f"Recording best episode {i+1} (target return: {target_return:.2f})...")
            
            filename = f"best_episode_{i+1:03d}_return_{target_return:.0f}.mp4"
            save_path = os.path.join(output_dir, filename)
            
            # Record until we get an episode close to the target
            attempts = 0
            while attempts < 10:  # Limit attempts to avoid infinite loop
                episode_return, episode_length, truncated = self.record_episode(
                    save_path, show_progress=False
                )
                
                # Check if this episode is close enough to the target
                if abs(episode_return - target_return) < 20:  # Allow some variance
                    print(f"  Recorded: Return={episode_return:.2f}, Length={episode_length}")
                    break
                
                attempts += 1
            
            if attempts >= 10:
                print(f"  Could not find episode close to target after {attempts} attempts")
    
    def create_compilation_video(self, episodes_data: List[Tuple[float, int, bool]], 
                                output_path: str, max_length: int = 30) -> None:
        """
        Create a compilation video from multiple episodes.
        
        Args:
            episodes_data: List of episode data
            output_path: Path to save compilation
            max_length: Maximum length per episode in seconds
        """
        all_frames = []
        
        for i, (episode_return, episode_length, truncated) in enumerate(episodes_data):
            print(f"Recording episode {i+1} for compilation...")
            
            state, _ = self.env.reset()
            frames = []
            episode_return_actual = 0.0
            episode_length_actual = 0
            
            while True:
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                if self.render_mode == 'rgb_array':
                    frame = self.env.render()
                    frames.append(frame)
                
                episode_return_actual += reward
                episode_length_actual += 1
                state = next_state
                
                if terminated or truncated:
                    break
            
            # Limit frames to max_length seconds
            max_frames = max_length * self.fps
            if len(frames) > max_frames:
                # Sample frames evenly
                indices = np.linspace(0, len(frames)-1, max_frames, dtype=int)
                frames = [frames[i] for i in indices]
            
            all_frames.extend(frames)
        
        # Save compilation
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimsave(output_path, all_frames, fps=self.fps)
        print(f"Saved compilation to: {output_path}")
        print(f"Total frames: {len(all_frames)} ({len(all_frames)/self.fps:.1f}s)")
    
    def close(self):
        """Clean up resources."""
        self.env.close()


def main():
    """Command line interface for video recording."""
    parser = argparse.ArgumentParser(description='Record DQN agent videos')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to record')
    parser.add_argument('--output', type=str, default='videos',
                       help='Output directory for videos')
    parser.add_argument('--best', action='store_true',
                       help='Record best performing episodes')
    parser.add_argument('--compilation', action='store_true',
                       help='Create compilation video')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for videos')
    parser.add_argument('--prefix', type=str, default='episode',
                       help='Prefix for video filenames')
    
    args = parser.parse_args()
    
    # Create video recorder
    recorder = VideoRecorder(args.model, args.config, args.fps)
    
    try:
        if args.best:
            # Record best episodes
            recorder.record_best_episodes(num_episodes=args.episodes, output_dir=args.output)
        elif args.compilation:
            # Create compilation
            print("Recording episodes for compilation...")
            results = recorder.record_multiple_episodes(
                num_episodes=args.episodes, 
                output_dir=args.output,
                prefix=args.prefix
            )
            
            compilation_path = os.path.join(args.output, 'compilation.mp4')
            recorder.create_compilation_video(results, compilation_path)
        else:
            # Record regular episodes
            recorder.record_multiple_episodes(
                num_episodes=args.episodes,
                output_dir=args.output,
                prefix=args.prefix
            )
    
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
