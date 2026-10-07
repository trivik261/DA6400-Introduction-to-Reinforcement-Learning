"""
Modified LunarLander-v3 environment with hover reward capability.
Supports adding bonus/penalty for hovering in a specified box.
"""
import gymnasium as gym
import numpy as np
class LunarLanderHoverEnv(gym.Wrapper):
    """
    LunarLander-v3 wrapper that adds hover reward.
    Hover reward is given once per episode when the lander enters a specified box.
    Box coordinates: |x| < 0.1 and 0.4 < y < 0.6
    """
    def __init__(self, hover_reward=0, hover_box=None, continuous=True):
        """
        Args:
            hover_reward: Reward given once when entering hover box (can be +200, -100, or 0)
            hover_box: Dict with x_min, x_max, y_min, y_max defining the hover region
            continuous: Whether to use continuous action space
        """
        env_name = "LunarLander-v3" if continuous else "LunarLander-v3"
        base_env = gym.make(env_name, continuous=continuous)
        super().__init__(base_env)
        self.hover_reward = hover_reward
        if hover_box is None:
            self.hover_box = {
                'x_min': -0.1,
                'x_max': 0.1,
                'y_min': 0.4,
                'y_max': 0.6
            }
        else:
            self.hover_box = hover_box
        self.hover_reward_given = False
        self.episode_step = 0
        self.total_hover_rewards = 0
    def reset(self, **kwargs):
        """Reset environment and hover tracking."""
        obs, info = self.env.reset(**kwargs)
        self.hover_reward_given = False
        self.episode_step = 0
        return obs, info
    def step(self, action):
        """Step environment and check for hover reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1
        if not self.hover_reward_given and self.hover_reward != 0:
            x_pos = obs[0]  # Horizontal position
            y_pos = obs[1]  # Vertical position
            in_hover_box = (
                self.hover_box['x_min'] <= x_pos <= self.hover_box['x_max'] and
                self.hover_box['y_min'] <= y_pos <= self.hover_box['y_max']
            )
            if in_hover_box:
                reward += self.hover_reward
                self.hover_reward_given = True
                self.total_hover_rewards += 1
                info['hover_reward_triggered'] = True
        info['hover_reward_given'] = self.hover_reward_given
        info['episode_step'] = self.episode_step
        return obs, reward, terminated, truncated, info
    def get_hover_stats(self):
        """Get statistics about hover rewards."""
        return {
            'total_hover_rewards': self.total_hover_rewards,
            'hover_reward_value': self.hover_reward,
            'hover_box': self.hover_box
        }
def make_lunar_lander_env(hover_reward=0, hover_box=None, continuous=True):
    """
    Factory function to create LunarLander environment.
    Args:
        hover_reward: Reward for entering hover box (0 for default, +200 or -100 for Q3)
        hover_box: Hover box coordinates
        continuous: Use continuous action space
    Returns:
        LunarLanderHoverEnv instance
    """
    return LunarLanderHoverEnv(
        hover_reward=hover_reward,
        hover_box=hover_box,
        continuous=continuous
    )
