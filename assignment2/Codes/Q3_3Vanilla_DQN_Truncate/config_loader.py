"""
Configuration loader for RL MountainCar experiments.
Handles loading and validation of JSON configuration files.
"""

import json
import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Configuration class for experiment parameters."""
    
    # Environment
    env_name: str
    max_episode_steps: int
    
    # Training
    num_episodes: int
    num_seeds: int
    batch_size: int
    learning_rate: float
    gamma: float
    target_update_freq: int
    replay_buffer_size: int
    update_every: int
    replay_factor: int
    
    # Exploration
    eps_start: float
    eps_end: float
    eps_decay: float
    
    # Experiment
    experiment_type: str
    truncation_values: list
    replay_factors: list
    enable_video: bool
    save_checkpoints: bool
    
    # System
    device: str
    fast_mode: bool


class ConfigLoader:
    """Loads and validates experiment configuration from JSON files."""
    
    @staticmethod
    def load_config(config_path: str) -> ExperimentConfig:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to the configuration JSON file
            
        Returns:
            ExperimentConfig: Parsed configuration object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is not valid JSON
            ValueError: If required fields are missing or invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Validate required sections
        required_sections = ['environment', 'training', 'exploration', 'experiment', 'system']
        for section in required_sections:
            if section not in config_dict:
                raise ValueError(f"Missing required section: {section}")
        
        # Extract and validate parameters
        env_config = config_dict['environment']
        training_config = config_dict['training']
        exploration_config = config_dict['exploration']
        experiment_config = config_dict['experiment']
        system_config = config_dict['system']
        
        # Validate environment
        if env_config['name'] != "MountainCar-v0":
            raise ValueError("Only MountainCar-v0 environment is supported")
        
        # Validate experiment type
        valid_types = ['Q3.1', 'Q3.2', 'Q3.3', 'Q3.4a']
        exp_type = experiment_config['experiment_type']
        # Allow Q3.4a_rho_X variants for replay factor experiments
        if exp_type not in valid_types and not exp_type.startswith('Q3.4a_rho_'):
            raise ValueError(f"Invalid experiment type. Must be one of: {valid_types} or Q3.4a_rho_X")
        
        # Validate number of seeds
        if training_config['num_seeds'] != 15:
            raise ValueError("Number of seeds must be exactly 15")
        
        # Validate gamma (should be 0.99)
        if training_config.get('gamma', 0.99) != 0.99:
            raise ValueError("Gamma must be fixed at 0.99")
        
        return ExperimentConfig(
            # Environment
            env_name=env_config['name'],
            max_episode_steps=env_config['max_episode_steps'],
            
            # Training
            num_episodes=training_config['num_episodes'],
            num_seeds=training_config['num_seeds'],
            batch_size=training_config['batch_size'],
            learning_rate=training_config['learning_rate'],
            gamma=training_config['gamma'],
            target_update_freq=training_config['target_update_freq'],
            replay_buffer_size=training_config['replay_buffer_size'],
            update_every=training_config['update_every'],
            replay_factor=training_config['replay_factor'],
            
            # Exploration
            eps_start=exploration_config['eps_start'],
            eps_end=exploration_config['eps_end'],
            eps_decay=exploration_config['eps_decay'],
            
            # Experiment
            experiment_type=experiment_config['experiment_type'],
            truncation_values=experiment_config['truncation_values'],
            replay_factors=experiment_config['replay_factors'],
            enable_video=experiment_config['enable_video'],
            save_checkpoints=experiment_config['save_checkpoints'],
            
            # System
            device=system_config['device'],
            fast_mode=system_config['fast_mode']
        )
    
    @staticmethod
    def save_default_config(config_path: str) -> None:
        """
        Save a default configuration file.
        
        Args:
            config_path: Path where to save the configuration file
        """
        default_config = {
            "environment": {
                "name": "MountainCar-v0",
                "max_episode_steps": 200
            },
            "training": {
                "num_episodes": 500,
                "num_seeds": 15,
                "batch_size": 64,
                "learning_rate": 0.001,
                "gamma": 0.99,
                "target_update_freq": 100,
                "replay_buffer_size": 10000,
                "update_every": 4,
                "replay_factor": 1
            },
            "exploration": {
                "eps_start": 1.0,
                "eps_end": 0.01,
                "eps_decay": 0.995
            },
            "experiment": {
                "experiment_type": "Q3.1",
                "truncation_values": [2000],
                "replay_factors": [1, 2, 4, 8],
                "enable_video": True,
                "save_checkpoints": True
            },
            "system": {
                "device": "cpu",
                "fast_mode": False
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    @staticmethod
    def validate_config_for_experiment(config: ExperimentConfig) -> bool:
        """
        Validate configuration for specific experiment type.
        
        Args:
            config: Experiment configuration
            
        Returns:
            bool: True if configuration is valid for the experiment type
        """
        if config.experiment_type == 'Q3.3':
            # Q3.3 requires specific truncation values
            required_values = {200, 1000, 2000}
            if set(config.truncation_values) != required_values:
                raise ValueError("Q3.3 requires truncation values: [200, 1000, 2000]")
        
        return True


if __name__ == "__main__":
    # Example usage
    config_path = "config.json"
    
    # Create default config if it doesn't exist
    if not os.path.exists(config_path):
        ConfigLoader.save_default_config(config_path)
        print(f"Created default configuration at: {config_path}")
    
    # Load and validate config
    try:
        config = ConfigLoader.load_config(config_path)
        print("Configuration loaded successfully:")
        print(f"  Environment: {config.env_name}")
        print(f"  Experiment: {config.experiment_type}")
        print(f"  Seeds: {config.num_seeds}")
        print(f"  Episodes: {config.num_episodes}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
