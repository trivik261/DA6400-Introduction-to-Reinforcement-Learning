"""
Experiment manager for DQN experiments.
Handles multi-seed training, experiment organization, and result aggregation.
"""

import os
import time
import json
import numpy as np
import copy
import torch
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config_loader import ExperimentConfig, ConfigLoader
from trainer import DQNTrainer
from plotter import ExperimentPlotter
from evaluator import DQNEvaluator
from video_recorder import VideoRecorder


class ExperimentManager:
    """
    Manager for DQN experiments.
    
    Handles:
    - Multi-seed training
    - Experiment organization
    - Result aggregation
    - Checkpoint management
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment manager.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.base_dir = 'experiments'
        
        # Create base experiments directory
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Setup logging for regular experiments
        self.setup_logging()
        
        # Print configuration details
        self._print_config_details()
        
        print(f"Initialized experiment manager")
        print(f"Experiment type: {config.experiment_type}")
        print(f"Number of seeds: {config.num_seeds}")
        print(f"Number of episodes: {config.num_episodes}")
    
    def _create_experiment_directory(self) -> str:
        """Create experiment directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{self.config.experiment_type}_{timestamp}"
        experiment_dir = os.path.join(self.base_dir, f"Experiment_{experiment_name}")
        
        # Create directory structure
        subdirs = ['config', 'models/checkpoints', 'logs', 'plots', 'videos', 'data']
        for subdir in subdirs:
            os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
        
        return experiment_dir
    
    def setup_logging(self):
        """Setup logging for experiment manager."""
        # Use base directory if experiment_dir not set yet
        if hasattr(self, 'experiment_dir'):
            # Update logging to use new experiment directory (but don't override existing logger setup)
            log_file = os.path.join(self.experiment_dir, 'logs', 'experiment.log')
            
            # Clear existing handlers and add new file handler
            self.logger.handlers.clear()
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Keep existing console handler
            console_handler = None
            for handler in self.logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    console_handler = handler
                    break
            
            if console_handler:
                self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            
            # Set formatter for all handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            if console_handler:
                console_handler.setFormatter(formatter)
        else:
            # Create a temporary log file in base directory
            log_dir = os.path.join(self.base_dir, 'logs')
            
            # Create logs directory
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'experiment.log')
            
            # Clear existing handlers and reconfigure
            logger = logging.getLogger('experiment_manager')
            logger.handlers.clear()  # Remove all existing handlers
            
            # Create new file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Configure logger
            logger.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            # Set formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger = logger
    
    def save_config(self):
        """Save experiment configuration."""
        config_path = os.path.join(self.experiment_dir, 'config', 'config.json')
        
        # Convert config to dictionary
        config_dict = {
            'environment': {
                'name': self.config.env_name,
                'max_episode_steps': self.config.max_episode_steps
            },
            'training': {
                'num_episodes': self.config.num_episodes,
                'num_seeds': self.config.num_seeds,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'gamma': self.config.gamma,
                'target_update_freq': self.config.target_update_freq,
                'replay_buffer_size': self.config.replay_buffer_size,
                'update_every': self.config.update_every,
                'replay_factor': self.config.replay_factor
            },
            'exploration': {
                'eps_start': self.config.eps_start,
                'eps_end': self.config.eps_end,
                'eps_decay': self.config.eps_decay
            },
            'experiment': {
                'experiment_type': self.config.experiment_type,
                'truncation_values': self.config.truncation_values,
                'replay_factors': self.config.replay_factors,
                'enable_video': self.config.enable_video,
                'save_checkpoints': self.config.save_checkpoints
            },
            'system': {
                'device': self.config.device,
                'fast_mode': self.config.fast_mode
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def train_single_seed(self, seed: int) -> Dict:
        """
        Train a single seed.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with training results
        """
        trainer = DQNTrainer(self.config, seed, self.experiment_dir)
        results = trainer.train()
        
        return {
            'seed': seed,
            'episode_returns': results['episode_returns'],
            'episode_lengths': results['episode_lengths'],
            'losses': results['losses'],
            'truncation_rate': results['truncation_rate'],
            'final_return': results['final_return'],
            'best_return': results['best_return'],
            'training_time': results['training_time']
        }
    
    def train_all_seeds(self, parallel: bool = True, max_workers: Optional[int] = None) -> Dict:
        """
        Train agents for all seeds.
        
        Args:
            parallel: Whether to use parallel training
            max_workers: Maximum number of parallel workers
            
        Returns:
            Aggregated training results
        """
        self.logger.info("Starting training for all seeds...")
        start_time = time.time()
        
        all_results = {}
        
        if parallel and self.config.num_seeds > 1:
            # Parallel training
            max_workers = max_workers or min(self.config.num_seeds, os.cpu_count())
            self.logger.info(f"Using parallel training with {max_workers} workers")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all seeds
                future_to_seed = {
                    executor.submit(self.train_single_seed, seed): seed 
                    for seed in range(self.config.num_seeds)
                }
                
                # Collect results
                for future in as_completed(future_to_seed):
                    seed = future_to_seed[future]
                    try:
                        result = future.result()
                        all_results[seed] = result
                        self.logger.info(f"Seed {seed} completed: Return={result['final_return']:.2f}")
                    except Exception as e:
                        self.logger.error(f"Seed {seed} failed: {e}")
                        all_results[seed] = {'error': str(e)}
        else:
            # Sequential training
            self.logger.info("Using sequential training")
            for seed in range(self.config.num_seeds):
                try:
                    result = self.train_single_seed(seed)
                    all_results[seed] = result
                    self.logger.info(f"Seed {seed} completed: Return={result['final_return']:.2f}")
                except Exception as e:
                    self.logger.error(f"Seed {seed} failed: {e}")
                    all_results[seed] = {'error': str(e)}
        
        # Aggregate results
        aggregated_results = self._aggregate_results(all_results)
        
        # Save aggregated results
        self._save_aggregated_results(aggregated_results)
        
        total_time = time.time() - start_time
        self.logger.info(f"All seeds completed in {total_time:.2f}s")
        
        return aggregated_results
    
    def _aggregate_results(self, all_results: Dict) -> Dict:
        """
        Aggregate results across all seeds.
        
        Args:
            all_results: Results for each seed
            
        Returns:
            Aggregated results
        """
        # Filter out failed seeds
        successful_results = {k: v for k, v in all_results.items() if 'error' not in v}
        failed_seeds = [k for k, v in all_results.items() if 'error' in v]
        
        if not successful_results:
            raise ValueError("All seeds failed")
        
        # Collect all returns
        all_returns = []
        all_final_returns = []
        all_best_returns = []
        all_training_times = []
        all_truncation_rates = []
        
        for seed, result in successful_results.items():
            all_returns.append(result['episode_returns'])
            all_final_returns.append(result['final_return'])
            all_best_returns.append(result['best_return'])
            all_training_times.append(result['training_time'])
            all_truncation_rates.append(result['truncation_rate'])
        
        # Convert to numpy array
        all_returns_array = np.array(all_returns)  # seeds × episodes
        
        # Calculate statistics
        aggregated = {
            'experiment_info': {
                'experiment_type': self.config.experiment_type,
                'num_seeds': len(successful_results),
                'failed_seeds': failed_seeds,
                'num_episodes': self.config.num_episodes,
                'total_runtime': sum(all_training_times),
                'device': self.config.device
            },
            'final_returns': {
                'mean': np.mean(all_final_returns),
                'std': np.std(all_final_returns),
                'min': np.min(all_final_returns),
                'max': np.max(all_final_returns),
                'median': np.median(all_final_returns),
                'all_values': all_final_returns
            },
            'best_returns': {
                'mean': np.mean(all_best_returns),
                'std': np.std(all_best_returns),
                'min': np.min(all_best_returns),
                'max': np.max(all_best_returns),
                'median': np.median(all_best_returns),
                'all_values': all_best_returns
            },
            'training_times': {
                'mean': np.mean(all_training_times),
                'std': np.std(all_training_times),
                'total': sum(all_training_times),
                'per_seed_mean': np.mean(all_training_times)
            },
            'truncation_rates': {
                'mean': np.mean(all_truncation_rates),
                'std': np.std(all_truncation_rates),
                'min': np.min(all_truncation_rates),
                'max': np.max(all_truncation_rates)
            },
            'convergence': {
                'episode_to_threshold': self._calculate_convergence(all_returns_array),
                'final_performance': np.mean(all_returns_array[:, -10:])  # Last 10 episodes
            }
        }
        
        # Save all returns array
        np.save(os.path.join(self.experiment_dir, 'data', 'all_returns.npy'), all_returns_array)
        
        return aggregated
    
    def _calculate_convergence(self, returns: np.ndarray, threshold: float = -120, 
                              window: int = 50) -> Optional[int]:
        """
        Calculate convergence episode.
        
        Args:
            returns: Returns array (seeds × episodes)
            threshold: Performance threshold
            window: Moving average window
            
        Returns:
            Convergence episode or None
        """
        mean_returns = np.mean(returns, axis=0)
        
        if len(mean_returns) < window:
            return None
        
        # Calculate moving average
        moving_avg = np.convolve(mean_returns, np.ones(window)/window, mode='valid')
        
        # Find first episode where moving average exceeds threshold
        convergence_episodes = np.where(moving_avg >= threshold)[0]
        
        if len(convergence_episodes) > 0:
            return convergence_episodes[0] + window
        else:
            return None
    
    def _save_aggregated_results(self, results: Dict):
        """Save aggregated results to files."""
        # Save JSON summary
        summary_path = os.path.join(self.experiment_dir, 'experiment_info.json')
        
        # Convert numpy types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        json_results[key][subkey] = subvalue.tolist()
                    elif isinstance(subvalue, (np.integer, np.floating)):
                        json_results[key][subkey] = float(subvalue)
                    else:
                        json_results[key][subkey] = subvalue
            else:
                json_results[key] = value
        
        with open(summary_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def _print_config_details(self):
        """Print configuration details."""
        if hasattr(self, 'logger'):
            self.logger.info("EXPERIMENT CONFIGURATION")
            self.logger.info("-" * 40)
        else:
            print("EXPERIMENT CONFIGURATION")
            print("----------------------------------------")
        
        # Environment config
        if hasattr(self, 'logger'):
            self.logger.info("ENVIRONMENT:")
            self.logger.info(f"  Name: {self.config.env_name}")
            self.logger.info(f"  Max Episode Steps: {self.config.max_episode_steps}")
        else:
            print("ENVIRONMENT:")
            print(f"  Name: {self.config.env_name}")
            print(f"  Max Episode Steps: {self.config.max_episode_steps}")
        
        # Training config
        if hasattr(self, 'logger'):
            self.logger.info("TRAINING:")
            self.logger.info(f"  Episodes: {self.config.num_episodes}")
            self.logger.info(f"  Seeds: {self.config.num_seeds}")
            self.logger.info(f"  Batch Size: {self.config.batch_size}")
            self.logger.info(f"  Learning Rate: {self.config.learning_rate}")
            self.logger.info(f"  Gamma: {self.config.gamma}")
            self.logger.info(f"  Target Update Freq: {self.config.target_update_freq}")
            self.logger.info(f"  Replay Buffer Size: {self.config.replay_buffer_size}")
            self.logger.info(f"  Update Every: {self.config.update_every}")
        else:
            print("TRAINING:")
            print(f"  Episodes: {self.config.num_episodes}")
            print(f"  Seeds: {self.config.num_seeds}")
            print(f"  Batch Size: {self.config.batch_size}")
            print(f"  Learning Rate: {self.config.learning_rate}")
            print(f"  Gamma: {self.config.gamma}")
            print(f"  Target Update Freq: {self.config.target_update_freq}")
            print(f"  Replay Buffer Size: {self.config.replay_buffer_size}")
            print(f"  Update Every: {self.config.update_every}")
        
        # Exploration config
        if hasattr(self, 'logger'):
            self.logger.info("EXPLORATION:")
            self.logger.info(f"  Epsilon Start: {self.config.eps_start}")
            self.logger.info(f"  Epsilon End: {self.config.eps_end}")
            self.logger.info(f"  Epsilon Decay: {self.config.eps_decay}")
        else:
            print("EXPLORATION:")
            print(f"  Epsilon Start: {self.config.eps_start}")
            print(f"  Epsilon End: {self.config.eps_end}")
            print(f"  Epsilon Decay: {self.config.eps_decay}")
        
        # Experiment config
        if hasattr(self, 'logger'):
            self.logger.info("EXPERIMENT:")
            self.logger.info(f"  Type: {self.config.experiment_type}")
            self.logger.info(f"  Truncation Values: {self.config.truncation_values}")
            self.logger.info(f"  Enable Video: {self.config.enable_video}")
            self.logger.info(f"  Save Checkpoints: {self.config.save_checkpoints}")
        else:
            print("EXPERIMENT:")
            print(f"  Type: {self.config.experiment_type}")
            print(f"  Truncation Values: {self.config.truncation_values}")
            print(f"  Enable Video: {self.config.enable_video}")
            print(f"  Save Checkpoints: {self.config.save_checkpoints}")
        
        # System config
        if hasattr(self, 'logger'):
            self.logger.info("SYSTEM:")
            self.logger.info(f"  Device: {self.config.device}")
            self.logger.info(f"  Fast Mode: {self.config.fast_mode}")
        else:
            print("SYSTEM:")
            print(f"  Device: {self.config.device}")
            print(f"  Fast Mode: {self.config.fast_mode}")
        
        if hasattr(self, 'logger'):
            self.logger.info("-" * 40)
            if hasattr(self, 'experiment_dir'):
                self.logger.info(f"Experiment Directory: {self.experiment_dir}")
            else:
                self.logger.info("Experiment Directory: Will be created per replay factor")
            self.logger.info("=" * 60)
        else:
            print("----------------------------------------")
            if hasattr(self, 'experiment_dir'):
                print(f"Experiment Directory: {self.experiment_dir}")
            else:
                print("Experiment Directory: Will be created per experiment")
    
    def generate_plots(self):
        """Generate plots for the experiment."""
        if hasattr(self, 'logger'):
            self.logger.info("Generating plots...")
        else:
            print("Generating plots...")
        
        plotter = ExperimentPlotter()
        
        # Load returns data
        returns_path = os.path.join(self.experiment_dir, 'data', 'all_returns.npy')
        returns = plotter.load_returns(returns_path)
        
        # Generate learning curve
        learning_curve_path = os.path.join(self.experiment_dir, 'plots', 'learning_curve.png')
        plotter.plot_learning_curve(
            returns, 
            title=f"Learning Curve - {self.config.experiment_type}",
            save_path=learning_curve_path,
            show=False
        )
        
        # Generate seed performance plot
        seed_performance_path = os.path.join(self.experiment_dir, 'plots', 'seed_performance.png')
        plotter.plot_seed_performance(
            returns,
            title=f"Seed Performance - {self.config.experiment_type}",
            save_path=seed_performance_path,
            show=False
        )
        
        # Generate statistics
        stats_path = os.path.join(self.experiment_dir, 'plots', 'statistics.json')
        statistics = plotter.create_summary_statistics(returns, stats_path)
        
        self.logger.info("Plots generated successfully")
    
    def save_best_model(self):
        """Save the best performing model."""
        self.logger.info("Saving best model...")
        
        # Find best seed
        returns_path = os.path.join(self.experiment_dir, 'data', 'all_returns.npy')
        returns = np.load(returns_path)
        
        final_returns = returns[:, -1]
        best_seed = np.argmax(final_returns)
        
        # Copy best model
        best_model_path = os.path.join(
            self.experiment_dir, 'models', f'final_model_seed_{best_seed}.pth'
        )
        best_model_dest = os.path.join(self.experiment_dir, 'models', 'best_model.pth')
        
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, best_model_dest)
            self.logger.info(f"Best model saved from seed {best_seed} (return: {final_returns[best_seed]:.2f})")
        else:
            self.logger.warning(f"Best model file not found: {best_model_path}")
    
    def generate_videos(self, num_episodes: int = 3):
        """Generate videos of the best model."""
        if not self.config.enable_video:
            self.logger.info("Video generation disabled in config")
            return
        
        self.logger.info(f"Generating {num_episodes} videos...")
        
        best_model_path = os.path.join(self.experiment_dir, 'models', 'best_model.pth')
        config_path = os.path.join(self.experiment_dir, 'config', 'config.json')
        video_dir = os.path.join(self.experiment_dir, 'videos')
        
        if not os.path.exists(best_model_path):
            self.logger.warning("Best model not found, skipping video generation")
            return
        
        try:
            recorder = VideoRecorder(best_model_path, config_path)
            recorder.record_multiple_episodes(num_episodes, video_dir)
            recorder.close()
            self.logger.info("Videos generated successfully")
        except Exception as e:
            self.logger.error(f"Video generation failed: {e}")
    
    def run_replay_factor_comparison(self, parallel: bool = True) -> Dict:
        """
        Run experiments comparing different replay factors.
        
        Args:
            parallel: Whether to use parallel training
            
        Returns:
            Dictionary with comparison results
        """
        print("Starting replay factor comparison experiments...")
        print(f"Replay factors: {self.config.replay_factors}")
        
        all_results = {}
        experiment_dirs = []
        
        for replay_factor in self.config.replay_factors:
            print(f"\n{'='*60}")
            print(f"RUNNING EXPERIMENT WITH REPLAY FACTOR rho = {replay_factor}")
            print(f"{'='*60}")
            
            # Update config for this replay factor
            config = copy.deepcopy(self.config)
            config.replay_factor = replay_factor
            config.experiment_type = f"Q3.4a_rho_{replay_factor}"
            
            # Create experiment directory for this replay factor
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            experiment_dir = os.path.join(
                self.base_dir, 
                f"Experiment_Q3.4a_rho_{replay_factor}_{timestamp}"
            )
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Set experiment_dir and create subdirectories directly
            self.experiment_dir = experiment_dir
            
            # Create required subdirectories
            subdirs = ['config', 'models/checkpoints', 'logs', 'plots', 'videos', 'data']
            for subdir in subdirs:
                os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
            
            # Temporarily update manager state for this replay factor
            original_config = self.config
            original_experiment_dir = getattr(self, 'experiment_dir', None)
            self.config = config
            self.experiment_dir = experiment_dir
            
            # Save config using the proper method
            self.save_config()
            
            # Update logging to use new experiment directory
            self.setup_logging()
            
            # Print configuration details for this replay factor
            self._print_config_details()
            
            # Train all seeds for this replay factor
            results = self.train_all_seeds(parallel=parallel)
            
            # Save best model
            self.save_best_model()
            
            # Generate plots
            self.generate_plots()
            
            # Generate videos if enabled
            if self.config.enable_video:
                self.generate_videos()
            
            # Restore original manager state
            self.config = original_config
            if original_experiment_dir is not None:
                self.experiment_dir = original_experiment_dir
            else:
                delattr(self, 'experiment_dir')
            
            # Store results
            all_results[f'rho_{replay_factor}'] = results
            experiment_dirs.append(experiment_dir)
            
            print(f"Completed replay factor {replay_factor}: Final return = {results['final_returns']['mean']:.2f}")
        
        # Generate comparison plot
        print("\nGenerating replay factor comparison plot...")
        self.generate_replay_factor_comparison_plot(all_results)
        
        return {
            'all_results': all_results,
            'experiment_dirs': experiment_dirs
        }
    
    def generate_replay_factor_comparison_plot(self, all_results: Dict):
        """Generate comparison plot for different replay factors."""
        plotter = ExperimentPlotter()
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot each replay factor
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (rho_key, results) in enumerate(all_results.items()):
            replay_factor = rho_key.split('_')[1]  # Extract number from 'rho_X'
            # Load returns from the experiment directory
            exp_dir = experiment_dirs[i]
            returns = np.load(os.path.join(exp_dir, 'data', 'all_returns.npy'))
            episodes = range(1, returns.shape[1] + 1)
            
            # Calculate mean and confidence intervals
            mean_returns = np.mean(returns, axis=0)
            std_returns = np.std(returns, axis=0)
            sem_returns = std_returns / np.sqrt(returns.shape[0])  # Standard error
            
            # 95% confidence interval
            ci_95 = 1.96 * sem_returns
            
            # Create dynamic label
            label = f"ρ={replay_factor}"
            if replay_factor == '1':
                label += ' (Vanilla)'
            
            # Plot mean line
            plt.plot(episodes, mean_returns, color=colors[i], label=label, linewidth=2)
            
            # Plot confidence interval
            plt.fill_between(episodes, 
                           mean_returns - ci_95, 
                           mean_returns + ci_95,
                           color=colors[i], alpha=0.2)
        
        plt.xlabel('Episodes')
        plt.ylabel('Mean Return')
        plt.title('DQN Performance: Replay Factor Comparison (ρ)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        comparison_path = os.path.join(self.base_dir, 'replay_factor_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Replay factor comparison plot saved to: {comparison_path}")
    
    def run_complete_experiment(self, parallel: bool = True, generate_plots: bool = True,
                              generate_videos: bool = True) -> Dict:
        """
        Run complete experiment from start to finish.
        
        Args:
            parallel: Whether to use parallel training
            generate_plots: Whether to generate plots
            generate_videos: Whether to generate videos
            
        Returns:
            Dictionary with experiment results
        """
        # Create experiment directory for this experiment FIRST
        self.experiment_dir = self._create_experiment_directory()
        
        # Save configuration to experiment folder
        self.save_config()
        
        # Setup logging AFTER creating experiment directory
        if not hasattr(self, 'logger'):
            self.setup_logging()
        else:
            # Update logging to use new experiment directory
            self.setup_logging()
        
        # Start timing
        start_time = time.time()
        
        self.logger.info(f"Running complete experiment: {self.config.experiment_type}")
        self.logger.info(f"Number of seeds: {self.config.num_seeds}")
        self.logger.info(f"Number of episodes: {self.config.num_episodes}")
        self.logger.info("=" * 60)
        
        # Print configuration details
        self._print_config_details()
            
        # Train all seeds for this replay factor
        results = self.train_all_seeds(parallel=parallel)
        
        # Save best model
        self.save_best_model()
        
        # Generate plots
        if generate_plots:
            self.generate_plots()
        
        # Generate videos
        if generate_videos and self.config.enable_video:
            self.generate_videos()
        
        total_time = time.time() - start_time
        self.logger.info(f"Complete experiment finished in {total_time:.2f}s")
        
        # Print summary
        self.logger.info("=" * 50)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Final return: {results['final_returns']['mean']:.2f} ± {results['final_returns']['std']:.2f}")
        self.logger.info(f"Best return: {results['best_returns']['mean']:.2f} ± {results['best_returns']['std']:.2f}")
        self.logger.info(f"Truncation rate: {results['truncation_rates']['mean']:.2%}")
        self.logger.info(f"Total runtime: {results['training_times']['total']:.2f}s")
        
        if results['convergence']['episode_to_threshold']:
            self.logger.info(f"Convergence episode: {results['convergence']['episode_to_threshold']}")
        
        self.logger.info(f"Results saved to: {self.experiment_dir}")
        
        return results


if __name__ == "__main__":
    # Example usage
    from config_loader import ConfigLoader
    
    # Load config
    config = ConfigLoader.load_config("config.json")
    
    # Create and run experiment
    manager = ExperimentManager(config)
    results = manager.run_complete_experiment()
    
    print("Experiment completed!")
    print(f"Results: {results}")
