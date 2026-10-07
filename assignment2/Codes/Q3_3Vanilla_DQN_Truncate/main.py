"""
Main entry point for RL MountainCar experiments.
Provides CLI interface for training, evaluation, plotting, and video generation.
"""

import argparse
import os
import sys
import json
from typing import Optional

from config_loader import ConfigLoader, ExperimentConfig
from experiment_manager import ExperimentManager
from evaluator import DQNEvaluator
from plotter import ExperimentPlotter
from video_recorder import VideoRecorder


def create_experiment_configs():
    """Create configurations for different experiment types."""
    configs = {
        'Q3.1': {
            'experiment_type': 'Q3.1',
            'truncation_values': [2000],
            'num_episodes': 500
        },
        'Q3.2': {
            'experiment_type': 'Q3.2',
            'truncation_values': [2000],
            'num_episodes': 500
        },
        'Q3.3': {
            'experiment_type': 'Q3.3',
            'truncation_values': [200, 1000, 2000],
            'num_episodes': 500
        },
        'Q3.4a': {
            'experiment_type': 'Q3.4a',
            'truncation_values': [2000],
            'num_episodes': 500
        }
    }
    return configs


def train_experiment(config_path: str, experiment_type: Optional[str] = None, 
                    parallel: bool = True, force_retrain: bool = False):
    """
    Run training experiment.
    
    Args:
        config_path: Path to configuration file
        experiment_type: Type of experiment to run
        parallel: Whether to use parallel training
        force_retrain: Whether to force retraining
    """
    # Load configuration
    config = ConfigLoader.load_config(config_path)
    
    # Override experiment type if specified (DO NOT save to config.json)
    if experiment_type:
        configs = create_experiment_configs()
        if experiment_type in configs:
            exp_config = configs[experiment_type]
            config.experiment_type = exp_config['experiment_type']
            config.truncation_values = exp_config['truncation_values']
            config.num_episodes = exp_config['num_episodes']
            # Config will be saved by ExperimentManager to the experiment folder
    
    # Validate configuration
    ConfigLoader.validate_config_for_experiment(config)
    
    # Create experiment manager and run
    manager = ExperimentManager(config)
    results = manager.run_complete_experiment(parallel=parallel)
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {manager.experiment_dir}")
    print(f"Final return: {results['final_returns']['mean']:.2f} ± {results['final_returns']['std']:.2f}")
    
    return manager.experiment_dir


def plot_results(data_path: str, compare_paths: Optional[list] = None, 
                truncation_dirs: Optional[list] = None, 
                truncation_values: Optional[list] = None,
                output_dir: Optional[str] = None, show: bool = False):
    """
    Generate plots from experiment results.
    
    Args:
        data_path: Path to returns data file
        compare_paths: List of additional data files for comparison
        truncation_dirs: List of experiment directories for truncation comparison
        truncation_values: List of truncation values
        output_dir: Output directory for plots
        show: Whether to display plots
    """
    plotter = ExperimentPlotter()
    
    if compare_paths:
        # Multiple curves comparison
        labels = [f'Model {i+1}' for i in range(len(compare_paths) + 1)]
        data_paths = [data_path] + compare_paths
        
        save_path = os.path.join(output_dir, 'comparison.png') if output_dir else None
        plotter.plot_multiple_curves(data_paths, labels, save_path=save_path, show=show)
        
    elif truncation_dirs and truncation_values:
        # Truncation comparison
        save_path = os.path.join(output_dir, 'truncation_comparison.png') if output_dir else None
        plotter.plot_truncation_comparison(truncation_dirs, truncation_values, 
                                        save_path=save_path, show=show)
    
    else:
        # Single learning curve
        returns = plotter.load_returns(data_path)
        
        save_path = os.path.join(output_dir, 'learning_curve.png') if output_dir else None
        plotter.plot_learning_curve(returns, save_path=save_path, show=show)
        
        # Create statistics
        stats_path = os.path.join(output_dir, 'statistics.json') if output_dir else None
        statistics = plotter.create_summary_statistics(returns, stats_path)
        print(f"Final episode return: {statistics['final_episode']['mean']:.2f} ± {statistics['final_episode']['std']:.2f}")


def evaluate_model(model_path: str, config_path: str, num_episodes: int = 100):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        num_episodes: Number of evaluation episodes
    """
    evaluator = DQNEvaluator(model_path, config_path)
    results = evaluator.evaluate(num_episodes)
    evaluator.close()
    
    return results


def generate_videos(model_path: str, config_path: str, num_episodes: int = 3, 
                   output_dir: str = "videos"):
    """
    Generate videos of trained model.
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        num_episodes: Number of episodes to record
        output_dir: Output directory for videos
    """
    recorder = VideoRecorder(model_path, config_path)
    recorder.record_multiple_episodes(num_episodes, output_dir)
    recorder.close()


def run_truncation_comparison(config_path: str, parallel: bool = True):
    """
    Run truncation comparison for Q3.3.
    
    Args:
        config_path: Path to configuration file
        parallel: Whether to use parallel training
    """
    truncation_values = [200, 1000, 2000]
    experiment_dirs = []
    
    print("Running truncation comparison experiments...")
    
    for trunc_val in truncation_values:
        print(f"\nRunning experiment with truncation: {trunc_val}")
        
        # Load and modify config
        config = ConfigLoader.load_config(config_path)
        config.experiment_type = 'Q3.3'
        config.truncation_values = [trunc_val]
        config.max_episode_steps = trunc_val  # CRITICAL: Actually update the truncation
        
        # Run experiment
        manager = ExperimentManager(config)
        results = manager.run_complete_experiment(parallel=parallel)
        experiment_dirs.append(manager.experiment_dir)
        
        print(f"Completed truncation {trunc_val}: Return = {results['final_returns']['mean']:.2f}")
    
    # Generate comparison plot
    print("\nGenerating truncation comparison plot...")
    plotter = ExperimentPlotter()
    comparison_path = os.path.join(os.path.dirname(experiment_dirs[0]), 'truncation_comparison.png')
    plotter.plot_truncation_comparison(experiment_dirs, truncation_values, 
                                      save_path=comparison_path, show=False)
    
    print(f"Truncation comparison completed!")
    print(f"Comparison plot saved to: {comparison_path}")
    
    return experiment_dirs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='DQN MountainCar Experiment Framework')
    
    # Config argument
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    
    # Mode argument
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'plot', 'eval', 'video', 'truncation', 'replay_factor'],
                       help='Experiment mode to run')
    
    # Training arguments
    parser.add_argument('--experiment', type=str,
                       help='Experiment name (for train mode)')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Use parallel training')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining even if results exist')
    
    # Plotting arguments
    parser.add_argument('--data', type=str,
                       help='Path to returns data (.npy file)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple experiments')
    parser.add_argument('--truncation-dirs', nargs='+', type=str,
                       help='Directories for truncation comparison')
    parser.add_argument('--truncation-values', nargs='+', type=int,
                       help='Truncation values for comparison')
    parser.add_argument('--output', type=str,
                       help='Output directory for plots')
    parser.add_argument('--show', action='store_true',
                       help='Show plots')
    
    # Evaluation arguments
    parser.add_argument('--model', type=str,
                       help='Path to trained model for evaluation')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes for evaluation/video')
    parser.add_argument('--video-dir', type=str, default='videos',
                       help='Output directory for videos')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Creating default configuration...")
        ConfigLoader.save_default_config(args.config)
        print(f"Created default config at: {args.config}")
        return
    
    try:
        if args.mode == 'train':
            # Training mode
            experiment_dir = train_experiment(
                args.config, 
                args.experiment, 
                args.parallel, 
                args.force_retrain
            )
            print(f"\nExperiment completed! Results saved to: {experiment_dir}")
            
        elif args.mode == 'plot':
            # Plotting mode
            if not args.data:
                print("Error: --data path required for plotting mode")
                return
            
            plot_results(
                args.data,
                args.compare,
                args.truncation_dirs,
                args.truncation_values,
                args.output,
                args.show
            )
            
        elif args.mode == 'replay_factor':
            config = ConfigLoader.load_config(args.config)
            manager = ExperimentManager(config)
            results = manager.run_replay_factor_comparison()
            
            print(f"\nReplay factor comparison completed!")
            print(f"Experiment directories: {results['experiment_dirs']}")
        
        elif args.mode == 'eval':
            # Evaluation mode
            if not args.model:
                print("Error: --model path required for evaluation mode")
                return
            
            results = evaluate_model(args.model, args.config, args.episodes)
            print(f"\nEvaluation completed!")
            print(f"Mean return: {results['returns']['mean']:.2f} ± {results['returns']['std']:.2f}")
            
        elif args.mode == 'video':
            # Video generation mode
            if not args.model:
                print("Error: --model path required for video mode")
                return
            
            generate_videos(args.model, args.config, args.episodes, args.video_dir)
            print(f"\nVideos generated in: {args.video_dir}")
            
        elif args.mode == 'truncation':
            # Truncation comparison mode
            experiment_dirs = run_truncation_comparison(args.config, args.parallel)
            
            print(f"\nTruncation comparison completed!")
            print(f"Experiment directories: {experiment_dirs}")
            
        else:
            print(f"Error: Unknown mode '{args.mode}'")
            print("Available modes: train, plot, eval, video, truncation, replay_factor")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
