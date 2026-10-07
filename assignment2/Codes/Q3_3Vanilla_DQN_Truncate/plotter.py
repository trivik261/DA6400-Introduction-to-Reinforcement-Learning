"""
Plotting module for DQN experiment results.
Handles learning curves, truncation comparisons, and statistical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
from typing import Dict, List, Tuple, Optional
from scipy import stats
import pandas as pd


class ExperimentPlotter:
    """
    Plotter for DQN experiment results.
    
    Handles:
    - Learning curves with confidence intervals
    - Truncation comparisons
    - Statistical analysis
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize plotter.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = plt.cm.tab10.colors
    
    def load_returns(self, data_path: str) -> np.ndarray:
        """
        Load returns data from numpy file.
        
        Args:
            data_path: Path to .npy file containing returns
            
        Returns:
            Array of returns (seeds × episodes)
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        returns = np.load(data_path)
        print(f"Loaded returns data: {returns.shape} (seeds × episodes)")
        return returns
    
    def calculate_confidence_interval(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence interval using t-distribution.
        
        Args:
            data: Data array (episodes × seeds)
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1, ddof=1)
        n = data.shape[1]
        
        # t-critical value for confidence interval
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        
        margin_error = t_critical * (std / np.sqrt(n))
        
        lower_bound = mean - margin_error
        upper_bound = mean + margin_error
        
        return lower_bound, upper_bound
    
    def plot_learning_curve(self, returns: np.ndarray, title: str = "Learning Curve",
                           save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Plot learning curve with 95% confidence interval.
        
        Args:
            returns: Array of returns (seeds × episodes)
            title: Plot title
            save_path: Path to save plot
            show: Whether to display plot
        """
        # Transpose for episode-wise statistics
        returns_T = returns.T  # episodes × seeds
        
        # Calculate statistics
        episodes = np.arange(1, returns_T.shape[0] + 1)
        mean_returns = np.mean(returns_T, axis=1)
        lower_ci, upper_ci = self.calculate_confidence_interval(returns_T)
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot mean line
        ax.plot(episodes, mean_returns, 'b-', linewidth=2, label='Mean Return')
        
        # Fill confidence interval
        ax.fill_between(episodes, lower_ci, upper_ci, alpha=0.3, color='blue', 
                       label='95% CI')
        
        # Formatting
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add target performance line (near-optimal for MountainCar)
        ax.axhline(y=-110, color='r', linestyle='--', alpha=0.7, label='Target (-110)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved learning curve to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_multiple_curves(self, data_paths: List[str], labels: List[str], 
                           title: str = "Learning Curves Comparison",
                           save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Plot multiple learning curves for comparison.
        
        Args:
            data_paths: List of paths to .npy files
            labels: List of labels for each curve
            title: Plot title
            save_path: Path to save plot
            show: Whether to display plot
        """
        if len(data_paths) != len(labels):
            raise ValueError("data_paths and labels must have same length")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (data_path, label) in enumerate(zip(data_paths, labels)):
            returns = self.load_returns(data_path)
            returns_T = returns.T  # episodes × seeds
            
            episodes = np.arange(1, returns_T.shape[0] + 1)
            mean_returns = np.mean(returns_T, axis=1)
            lower_ci, upper_ci = self.calculate_confidence_interval(returns_T)
            
            color = self.colors[i % len(self.colors)]
            
            # Plot mean line
            ax.plot(episodes, mean_returns, color=color, linewidth=2, label=label)
            
            # Fill confidence interval
            ax.fill_between(episodes, lower_ci, upper_ci, alpha=0.2, color=color)
        
        # Formatting
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add target performance line
        ax.axhline(y=-110, color='r', linestyle='--', alpha=0.7, label='Target (-110)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_truncation_comparison(self, experiment_dirs: List[str], 
                                 truncation_values: List[int],
                                 save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Plot truncation comparison for Q3.3.
        
        Args:
            experiment_dirs: List of experiment directories
            truncation_values: List of truncation values
            save_path: Path to save plot
            show: Whether to display plot
        """
        if len(experiment_dirs) != len(truncation_values):
            raise ValueError("experiment_dirs and truncation_values must have same length")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (exp_dir, trunc_val) in enumerate(zip(experiment_dirs, truncation_values)):
            data_path = os.path.join(exp_dir, 'data', 'all_returns.npy')
            
            if not os.path.exists(data_path):
                print(f"Warning: Data file not found for truncation {trunc_val}: {data_path}")
                continue
            
            returns = self.load_returns(data_path)
            returns_T = returns.T  # episodes × seeds
            
            episodes = np.arange(1, returns_T.shape[0] + 1)
            mean_returns = np.mean(returns_T, axis=1)
            lower_ci, upper_ci = self.calculate_confidence_interval(returns_T)
            
            color = self.colors[i % len(self.colors)]
            
            # Plot mean line
            ax.plot(episodes, mean_returns, color=color, linewidth=2, 
                   label=f'Truncation: {trunc_val}')
            
            # Fill confidence interval
            ax.fill_between(episodes, lower_ci, upper_ci, alpha=0.2, color=color)
        
        # Formatting
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title('Truncation Comparison (Q3.3)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add target performance line
        ax.axhline(y=-110, color='r', linestyle='--', alpha=0.7, label='Target (-110)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved truncation comparison to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_seed_performance(self, returns: np.ndarray, title: str = "Seed Performance",
                             save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Plot performance of individual seeds.
        
        Args:
            returns: Array of returns (seeds × episodes)
            title: Plot title
            save_path: Path to save plot
            show: Whether to display plot
        """
        episodes = np.arange(1, returns.shape[1] + 1)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot individual seeds with low alpha
        for seed_idx in range(returns.shape[0]):
            ax.plot(episodes, returns[seed_idx], alpha=0.3, linewidth=1)
        
        # Plot mean and confidence interval
        returns_T = returns.T  # episodes × seeds
        mean_returns = np.mean(returns_T, axis=1)
        lower_ci, upper_ci = self.calculate_confidence_interval(returns_T)
        
        ax.plot(episodes, mean_returns, 'b-', linewidth=3, label='Mean')
        ax.fill_between(episodes, lower_ci, upper_ci, alpha=0.3, color='blue')
        
        # Formatting
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title(f'{title} ({returns.shape[0]} seeds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add target performance line
        ax.axhline(y=-110, color='r', linestyle='--', alpha=0.7, label='Target (-110)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved seed performance plot to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_summary_statistics(self, returns: np.ndarray, save_path: Optional[str] = None) -> Dict:
        """
        Create summary statistics for experiment results.
        
        Args:
            returns: Array of returns (seeds × episodes)
            save_path: Path to save statistics
            
        Returns:
            Dictionary with statistics
        """
        # Final episode statistics
        final_returns = returns[:, -1]
        
        # Best episode statistics
        best_returns = np.max(returns, axis=1)
        
        # Mean episode statistics
        mean_returns = np.mean(returns, axis=1)
        
        statistics = {
            'final_episode': {
                'mean': float(np.mean(final_returns)),
                'std': float(np.std(final_returns)),
                'min': float(np.min(final_returns)),
                'max': float(np.max(final_returns)),
                'median': float(np.median(final_returns))
            },
            'best_episode': {
                'mean': float(np.mean(best_returns)),
                'std': float(np.std(best_returns)),
                'min': float(np.min(best_returns)),
                'max': float(np.max(best_returns)),
                'median': float(np.median(best_returns))
            },
            'mean_episode': {
                'mean': float(np.mean(mean_returns)),
                'std': float(np.std(mean_returns)),
                'min': float(np.min(mean_returns)),
                'max': float(np.max(mean_returns)),
                'median': float(np.median(mean_returns))
            },
            'convergence_episode': self._estimate_convergence(returns)
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(statistics, f, indent=2)
            print(f"Saved statistics to: {save_path}")
        
        return statistics
    
    def _estimate_convergence(self, returns: np.ndarray, window: int = 50) -> int:
        """
        Estimate convergence episode based on moving average.
        
        Args:
            returns: Array of returns (seeds × episodes)
            window: Moving average window size
            
        Returns:
            Estimated convergence episode
        """
        returns_T = returns.T  # episodes × seeds
        mean_returns = np.mean(returns_T, axis=1)
        
        # Calculate moving average
        if len(mean_returns) < window:
            return len(mean_returns)
        
        moving_avg = np.convolve(mean_returns, np.ones(window)/window, mode='valid')
        
        # Find first episode where moving average exceeds threshold
        threshold = -120  # Near-optimal performance
        convergence_episodes = np.where(moving_avg >= threshold)[0]
        
        if len(convergence_episodes) > 0:
            return convergence_episodes[0] + window  # Convert back to original episode index
        else:
            return len(mean_returns)


def main():
    """Command line interface for plotting."""
    parser = argparse.ArgumentParser(description='Plot DQN experiment results')
    parser.add_argument('--data', type=str, help='Path to returns .npy file')
    parser.add_argument('--compare', nargs='+', default=[], help='Multiple data files to compare')
    parser.add_argument('--labels', nargs='+', default=[], help='Labels for comparison')
    parser.add_argument('--truncation', nargs='+', default=[], help='Experiment dirs for truncation comparison')
    parser.add_argument('--truncation_values', nargs='+', type=int, default=[], help='Truncation values')
    parser.add_argument('--output', type=str, help='Output directory for plots')
    parser.add_argument('--show', action='store_true', help='Show plots')
    parser.add_argument('--seeds', action='store_true', help='Plot individual seed performance')
    
    args = parser.parse_args()
    
    # Create plotter
    plotter = ExperimentPlotter()
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    if args.compare:
        # Multiple curves comparison
        labels = args.labels if args.labels else [f'Model {i+1}' for i in range(len(args.compare))]
        save_path = os.path.join(args.output, 'comparison.png') if args.output else None
        plotter.plot_multiple_curves(args.compare, labels, save_path=save_path, show=args.show)
    
    elif args.truncation:
        # Truncation comparison
        save_path = os.path.join(args.output, 'truncation_comparison.png') if args.output else None
        plotter.plot_truncation_comparison(args.truncation, args.truncation_values, 
                                        save_path=save_path, show=args.show)
    
    elif args.data:
        # Single learning curve
        returns = plotter.load_returns(args.data)
        
        if args.seeds:
            # Plot individual seed performance
            save_path = os.path.join(args.output, 'seed_performance.png') if args.output else None
            plotter.plot_seed_performance(returns, save_path=save_path, show=args.show)
        else:
            # Standard learning curve
            save_path = os.path.join(args.output, 'learning_curve.png') if args.output else None
            plotter.plot_learning_curve(returns, save_path=save_path, show=args.show)
        
        # Create summary statistics
        stats_path = os.path.join(args.output, 'statistics.json') if args.output else None
        statistics = plotter.create_summary_statistics(returns, stats_path)
        print(f"Final episode return: {statistics['final_episode']['mean']:.2f} ± {statistics['final_episode']['std']:.2f}")
    
    else:
        print("Please specify --data, --compare, or --truncation")


if __name__ == "__main__":
    main()
