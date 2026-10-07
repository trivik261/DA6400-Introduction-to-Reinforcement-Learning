# TD-based Control in Acrobot-v1

Implements SARSA and Q-Learning with ε-greedy exploration for the Gymnasium Acrobot-v1 environment.

## Setup

**Python version:** 3.13.2

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Code

Each part of the assignment has its own folder and script. Shared utilities (discretisation, SARSA, Q-Learning, evaluation) are in `utils.py` at the project root.

```bash
# Part 1: SARSA & Q-Learning implementation + learning curves
python Part1_SARSA_and_QLearning/part1_sarsa_qlearning.py

# Part 2: Hyperparameter tuning (2a) & multi-seed evaluation (2b)
python Part2_Tuning_and_MultiSeed/part2_tuning_and_multiseed.py

# Part 3: Online vs greedy performance (ε: 1 → 0.1, fixed)
python Part3_Online_vs_Greedy/part3_online_vs_greedy.py

# Part 6 (Bonus): Modified reward experiments (η = 0, 0.5, 1, 2, 5)
python Part6_Bonus_Empirical_Reward/part6_bonus_modified_reward.py
```

## File Structure

| Path | Description |
|------|-------------|
| `utils.py` | Shared utilities: environment setup, discretisation (10 bins/dim), ε-greedy action selection, SARSA, Q-Learning, greedy evaluation, smoothing |
| **Part1_SARSA_and_QLearning/** | |
| `├── part1_sarsa_qlearning.py` | Trains SARSA & Q-Learning, plots learning curves |
| `├── part1_learning_curves.png` | Learning curve plot |
| `└── part1-answer.txt` | Training results log |
| **Part2_Tuning_and_MultiSeed/** | |
| `├── part2_tuning_and_multiseed.py` | Grid search over (α, ε-schedule), multi-seed runs with best configs |
| `├── part2a_tuning_heatmap.png` | Hyperparameter tuning heatmap |
| `├── part2a_top3_return_vs_episodes.png` | Top-3 configs: return vs episodes |
| `├── part2a_top3_return_vs_timesteps.png` | Top-3 configs: return vs timesteps |
| `├── part2b_return_vs_episodes.png` | Multi-seed: return vs episodes |
| `├── part2b_return_vs_timesteps.png` | Multi-seed: return vs timesteps |
| `├── part2-results.txt` | Tuning results log |
| `└── part-2b.txt` | Multi-seed results log |
| **Part3_Online_vs_Greedy/** | |
| `├── part3_online_vs_greedy.py` | Online vs greedy comparison with ε fixed at 0.1 |
| `├── part3_online_vs_greedy.png` | Comparison plot |
| `└── part-3.txt` | Results log |
| **Part6_Bonus_Empirical_Reward/** | |
| `├── part6_bonus_modified_reward.py` | Empirical study of modified reward with varying η |
| `├── part6_reward_function.png` | Reward function visualization |
| `├── part6_curves_default.png` | Learning curves (default env) |
| `├── part6_curves_hard.png` | Learning curves (hard env) |
| `├── part6_heatmap.png` | η vs decay-rate heatmap |
| `├── part6_env_comparison.png` | Default vs hard env comparison |
| `├── part6_greedy_return.png` | Greedy evaluation returns |
| `├── part6_greedy_goal.png` | Greedy goal-reach rates |
| `└── part6-result.txt` | Results log |

## Key Parameters

- **Discretisation:** 10 bins per observation dimension → 10^6 states
- **Discount factor:** γ = 0.99
- **Default hyperparameters:** α = 0.1, ε: 1.0 → 0.01 (multiplicative decay)
- **Training:** 3000 episodes per run
- **Multi-seed:** 10 seeds (0–9) for statistical evaluation
