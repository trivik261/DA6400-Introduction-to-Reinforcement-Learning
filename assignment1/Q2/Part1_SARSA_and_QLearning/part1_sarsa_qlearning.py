"""
Part 1 — SARSA and Q-Learning with ε-greedy Exploration
=========================================================
Trains both algorithms on Acrobot-v1 with discretised state space
(10 bins/dim → 10^6 = 1,000,000 states, 3 actions, γ = 0.99).

Produces:
  - Training log with periodic reward averages
  - Greedy policy evaluation after training
  - Learning curve plot comparing SARSA vs Q-Learning  (part1_learning_curves.png)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import (make_env, sarsa, q_learning, evaluate_policy, smooth,
                   GAMMA, NUM_BINS, N_ACTIONS, OBS_DIM)

# ─── Hyperparameters ─────────────────────────────────────────────────────
NUM_EPISODES = 3000
ALPHA = 0.1
EPS_START = 1.0
EPS_MIN = 0.01
EPS_DECAY = 0.995
EVAL_EPISODES = 50
SEED = 42


if __name__ == "__main__":
    num_states = NUM_BINS ** OBS_DIM

    print("=" * 60)
    print("Part 1 — SARSA & Q-Learning on Acrobot-v1")
    print(f"  Obs dimensions  : {OBS_DIM}")
    print(f"  Bins/dim        : {NUM_BINS}")
    print(f"  Total states    : {num_states:,}  ({NUM_BINS}^{OBS_DIM})")
    print(f"  Actions         : {N_ACTIONS}  (torque -1, 0, +1)")
    print(f"  γ = {GAMMA}   α = {ALPHA}")
    print(f"  ε : {EPS_START} → {EPS_MIN} (decay = {EPS_DECAY})")
    print(f"  Training episodes : {NUM_EPISODES}")
    print("=" * 60)

    env = make_env()

    # ── Train SARSA ───────────────────────────────────────────────────────
    print("\n>>> Training SARSA ...")
    Q_sarsa, sarsa_rewards, _ = sarsa(
        env, NUM_EPISODES, ALPHA, GAMMA,
        EPS_START, EPS_MIN, EPS_DECAY, SEED, verbose=True
    )

    # ── Train Q-Learning ──────────────────────────────────────────────────
    print("\n>>> Training Q-Learning ...")
    Q_ql, ql_rewards, _ = q_learning(
        env, NUM_EPISODES, ALPHA, GAMMA,
        EPS_START, EPS_MIN, EPS_DECAY, SEED, verbose=True
    )

    # ── Evaluate greedy policies (ε = 0) ──────────────────────────────────
    print("\n>>> Evaluating greedy policies ...")
    sarsa_eval = evaluate_policy(env, Q_sarsa, EVAL_EPISODES)
    ql_eval = evaluate_policy(env, Q_ql, EVAL_EPISODES)

    print(f"\n  SARSA      — Greedy mean reward : {np.mean(sarsa_eval):.2f} ± {np.std(sarsa_eval):.2f}")
    print(f"  Q-Learning — Greedy mean reward : {np.mean(ql_eval):.2f} ± {np.std(ql_eval):.2f}")

    # ── Plot learning curves ──────────────────────────────────────────────
    window = 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(smooth(sarsa_rewards, window), label='SARSA', color='#2196F3',
            linewidth=1.5, alpha=0.85)
    ax.plot(smooth(ql_rewards, window), label='Q-Learning', color='#FF5722',
            linewidth=1.5, alpha=0.85)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(f'Return ({window}-episode running avg)', fontsize=12)
    ax.set_title('Part 1 — SARSA vs Q-Learning  (Acrobot-v1)', fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('part1_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    env.close()
    print("\nPlot saved → part1_learning_curves.png")
    print("Done.")
