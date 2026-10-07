"""
Part 3 — ε: 1 → 0.1 (fixed thereafter)
=========================================
Run SARSA and Q-Learning with ε decaying from 1.0 to 0.1 then held constant.
Compare:
  (i)  Online performance (returns while learning with ε=0.1 exploration)
  (ii) Greedy policy performance after training (ε=0, no exploration)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from utils import (make_env, sarsa, q_learning, evaluate_policy, smooth,
                   GAMMA, NUM_BINS)

NUM_EPISODES = 3000
EVAL_EPISODES = 50

# ε schedule: 1.0 → 0.1, then fixed at 0.1
EPS_START = 1.0
EPS_MIN = 0.1
EPS_DECAY = 0.995
ALPHA = 0.1


def part3(env):
    print("=" * 70)
    print("PART 3 — ε: 1 → 0.1 (then fixed)")
    print(f"  α={ALPHA}, γ={GAMMA}, ε: {EPS_START} → {EPS_MIN} (decay={EPS_DECAY})")
    print("=" * 70)

    seeds = list(range(10))
    online_sarsa, online_ql = [], []
    greedy_sarsa, greedy_ql = [], []

    for i, s in enumerate(seeds):
        print(f"  Seed {s} ({i+1}/10) ...", end=" ", flush=True)
        t0 = time.time()

        Q_s, sr, _ = sarsa(env, NUM_EPISODES, ALPHA, GAMMA,
                           EPS_START, EPS_MIN, EPS_DECAY, seed=s)
        Q_q, qr, _ = q_learning(env, NUM_EPISODES, ALPHA, GAMMA,
                                 EPS_START, EPS_MIN, EPS_DECAY, seed=s)

        online_sarsa.append(sr)
        online_ql.append(qr)
        greedy_sarsa.append(evaluate_policy(env, Q_s, EVAL_EPISODES, seed=s + 100))
        greedy_ql.append(evaluate_policy(env, Q_q, EVAL_EPISODES, seed=s + 100))

        print(f"done ({time.time()-t0:.1f}s)")

    online_sarsa = np.array(online_sarsa)   # (10, NUM_EPISODES)
    online_ql = np.array(online_ql)
    greedy_sarsa = np.array(greedy_sarsa)   # (10, EVAL_EPISODES)
    greedy_ql = np.array(greedy_ql)

    # ── Smoothed online curves ────────────────────────────────────────────
    window = 100
    sm_s = np.array([smooth(online_sarsa[i], window) for i in range(10)])
    sm_q = np.array([smooth(online_ql[i], window) for i in range(10)])
    episodes = np.arange(sm_s.shape[1]) + window

    mean_s, std_s = sm_s.mean(0), sm_s.std(0)
    mean_q, std_q = sm_q.mean(0), sm_q.std(0)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (i) Online performance
    ax = axes[0]
    ax.plot(episodes, mean_s, label='SARSA', color='#2196F3', linewidth=1.5)
    ax.fill_between(episodes, mean_s - std_s, mean_s + std_s,
                    alpha=0.2, color='#2196F3')
    ax.plot(episodes, mean_q, label='Q-Learning', color='#FF5722', linewidth=1.5)
    ax.fill_between(episodes, mean_q - std_q, mean_q + std_q,
                    alpha=0.2, color='#FF5722')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Return (100-ep running avg)', fontsize=12)
    ax.set_title('(i) Online Performance (ε: 1 → 0.1, fixed)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # (ii) Greedy evaluation
    ax = axes[1]
    s_means = greedy_sarsa.mean(axis=1)   # mean per seed
    q_means = greedy_ql.mean(axis=1)
    x = np.arange(2)
    bars = ax.bar(x,
                  [s_means.mean(), q_means.mean()],
                  yerr=[s_means.std(), q_means.std()],
                  capsize=10, width=0.5,
                  color=['#2196F3', '#FF5722'], edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['SARSA', 'Q-Learning'], fontsize=12)
    ax.set_ylabel('Avg Return (greedy, ε=0)', fontsize=12)
    ax.set_title('(ii) Greedy Policy After Learning', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, [s_means.mean(), q_means.mean()]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle(f'Part 3: ε = 1 → 0.1 (fixed) | α={ALPHA} | γ={GAMMA}',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('part3_online_vs_greedy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved → part3_online_vs_greedy.png")

    # ── Numerical results ─────────────────────────────────────────────────
    ol_s = online_sarsa[:, -200:].mean(axis=1)
    ol_q = online_ql[:, -200:].mean(axis=1)
    print(f"\n  (i)  Online performance (last 200 eps, ε ≈ 0.1):")
    print(f"         SARSA      : {ol_s.mean():.2f} ± {ol_s.std():.2f}")
    print(f"         Q-Learning : {ol_q.mean():.2f} ± {ol_q.std():.2f}")
    print(f"  (ii) Greedy policy (ε = 0, after training):")
    print(f"         SARSA      : {s_means.mean():.2f} ± {s_means.std():.2f}")
    print(f"         Q-Learning : {q_means.mean():.2f} ± {q_means.std():.2f}")

    print("\n--- Explanation ---")
    print("  Online vs Greedy gap:")
    print("    The greedy policy (ε=0) outperforms the online performance (ε=0.1)")
    print("    because 10% random actions during training cause suboptimal steps,")
    print("    increasing episode length and accumulating more negative reward.")
    print()
    print("  SARSA vs Q-Learning — online:")
    print("    SARSA (on-policy) learns values under the ε-greedy behaviour policy")
    print("    it actually follows. Its online returns directly reflect the quality")
    print("    of its learned Q-values. With ε fixed at 0.1, the agent never fully")
    print("    exploits and online returns plateau.")
    print()
    print("  SARSA vs Q-Learning — greedy:")
    print("    Q-Learning (off-policy) always updates towards max_a Q(s',a),")
    print("    directly estimating the optimal (greedy) value function regardless")
    print("    of the exploratory actions taken. Therefore its greedy policy after")
    print("    training tends to be better than SARSA's, whose Q-values were shaped")
    print("    by the ε-greedy behaviour and are slightly conservative.")
    print()
    print("  Since ε never reaches 0 (fixed at 0.1), the persistent exploration")
    print("  creates a permanent gap between online and greedy performance. This")
    print("  contrasts with a decaying schedule (ε→0) where the gap eventually")
    print("  vanishes as the agent becomes fully greedy.")


if __name__ == "__main__":
    print(f"Acrobot-v1 | Bins={NUM_BINS} | States={NUM_BINS**6:,} | γ={GAMMA}")
    print(f"Training episodes = {NUM_EPISODES}\n")

    env = make_env()
    part3(env)
    env.close()

    print("\nDone. Plot: part3_online_vs_greedy.png")
