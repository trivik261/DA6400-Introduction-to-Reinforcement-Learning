"""
Part 2 — Hyperparameter Tuning & Multi-Seed Evaluation
========================================================
2a) Grid-search over (α, ε-schedule), report top-3 configs for SARSA & Q-Learning.
    Discuss whether constant ε is sufficient; implement ε-decay.
2b) With tuned hyperparameters, run 10 seeds and plot mean ± std confidence bands.
    Plots: Return vs Episodes AND Return vs Timesteps.
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


# ═══════════════════════════════════════════════════════════════════════════
# 2a — Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════════
def hyperparameter_tuning(env):
    print("=" * 70)
    print("PART 2a — Hyperparameter Tuning (step-size α and ε schedule)")
    print("=" * 70)

    alphas = [0.05, 0.1, 0.2, 0.5]

    eps_configs = [
        # (eps_start, eps_min, eps_decay, label)
        (0.3, 0.3,  1.0,   "constant ε=0.3"),
        (0.1, 0.1,  1.0,   "constant ε=0.1"),
        (1.0, 0.01, 0.995, "decay 1→0.01, d=0.995"),
        (1.0, 0.01, 0.999, "decay 1→0.01, d=0.999"),
        (1.0, 0.01, 0.99,  "decay 1→0.01, d=0.99"),
    ]

    tuning_seed = 42
    results_sarsa, results_ql = [], []
    curves_sarsa, curves_ql = {}, {}       # keyed by label
    ts_sarsa, ts_ql = {}, {}               # cumulative timesteps per config
    total = len(alphas) * len(eps_configs)
    counter = 0

    for alpha in alphas:
        for eps_start, eps_min, eps_decay, eps_label in eps_configs:
            counter += 1
            label = f"α={alpha}, {eps_label}"
            print(f"  [{counter:>2d}/{total}] {label} ...", end=" ", flush=True)
            t0 = time.time()

            _, sr, s_ts = sarsa(env, NUM_EPISODES, alpha, GAMMA,
                                eps_start, eps_min, eps_decay, tuning_seed)
            _, qr, q_ts = q_learning(env, NUM_EPISODES, alpha, GAMMA,
                                     eps_start, eps_min, eps_decay, tuning_seed)

            s_avg = np.mean(sr[-200:])
            q_avg = np.mean(qr[-200:])
            results_sarsa.append((s_avg, alpha, eps_start, eps_min, eps_decay, eps_label))
            results_ql.append((q_avg, alpha, eps_start, eps_min, eps_decay, eps_label))
            curves_sarsa[label] = sr
            curves_ql[label] = qr
            ts_sarsa[label] = np.cumsum(s_ts)
            ts_ql[label] = np.cumsum(q_ts)

            print(f"SARSA={s_avg:.1f}  Q-Lrn={q_avg:.1f}  ({time.time()-t0:.1f}s)")

    results_sarsa.sort(key=lambda x: x[0], reverse=True)
    results_ql.sort(key=lambda x: x[0], reverse=True)

    print("\n--- Top 3 Configurations: SARSA ---")
    for rank, (avg, a, es, em, ed, lbl) in enumerate(results_sarsa[:3], 1):
        print(f"  #{rank}  α={a}, {lbl}  →  Avg return (last 200 eps) = {avg:.2f}")

    print("\n--- Top 3 Configurations: Q-Learning ---")
    for rank, (avg, a, es, em, ed, lbl) in enumerate(results_ql[:3], 1):
        print(f"  #{rank}  α={a}, {lbl}  →  Avg return (last 200 eps) = {avg:.2f}")

    print("\n--- Constant ε vs Decaying ε ---")
    best_const_s = max(r for r in results_sarsa if r[4] == 1.0)
    best_decay_s = max(r for r in results_sarsa if r[4] != 1.0)
    best_const_q = max(r for r in results_ql if r[4] == 1.0)
    best_decay_q = max(r for r in results_ql if r[4] != 1.0)
    print(f"  SARSA  best constant : {best_const_s[5]}, α={best_const_s[1]} → {best_const_s[0]:.2f}")
    print(f"  SARSA  best decaying : {best_decay_s[5]}, α={best_decay_s[1]} → {best_decay_s[0]:.2f}")
    print(f"  Q-Lrn  best constant : {best_const_q[5]}, α={best_const_q[1]} → {best_const_q[0]:.2f}")
    print(f"  Q-Lrn  best decaying : {best_decay_q[5]}, α={best_decay_q[1]} → {best_decay_q[0]:.2f}")
    print()
    print("  A constant ε forces the agent to keep taking random actions even after")
    print("  it has learned a good policy, degrading final performance. Decaying ε")
    print("  (→ 0) allows early exploration and later exploitation, which is needed")
    print("  to converge to a near-optimal greedy policy.")

    # Heatmap plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, results, title in [(axes[0], results_sarsa, "SARSA"),
                                (axes[1], results_ql, "Q-Learning")]:
        grid = np.zeros((len(alphas), len(eps_configs)))
        for avg, a, es, em, ed, lbl in results:
            ai = alphas.index(a)
            ei = next(j for j, c in enumerate(eps_configs) if c[3] == lbl)
            grid[ai, ei] = avg
        im = ax.imshow(grid, aspect='auto', cmap='coolwarm_r')
        ax.set_xticks(range(len(eps_configs)))
        ax.set_xticklabels([c[3] for c in eps_configs], rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels([str(a) for a in alphas])
        ax.set_xlabel("ε Configuration")
        ax.set_ylabel("α (step-size)")
        ax.set_title(f"{title} — Avg Return (last 200 eps)")
        for i in range(len(alphas)):
            for j in range(len(eps_configs)):
                ax.text(j, i, f"{grid[i,j]:.0f}", ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white' if abs(grid[i, j]) > abs(np.mean(grid)) else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig('part2a_tuning_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved → part2a_tuning_heatmap.png")

    # ── Top-3 learning curves: Return vs Episodes ─────────────────────────
    window = 100
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, results, curves, title in [
        (axes[0], results_sarsa, curves_sarsa, "SARSA"),
        (axes[1], results_ql, curves_ql, "Q-Learning"),
    ]:
        for rank, (avg, a, es, em, ed, lbl) in enumerate(results[:3]):
            key = f"α={a}, {lbl}"
            sm = smooth(curves[key], window)
            eps_x = np.arange(len(sm)) + window
            ax.plot(eps_x, sm, label=f"#{rank+1} {key} ({avg:.0f})",
                    color=colors[rank], linewidth=1.5, alpha=0.85)
        ax.set_xlabel('Episodes', fontsize=12)
        ax.set_ylabel('Return (100-ep running avg)', fontsize=12)
        ax.set_title(f'{title} — Top 3 Configs (Return vs Episodes)', fontsize=11)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('part2a_top3_return_vs_episodes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved → part2a_top3_return_vs_episodes.png")

    # ── Top-3 learning curves: Return vs Timesteps ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, results, curves, ts_dict, title in [
        (axes[0], results_sarsa, curves_sarsa, ts_sarsa, "SARSA"),
        (axes[1], results_ql, curves_ql, ts_ql, "Q-Learning"),
    ]:
        for rank, (avg, a, es, em, ed, lbl) in enumerate(results[:3]):
            key = f"α={a}, {lbl}"
            sm = smooth(curves[key], window)
            cum_ts = ts_dict[key][window-1:]
            ax.plot(cum_ts, sm, label=f"#{rank+1} {key} ({avg:.0f})",
                    color=colors[rank], linewidth=1.5, alpha=0.85)
        ax.set_xlabel('Timesteps (cumulative)', fontsize=12)
        ax.set_ylabel('Return (100-ep running avg)', fontsize=12)
        ax.set_title(f'{title} — Top 3 Configs (Return vs Timesteps)', fontsize=11)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('part2a_top3_return_vs_timesteps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved → part2a_top3_return_vs_timesteps.png")

    best_s = results_sarsa[0]
    best_q = results_ql[0]
    sarsa_params = dict(alpha=best_s[1], eps_start=best_s[2],
                        eps_min=best_s[3], eps_decay=best_s[4])
    ql_params = dict(alpha=best_q[1], eps_start=best_q[2],
                     eps_min=best_q[3], eps_decay=best_q[4])
    return sarsa_params, ql_params


# ═══════════════════════════════════════════════════════════════════════════
# 2b — Multi-Seed Runs + Confidence Interval Plots
# ═══════════════════════════════════════════════════════════════════════════
def multiseed_runs(env, sarsa_params, ql_params):
    print("\n" + "=" * 70)
    print("PART 2b — 10-Seed Runs with Tuned Hyperparameters")
    print(f"  SARSA  : {sarsa_params}")
    print(f"  Q-Lrn  : {ql_params}")
    print("=" * 70)

    seeds = list(range(10))
    all_sarsa, all_ql = [], []
    all_sarsa_ts, all_ql_ts = [], []

    for i, s in enumerate(seeds):
        print(f"  Seed {s} ({i+1}/10) ...", end=" ", flush=True)
        t0 = time.time()

        _, sr, s_ts = sarsa(env, NUM_EPISODES, sarsa_params['alpha'], GAMMA,
                            sarsa_params['eps_start'], sarsa_params['eps_min'],
                            sarsa_params['eps_decay'], seed=s)
        _, qr, q_ts = q_learning(env, NUM_EPISODES, ql_params['alpha'], GAMMA,
                                 ql_params['eps_start'], ql_params['eps_min'],
                                 ql_params['eps_decay'], seed=s)

        all_sarsa.append(sr)
        all_ql.append(qr)
        all_sarsa_ts.append(np.cumsum(s_ts))
        all_ql_ts.append(np.cumsum(q_ts))
        print(f"done ({time.time()-t0:.1f}s)")

    all_sarsa = np.array(all_sarsa)       # (10, NUM_EPISODES)
    all_ql = np.array(all_ql)
    all_sarsa_ts = np.array(all_sarsa_ts) # (10, NUM_EPISODES)
    all_ql_ts = np.array(all_ql_ts)

    window = 100

    # ── Smoothed returns ──────────────────────────────────────────────────
    sm_sarsa = np.array([smooth(all_sarsa[i], window) for i in range(10)])
    sm_ql = np.array([smooth(all_ql[i], window) for i in range(10)])

    mean_s, std_s = sm_sarsa.mean(0), sm_sarsa.std(0)
    mean_q, std_q = sm_ql.mean(0), sm_ql.std(0)

    # ── Plot 1: Return vs Episodes ────────────────────────────────────────
    episodes = np.arange(len(mean_s)) + window

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episodes, mean_s, label='SARSA', color='#2196F3', linewidth=1.5)
    ax.fill_between(episodes, mean_s - std_s, mean_s + std_s,
                    alpha=0.2, color='#2196F3')
    ax.plot(episodes, mean_q, label='Q-Learning', color='#FF5722', linewidth=1.5)
    ax.fill_between(episodes, mean_q - std_q, mean_q + std_q,
                    alpha=0.2, color='#FF5722')
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Return (100-episode running avg)', fontsize=12)
    ax.set_title(
        'Return vs Episodes — SARSA vs Q-Learning (Mean ± 1 Std Dev, 10 Seeds)\n'
        f'(SARSA: α={sarsa_params["alpha"]}, '
        f'ε: {sarsa_params["eps_start"]}→{sarsa_params["eps_min"]} decay={sarsa_params["eps_decay"]}  |  '
        f'Q-Lrn: α={ql_params["alpha"]}, '
        f'ε: {ql_params["eps_start"]}→{ql_params["eps_min"]} decay={ql_params["eps_decay"]})',
        fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('part2b_return_vs_episodes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved → part2b_return_vs_episodes.png")

    # ── Plot 2: Return vs Timesteps ───────────────────────────────────────
    # Use the cumulative timestep at the end of each episode's smoothing window.
    # Average the cumulative timestep curves across seeds for a shared x-axis.
    ts_sarsa_smooth = np.array([all_sarsa_ts[i][window-1:] for i in range(10)])
    ts_ql_smooth = np.array([all_ql_ts[i][window-1:] for i in range(10)])
    mean_ts_s = ts_sarsa_smooth.mean(0)
    mean_ts_q = ts_ql_smooth.mean(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(mean_ts_s, mean_s, label='SARSA', color='#2196F3', linewidth=1.5)
    ax.fill_between(mean_ts_s, mean_s - std_s, mean_s + std_s,
                    alpha=0.2, color='#2196F3')
    ax.plot(mean_ts_q, mean_q, label='Q-Learning', color='#FF5722', linewidth=1.5)
    ax.fill_between(mean_ts_q, mean_q - std_q, mean_q + std_q,
                    alpha=0.2, color='#FF5722')
    ax.set_xlabel('Timesteps (cumulative)', fontsize=12)
    ax.set_ylabel('Return (100-episode running avg)', fontsize=12)
    ax.set_title(
        'Return vs Timesteps — SARSA vs Q-Learning (Mean ± 1 Std Dev, 10 Seeds)\n'
        f'(SARSA: α={sarsa_params["alpha"]}, '
        f'ε: {sarsa_params["eps_start"]}→{sarsa_params["eps_min"]} decay={sarsa_params["eps_decay"]}  |  '
        f'Q-Lrn: α={ql_params["alpha"]}, '
        f'ε: {ql_params["eps_start"]}→{ql_params["eps_min"]} decay={ql_params["eps_decay"]})',
        fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('part2b_return_vs_timesteps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved → part2b_return_vs_timesteps.png")

    # ── Numerical summary ─────────────────────────────────────────────────
    final_s = all_sarsa[:, -200:].mean(axis=1)
    final_q = all_ql[:, -200:].mean(axis=1)
    print(f"\n  Final performance (mean of last 200 eps across 10 seeds):")
    print(f"    SARSA      : {final_s.mean():.2f} ± {final_s.std():.2f}")
    print(f"    Q-Learning : {final_q.mean():.2f} ± {final_q.std():.2f}")

    print("\n--- Comparison ---")
    print("  Q-Learning (off-policy) updates towards the max Q-value of the next")
    print("  state, directly estimating the optimal action-value function. This")
    print("  usually leads to faster convergence and higher final returns.")
    print()
    print("  SARSA (on-policy) updates towards the Q-value of the action actually")
    print("  taken under ε-greedy, so it learns a policy that accounts for ongoing")
    print("  exploration. This makes it more conservative but also more stable in")
    print("  environments where exploratory actions can be costly.")
    print()
    print("  As ε decays towards 0, both algorithms converge towards the same")
    print("  optimal Q-values, but Q-Learning typically gets there faster because")
    print("  its target is always the greedy (max) value.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Acrobot-v1 | Bins={NUM_BINS} | States={NUM_BINS**6:,} | γ={GAMMA}")
    print(f"Training episodes = {NUM_EPISODES}\n")

    env = make_env()

    sarsa_params, ql_params = hyperparameter_tuning(env)
    multiseed_runs(env, sarsa_params, ql_params)

    env.close()
    print("\nDone. Plots:")
    print("  part2a_tuning_heatmap.png")
    print("  part2a_top3_return_vs_episodes.png")
    print("  part2a_top3_return_vs_timesteps.png")
    print("  part2b_return_vs_episodes.png")
    print("  part2b_return_vs_timesteps.png")
