"""
Q4 Master plotting script: runs all Q4 sub-part plots.
  (a) Replay factor comparison — learning curves with CI
  (b) Distribution of aggregate performance
  (c) Tolerance intervals
  (d) Sensitivity analysis

Each sub-part also has its own standalone script:
  plot_q4b.py, plot_q4c.py, plot_q4d.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from train import load_all_logs
from plot_utils import (
    interpolate_to_common_timesteps,
    interpolate_eval,
    mean_and_ci,
    aggregate_performance,
)

OUT_DIR = "results/plots/q4a"
RHOS = [1, 2, 4, 8]
RHO_COLORS = {1: "tab:blue", 2: "tab:orange", 4: "tab:green", 8: "tab:red"}


def plot_q4a():
    """Q4a: Learning curves with confidence intervals for each rho."""
    os.makedirs(OUT_DIR, exist_ok=True)
    log_dir = "results/q4a"

    # --- Return vs timesteps ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for rho in RHOS:
        tag = f"q4a_rho{rho}"
        logs = load_all_logs(log_dir, tag)
        if not logs:
            print(f"No logs for {tag}")
            continue
        print(f"Loaded {len(logs)} logs for rho={rho}")
        max_ts = max(log["episode_end_timesteps"][-1] for log in logs if log["episode_end_timesteps"])
        common_ts, matrix = interpolate_to_common_timesteps(logs, max_timestep=max_ts)
        mean, lo, hi = mean_and_ci(matrix)
        ax.plot(common_ts, mean, label=rf"$\rho={rho}$", color=RHO_COLORS[rho])
        ax.fill_between(common_ts, lo, hi, alpha=0.12, color=RHO_COLORS[rho])

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Return")
    ax.set_title(r"Q4a: DQN Return vs Timesteps for Different $\rho$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q4a_rho_comparison_timesteps.png"), dpi=150)
    plt.close(fig)

    # --- Eval return vs timesteps ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for rho in RHOS:
        tag = f"q4a_rho{rho}"
        logs = load_all_logs(log_dir, tag)
        if not logs:
            continue
        max_ts = max(log["eval_timesteps"][-1] for log in logs if log["eval_timesteps"])
        eval_ts, eval_matrix = interpolate_eval(logs, max_timestep=max_ts)
        mean, lo, hi = mean_and_ci(eval_matrix)
        ax.plot(eval_ts, mean, label=rf"$\rho={rho}$", color=RHO_COLORS[rho])
        ax.fill_between(eval_ts, lo, hi, alpha=0.12, color=RHO_COLORS[rho])

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Eval Return")
    ax.set_title(r"Q4a: DQN Eval Performance for Different $\rho$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q4a_rho_comparison_eval.png"), dpi=150)
    plt.close(fig)

    # --- Aggregate bar chart (Figure 3b style) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    agg_means, agg_cis = [], []
    for rho in RHOS:
        tag = f"q4a_rho{rho}"
        logs = load_all_logs(log_dir, tag)
        if not logs:
            agg_means.append(0)
            agg_cis.append(0)
            continue
        aucs = aggregate_performance(logs, key="eval_mean_returns")
        m = np.nanmean(aucs)
        se = sp_stats.sem(aucs, nan_policy="omit")
        h = se * sp_stats.t.ppf(0.975, len(aucs) - 1)
        agg_means.append(m)
        agg_cis.append(h)

    x = np.arange(len(RHOS))
    ax.bar(x, agg_means, yerr=agg_cis, capsize=5,
           color=[RHO_COLORS[r] for r in RHOS], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\rho={r}$" for r in RHOS])
    ax.set_ylabel("Aggregate Performance (Mean Eval Return)")
    ax.set_title(r"Q4a: Aggregate Performance for Different $\rho$")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q4a_aggregate_bar.png"), dpi=150)
    plt.close(fig)

    print(f"Q4a plots saved to {OUT_DIR}/")


def main():
    print("=== Q4a: Replay factor comparison ===")
    plot_q4a()

    print("\n=== Q4b: Performance distributions ===")
    from plot_q4b import plot_q4b
    plot_q4b()

    print("\n=== Q4c: Tolerance intervals ===")
    from plot_q4c import plot_q4c
    plot_q4c()

    print("\n=== Q4d: Sensitivity analysis ===")
    from plot_q4d import plot_q4d
    plot_q4d()

    print("\nAll Q4 plots generated.")


if __name__ == "__main__":
    main()
