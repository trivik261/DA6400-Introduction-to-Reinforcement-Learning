"""
Q5 Plots: PER vs Uniform Replay for different replay factors.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

from train import load_all_logs
from plot_utils import interpolate_eval, mean_and_ci, aggregate_performance

LOG_DIR = "results/q5"
OUT_DIR = "results/plots/q5"
RHOS = [1, 2, 4, 8]
COLORS_PER = {1: "tab:blue", 2: "tab:orange", 4: "tab:green", 8: "tab:red"}
COLORS_UNI = {1: "lightblue", 2: "moccasin", 4: "lightgreen", 8: "lightcoral"}


def plot_q5():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Learning curves: PER vs Uniform for each rho ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, rho in enumerate(RHOS):
        ax = axes[i]
        for mode, ls, alpha_fill in [("per", "-", 0.15), ("uniform", "--", 0.08)]:
            tag = f"q5_{mode}_rho{rho}"
            logs = load_all_logs(LOG_DIR, tag)
            if not logs:
                continue
            max_ts = max(log["eval_timesteps"][-1] for log in logs if log["eval_timesteps"])
            eval_ts, eval_matrix = interpolate_eval(logs, max_timestep=max_ts)
            mean, lo, hi = mean_and_ci(eval_matrix)
            label = f"{mode.upper()}"
            color = COLORS_PER[rho] if mode == "per" else "gray"
            ax.plot(eval_ts, mean, label=label, color=color, linestyle=ls)
            ax.fill_between(eval_ts, lo, hi, alpha=alpha_fill, color=color)

        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Eval Return")
        ax.set_title(rf"$\rho={rho}$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Q5: PER vs Uniform Replay for Different Replay Factors", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q5_per_vs_uniform_curves.png"), dpi=150)
    plt.close(fig)

    # --- Aggregate bar chart: PER vs Uniform ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(RHOS))
    width = 0.35

    for offset, mode, color_base in [(-width / 2, "per", "tab:blue"), (width / 2, "uniform", "gray")]:
        means, cis = [], []
        for rho in RHOS:
            tag = f"q5_{mode}_rho{rho}"
            logs = load_all_logs(LOG_DIR, tag)
            if not logs:
                means.append(np.nan)
                cis.append(0)
                continue
            aucs = aggregate_performance(logs, key="eval_mean_returns")
            m = np.nanmean(aucs)
            se = sp_stats.sem(aucs, nan_policy="omit")
            h = se * sp_stats.t.ppf(0.975, max(len(aucs) - 1, 1))
            means.append(m)
            cis.append(h)
        ax.bar(x + offset, means, width, yerr=cis, capsize=4, label=mode.upper(),
               color=color_base, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\rho={r}$" for r in RHOS])
    ax.set_ylabel("Aggregate Performance")
    ax.set_title("Q5: PER vs Uniform — Aggregate Performance by Replay Factor")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q5_per_vs_uniform_aggregate.png"), dpi=150)
    plt.close(fig)

    # --- Overlay: all PER curves together, all Uniform curves together ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mode, title in zip(axes, ["per", "uniform"], ["PER", "Uniform"]):
        for rho in RHOS:
            tag = f"q5_{mode}_rho{rho}"
            logs = load_all_logs(LOG_DIR, tag)
            if not logs:
                continue
            max_ts = max(log["eval_timesteps"][-1] for log in logs if log["eval_timesteps"])
            eval_ts, eval_matrix = interpolate_eval(logs, max_timestep=max_ts)
            mean, lo, hi = mean_and_ci(eval_matrix)
            ax.plot(eval_ts, mean, label=rf"$\rho={rho}$", color=COLORS_PER[rho])
            ax.fill_between(eval_ts, lo, hi, alpha=0.12, color=COLORS_PER[rho])
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Eval Return")
        ax.set_title(f"{title} — Replay Factor Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Q5: Effect of Replay Factor under PER vs Uniform", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q5_rho_comparison_per_uniform.png"), dpi=150)
    plt.close(fig)

    print(f"Q5 plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    plot_q5()
