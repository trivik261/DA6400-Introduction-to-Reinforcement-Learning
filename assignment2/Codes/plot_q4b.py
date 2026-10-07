"""
Q4b Plots: Distribution of aggregate performance for each replay factor rho.
Uses logs from run_q4a.py (no separate training needed).

Generates:
  - Per-rho histograms with KDE (2x2 grid)
  - Overlay KDE plot comparing all rho values
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

from train import load_all_logs
from plot_utils import aggregate_performance

LOG_DIR = "results/q4a"
OUT_DIR = "results/plots/q4b"
RHOS = [1, 2, 4, 8]
RHO_COLORS = {1: "tab:blue", 2: "tab:orange", 4: "tab:green", 8: "tab:red"}


def plot_q4b():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_aucs = {}
    for rho in RHOS:
        tag = f"q4a_rho{rho}"
        logs = load_all_logs(LOG_DIR, tag)
        if not logs:
            print(f"No logs found for {tag} in {LOG_DIR}. Run run_q4a.py first.")
            continue
        aucs = aggregate_performance(logs, key="eval_mean_returns")
        all_aucs[rho] = aucs
        print(f"rho={rho}: loaded {len(logs)} seeds, mean agg={np.mean(aucs):.1f}, std={np.std(aucs):.1f}")

    if not all_aucs:
        print("No data to plot. Exiting.")
        return

    # ---- Plot 1: Individual histograms per rho (2x2 grid) ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, rho in enumerate(RHOS):
        ax = axes[i]
        if rho not in all_aucs:
            ax.set_title(rf"$\rho={rho}$ — no data")
            continue
        aucs = all_aucs[rho]
        sns.histplot(aucs, kde=True, ax=ax, color=RHO_COLORS[rho], bins=8, stat="density")
        ax.axvline(np.mean(aucs), color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean = {np.mean(aucs):.1f}")
        ax.axvline(np.median(aucs), color="gray", linestyle=":", linewidth=1.5,
                   label=f"Median = {np.median(aucs):.1f}")

        skewness = sp_stats.skew(aucs)
        ax.set_title(rf"$\rho={rho}$ (n={len(aucs)}, skew={skewness:.2f})")
        ax.set_xlabel("Aggregate Performance (Mean Eval Return)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Q4b: Distribution of Aggregate Performance per Replay Factor", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q4b_distributions_grid.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/q4b_distributions_grid.png")

    # ---- Plot 2: Overlay KDE for all rho values ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for rho in RHOS:
        if rho not in all_aucs:
            continue
        sns.kdeplot(all_aucs[rho], ax=ax, label=rf"$\rho={rho}$",
                    color=RHO_COLORS[rho], fill=True, alpha=0.2, linewidth=2)
    ax.set_xlabel("Aggregate Performance (Mean Eval Return)")
    ax.set_ylabel("Density")
    ax.set_title("Q4b: Overlay of Performance Distributions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q4b_distributions_overlay.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/q4b_distributions_overlay.png")

    # ---- Print summary statistics for the report ----
    print("\n--- Q4b Summary Statistics ---")
    for rho in RHOS:
        if rho not in all_aucs:
            continue
        aucs = all_aucs[rho]
        print(f"  rho={rho}: mean={np.mean(aucs):.2f}, median={np.median(aucs):.2f}, "
              f"std={np.std(aucs, ddof=1):.2f}, skew={sp_stats.skew(aucs):.2f}, "
              f"kurtosis={sp_stats.kurtosis(aucs):.2f}, "
              f"shapiro_p={sp_stats.shapiro(aucs).pvalue:.4f}")


if __name__ == "__main__":
    plot_q4b()
