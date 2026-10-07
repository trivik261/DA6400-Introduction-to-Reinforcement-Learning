"""
Q4d Plots: Sensitivity analysis for mini-batch size and target network refresh rate.
Uses logs from run_q4d.py.

Generates:
  - Batch size sensitivity: side-by-side (rho=1, rho=4) + overlay
  - Target refresh sensitivity: side-by-side (rho=1, rho=4) + overlay
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from train import load_all_logs
from plot_utils import aggregate_performance

LOG_DIR = "results/q4d"
OUT_DIR = "results/plots/q4d"

BATCH_SIZES = [16, 32, 64, 128, 256]
TARGET_FREQS = [100, 250, 500, 1000, 2000]
RHOS = [1, 4]
RHO_COLORS = {1: "tab:blue", 4: "tab:green"}


def _load_sensitivity(param_name, param_values, rho):
    """Load aggregate performance for a sweep of one hyperparameter."""
    agg_means, agg_cis = [], []
    for val in param_values:
        tag = f"q4d_rho{rho}_{param_name}{val}"
        logs = load_all_logs(LOG_DIR, tag)
        if not logs:
            agg_means.append(np.nan)
            agg_cis.append(0)
            continue
        aucs = aggregate_performance(logs, key="eval_mean_returns")
        m = np.nanmean(aucs)
        se = sp_stats.sem(aucs, nan_policy="omit")
        h = se * sp_stats.t.ppf(0.975, max(len(aucs) - 1, 1))
        agg_means.append(m)
        agg_cis.append(h)
    return agg_means, agg_cis


def plot_q4d():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Batch size: side-by-side panels ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, rho in enumerate(RHOS):
        ax = axes[idx]
        means, cis = _load_sensitivity("bs", BATCH_SIZES, rho)
        ax.errorbar(BATCH_SIZES, means, yerr=cis, marker="o", capsize=4,
                     color=RHO_COLORS[rho], linewidth=2)
        ax.set_xlabel("Mini-batch Size")
        ax.set_ylabel("Aggregate Performance")
        ax.set_title(rf"Batch Size Sensitivity ($\rho={rho}$)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCH_SIZES)
        ax.set_xticklabels([str(b) for b in BATCH_SIZES])
        ax.grid(True, alpha=0.3)
    fig.suptitle("Q4d: Mini-batch Size Sensitivity", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q4d_sensitivity_batch_size.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/q4d_sensitivity_batch_size.png")

    # ---- Target refresh: side-by-side panels ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, rho in enumerate(RHOS):
        ax = axes[idx]
        means, cis = _load_sensitivity("tf", TARGET_FREQS, rho)
        ax.errorbar(TARGET_FREQS, means, yerr=cis, marker="s", capsize=4,
                     color=RHO_COLORS[rho], linewidth=2)
        ax.set_xlabel("Target Network Refresh Rate (steps)")
        ax.set_ylabel("Aggregate Performance")
        ax.set_title(rf"Target Refresh Sensitivity ($\rho={rho}$)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(TARGET_FREQS)
        ax.set_xticklabels([str(t) for t in TARGET_FREQS])
        ax.grid(True, alpha=0.3)
    fig.suptitle("Q4d: Target Network Refresh Rate Sensitivity", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q4d_sensitivity_target_freq.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/q4d_sensitivity_target_freq.png")

    # ---- Batch size: overlay rho=1 vs rho=4 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for rho in RHOS:
        means, cis = _load_sensitivity("bs", BATCH_SIZES, rho)
        ax.errorbar(BATCH_SIZES, means, yerr=cis, marker="o", capsize=4,
                     color=RHO_COLORS[rho], linewidth=2, label=rf"$\rho={rho}$")
    ax.set_xlabel("Mini-batch Size")
    ax.set_ylabel("Aggregate Performance")
    ax.set_title("Q4d: Batch Size Sensitivity — rho=1 vs rho=4")
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCH_SIZES)
    ax.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q4d_batch_size_overlay.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/q4d_batch_size_overlay.png")

    # ---- Target refresh: overlay rho=1 vs rho=4 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for rho in RHOS:
        means, cis = _load_sensitivity("tf", TARGET_FREQS, rho)
        ax.errorbar(TARGET_FREQS, means, yerr=cis, marker="s", capsize=4,
                     color=RHO_COLORS[rho], linewidth=2, label=rf"$\rho={rho}$")
    ax.set_xlabel("Target Network Refresh Rate (steps)")
    ax.set_ylabel("Aggregate Performance")
    ax.set_title("Q4d: Target Refresh Rate Sensitivity — rho=1 vs rho=4")
    ax.set_xscale("log", base=2)
    ax.set_xticks(TARGET_FREQS)
    ax.set_xticklabels([str(t) for t in TARGET_FREQS])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q4d_target_freq_overlay.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/q4d_target_freq_overlay.png")


if __name__ == "__main__":
    plot_q4d()
