"""
Q4c Plots: Tolerance intervals (alpha=0.05, beta=0.9) for each replay factor rho.
Uses logs from run_q4a.py (no separate training needed).

Generates:
  - Combined tolerance interval plot (all rho on one axis)
  - 2x2 grid comparing confidence intervals vs tolerance intervals per rho
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from train import load_all_logs
from plot_utils import interpolate_eval, mean_and_ci, tolerance_interval_per_timestep

LOG_DIR = "results/q4a"
OUT_DIR = "results/plots/q4c"
RHOS = [1, 2, 4, 8]
RHO_COLORS = {1: "tab:blue", 2: "tab:orange", 4: "tab:green", 8: "tab:red"}


def plot_q4c():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Plot 1: Combined tolerance intervals (all rho, one axis) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for rho in RHOS:
        tag = f"q4a_rho{rho}"
        logs = load_all_logs(LOG_DIR, tag)
        if not logs:
            print(f"No logs found for {tag} in {LOG_DIR}. Run run_q4a.py first.")
            continue
        print(f"rho={rho}: loaded {len(logs)} seeds for tolerance intervals")
        max_ts = max(log["eval_timesteps"][-1] for log in logs if log["eval_timesteps"])
        eval_ts, eval_matrix = interpolate_eval(logs, max_timestep=max_ts)
        means, lowers, uppers = tolerance_interval_per_timestep(eval_matrix, alpha=0.05, beta=0.9)
        ax.plot(eval_ts, means, label=rf"$\rho={rho}$", color=RHO_COLORS[rho])
        ax.fill_between(eval_ts, lowers, uppers, alpha=0.1, color=RHO_COLORS[rho])

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Eval Return")
    ax.set_title(r"Q4c: Tolerance Intervals ($\alpha=0.05, \beta=0.9$) for Different $\rho$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q4c_tolerance_intervals.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/q4c_tolerance_intervals.png")

    # ---- Plot 2: CI vs TI side-by-side per rho (2x2 grid) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for i, rho in enumerate(RHOS):
        tag = f"q4a_rho{rho}"
        logs = load_all_logs(LOG_DIR, tag)
        ax = axes[i]
        if not logs:
            ax.set_title(rf"$\rho={rho}$ — no data")
            continue
        max_ts = max(log["eval_timesteps"][-1] for log in logs if log["eval_timesteps"])
        eval_ts, eval_matrix = interpolate_eval(logs, max_timestep=max_ts)

        mean_ci, lo_ci, hi_ci = mean_and_ci(eval_matrix)
        mean_ti, lo_ti, hi_ti = tolerance_interval_per_timestep(eval_matrix, 0.05, 0.9)

        ax.plot(eval_ts, mean_ci, color=RHO_COLORS[rho], linewidth=1.5)
        ax.fill_between(eval_ts, lo_ci, hi_ci, alpha=0.35, color=RHO_COLORS[rho], label="95% CI")
        ax.fill_between(eval_ts, lo_ti, hi_ti, alpha=0.12, color=RHO_COLORS[rho],
                        label=r"TI ($\alpha$=0.05, $\beta$=0.9)")
        ax.set_title(rf"$\rho={rho}$")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Eval Return")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Q4c: Confidence Intervals vs Tolerance Intervals", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q4c_ci_vs_ti.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/q4c_ci_vs_ti.png")

    # ---- Plot 3: Only TI per rho (individual panels) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for i, rho in enumerate(RHOS):
        tag = f"q4a_rho{rho}"
        logs = load_all_logs(LOG_DIR, tag)
        ax = axes[i]
        if not logs:
            ax.set_title(rf"$\rho={rho}$ — no data")
            continue
        max_ts = max(log["eval_timesteps"][-1] for log in logs if log["eval_timesteps"])
        eval_ts, eval_matrix = interpolate_eval(logs, max_timestep=max_ts)

        means, lowers, uppers = tolerance_interval_per_timestep(eval_matrix, 0.05, 0.9)

        ax.plot(eval_ts, means, color=RHO_COLORS[rho], linewidth=1.5, label="Mean")
        ax.fill_between(eval_ts, lowers, uppers, alpha=0.2, color=RHO_COLORS[rho],
                        label=r"TI ($\alpha$=0.05, $\beta$=0.9)")

        final_width = uppers[-1] - lowers[-1]
        ax.set_title(rf"$\rho={rho}$ | Final TI width = {final_width:.0f}")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Eval Return")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(r"Q4c: Tolerance Intervals ($\alpha=0.05, \beta=0.9$) per $\rho$", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q4c_ti_per_rho.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/q4c_ti_per_rho.png")


if __name__ == "__main__":
    plot_q4c()
