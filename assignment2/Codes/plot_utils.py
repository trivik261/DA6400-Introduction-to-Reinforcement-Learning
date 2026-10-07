"""
Shared plotting utilities: interpolation, confidence intervals, tolerance intervals,
and performance distribution helpers.
"""

import numpy as np
from scipy import stats


def interpolate_to_common_timesteps(logs, key_x="episode_end_timesteps", key_y="episode_returns",
                                     num_points=200, max_timestep=None):
    """
    Interpolate per-seed episode data onto a common set of timesteps.
    Returns (common_timesteps, matrix of shape [num_seeds, num_points]).
    """
    if max_timestep is None:
        max_timestep = max(log[key_x][-1] for log in logs if log[key_x])
    common_ts = np.linspace(0, max_timestep, num_points)
    matrix = []
    for log in logs:
        xs = np.array(log[key_x])
        ys = np.array(log[key_y])
        if len(xs) == 0:
            matrix.append(np.full(num_points, np.nan))
            continue
        interp = np.interp(common_ts, xs, ys)
        matrix.append(interp)
    return common_ts, np.array(matrix)


def interpolate_eval(logs, num_points=200, max_timestep=None):
    """Interpolate eval returns to common timesteps."""
    return interpolate_to_common_timesteps(
        logs, key_x="eval_timesteps", key_y="eval_mean_returns",
        num_points=num_points, max_timestep=max_timestep,
    )


def mean_and_ci(matrix, confidence=0.95):
    """Compute mean and confidence interval across axis=0."""
    n = matrix.shape[0]
    mean = np.nanmean(matrix, axis=0)
    se = stats.sem(matrix, axis=0, nan_policy="omit")
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h


def compute_auc(returns, normalize=True):
    """Compute area under the return curve for a single seed. Optionally normalize by length."""
    arr = np.array(returns)
    auc = np.trapz(arr, dx=1)
    if normalize and len(arr) > 0:
        auc /= len(arr)
    return auc


def aggregate_performance(logs, key="eval_mean_returns"):
    """Get aggregate (AUC-based) performance per seed."""
    aucs = []
    for log in logs:
        vals = np.array(log[key])
        if len(vals) == 0:
            aucs.append(np.nan)
        else:
            aucs.append(np.mean(vals))
    return np.array(aucs)


def tolerance_interval(data, alpha=0.05, beta=0.9):
    """
    Compute (alpha, beta)-tolerance interval for normally distributed data.
    Returns (mean, lower, upper) where the interval covers at least beta fraction
    of the population with 1-alpha confidence.

    Uses the k-factor approach: mean ± k * std
    k = z_{(1+beta)/2} * sqrt(1 + 1/n) * correction
    More precisely, uses scipy's tolerance factor.
    """
    n = len(data)
    if n < 2:
        m = np.mean(data)
        return m, m, m

    mean = np.mean(data)
    std = np.std(data, ddof=1)

    z_beta = stats.norm.ppf((1 + beta) / 2)
    chi2_val = stats.chi2.ppf(1 - alpha, n - 1)
    k = z_beta * np.sqrt((n * (n - 1)) / chi2_val) * np.sqrt(1.0 / n + 1)

    lower = mean - k * std
    upper = mean + k * std
    return mean, lower, upper


def tolerance_interval_per_timestep(matrix, alpha=0.05, beta=0.9):
    """Compute tolerance intervals at each timestep."""
    means, lowers, uppers = [], [], []
    for i in range(matrix.shape[1]):
        col = matrix[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            means.append(np.nanmean(col))
            lowers.append(np.nanmean(col))
            uppers.append(np.nanmean(col))
        else:
            m, lo, hi = tolerance_interval(valid, alpha, beta)
            means.append(m)
            lowers.append(lo)
            uppers.append(hi)
    return np.array(means), np.array(lowers), np.array(uppers)
