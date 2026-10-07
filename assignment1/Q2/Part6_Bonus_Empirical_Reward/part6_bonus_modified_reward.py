"""
Part 6 — Bonus: Empirical evidence for the modified reward function
=====================================================================
Reward:  r = ηh/2  +  sign(-1 + ηh) * (2 - ηh) / 2
where h = -cos(θ1) - cos(θ1 + θ2) is the tip height (∈ [-2, 2]).

We run Q-Learning sweeping THREE axes:
  1. η ∈ {0, 0.5, 1, 2, 5}  (reward shaping strength)
  2. ε-decay ∈ {0.99, 0.995, 0.999}  (exploration schedule)
  3. Environment difficulty: default vs hard (heavier links + torque noise)

η = 0 is the baseline (original Acrobot reward: -1 per step).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import multiprocessing as mp

from utils import (discretise, epsilon_greedy_action, smooth,
                   GAMMA, NUM_BINS, N_ACTIONS, BIN_EDGES)

N_WORKERS = min(mp.cpu_count(), 10)

# ─── Experiment parameters ────────────────────────────────────────────────
NUM_EPISODES   = 3000
ALPHA          = 0.1
EPS_START      = 1.0
EPS_MIN        = 0.01
MAX_EP_STEPS   = 500
SEEDS          = list(range(5))
EVAL_EPISODES  = 50

ETAS           = [0, 0.5, 1, 2, 5]
EPS_DECAYS     = [0.99, 0.995, 0.999]

ENV_CONFIGS = {
    'default': {},
    'hard': {
        'LINK_MASS_1': 1.5,
        'LINK_MASS_2': 1.5,
        'torque_noise_max': 0.1,
    },
}


# ─── Modified reward function ─────────────────────────────────────────────
def compute_height(obs):
    """Compute tip height h = -cos(θ1) - cos(θ1 + θ2) from observation."""
    cos_t1, sin_t1 = obs[0], obs[1]
    cos_t2, sin_t2 = obs[2], obs[3]
    cos_t1t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
    return -cos_t1 - cos_t1t2


def modified_reward(h, eta):
    """r = ηh/2 + sign(-1 + ηh) * (2 - ηh) / 2"""
    eta_h = eta * h
    return eta_h / 2.0 + np.sign(-1.0 + eta_h) * (2.0 - eta_h) / 2.0


# ─── Environment factory ─────────────────────────────────────────────────
def make_env(env_config, max_steps=500):
    """Create Acrobot-v1 and override internal physics parameters."""
    env = gym.make("Acrobot-v1", max_episode_steps=max_steps)
    for attr, val in env_config.items():
        setattr(env.unwrapped, attr, val)
    return env


# ─── Q-Learning with modified reward ─────────────────────────────────────
def q_learning_modified(num_episodes, alpha, gamma,
                        eps_start, eps_min, eps_decay,
                        eta, seed, env_config, max_steps=500):
    """
    Q-Learning with the modified reward function.
    When eta=0, falls back to the standard Acrobot reward (-1 per step).
    """
    env = make_env(env_config, max_steps)
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(n_actions))
    epsilon = eps_start

    original_returns = []
    modified_returns = []
    max_heights = []
    timesteps_per_ep = []
    goal_reached = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1e6)))
        state = discretise(obs)
        total_orig = 0.0
        total_mod  = 0.0
        ep_max_h   = -2.0
        steps      = 0
        done       = False

        while not done:
            action = epsilon_greedy_action(Q, state, n_actions, epsilon, rng)
            next_obs, orig_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretise(next_obs)

            h = compute_height(next_obs)
            ep_max_h = max(ep_max_h, h)

            if eta == 0:
                reward = orig_reward
            else:
                reward = modified_reward(h, eta)

            td_target = reward + gamma * np.max(Q[next_state]) * (1 - int(terminated))
            Q[state][action] += alpha * (td_target - Q[state][action])

            state = next_state
            total_orig += orig_reward
            total_mod  += reward
            steps += 1

        original_returns.append(total_orig)
        modified_returns.append(total_mod)
        max_heights.append(ep_max_h)
        timesteps_per_ep.append(steps)
        goal_reached.append(terminated and not truncated)
        epsilon = max(eps_min, epsilon * eps_decay)

    env.close()
    return Q, {
        'original_returns': original_returns,
        'modified_returns': modified_returns,
        'max_heights': max_heights,
        'timesteps_per_ep': timesteps_per_ep,
        'goal_reached': goal_reached,
    }


def evaluate_greedy(Q, env_config, num_episodes=50, seed=9999, max_steps=500):
    """Greedy evaluation: returns original returns and heights."""
    env = make_env(env_config, max_steps)
    rng = np.random.default_rng(seed)
    returns, heights, goals = [], [], []

    for _ in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1e6)))
        state = discretise(obs)
        done, ep_ret, ep_max_h = False, 0.0, -2.0
        terminated_flag = False
        while not done:
            action = int(np.argmax(Q[state]))
            obs, reward, terminated, truncated, _ = env.step(action)
            state = discretise(obs)
            done = terminated or truncated
            ep_ret += reward
            h = compute_height(obs)
            ep_max_h = max(ep_max_h, h)
            if terminated:
                terminated_flag = True
        returns.append(ep_ret)
        heights.append(ep_max_h)
        goals.append(terminated_flag)
    env.close()
    return np.array(returns), np.array(heights), np.array(goals)


# ─── Analysis: Plot the reward function itself ───────────────────────────
def plot_reward_function():
    """Visualise r(h) for each η value to build intuition."""
    fig, ax = plt.subplots(figsize=(10, 5))
    h_vals = np.linspace(-2, 2, 500)
    colors = {0: '#888888', 0.5: '#2196F3', 1: '#4CAF50', 2: '#FF9800', 5: '#F44336'}

    for eta in ETAS:
        if eta == 0:
            r_vals = np.full_like(h_vals, -1.0)
            label = 'η=0 (default: −1)'
        else:
            r_vals = np.array([modified_reward(h, eta) for h in h_vals])
            label = f'η={eta}'
        ax.plot(h_vals, r_vals, label=label, color=colors[eta], linewidth=2)

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.axvline(1.0, color='gray', linewidth=0.8, linestyle=':', label='Goal height (h=1)')
    ax.set_xlabel('Height h', fontsize=12)
    ax.set_ylabel('Reward r(h)', fontsize=12)
    ax.set_title('Modified Reward Function r(h) for Different η Values', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('part6_reward_function.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved → part6_reward_function.png")


# ─── Single-run worker (top-level for pickling) ──────────────────────────
def _run_one(args):
    """Train + evaluate one (env_name, eps_decay, eta, seed) combo.
    Returns (key, seed, data_dict) — no Q-table (unpicklable & large)."""
    env_name, env_config, eps_decay, eta, seed = args
    Q, data = q_learning_modified(
        NUM_EPISODES, ALPHA, GAMMA,
        EPS_START, EPS_MIN, eps_decay,
        eta=eta, seed=seed, env_config=env_config)

    greedy_ret, greedy_h, greedy_g = evaluate_greedy(
        Q, env_config, EVAL_EPISODES, seed=seed + 100)

    data['greedy_returns'] = greedy_ret.tolist()
    data['greedy_heights'] = greedy_h.tolist()
    data['greedy_goals'] = greedy_g.tolist()

    return ((env_name, eps_decay, eta), seed, data)


# ─── Main experiment ─────────────────────────────────────────────────────
def run_experiments():
    jobs = []
    for env_name, env_config in ENV_CONFIGS.items():
        for eps_decay in EPS_DECAYS:
            for eta in ETAS:
                for s in SEEDS:
                    jobs.append((env_name, env_config, eps_decay, eta, s))

    total = len(jobs)
    print("=" * 70)
    print("PART 6 — Bonus: Modified Reward — Full Ablation Study")
    print(f"  α={ALPHA}, γ={GAMMA}, ε: {EPS_START}→{EPS_MIN}")
    print(f"  η values:       {ETAS}")
    print(f"  ε-decay values: {EPS_DECAYS}")
    print(f"  Env configs:    {list(ENV_CONFIGS.keys())}")
    print(f"  Seeds: {SEEDS}, Episodes: {NUM_EPISODES}")
    print(f"  Total runs: {total}  |  Workers: {N_WORKERS}")
    print("=" * 70)

    all_results = {}
    t0 = time.time()

    with mp.Pool(N_WORKERS) as pool:
        for i, (key, seed, data) in enumerate(pool.imap_unordered(_run_one, jobs)):
            elapsed = time.time() - t0
            eta_str = f"η={key[2]}" if key[2] > 0 else "η=0"
            print(f"  [{i+1}/{total}] env={key[0]} ε-d={key[1]} {eta_str} "
                  f"seed={seed}  ({elapsed:.0f}s elapsed)", flush=True)

            if key not in all_results:
                all_results[key] = []
            all_results[key].append(data)

    wall = time.time() - t0
    print(f"\nAll {total} runs done in {wall:.0f}s "
          f"(~{wall/total:.1f}s/run effective, {N_WORKERS} workers)")

    return all_results


# ─── Summary statistics ──────────────────────────────────────────────────
def _summarise_key(seed_data):
    """Return a dict of aggregate statistics for one (env, eps_decay, eta) combo."""
    orig_last200 = np.array([np.mean(d['original_returns'][-200:]) for d in seed_data])
    goal_last200 = np.array([np.mean(d['goal_reached'][-200:]) for d in seed_data])
    ep_len_last200 = np.array([np.mean(d['timesteps_per_ep'][-200:]) for d in seed_data])
    greedy_ret = np.array([np.mean(np.asarray(d['greedy_returns'])) for d in seed_data])
    greedy_goal = np.array([np.mean(np.asarray(d['greedy_goals'])) for d in seed_data])
    return {
        'orig_ret':   (orig_last200.mean(), orig_last200.std()),
        'goal_rate':  (goal_last200.mean(), goal_last200.std()),
        'ep_len':     (ep_len_last200.mean(), ep_len_last200.std()),
        'greedy_ret': (greedy_ret.mean(), greedy_ret.std()),
        'greedy_goal':(greedy_goal.mean(), greedy_goal.std()),
    }


def print_summary(all_results):
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)

    for env_name in ENV_CONFIGS:
        for eps_decay in EPS_DECAYS:
            print(f"\n{'─'*60}")
            print(f"  Env: {env_name} | ε-decay: {eps_decay}")
            print(f"{'─'*60}")
            for eta in ETAS:
                key = (env_name, eps_decay, eta)
                s = _summarise_key(all_results[key])
                label = f"η={eta}" if eta > 0 else "η=0 (default)"
                print(f"    {label}:")
                print(f"      Online(last 200): return={s['orig_ret'][0]:.1f}±{s['orig_ret'][1]:.1f}"
                      f"  goal={s['goal_rate'][0]:.1%}±{s['goal_rate'][1]:.1%}"
                      f"  len={s['ep_len'][0]:.0f}±{s['ep_len'][1]:.0f}")
                print(f"      Greedy (ε=0):     return={s['greedy_ret'][0]:.1f}±{s['greedy_ret'][1]:.1f}"
                      f"  goal={s['greedy_goal'][0]:.1%}±{s['greedy_goal'][1]:.1%}")


# ─── Plotting ─────────────────────────────────────────────────────────────
ETA_COLORS = {0: '#888888', 0.5: '#2196F3', 1: '#4CAF50', 2: '#FF9800', 5: '#F44336'}
DECAY_STYLES = {0.99: '-', 0.995: '--', 0.999: ':'}


def _plot_learning_curve(ax, all_results, env_name, eps_decay, metric_key,
                         ylabel, window=100):
    """Plot one learning-curve panel for a given (env, eps_decay) combo."""
    for eta in ETAS:
        key = (env_name, eps_decay, eta)
        seed_data = all_results[key]
        raw = np.array([d[metric_key] for d in seed_data], dtype=float)
        sm = np.array([smooth(raw[i], window) for i in range(len(seed_data))])
        eps = np.arange(sm.shape[1]) + window
        mean, std = sm.mean(0), sm.std(0)
        label = f'η={eta}' if eta > 0 else 'η=0 (default)'
        ax.plot(eps, mean, label=label, color=ETA_COLORS[eta], linewidth=1.5)
        ax.fill_between(eps, mean - std, mean + std, alpha=0.12, color=ETA_COLORS[eta])
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_learning_curves(all_results):
    """One figure per env config: 3 rows (ε-decay) × 4 columns (metric)."""
    metrics = [
        ('original_returns', 'Original Return'),
        ('timesteps_per_ep', 'Episode Length'),
        ('max_heights', 'Max Height'),
        ('goal_reached', 'Goal Rate'),
    ]

    for env_name in ENV_CONFIGS:
        fig, axes = plt.subplots(len(EPS_DECAYS), len(metrics),
                                 figsize=(20, 4 * len(EPS_DECAYS)),
                                 squeeze=False)

        for row, eps_decay in enumerate(EPS_DECAYS):
            for col, (mkey, mlabel) in enumerate(metrics):
                ax = axes[row, col]
                _plot_learning_curve(ax, all_results, env_name, eps_decay,
                                     mkey, f'{mlabel} (100-ep avg)')
                if mkey == 'max_heights':
                    ax.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
                if mkey == 'goal_reached':
                    ax.set_ylim(-0.05, 1.05)
                if row == 0:
                    ax.set_title(mlabel, fontsize=11, fontweight='bold')

            axes[row, 0].annotate(
                f'ε-decay={eps_decay}', xy=(0, 0.5),
                xytext=(-0.25, 0.5), xycoords='axes fraction',
                textcoords='axes fraction', fontsize=11, fontweight='bold',
                rotation=90, va='center', ha='center')

        fig.suptitle(f'Learning Curves — env={env_name} | α={ALPHA}, γ={GAMMA}',
                     fontsize=14, y=1.01)
        plt.tight_layout()
        fname = f'part6_curves_{env_name}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved → {fname}")


def plot_greedy_comparison(all_results):
    """Bar chart: greedy return & goal rate for each (env, eps_decay) combo."""
    n_env = len(ENV_CONFIGS)
    n_decay = len(EPS_DECAYS)

    fig, axes = plt.subplots(n_env, n_decay, figsize=(6 * n_decay, 5 * n_env),
                             squeeze=False)

    x = np.arange(len(ETAS))
    eta_labels = [f'η={e}' if e > 0 else 'η=0\n(def)' for e in ETAS]
    bar_colors = [ETA_COLORS[e] for e in ETAS]

    for row, env_name in enumerate(ENV_CONFIGS):
        for col, eps_decay in enumerate(EPS_DECAYS):
            ax = axes[row, col]
            means, stds = [], []
            for eta in ETAS:
                key = (env_name, eps_decay, eta)
                s = _summarise_key(all_results[key])
                means.append(s['greedy_ret'][0])
                stds.append(s['greedy_ret'][1])

            bars = ax.bar(x, means, yerr=stds, capsize=6, color=bar_colors,
                          edgecolor='black', linewidth=0.7, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(eta_labels, fontsize=9)
            ax.set_ylabel('Greedy Return', fontsize=10)
            ax.set_title(f'env={env_name} | ε-d={eps_decay}', fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + abs(bar.get_height()) * 0.02 + 1,
                        f'{val:.0f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

    fig.suptitle('Greedy Return After Training (ε=0)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('part6_greedy_return.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved → part6_greedy_return.png")

    fig, axes = plt.subplots(n_env, n_decay, figsize=(6 * n_decay, 5 * n_env),
                             squeeze=False)

    for row, env_name in enumerate(ENV_CONFIGS):
        for col, eps_decay in enumerate(EPS_DECAYS):
            ax = axes[row, col]
            means, stds = [], []
            for eta in ETAS:
                key = (env_name, eps_decay, eta)
                s = _summarise_key(all_results[key])
                means.append(s['greedy_goal'][0])
                stds.append(s['greedy_goal'][1])

            bars = ax.bar(x, means, yerr=stds, capsize=6, color=bar_colors,
                          edgecolor='black', linewidth=0.7, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(eta_labels, fontsize=9)
            ax.set_ylabel('Goal Success Rate', fontsize=10)
            ax.set_title(f'env={env_name} | ε-d={eps_decay}', fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_ylim(0, 1.15)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f'{val:.0%}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

    fig.suptitle('Greedy Goal Success Rate (ε=0)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('part6_greedy_goal.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved → part6_greedy_goal.png")


def plot_eta_vs_decay_heatmaps(all_results):
    """Heatmaps: greedy return as f(η, ε-decay) for each env config."""
    fig, axes = plt.subplots(1, len(ENV_CONFIGS),
                             figsize=(7 * len(ENV_CONFIGS), 5), squeeze=False)

    for col, env_name in enumerate(ENV_CONFIGS):
        ax = axes[0, col]
        grid = np.zeros((len(EPS_DECAYS), len(ETAS)))
        for i, eps_decay in enumerate(EPS_DECAYS):
            for j, eta in enumerate(ETAS):
                key = (env_name, eps_decay, eta)
                s = _summarise_key(all_results[key])
                grid[i, j] = s['greedy_ret'][0]

        im = ax.imshow(grid, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(ETAS)))
        ax.set_xticklabels([str(e) for e in ETAS])
        ax.set_yticks(range(len(EPS_DECAYS)))
        ax.set_yticklabels([str(d) for d in EPS_DECAYS])
        ax.set_xlabel('η', fontsize=11)
        ax.set_ylabel('ε-decay', fontsize=11)
        ax.set_title(f'Env: {env_name}', fontsize=12)

        for i in range(len(EPS_DECAYS)):
            for j in range(len(ETAS)):
                ax.text(j, i, f'{grid[i, j]:.0f}', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='white' if abs(grid[i, j]) > abs(grid).mean() else 'black')

        plt.colorbar(im, ax=ax, label='Greedy Return')

    fig.suptitle('Greedy Return Heatmap: η vs ε-decay', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('part6_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved → part6_heatmap.png")


def plot_env_comparison(all_results):
    """Side-by-side: default vs hard env for the best ε-decay, across η."""
    best_decay = 0.995
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(ETAS))
    width = 0.35
    eta_labels = [f'η={e}' if e > 0 else 'η=0' for e in ETAS]

    for ax_idx, (metric, ylabel, key_name) in enumerate([
        ('greedy_ret', 'Greedy Return', 'Greedy Return'),
        ('greedy_goal', 'Goal Success Rate', 'Greedy Goal Rate'),
    ]):
        ax = axes[ax_idx]
        for i, env_name in enumerate(ENV_CONFIGS):
            means, stds = [], []
            for eta in ETAS:
                key = (env_name, best_decay, eta)
                s = _summarise_key(all_results[key])
                means.append(s[metric][0])
                stds.append(s[metric][1])
            offset = -width / 2 + i * width
            bars = ax.bar(x + offset, means, width, yerr=stds, capsize=5,
                          label=env_name, edgecolor='black', linewidth=0.6,
                          alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(eta_labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{key_name} (ε-decay={best_decay})', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        if metric == 'greedy_goal':
            ax.set_ylim(0, 1.15)

    fig.suptitle('Default vs Hard Environment Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('part6_env_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved → part6_env_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    total = len(ENV_CONFIGS) * len(EPS_DECAYS) * len(ETAS) * len(SEEDS)
    print(f"Acrobot-v1 | Bins={NUM_BINS} | γ={GAMMA}")
    print(f"Episodes={NUM_EPISODES} | Seeds={len(SEEDS)} | Total runs={total}\n")

    plot_reward_function()
    all_results = run_experiments()
    print_summary(all_results)

    print("\nGenerating plots ...")
    plot_learning_curves(all_results)
    plot_greedy_comparison(all_results)
    plot_eta_vs_decay_heatmaps(all_results)
    plot_env_comparison(all_results)

    print("\nDone. Plots saved:")
    print("  part6_reward_function.png")
    print("  part6_curves_default.png")
    print("  part6_curves_hard.png")
    print("  part6_greedy_return.png")
    print("  part6_greedy_goal.png")
    print("  part6_heatmap.png")
    print("  part6_env_comparison.png")
