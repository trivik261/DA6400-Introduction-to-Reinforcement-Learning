#!/usr/bin/env python3
"""
Question 2.3 — Reacher experiments.

Trains SAC with auto-alpha on three reward formulations (Ra, Rb, Rc).
Cross-evaluates each policy under all three reward formulations.
Then runs final-policy evaluation (500 episodes × 5000 steps) for bar charts.

Usage:
    python run_q2_3.py                             # run everything
    python run_q2_3.py --variants Ra Rb            # only Ra and Rb
    python run_q2_3.py --num-seeds 3 --parallel 3 --total-steps 5000  # quick test
    python run_q2_3.py --resume                    # resume from checkpoints
    python run_q2_3.py --plot-only                 # just regenerate plots
    python run_q2_3.py --eval-only                 # only final-policy eval + plots
"""
from __future__ import annotations

import argparse, csv, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"

# ───────────────────────── subprocess runner ─────────────────────────

def _run_seed(module, args_list, logdir, seed):
    seed_log = logdir / f"seed_{seed}.log"
    cmd = [sys.executable, "-u", "-m", module] + args_list + [
        "--seed", str(seed), "--logdir", str(logdir),
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "PA3_TORCH_THREADS": "2", "MPLBACKEND": "Agg"}
    t0 = time.time()
    with open(seed_log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    return f"{logdir.name}/seed_{seed}", rc, time.time() - t0


def _is_done(logdir, seed):
    return (logdir / f"seed_{seed}" / "checkpoints" / "final.pt").exists()


def _fmt_time(secs):
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs/60:.0f}m {secs%60:.0f}s"
    return f"{secs/3600:.0f}h {(secs%3600)/60:.0f}m"


def run_experiment(name, module, extra_args, logdir, seeds, parallel, resume, skip_done):
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    args_list = extra_args + (["--resume", "latest"] if resume else [])
    todo = [s for s in seeds if not (skip_done and _is_done(logdir, s))]
    if not todo:
        print(f"  [{name}] all {len(seeds)} seeds done, skipping")
        return
    print(f"\n  ┌─ {name}: {len(todo)} seeds, {parallel} parallel")
    done, fail = 0, 0
    t_exp = time.time()
    with ProcessPoolExecutor(max_workers=parallel) as pool:
        futs = {pool.submit(_run_seed, module, args_list, logdir, s): s for s in todo}
        for fut in as_completed(futs):
            tag, rc, elapsed = fut.result()
            done += 1
            wall = time.time() - t_exp
            pct = done / len(todo) * 100
            eta = wall / done * (len(todo) - done) if done > 0 else 0
            bar = f"{'█' * int(pct // 5)}{'░' * (20 - int(pct // 5))}"
            status = "✓" if rc == 0 else "✗"
            if rc != 0:
                fail += 1
                tail = (logdir / f"seed_{futs[fut]}.log").read_text().strip().split("\n")[-3:]
                print(f"  │ {status} {tag} ({_fmt_time(elapsed)}) FAIL: {' | '.join(tail)}")
            else:
                print(f"  │ {status} {tag} ({_fmt_time(elapsed)})  "
                      f"[{bar}] {done}/{len(todo)} | "
                      f"elapsed {_fmt_time(wall)} | ETA {_fmt_time(eta)}")
    wall = time.time() - t_exp
    print(f"  └─ {name}: {done - fail}/{done} ok in {_fmt_time(wall)}"
          + (f"  ({fail} failed)" if fail else ""))

# ──────────────────── final-policy evaluation (Q2.3.3a) ─────────────

def evaluate_final_policies(variants, seeds, episodes=500, horizon=5000, device="cpu"):
    """Load final checkpoints and compute steps-to-goal / steps-in-target."""
    from pa3.agents.sac import SACAgent, SACConfig
    from pa3.envs.make_env import make_env

    rows = []
    for variant in variants:
        for seed in seeds:
            ckpt = RESULTS / f"reacher_{variant}" / f"seed_{seed}" / "checkpoints" / "final.pt"
            if not ckpt.exists():
                print(f"  missing {ckpt}; skipping")
                continue
            env = make_env(f"ReacherEasy-{variant}", seed=seed + 30_000)
            obs_dim = int(np.prod(env.observation_space.shape))
            action_dim = int(env.action_space.shape[0])
            cfg = SACConfig(obs_dim=obs_dim, action_dim=action_dim, device=device)
            agent = SACAgent(cfg)
            blob = torch.load(ckpt, map_location=device, weights_only=False)
            agent.load_full_state_dict(blob["agent"])
            lo, hi = env.action_space.low, env.action_space.high
            print(f"  evaluating {variant} seed={seed} ({episodes} eps × {horizon} steps)...")
            for ep in range(episodes):
                obs, _ = env.reset()
                first_in, in_count = None, 0
                for t in range(horizon):
                    with torch.no_grad():
                        a = agent.select_action(obs, deterministic=True)
                    a_env = lo + (a + 1.0) * 0.5 * (hi - lo)
                    obs, _, terminated, truncated, info = env.step(a_env)
                    if info.get("in_target", False):
                        if first_in is None:
                            first_in = t
                        in_count += 1
                    if terminated or truncated:
                        break
                rows.append({
                    "variant": variant, "seed": seed, "episode": ep,
                    "steps_to_goal": first_in if first_in is not None else horizon,
                    "steps_in_target": in_count,
                })
    out = RESULTS / "reacher_final_metrics.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "seed", "episode",
                                           "steps_to_goal", "steps_in_target"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows)} rows to {out}")

# ──────────────────────────── plotting ───────────────────────────────

def _load_curves(path, column="return"):
    series = []
    for sd in sorted(Path(path).glob("seed_*")):
        csv_f = sd / "train.csv"
        if not csv_f.exists():
            continue
        df = pd.read_csv(csv_f)
        if column not in df.columns:
            continue
        series.append(pd.Series(df[column].values, index=df["step"].values))
    if not series:
        return None, None
    df = pd.concat(series, axis=1, join="outer").sort_index().interpolate("index").dropna(how="all")
    return df.index.values, df.values


def _load_cross(exp_dir, target):
    cols = []
    for sd in sorted(Path(exp_dir).glob("seed_*")):
        csv_f = sd / "eval_cross.csv"
        if not csv_f.exists():
            continue
        df = pd.read_csv(csv_f)
        col = f"return_{target}"
        if col not in df.columns:
            continue
        cols.append(pd.Series(df[col].values, index=df["step"].values))
    if not cols:
        return None, None
    df = pd.concat(cols, axis=1, join="outer").sort_index().interpolate("index").dropna(how="all")
    return df.index.values, df.values


def _ci(y):
    m = np.nanmean(y, axis=1)
    sem = np.nanstd(y, axis=1, ddof=1) / np.sqrt(np.sum(~np.isnan(y), axis=1).clip(1))
    return m, 1.96 * sem


def make_plots(variants):
    print("\n=== Generating Q2.3 plots ===")
    PLOTS.mkdir(parents=True, exist_ok=True)

    # Q2.3.2 — one plot per formulation: SAC-Ri vs Ri
    for v in variants:
        path = RESULTS / f"reacher_{v}"
        x, y = _load_curves(path)
        if x is None:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        m, ci = _ci(y)
        ax.plot(x, m, label=f"SAC-{v}", linewidth=2)
        ax.fill_between(x, m - ci, m + ci, alpha=0.2)
        ax.set_xlabel("Environment Timesteps")
        ax.set_ylabel("Average Undiscounted Return")
        ax.set_title(f"Q2.3.2 — SAC-{v} evaluated under {v}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        out = PLOTS / f"q2_3_2_{v}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"  saved {out}")

    # Q2.3.3c — cross-evaluation: 3 panels, each shows all SAC-Rj eval'd under Ri
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, target in zip(axes, variants):
        for trained_on in variants:
            x, y = _load_cross(RESULTS / f"reacher_{trained_on}", target)
            if x is None:
                continue
            m, ci = _ci(y)
            ax.plot(x, m, label=f"SAC-{trained_on}", linewidth=2)
            ax.fill_between(x, m - ci, m + ci, alpha=0.2)
        ax.set_title(f"Evaluated under {target}")
        ax.set_xlabel("Environment Timesteps")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Average Undiscounted Return")
    axes[-1].legend()
    fig.tight_layout()
    out = PLOTS / "q2_3_3c_cross_eval.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  saved {out}")

    # Q2.3.3a — bar charts for steps_to_goal and steps_in_target
    metrics_csv = RESULTS / "reacher_final_metrics.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        if len(df) > 0:
            # CI over seeds (average episodes per seed first, then CI across seeds)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            present = sorted(df["variant"].unique())
            for ax, metric in zip(axes, ["steps_to_goal", "steps_in_target"]):
                seed_means = df.groupby(["variant", "seed"])[metric].mean().reset_index()
                means, cis = [], []
                for v in present:
                    vals = seed_means.loc[seed_means["variant"] == v, metric].values
                    means.append(float(np.mean(vals)))
                    sem = float(np.std(vals, ddof=1) / np.sqrt(max(len(vals), 1))) if len(vals) > 1 else 0.0
                    cis.append(1.96 * sem)
                pos = np.arange(len(present))
                ax.bar(pos, means, yerr=cis, capsize=5, width=0.6)
                ax.set_xticks(pos)
                ax.set_xticklabels([f"SAC-{v}" for v in present])
                ax.set_title(metric.replace("_", " ").title())
                ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            out = PLOTS / "q2_3_3a_bar.png"
            fig.savefig(out, dpi=140); plt.close(fig)
            print(f"  saved {out}")

# ────────────────────────────── main ─────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Q2.3 — Reacher experiments")
    p.add_argument("--variants", nargs="+", default=["Ra", "Rb", "Rc"])
    p.add_argument("--num-seeds", type=int, default=15)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-done", action="store_true", default=True)
    p.add_argument("--no-skip-done", dest="skip_done", action="store_false")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training, run final-policy evaluation + plots")

    g = p.add_argument_group("step counts")
    g.add_argument("--total-steps", type=int, default=500_000)
    g.add_argument("--video-every", type=int, default=50_000)
    g.add_argument("--checkpoint-every", type=int, default=50_000)

    g2 = p.add_argument_group("final evaluation (Q2.3.3a)")
    g2.add_argument("--eval-episodes", type=int, default=500)
    g2.add_argument("--eval-horizon", type=int, default=5000)
    args = p.parse_args()

    if args.plot_only:
        make_plots(args.variants)
        return

    seeds = list(range(args.num_seeds))

    if not args.eval_only:
        common = ["--device", args.device, "--start-steps", "10000",
                  "--video-every", str(args.video_every),
                  "--checkpoint-every", str(args.checkpoint_every)]

        print(f"=== Q2.3 Reacher — {args.num_seeds} seeds, {args.parallel} parallel ===\n")

        for variant in args.variants:
            run_experiment(
                f"reacher_{variant}", "train.train_sac",
                ["--env", f"ReacherEasy-{variant}",
                 "--total-steps", str(args.total_steps), "--auto-alpha",
                 "--extra-eval-rewards", "Ra", "Rb", "Rc"] + common,
                RESULTS / f"reacher_{variant}", seeds, args.parallel,
                args.resume, args.skip_done,
            )

    # Final-policy evaluation
    print("\n=== Final-policy evaluation (Q2.3.3a) ===")
    evaluate_final_policies(args.variants, seeds, args.eval_episodes,
                            args.eval_horizon, args.device)

    make_plots(args.variants)
    print("\n=== Q2.3 done ===")


if __name__ == "__main__":
    main()
