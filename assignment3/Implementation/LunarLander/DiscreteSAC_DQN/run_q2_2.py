#!/usr/bin/env python3
"""
Question 2.2 — LunarLander experiments.

Runs 5 sub-experiments, 15 seeds each, with parallel workers:
  1) sac_cont   — Continuous SAC with auto-alpha          (Q2.2.1)
  2) hover_fixed — Hover variant, fixed alpha=0.01        (Q2.2.3)
  3) hover_auto  — Hover variant, auto-alpha              (Q2.2.3)
  4) sac_disc   — Discrete SAC with auto-alpha            (Q2.2.4b)
  5) dqn        — DQN baseline                            (Q2.2.4c)

Usage:
    python run_q2_2.py                        # run everything
    python run_q2_2.py --experiments sac_cont # run only one
    python run_q2_2.py --num-seeds 3 --parallel 3 --total-steps-cont 5000  # quick test
    python run_q2_2.py --resume               # resume from checkpoints
    python run_q2_2.py --plot-only            # just regenerate plots
"""
from __future__ import annotations

import argparse, csv, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"

# ───────────────────────────── training ──────────────────────────────

def _run_seed(module, args_list, logdir, seed):
    """Run one seed as a subprocess; returns (name, returncode, elapsed)."""
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


def _ci(y):
    m = np.nanmean(y, axis=1)
    sem = np.nanstd(y, axis=1, ddof=1) / np.sqrt(np.sum(~np.isnan(y), axis=1).clip(1))
    return m, 1.96 * sem


def _plot_multi(groups, out, title=None, ylabel="Average Undiscounted Return"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for path, label in groups:
        x, y = _load_curves(path)
        if x is None:
            continue
        m, ci = _ci(y)
        ax.plot(x, m, label=label, linewidth=2)
        ax.fill_between(x, m - ci, m + ci, alpha=0.2)
    ax.set_xlabel("Environment Timesteps")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  saved {out}")


def make_plots():
    print("\n=== Generating Q2.2 plots ===")
    r = RESULTS
    if (r / "ll_sac_cont").exists():
        _plot_multi([(r / "ll_sac_cont", "SAC continuous")],
                    PLOTS / "q2_2_1_sac_cont.png",
                    title="Q2.2.1 — Continuous SAC on LunarLander")
    if (r / "ll_sac_disc").exists() and (r / "ll_dqn").exists():
        _plot_multi([(r / "ll_sac_disc", "Discrete-SAC"), (r / "ll_dqn", "DQN")],
                    PLOTS / "q2_2_4_disc_vs_dqn.png",
                    title="Q2.2.4 — Discrete-SAC vs DQN")
    if (r / "ll_hover_fixed").exists() and (r / "ll_hover_auto").exists():
        _plot_multi([(r / "ll_hover_fixed", "α=0.01 (fixed)"),
                     (r / "ll_hover_auto", "α auto")],
                    PLOTS / "q2_2_3_hover.png",
                    title="Q2.2.3 — Hover Reward (+200→−100)")

# ────────────────────────────── main ─────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Q2.2 — LunarLander experiments")
    p.add_argument("--experiments", nargs="+",
                   default=["sac_cont", "sac_disc", "dqn", "hover_fixed", "hover_auto"],
                   help="which experiments to run")
    p.add_argument("--num-seeds", type=int, default=15)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-done", action="store_true", default=True)
    p.add_argument("--no-skip-done", dest="skip_done", action="store_false")
    p.add_argument("--plot-only", action="store_true")

    g = p.add_argument_group("step counts")
    g.add_argument("--total-steps-cont", type=int, default=500_000)
    g.add_argument("--total-steps-disc", type=int, default=400_000)
    g.add_argument("--total-steps-hover", type=int, default=600_000)
    g.add_argument("--hover-flip", type=int, default=300_000)
    g.add_argument("--video-every", type=int, default=50_000)
    g.add_argument("--checkpoint-every", type=int, default=50_000)
    args = p.parse_args()

    if args.plot_only:
        make_plots()
        return

    seeds = list(range(args.num_seeds))
    common = ["--device", args.device, "--start-steps", "10000",
              "--video-every", str(args.video_every),
              "--checkpoint-every", str(args.checkpoint_every)]

    print(f"=== Q2.2 LunarLander — {args.num_seeds} seeds, {args.parallel} parallel ===\n")

    defs = {
        "sac_cont": ("train.train_sac",
            ["--env", "LunarLanderContinuous-v3", "--total-steps", str(args.total_steps_cont),
             "--auto-alpha"] + common,
            RESULTS / "ll_sac_cont"),
        "sac_disc": ("train.train_sac",
            ["--env", "LunarLander-v3", "--discrete", "--total-steps", str(args.total_steps_disc),
             "--auto-alpha"] + common,
            RESULTS / "ll_sac_disc"),
        "dqn": ("train.train_dqn",
            ["--env", "LunarLander-v3", "--total-steps", str(args.total_steps_disc)] + common,
            RESULTS / "ll_dqn"),
        "hover_fixed": ("train.train_sac",
            ["--env", "LunarLanderHover-v3", "--total-steps", str(args.total_steps_hover),
             "--alpha", "0.01", "--hover-flip-step", str(args.hover_flip),
             "--buffer-size", "100000"] + common,
            RESULTS / "ll_hover_fixed"),
        "hover_auto": ("train.train_sac",
            ["--env", "LunarLanderHover-v3", "--total-steps", str(args.total_steps_hover),
             "--auto-alpha", "--hover-flip-step", str(args.hover_flip),
             "--buffer-size", "100000"] + common,
            RESULTS / "ll_hover_auto"),
    }

    for name in args.experiments:
        if name not in defs:
            print(f"Unknown experiment '{name}', choices: {list(defs)}")
            continue
        module, extra, logdir = defs[name]
        run_experiment(name, module, extra, logdir, seeds, args.parallel,
                       args.resume, args.skip_done)

    make_plots()
    print("\n=== Q2.2 done ===")


if __name__ == "__main__":
    main()
