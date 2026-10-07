#!/usr/bin/env python3
"""
Question 3 — PEBBLE (bonus).

Sub-experiments:
  1) SAC with ground-truth reward on Pendulum (baseline)
  2) PEBBLE on Pendulum with budgets {100, 500, 1000, 2000}
  3) PEBBLE on Reacher with simulated teachers Ra, Rb, Rc

Usage:
    python run_q3.py                              # run everything
    python run_q3.py --experiments sac_gt pebble_1000  # selective
    python run_q3.py --num-seeds 3 --parallel 3 --pendulum-steps 5000  # quick test
    python run_q3.py --resume                     # resume from checkpoints
    python run_q3.py --plot-only                  # just regenerate plots
"""
from __future__ import annotations

import argparse, os, subprocess, sys, time
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


def _plot_multi(groups, out, title=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for path, label in groups:
        x, y = _load_curves(path)
        if x is None:
            continue
        m, ci = _ci(y)
        ax.plot(x, m, label=label, linewidth=2)
        ax.fill_between(x, m - ci, m + ci, alpha=0.2)
    ax.set_xlabel("Environment Timesteps")
    ax.set_ylabel("Average Undiscounted Return (GT)")
    if title:
        ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  saved {out}")


def make_plots():
    print("\n=== Generating Q3 plots ===")
    r = RESULTS

    # SAC GT vs PEBBLE on Pendulum
    groups = []
    if (r / "pendulum_sac_gt").exists():
        groups.append((r / "pendulum_sac_gt", "SAC (GT reward)"))
    for b in [100, 500, 1000, 2000]:
        d = r / f"pebble_pendulum_b{b}"
        if d.exists():
            groups.append((d, f"PEBBLE ({b})"))
    if groups:
        _plot_multi(groups, PLOTS / "q3_1_pebble_vs_sac.png",
                    title="Q3.1 — SAC vs PEBBLE on Pendulum")

    # Budget sweep
    budget_groups = [(r / f"pebble_pendulum_b{b}", str(b))
                     for b in [100, 500, 1000, 2000] if (r / f"pebble_pendulum_b{b}").exists()]
    if len(budget_groups) > 1:
        _plot_multi(budget_groups, PLOTS / "q3_2_budget_sweep.png",
                    title="Q3.2 — PEBBLE Budget Sweep on Pendulum")

    # PEBBLE on Reacher
    reacher_groups = [(r / f"pebble_reacher_{v}", f"PEBBLE-{v}")
                      for v in ["Ra", "Rb", "Rc"] if (r / f"pebble_reacher_{v}").exists()]
    if reacher_groups:
        _plot_multi(reacher_groups, PLOTS / "q3_3_pebble_reacher.png",
                    title="Q3.3 — PEBBLE on Reacher")

# ────────────────────────────── main ─────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Q3 — PEBBLE (bonus)")
    ALL_EXPS = ["sac_gt", "pebble_100", "pebble_500", "pebble_1000", "pebble_2000",
                "pebble_Ra", "pebble_Rb", "pebble_Rc"]
    p.add_argument("--experiments", nargs="+", default=ALL_EXPS)
    p.add_argument("--num-seeds", type=int, default=15)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-done", action="store_true", default=True)
    p.add_argument("--no-skip-done", dest="skip_done", action="store_false")
    p.add_argument("--plot-only", action="store_true")

    g = p.add_argument_group("step counts")
    g.add_argument("--pendulum-steps", type=int, default=200_000)
    g.add_argument("--reacher-steps", type=int, default=500_000)
    g.add_argument("--video-every", type=int, default=50_000)
    g.add_argument("--checkpoint-every", type=int, default=50_000)
    args = p.parse_args()

    if args.plot_only:
        make_plots()
        return

    seeds = list(range(args.num_seeds))
    pend_common = ["--device", args.device, "--start-steps", "500",
                   "--video-every", str(args.video_every),
                   "--checkpoint-every", str(args.checkpoint_every)]
    reach_common = ["--device", args.device, "--start-steps", "10000",
                    "--video-every", str(args.video_every),
                    "--checkpoint-every", str(args.checkpoint_every)]

    print(f"=== Q3 PEBBLE — {args.num_seeds} seeds, {args.parallel} parallel ===\n")

    defs = {
        "sac_gt": ("train.train_sac",
            ["--env", "Pendulum-dense", "--total-steps", str(args.pendulum_steps),
             "--auto-alpha"] + pend_common,
            RESULTS / "pendulum_sac_gt"),
    }
    for budget in [100, 500, 1000, 2000]:
        defs[f"pebble_{budget}"] = ("train.train_pebble",
            ["--env", "Pendulum-dense", "--total-steps", str(args.pendulum_steps),
             "--pref-budget", str(budget)] + pend_common,
            RESULTS / f"pebble_pendulum_b{budget}")
    for v in ["Ra", "Rb", "Rc"]:
        defs[f"pebble_{v}"] = ("train.train_pebble",
            ["--env", f"ReacherEasy-{v}", "--total-steps", str(args.reacher_steps),
             "--pref-budget", "2000", "--teacher", v] + reach_common,
            RESULTS / f"pebble_reacher_{v}")

    for name in args.experiments:
        if name not in defs:
            print(f"Unknown experiment '{name}', choices: {list(defs)}")
            continue
        module, extra, logdir = defs[name]
        run_experiment(name, module, extra, logdir, seeds, args.parallel,
                       args.resume, args.skip_done)

    make_plots()
    print("\n=== Q3 done ===")


if __name__ == "__main__":
    main()
