"""
Master script to run ALL experiments and generate ALL plots.
Usage:
    python run_all.py                    # full run (15 seeds each)
    python run_all.py --num_seeds 2      # quick test (2 seeds)
    python run_all.py --plots_only       # only generate plots from existing logs
"""

import argparse
import subprocess
import sys
import time


def run_script(script, num_seeds, max_timesteps):
    cmd = [sys.executable, script, "--num_seeds", str(num_seeds), "--max_timesteps", str(max_timesteps)]
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start
    print(f"Finished {script} in {elapsed:.1f}s")


def run_plots(script):
    cmd = [sys.executable, script]
    print(f"\n{'='*60}")
    print(f"Plotting: {' '.join(cmd)}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run all PA2 experiments")
    parser.add_argument("--num_seeds", type=int, default=15)
    parser.add_argument("--max_timesteps", type=int, default=100000)
    parser.add_argument("--plots_only", action="store_true",
                        help="Only generate plots from existing logs")
    parser.add_argument("--skip_q5", action="store_true",
                        help="Skip Q5 bonus (PER)")
    args = parser.parse_args()

    total_start = time.time()

    if not args.plots_only:
        print("\n" + "#" * 60)
        print("# RUNNING ALL EXPERIMENTS")
        print("#" * 60)

        run_script("run_q4a.py", args.num_seeds, args.max_timesteps)
        run_script("run_q4d.py", args.num_seeds, args.max_timesteps)

        if not args.skip_q5:
            run_script("run_q5.py", args.num_seeds, args.max_timesteps)

    print("\n" + "#" * 60)
    print("# GENERATING ALL PLOTS")
    print("#" * 60)

    run_plots("plot_q4.py")    # Q4a plots
    run_plots("plot_q4b.py")   # Q4b: performance distributions
    run_plots("plot_q4c.py")   # Q4c: tolerance intervals
    run_plots("plot_q4d.py")   # Q4d: sensitivity analysis
    if not args.skip_q5:
        run_plots("plot_q5.py")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
