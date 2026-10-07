"""
Q4d: Sensitivity analysis for DQN with rho=1 and rho=4.
Varies mini-batch size and target network refresh rate.
Two values smaller and two values larger than the default for each.
"""

import argparse
from tqdm import tqdm
from train import train_dqn

BASE_PARAMS = dict(
    max_timesteps=100000,
    max_episode_steps=2000,
    lr=1e-3,
    gamma=0.99,
    buffer_capacity=50000,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_steps=10000,
    hidden_dims=(128, 128),
    eval_interval=1000,
    eval_episodes=10,
)

DEFAULT_BATCH_SIZE = 64
DEFAULT_TARGET_FREQ = 500

BATCH_SIZES = [16, 32, 64, 128, 256]
TARGET_FREQS = [100, 250, 500, 1000, 2000]

RHOS = [1, 4]


def main():
    parser = argparse.ArgumentParser(description="Q4d: Sensitivity analysis")
    parser.add_argument("--num_seeds", type=int, default=15)
    parser.add_argument("--max_timesteps", type=int, default=BASE_PARAMS["max_timesteps"])
    args = parser.parse_args()

    log_dir = "results/q4d"

    # --- Sensitivity: batch size ---
    for rho in RHOS:
        for bs in BATCH_SIZES:
            tag = f"q4d_rho{rho}_bs{bs}"
            print(f"\n=== Q4d: rho={rho}, batch_size={bs} | {args.num_seeds} seeds ===")
            for seed in tqdm(range(args.num_seeds), desc=f"rho={rho},bs={bs}"):
                train_dqn(
                    seed=seed,
                    replay_factor=rho,
                    batch_size=bs,
                    target_update_freq=DEFAULT_TARGET_FREQ,
                    log_dir=log_dir,
                    tag=tag,
                    max_timesteps=args.max_timesteps,
                    **{k: v for k, v in BASE_PARAMS.items()
                       if k not in ("max_timesteps",)},
                )

    # --- Sensitivity: target update frequency ---
    for rho in RHOS:
        for tf in TARGET_FREQS:
            tag = f"q4d_rho{rho}_tf{tf}"
            print(f"\n=== Q4d: rho={rho}, target_freq={tf} | {args.num_seeds} seeds ===")
            for seed in tqdm(range(args.num_seeds), desc=f"rho={rho},tf={tf}"):
                train_dqn(
                    seed=seed,
                    replay_factor=rho,
                    batch_size=DEFAULT_BATCH_SIZE,
                    target_update_freq=tf,
                    log_dir=log_dir,
                    tag=tag,
                    max_timesteps=args.max_timesteps,
                    **{k: v for k, v in BASE_PARAMS.items()
                       if k not in ("max_timesteps",)},
                )

    print(f"\nLogs saved to {log_dir}/")


if __name__ == "__main__":
    main()
