"""
Q4a: Replay factor experiments with rho = {1, 2, 4, 8}.
All hyperparameters except rho remain the same.
"""

import argparse
from tqdm import tqdm
from train import train_dqn

DEFAULTS = dict(
    max_timesteps=100000,
    max_episode_steps=2000,
    lr=1e-3,
    gamma=0.99,
    buffer_capacity=50000,
    batch_size=64,
    target_update_freq=500,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_steps=10000,
    hidden_dims=(128, 128),
    eval_interval=1000,
    eval_episodes=10,
)

REPLAY_FACTORS = [1, 2, 4, 8]


def main():
    parser = argparse.ArgumentParser(description="Q4a: Replay factor experiments")
    parser.add_argument("--num_seeds", type=int, default=15)
    parser.add_argument("--max_timesteps", type=int, default=DEFAULTS["max_timesteps"])
    args = parser.parse_args()

    log_dir = "results/q4a"

    for rho in REPLAY_FACTORS:
        tag = f"q4a_rho{rho}"
        print(f"\n=== Q4a: DQN rho={rho} | {args.num_seeds} seeds ===")
        for seed in tqdm(range(args.num_seeds), desc=f"rho={rho}"):
            train_dqn(
                seed=seed,
                replay_factor=rho,
                log_dir=log_dir,
                tag=tag,
                max_timesteps=args.max_timesteps,
                **{k: v for k, v in DEFAULTS.items() if k != "max_timesteps"},
            )
    print(f"\nLogs saved to {log_dir}/")


if __name__ == "__main__":
    main()
