"""DQN training entry-point for the LunarLander-v3 (discrete) comparison."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from pa3.agents.dqn import DQNAgent, DQNConfig
from pa3.envs.make_env import make_env
from pa3.utils.checkpointing import find_latest, load_checkpoint, save_checkpoint
from pa3.utils.eval import evaluate_policy
from pa3.utils.logging import CSVLogger
from pa3.utils.replay import ReplayBuffer
from pa3.utils.seeding import seed_everything
from pa3.utils.threads import configure_threads
from pa3.utils.video import record_episode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=400_000)
    p.add_argument("--start-steps", type=int, default=10_000,
                   help="random exploration timesteps before any DQN update.")
    p.add_argument("--update-after", type=int, default=-1,
                   help="defaults to --start-steps")
    p.add_argument("--updates-per-step", type=int, default=1,
                   help="DQN replay factor; TA recommends 1.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-update", type=int, default=1_000)
    p.add_argument("--eps-decay", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--video-every", type=int, default=50_000)
    p.add_argument("--logdir", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--checkpoint-every", type=int, default=50_000)
    p.add_argument("--save-buffer", action="store_true")
    p.add_argument("--resume", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.update_after < 0:
        args.update_after = args.start_steps
    configure_threads()
    seed_everything(args.seed)
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logdir = Path(args.logdir) / f"seed_{args.seed}"
    logdir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env, seed=args.seed)
    eval_env = make_env(args.env, seed=args.seed + 10_000)
    video_env = None  # created lazily
    obs_dim = int(np.prod(env.observation_space.shape))
    num_actions = int(env.action_space.n)

    cfg = DQNConfig(
        obs_dim=obs_dim,
        num_actions=num_actions,
        gamma=args.gamma,
        lr=args.lr,
        target_update_every=args.target_update,
        eps_decay_steps=args.eps_decay,
        device=device,
    )
    agent = DQNAgent(cfg)
    buf = ReplayBuffer(args.buffer_size, obs_dim, (), np.int64, device=device)
    logger = CSVLogger(logdir / "train.csv")

    obs, _ = env.reset(seed=args.seed)
    next_eval = args.eval_every
    next_video = args.video_every if args.video_every > 0 else None
    next_ckpt = args.checkpoint_every if args.checkpoint_every > 0 else None
    ckpt_dir = logdir / "checkpoints"
    start_step = 0
    t0 = time.time()

    if args.resume:
        rpath = args.resume
        if rpath == "latest":
            found = find_latest(ckpt_dir)
            rpath = str(found) if found else ""
        if rpath:
            blob = load_checkpoint(rpath, map_location=device)
            agent.load_full_state_dict(blob["agent"])
            if blob.get("buffer") is not None:
                buf.load_state_dict(blob["buffer"])
            start_step = int(blob["step"])
            print(f"[resume] loaded {rpath} at step={start_step}")
            next_eval = ((start_step // args.eval_every) + 1) * args.eval_every
            if next_video is not None:
                next_video = ((start_step // args.video_every) + 1) * args.video_every
            if next_ckpt is not None:
                next_ckpt = ((start_step // args.checkpoint_every) + 1) * args.checkpoint_every

    if start_step == 0:
        mean_ret_0 = evaluate_policy(eval_env,
                                     lambda o: agent.select_action(o, deterministic=True),
                                     n_episodes=args.eval_episodes)
        logger.log({"step": 0, "return": mean_ret_0, "epsilon": agent.epsilon(0), "elapsed": 0.0})
        print(f"[seed={args.seed} step=0] return={mean_ret_0:.2f}  (random-policy baseline)")

    for step in range(start_step + 1, args.total_steps + 1):
        if step < args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, env_step=step)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buf.add(obs.astype(np.float32), np.int64(action), float(reward), next_obs.astype(np.float32), bool(terminated), bool(terminated or truncated))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
        if step >= args.update_after and len(buf) >= args.batch_size:
            for _ in range(args.updates_per_step):
                batch = buf.sample(args.batch_size)
                agent.update(batch)
        if step >= next_eval:
            mean_ret = evaluate_policy(
                eval_env,
                lambda o: agent.select_action(o, deterministic=True),
                n_episodes=args.eval_episodes,
            )
            logger.log({"step": step, "return": mean_ret, "epsilon": agent.epsilon(step), "elapsed": time.time() - t0})
            pct = step / args.total_steps * 100
            el = time.time() - t0
            eta = el / step * (args.total_steps - step) if step > 0 else 0
            bar = f"[{'█' * int(pct // 5)}{'░' * (20 - int(pct // 5))}]"
            print(f"[seed={args.seed}] {bar} {pct:5.1f}% | step {step}/{args.total_steps} | "
                  f"return={mean_ret:.2f} | ε={agent.epsilon(step):.3f} | "
                  f"elapsed {el/60:.0f}m{el%60:.0f}s | ETA {eta/60:.0f}m{eta%60:.0f}s")
            next_eval += args.eval_every
        if next_video is not None and step >= next_video:
            if video_env is None:
                video_env = make_env(args.env, seed=args.seed + 20_000, render_mode="rgb_array")
            try:
                record_episode(
                    video_env,
                    lambda o: agent.select_action(o, deterministic=True),
                    str(logdir / "videos" / f"step_{step:08d}.mp4"),
                )
            except Exception as e:  # noqa: BLE001
                print(f"video recording failed: {e}")
            next_video += args.video_every

        if next_ckpt is not None and step >= next_ckpt:
            cpath = ckpt_dir / f"step_{step:08d}.pt"
            try:
                save_checkpoint(
                    cpath,
                    agent_state=agent.full_state_dict(),
                    step=step,
                    buffer_state=buf.state_dict() if args.save_buffer else None,
                )
                print(f"[ckpt] step={step} -> {cpath}")
            except Exception as e:  # noqa: BLE001
                print(f"checkpoint write failed: {e}")
            next_ckpt += args.checkpoint_every

    logger.close()
    torch.save(agent.q.state_dict(), logdir / "final.pt")
    save_checkpoint(
        ckpt_dir / "final.pt",
        agent_state=agent.full_state_dict(),
        step=args.total_steps,
        buffer_state=buf.state_dict() if args.save_buffer else None,
    )
    el = time.time() - t0
    print(f"[seed={args.seed}] done in {el/60:.1f}m ({el:.0f}s)")


if __name__ == "__main__":
    main()
