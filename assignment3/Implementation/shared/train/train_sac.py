"""Generic SAC training entry-point.

Examples
--------
2.2.1  ``--env LunarLanderContinuous-v3 --auto-alpha``
2.2.3  ``--env LunarLanderHover-v3 --hover-flip-step 300000 --alpha 0.01``
2.2.4  ``--env LunarLander-v3 --discrete --auto-alpha``
2.3    ``--env ReacherEasy-Ra --extra-eval-rewards Ra Rb Rc``
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from pa3.agents.sac import SACAgent, SACConfig
from pa3.envs.make_env import make_env
from pa3.utils.checkpointing import find_latest, load_checkpoint, save_checkpoint
from pa3.utils.eval import evaluate_policy, evaluate_with_rewards
from pa3.utils.logging import CSVLogger
from pa3.utils.replay import ReplayBuffer
from pa3.utils.seeding import seed_everything
from pa3.utils.threads import configure_threads
from pa3.utils.video import record_episode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--start-steps", type=int, default=10_000,
                   help="random exploration timesteps; no SAC update happens before this. "
                        "TA recommends 10K for LunarLander/Reacher and 500 for Pendulum.")
    p.add_argument("--update-after", type=int, default=-1,
                   help="when to start SAC updates. Defaults to --start-steps (per TA's "
                        "clarification: random action only at the beginning, no updates during).")
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha", type=float, default=0.2, help="initial alpha (and fixed value if --auto-alpha not set)")
    p.add_argument("--auto-alpha", action="store_true")
    p.add_argument("--target-entropy", type=float, default=None)
    p.add_argument("--discrete", action="store_true")
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--video-every", type=int, default=50_000, help="env steps between recorded videos (0 disables)")
    p.add_argument("--logdir", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--hover-flip-step", type=int, default=-1, help="env step at which to flip the hover bonus +200 -> -100")
    p.add_argument("--extra-eval-rewards", nargs="*", default=[], help="cross-evaluation reward ids (Reacher only)")
    p.add_argument("--pendulum-targets-deg", nargs="*", type=float, default=None,
                   help="Pendulum-only: list of target theta values in degrees. "
                        "Sec 2.1 uses 8 targets, Sec 3 PEBBLE uses {0,-60,90,120,-150}.")
    p.add_argument("--checkpoint-every", type=int, default=50_000,
                   help="env-steps between checkpoint saves; 0 disables.")
    p.add_argument("--save-buffer", action="store_true",
                   help="include the replay buffer in each checkpoint (large but required for "
                        "byte-for-byte resume).  Off by default.")
    p.add_argument("--resume", type=str, default="",
                   help="path to a checkpoint .pt to resume from. Use \"latest\" to pick the "
                        "most-recent checkpoint inside <logdir>/checkpoints/.")
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

    env_kwargs = {}
    if args.pendulum_targets_deg is not None and args.env.startswith("Pendulum"):
        env_kwargs["theta_targets_deg"] = tuple(args.pendulum_targets_deg)
    env = make_env(args.env, seed=args.seed, **env_kwargs)
    eval_env = make_env(args.env, seed=args.seed + 10_000, **env_kwargs)
    video_env = None  # created lazily on first video to avoid macOS pygame init cost
    # TA middleground: Rc eval caps episodes at 1000 steps with -20 timeout penalty
    eval_max_steps = 1000 if getattr(env, "reward_id", None) == "Rc" else 0
    obs_dim = int(np.prod(env.observation_space.shape))

    if args.discrete:
        action_dim = int(env.action_space.n)
        action_shape = ()
        action_dtype = np.int64
    else:
        action_dim = int(env.action_space.shape[0])
        action_shape = env.action_space.shape
        action_dtype = np.float32

    cfg = SACConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        discrete=args.discrete,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.lr,
        critic_lr=args.lr,
        alpha_lr=args.lr,
        auto_alpha=args.auto_alpha,
        init_alpha=args.alpha,
        target_entropy=args.target_entropy,
        device=device,
    )
    agent = SACAgent(cfg)
    buf = ReplayBuffer(args.buffer_size, obs_dim, action_shape, action_dtype, device=device)

    logger = CSVLogger(logdir / "train.csv")
    extra_eval_logger = None
    if args.extra_eval_rewards:
        extra_eval_logger = CSVLogger(logdir / "eval_cross.csv")

    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_len = 0
    t0 = time.time()

    def act_for_env(o, det=False):
        if args.discrete:
            a = agent.select_action(o, deterministic=det)
            return int(a)
        a = agent.select_action(o, deterministic=det)
        # SAC outputs in [-1,1] for continuous — rescale to env action space.
        lo, hi = env.action_space.low, env.action_space.high
        return lo + (a + 1.0) * 0.5 * (hi - lo)

    next_eval = args.eval_every
    next_video = args.video_every if args.video_every > 0 else None
    next_ckpt = args.checkpoint_every if args.checkpoint_every > 0 else None
    ckpt_dir = logdir / "checkpoints"
    start_step = 0

    # ----- resume -----
    if args.resume:
        rpath = args.resume
        if rpath == "latest":
            found = find_latest(ckpt_dir)
            if found is None:
                print(f"--resume latest: no checkpoints under {ckpt_dir}; starting fresh")
                rpath = ""
            else:
                rpath = str(found)
        if rpath:
            blob = load_checkpoint(rpath, map_location=device)
            agent.load_full_state_dict(blob["agent"])
            if blob.get("buffer") is not None:
                buf.load_state_dict(blob["buffer"])
            start_step = int(blob["step"])
            print(f"[resume] loaded {rpath} at step={start_step} (buffer={'yes' if blob.get('buffer') else 'no'})")
            # advance the next-* schedules so we don't immediately re-eval.
            next_eval = ((start_step // args.eval_every) + 1) * args.eval_every
            if next_video is not None:
                next_video = ((start_step // args.video_every) + 1) * args.video_every
            if next_ckpt is not None:
                next_ckpt = ((start_step // args.checkpoint_every) + 1) * args.checkpoint_every

    # ----- step-0 evaluation (random-policy baseline) so plots start at x=0 -----
    if start_step == 0:
        mean_ret_0 = evaluate_policy(eval_env, lambda o: act_for_env(o, det=True),
                                     n_episodes=args.eval_episodes, max_steps=eval_max_steps)
        logger.log({"step": 0, "return": mean_ret_0,
                    "alpha": float(agent.alpha.detach().cpu()), "elapsed": 0.0})
        if extra_eval_logger is not None:
            cross0 = evaluate_with_rewards(eval_env, lambda o: act_for_env(o, det=True),
                                           args.extra_eval_rewards, n_episodes=args.eval_episodes,
                                           max_steps=eval_max_steps)
            extra_eval_logger.log({"step": 0, **{f"return_{k}": v for k, v in cross0.items()}})
        print(f"[seed={args.seed} step=0] return={mean_ret_0:.2f}  (random-policy baseline)")

    for step in range(start_step + 1, args.total_steps + 1):
        # Section 2.2.3: flip the hover bonus at the requested env step.
        if args.hover_flip_step > 0 and step == args.hover_flip_step:
            try:
                env.set_bonus(-100.0)
                eval_env.set_bonus(-100.0)
                if video_env is not None:
                    video_env.set_bonus(-100.0)
                print(f"[step {step}] hover bonus flipped to -100")
            except AttributeError:
                pass

        if step < args.start_steps:
            action = env.action_space.sample()
        else:
            action = act_for_env(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated  # do not bootstrap-stop on truncation

        # Store the *agent-space* action, not the rescaled one.
        if args.discrete:
            stored_action = np.int64(action)
        else:
            lo, hi = env.action_space.low, env.action_space.high
            stored_action = 2.0 * (np.asarray(action, dtype=np.float32) - lo) / (hi - lo) - 1.0
        buf.add(obs.astype(np.float32), stored_action, float(reward), next_obs.astype(np.float32), bool(done), bool(terminated or truncated))

        obs = next_obs
        ep_ret += float(reward)
        ep_len += 1
        if terminated or truncated:
            obs, _ = env.reset()
            ep_ret = 0.0
            ep_len = 0

        if step >= args.update_after and len(buf) >= args.batch_size:
            for _ in range(args.updates_per_step):
                batch = buf.sample(args.batch_size)
                metrics = agent.update(batch)

        # ----- evaluation -----
        if step >= next_eval:
            mean_ret = evaluate_policy(eval_env, lambda o: act_for_env(o, det=True),
                                       n_episodes=args.eval_episodes, max_steps=eval_max_steps)
            row = {
                "step": step,
                "return": mean_ret,
                "alpha": float(agent.alpha.detach().cpu()),
                "elapsed": time.time() - t0,
            }
            logger.log(row)
            pct = step / args.total_steps * 100
            el = time.time() - t0
            eta = el / step * (args.total_steps - step) if step > 0 else 0
            bar = f"[{'█' * int(pct // 5)}{'░' * (20 - int(pct // 5))}]"
            print(f"[seed={args.seed}] {bar} {pct:5.1f}% | step {step}/{args.total_steps} | "
                  f"return={mean_ret:.2f} | α={row['alpha']:.4f} | "
                  f"elapsed {el/60:.0f}m{el%60:.0f}s | ETA {eta/60:.0f}m{eta%60:.0f}s")

            if extra_eval_logger is not None:
                cross = evaluate_with_rewards(
                    eval_env,
                    lambda o: act_for_env(o, det=True),
                    args.extra_eval_rewards,
                    n_episodes=args.eval_episodes,
                    max_steps=eval_max_steps,
                )
                cross_row = {"step": step, **{f"return_{k}": v for k, v in cross.items()}}
                extra_eval_logger.log(cross_row)
            next_eval += args.eval_every

        # ----- video -----
        if next_video is not None and step >= next_video:
            if video_env is None:
                video_env = make_env(args.env, seed=args.seed + 20_000, render_mode="rgb_array")
                if args.hover_flip_step > 0 and step >= args.hover_flip_step:
                    try:
                        video_env.set_bonus(-100.0)
                    except AttributeError:
                        pass
            vpath = logdir / "videos" / f"step_{step:08d}.mp4"
            try:
                record_episode(video_env, lambda o: act_for_env(o, det=True), str(vpath))
            except Exception as e:  # noqa: BLE001
                print(f"video recording failed: {e}")
            next_video += args.video_every

        # ----- checkpoint -----
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
    if extra_eval_logger is not None:
        extra_eval_logger.close()
    torch.save(agent.state_dict(), logdir / "final.pt")
    # Always write a final checkpoint, even if --checkpoint-every is 0.
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
