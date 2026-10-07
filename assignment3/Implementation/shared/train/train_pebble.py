"""PEBBLE training loop.

Outline (matches Lee et al. 2021):
    1.  Unsupervised pre-training:
        run a vanilla SAC update using a state-entropy intrinsic reward
        (k-NN log-distance) for `unsup_steps` env steps.  No preferences yet.
    2.  Phase 2:  every `sessions_every` env steps, sample `queries_per_session`
        segment pairs from the replay buffer, ask the simulated teacher
        (ground-truth segment-return comparator), train the reward ensemble,
        relabel the entire replay buffer with the new mean predicted reward,
        keep doing SAC updates on those relabelled rewards.
    3.  Stop querying once the preference budget is exhausted; SAC continues to
        train on the *fixed* learned reward.

Logged metrics (env-step indexed):
* `return`     — *ground-truth* (assignment) return of greedy policy (20 eps)
* `learned_return` — return under the learned reward (just for sanity)
* `queries`    — cumulative preference queries used
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from pa3.agents.pebble import PEBBLEAgent, PEBBLEConfig
from pa3.agents.sac import SACConfig
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
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--unsup-steps", type=int, default=9_000)
    p.add_argument("--start-steps", type=int, default=500,
                   help="random exploration steps; 500 for Pendulum, 10K for LL/Reacher.")
    p.add_argument("--update-after", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--seg-len", type=int, default=50)
    p.add_argument("--pref-budget", type=int, default=1_000)
    p.add_argument("--queries-per-session", type=int, default=64)
    p.add_argument("--sessions-every", type=int, default=5_000)
    p.add_argument("--reward-train-epochs", type=int, default=50)
    p.add_argument("--ensemble-size", type=int, default=3)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--auto-alpha", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--video-every", type=int, default=50_000)
    p.add_argument("--logdir", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--teacher", type=str, default=None,
                   help="Reacher only: which reward formulation the simulated teacher uses (Ra/Rb/Rc).")
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
    assert hasattr(env.action_space, "shape"), "PEBBLE in this assignment is continuous-action only"
    action_dim = int(env.action_space.shape[0])

    sac_cfg = SACConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        discrete=False,
        actor_lr=args.lr,
        critic_lr=args.lr,
        alpha_lr=args.lr,
        auto_alpha=args.auto_alpha,
        device=device,
    )
    cfg = PEBBLEConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        ensemble_size=args.ensemble_size,
        seg_len=args.seg_len,
        pref_budget=args.pref_budget,
        queries_per_session=args.queries_per_session,
        sessions_every=args.sessions_every,
        reward_train_epochs=args.reward_train_epochs,
        unsup_steps=args.unsup_steps,
        knn_k=args.knn_k,
        sac=sac_cfg,
    )
    pebble = PEBBLEAgent(cfg)
    buf = ReplayBuffer(args.buffer_size, obs_dim, (action_dim,), np.float32, device=device)
    # Ground-truth reward (used by the simulated teacher only).
    gt_buf_reward = np.zeros(args.buffer_size, dtype=np.float32)

    logger = CSVLogger(logdir / "train.csv")

    obs, _ = env.reset(seed=args.seed)
    t0 = time.time()
    next_eval = args.eval_every
    next_video = args.video_every if args.video_every > 0 else None
    next_session = args.unsup_steps + args.sessions_every
    next_ckpt = args.checkpoint_every if args.checkpoint_every > 0 else None
    ckpt_dir = logdir / "checkpoints"
    start_step = 0

    def act_for_env(o, det=False):
        a = pebble.sac.select_action(o, deterministic=det)
        lo, hi = env.action_space.low, env.action_space.high
        return lo + (a + 1.0) * 0.5 * (hi - lo)

    write_idx = 0  # replay-buffer write pointer (used for ground-truth rewards)

    if args.resume:
        rpath = args.resume
        if rpath == "latest":
            found = find_latest(ckpt_dir)
            rpath = str(found) if found else ""
        if rpath:
            blob = load_checkpoint(rpath, map_location=device)
            extra = blob.get("extra", {}) or {}
            pebble.sac.load_full_state_dict(blob["agent"])
            if "ensemble" in extra:
                pebble.ensemble.load_state_dict(extra["ensemble"])
            if "ensemble_opt" in extra:
                pebble.opt.load_state_dict(extra["ensemble_opt"])
            if "queries_used" in extra:
                pebble.queries_used = int(extra["queries_used"])
            if "pref_segs1" in extra:
                pebble.pref_segs1 = list(extra["pref_segs1"])
                pebble.pref_segs2 = list(extra["pref_segs2"])
                pebble.pref_labels = list(extra["pref_labels"])
            if blob.get("buffer") is not None:
                buf.load_state_dict(blob["buffer"])
                if "gt_buf_reward" in extra:
                    gt_buf_reward[: len(extra["gt_buf_reward"])] = extra["gt_buf_reward"]
                    write_idx = int(extra.get("write_idx", 0))
            start_step = int(blob["step"])
            print(f"[resume] loaded {rpath} at step={start_step} queries={pebble.queries_used}")
            next_eval = ((start_step // args.eval_every) + 1) * args.eval_every
            if next_video is not None:
                next_video = ((start_step // args.video_every) + 1) * args.video_every
            if next_ckpt is not None:
                next_ckpt = ((start_step // args.checkpoint_every) + 1) * args.checkpoint_every

    if start_step == 0:
        mean_ret_0 = evaluate_policy(eval_env, lambda o: act_for_env(o, det=True),
                                     n_episodes=args.eval_episodes)
        logger.log({"step": 0, "return": mean_ret_0, "queries": 0,
                    "alpha": float(pebble.sac.alpha.detach().cpu()), "elapsed": 0.0})
        print(f"[seed={args.seed} step=0] return={mean_ret_0:.2f}  (random-policy baseline)")

    for step in range(start_step + 1, args.total_steps + 1):
        if step < args.start_steps:
            action = env.action_space.sample()
        else:
            action = act_for_env(obs)
        next_obs, reward_gt, terminated, truncated, _ = env.step(action)
        done = terminated

        lo, hi = env.action_space.low, env.action_space.high
        stored_action = 2.0 * (np.asarray(action, dtype=np.float32) - lo) / (hi - lo) - 1.0
        buf.add(obs.astype(np.float32), stored_action, float(reward_gt), next_obs.astype(np.float32), bool(done), bool(terminated or truncated))
        gt_buf_reward[write_idx] = float(reward_gt)
        write_idx = (write_idx + 1) % args.buffer_size

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

        # ----- SAC update -----
        if step >= args.update_after and len(buf) >= args.batch_size:
            batch = buf.sample(args.batch_size)
            if step <= args.unsup_steps:
                # Unsupervised pretraining: replace reward with k-NN state-entropy bonus.
                obs_np = batch.obs.cpu().numpy()
                bonus = pebble.state_entropy_bonus(obs_np, k=args.knn_k)
                batch.reward[:] = torch.as_tensor(bonus, dtype=torch.float32, device=device)
            pebble.sac.update(batch)

        # ----- preference session -----
        if (
            step > args.unsup_steps
            and step >= next_session
            and pebble.queries_used < args.pref_budget
        ):
            sample = buf.sample_segments(args.queries_per_session * 2, args.seg_len)
            if sample is not None:
                seg_obs, seg_act, _, idx = sample
                # Use the *stored ground-truth rewards* for the segments (the simulated teacher).
                # We need to fetch ground-truth reward at those indices.
                seg_rew_gt = gt_buf_reward[idx]
                half = args.queries_per_session
                want = min(args.queries_per_session, args.pref_budget - pebble.queries_used)
                for k in range(want):
                    s1, s2 = 2 * k, 2 * k + 1
                    r1, r2 = seg_rew_gt[s1].sum(), seg_rew_gt[s2].sum()
                    if abs(r1 - r2) < 1e-6:
                        label = 0.5
                    else:
                        label = 1.0 if r2 > r1 else 0.0
                    pebble.add_pref_query(seg_obs[s1], seg_act[s1], seg_obs[s2], seg_act[s2], label)
                metrics = pebble.train_reward()
                # Relabel buffer with the new learned reward (mean of ensemble).
                def _relabel(o, a, _agent=pebble):
                    return _agent.predict_reward(o, a)
                buf.relabel_rewards(_relabel)
                print(f"[seed={args.seed} step={step}] queries={pebble.queries_used} "
                      f"reward_loss={metrics['reward_loss']:.3f} acc={metrics['reward_acc']:.3f}")
            next_session += args.sessions_every

        # ----- evaluation -----
        if step >= next_eval:
            mean_ret = evaluate_policy(
                eval_env,
                lambda o: act_for_env(o, det=True),
                n_episodes=args.eval_episodes,
            )
            logger.log({
                "step": step,
                "return": mean_ret,
                "queries": pebble.queries_used,
                "alpha": float(pebble.sac.alpha.detach().cpu()),
                "elapsed": time.time() - t0,
            })
            pct = step / args.total_steps * 100
            el = time.time() - t0
            eta = el / step * (args.total_steps - step) if step > 0 else 0
            bar = f"[{'█' * int(pct // 5)}{'░' * (20 - int(pct // 5))}]"
            print(f"[seed={args.seed}] {bar} {pct:5.1f}% | step {step}/{args.total_steps} | "
                  f"return={mean_ret:.2f} | queries={pebble.queries_used} | "
                  f"elapsed {el/60:.0f}m{el%60:.0f}s | ETA {eta/60:.0f}m{eta%60:.0f}s")
            next_eval += args.eval_every

        if next_video is not None and step >= next_video:
            if video_env is None:
                video_env = make_env(args.env, seed=args.seed + 20_000, render_mode="rgb_array")
            try:
                record_episode(
                    video_env,
                    lambda o: act_for_env(o, det=True),
                    str(logdir / "videos" / f"step_{step:08d}.mp4"),
                )
            except Exception as e:  # noqa: BLE001
                print(f"video recording failed: {e}")
            next_video += args.video_every

        if next_ckpt is not None and step >= next_ckpt:
            cpath = ckpt_dir / f"step_{step:08d}.pt"
            extra = {
                "ensemble": pebble.ensemble.state_dict(),
                "ensemble_opt": pebble.opt.state_dict(),
                "queries_used": pebble.queries_used,
                "pref_segs1": pebble.pref_segs1,
                "pref_segs2": pebble.pref_segs2,
                "pref_labels": pebble.pref_labels,
                "gt_buf_reward": gt_buf_reward[: buf.size].copy(),
                "write_idx": write_idx,
            }
            try:
                save_checkpoint(
                    cpath,
                    agent_state=pebble.sac.full_state_dict(),
                    step=step,
                    buffer_state=buf.state_dict() if args.save_buffer else None,
                    extra=extra,
                )
                print(f"[ckpt] step={step} queries={pebble.queries_used} -> {cpath}")
            except Exception as e:  # noqa: BLE001
                print(f"checkpoint write failed: {e}")
            next_ckpt += args.checkpoint_every

    logger.close()
    save_checkpoint(
        ckpt_dir / "final.pt",
        agent_state=pebble.sac.full_state_dict(),
        step=args.total_steps,
        buffer_state=buf.state_dict() if args.save_buffer else None,
        extra={
            "ensemble": pebble.ensemble.state_dict(),
            "queries_used": pebble.queries_used,
            "pref_labels": pebble.pref_labels,
        },
    )
    el = time.time() - t0
    print(f"[seed={args.seed}] done in {el/60:.1f}m ({el:.0f}s)")


if __name__ == "__main__":
    main()
