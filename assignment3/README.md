# RL Programming Assignment 3 — Experiment Guide

## Environment Setup

Requires **Python 3.12** (not 3.13 — `labmaze`/`dm_control` wheels don't build on 3.13).

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r Implementation/shared/requirements.txt

# macOS only: needed for box2d-py (LunarLander)
brew install swig
```

## Project Structure

```
Implementation/
├── shared/                        # Core library (used by all experiments)
│   ├── pa3/
│   │   ├── agents/
│   │   │   ├── sac.py             # SAC (continuous + discrete, auto-temperature)
│   │   │   ├── dqn.py             # DQN with target network + ε-greedy
│   │   │   └── pebble.py          # PEBBLE: preference-based RL on top of SAC
│   │   ├── envs/
│   │   │   ├── reacher_dmc.py     # dm_control reacher with Ra/Rb/Rc rewards
│   │   │   ├── lunarlander_hover.py  # LunarLander hover-bonus wrapper
│   │   │   ├── pendulum_variants.py  # Multi-target Pendulum
│   │   │   └── make_env.py        # Environment factory
│   │   └── utils/
│   │       ├── networks.py        # MLP networks (Gaussian/Categorical actor, Q-nets, reward predictor)
│   │       ├── replay.py          # Replay buffer with segment sampling for PEBBLE
│   │       ├── eval.py            # Offline evaluation
│   │       ├── checkpointing.py   # Save/load checkpoints
│   │       ├── video.py           # MP4 recording
│   │       ├── logging.py         # CSV logger
│   │       ├── seeding.py         # seed_everything()
│   │       └── threads.py         # Limits torch to 2 threads (critical for perf)
│   ├── train/
│   │   ├── train_sac.py           # SAC training loop
│   │   ├── train_dqn.py           # DQN training loop
│   │   └── train_pebble.py        # PEBBLE training loop
│   └── requirements.txt
├── LunarLander/DiscreteSAC_DQN/
│   └── run_q2_2.py                # Q2.2 runner (all LunarLander experiments)
├── Reacher/
│   └── run_q2_3.py                # Q2.3 runner (Reacher Ra/Rb/Rc)
└── PEBBLE/
    └── run_q3.py                  # Q3 runner (PEBBLE on Pendulum + Reacher)

```

Generated plots, videos, and checkpoints are produced by the experiment scripts and are not included in this source snapshot.

## Running Experiments

All runner scripts should be executed from the **project root** (the directory containing `pa3/`, `train/`, and `results/`). Copy the runner script there or run with the correct working directory.

### Q2.2.4 — Discrete SAC vs DQN (LunarLander)

```bash
# Full run: 15 seeds, 400K steps each, 4 parallel workers
python run_q2_2.py --experiments sac_disc dqn

# Quick smoke test: 2 seeds, 5K steps
python run_q2_2.py --experiments sac_disc dqn --num-seeds 2 --parallel 2 \
    --total-steps-disc 5000 --video-every 0 --checkpoint-every 0

# Resume interrupted runs
python run_q2_2.py --experiments sac_disc dqn --resume

# Only regenerate plots from existing results
python run_q2_2.py --plot-only
```

**Key parameters:**
| Flag | Default | Description |
|------|---------|-------------|
| `--experiments` | all 5 | Which sub-experiments to run (`sac_cont`, `sac_disc`, `dqn`, `hover_fixed`, `hover_auto`) |
| `--num-seeds` | 15 | Number of random seeds |
| `--parallel` | 4 | Concurrent seed workers |
| `--total-steps-disc` | 400,000 | Training steps for discrete experiments |
| `--resume` | off | Auto-resume from latest checkpoint |
| `--video-every` | 50,000 | Record video every N steps (0 to disable) |
| `--checkpoint-every` | 50,000 | Save checkpoint every N steps |

**Expected runtime:** ~3 hours for `sac_disc` + `dqn` (15 seeds × 400K steps, 4 parallel).

### Q2.3 — Reacher (SAC with Ra/Rb/Rc)

```bash
# Full run: all 3 reward formulations, 15 seeds, 500K steps each
python run_q2_3.py

# Run only specific variants
python run_q2_3.py --variants Ra Rb

# Quick smoke test
python run_q2_3.py --num-seeds 2 --parallel 2 --total-steps 5000

# Only final-policy evaluation (500 episodes × 5000 steps) + plots
python run_q2_3.py --eval-only

# Only regenerate plots
python run_q2_3.py --plot-only

# Resume interrupted runs
python run_q2_3.py --resume
```

**Key parameters:**
| Flag | Default | Description |
|------|---------|-------------|
| `--variants` | Ra Rb Rc | Which reward formulations to train |
| `--num-seeds` | 15 | Number of random seeds |
| `--parallel` | 4 | Concurrent seed workers |
| `--total-steps` | 500,000 | Training steps per variant |
| `--eval-only` | off | Skip training, run final eval + plots only |
| `--plot-only` | off | Skip everything, regenerate plots from CSVs |

**Expected runtime:** ~8 hours for all 3 variants (15 seeds × 500K steps, 4 parallel).

**Outputs:**
- `plots/q2_3_2_Ra.png`, `q2_3_2_Rb.png`, `q2_3_2_Rc.png` — Self-evaluation learning curves
- `plots/q2_3_3a_bar.png` — Steps-to-goal / steps-in-target bar chart
- `plots/q2_3_3c_cross_eval.png` — Cross-evaluation (3 subplots)
- `results/reacher_final_metrics.csv` — Raw bar chart data (500 episodes per seed)

### Q3 — PEBBLE (Bonus)

```bash
# Full run: SAC ground-truth + PEBBLE on Pendulum (4 budgets) + PEBBLE on Reacher (3 teachers)
python run_q3.py

# Run only specific experiments
python run_q3.py --experiments sac_gt pebble_1000       # Pendulum baseline + budget=1000
python run_q3.py --experiments pebble_Ra pebble_Rb      # Reacher PEBBLE only

# Quick smoke test
python run_q3.py --experiments sac_gt pebble_1000 --num-seeds 2 --parallel 2 \
    --pendulum-steps 5000 --reacher-steps 5000

# Resume interrupted runs
python run_q3.py --resume

# Only regenerate plots
python run_q3.py --plot-only
```

**Key parameters:**
| Flag | Default | Description |
|------|---------|-------------|
| `--experiments` | all | Sub-experiments: `sac_gt`, `pebble_100`, `pebble_500`, `pebble_1000`, `pebble_2000`, `pebble_Ra`, `pebble_Rb`, `pebble_Rc` |
| `--num-seeds` | 15 | Number of random seeds |
| `--parallel` | 4 | Concurrent seed workers |
| `--pendulum-steps` | 200,000 | Training steps for Pendulum experiments |
| `--reacher-steps` | 500,000 | Training steps for Reacher experiments |

**Expected runtime:**
- Pendulum (SAC GT + 4 PEBBLE budgets): ~4 hours
- Reacher (3 PEBBLE teachers): ~12 hours

**Outputs:**
- `plots/q3_1_pebble_vs_sac.png` — PEBBLE vs SAC on Pendulum
- `plots/q3_2_budget_sweep.png` — Budget sweep comparison
- `plots/q3_3_pebble_reacher.png` — PEBBLE on Reacher (Ra/Rb/Rc teachers)

## Common Flags (All Runners)

| Flag | Description |
|------|-------------|
| `--parallel N` | Number of concurrent seed workers (tune to CPU cores / scripts running) |
| `--resume` | Auto-resume from latest checkpoint per seed |
| `--plot-only` | Skip training, regenerate plots from existing `results/` CSVs |
| `--no-skip-done` | Re-run even seeds that already have `final.pt` |

## Results Directory

All training outputs go to `results/<experiment>/seed_N/`:
```
results/
├── ll_sac_disc/seed_0/
│   ├── train.csv          # step, return, alpha, elapsed
│   ├── checkpoints/       # step_XXXXXXXX.pt + final.pt
│   └── videos/            # step_XXXXXXXX.mp4
├── reacher_Ra/seed_0/
│   ├── train.csv          # step, return, alpha, elapsed
│   ├── eval_cross.csv     # step, return_Ra, return_Rb, return_Rc
│   ├── checkpoints/
│   └── videos/
└── pebble_pendulum_b1000/seed_0/
    ├── train.csv          # step, return, queries, alpha, elapsed
    ├── checkpoints/
    └── videos/
```

## Performance Notes

- **CPU only** — MPS/GPU is slower for these small networks due to data transfer overhead.
- `torch.set_num_threads(2)` is critical — set automatically by the runners via `PA3_TORCH_THREADS=2`.
- Approximate speeds: LunarLander ~3.8 ms/step, Reacher ~4.6 ms/step, DQN ~3 ms/step.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `labmaze` build fails | Use Python 3.12, not 3.13 |
| `swig` not found (macOS) | `brew install swig` |
| Video recording hangs | Already fixed — lazy `video_env` creation avoids Pygame init issues |
| `torch.load` errors | Checkpoints use `weights_only=False` (already set) |
| Training very slow | Check `torch.get_num_threads()` returns 2, not your total core count |
