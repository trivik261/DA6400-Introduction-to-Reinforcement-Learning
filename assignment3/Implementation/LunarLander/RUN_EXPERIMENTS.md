# LunarLander Experiment Commands

All experiments use the unified `train_sac.py` script with different configs.

## Quick Reference

| Experiment | Command | Buffer | Seeds |
|------------|---------|--------|-------|
| **Q1/Q2** | `--experiment q1` | 1M (infinite-like) | 8 |
| **Q3 Fixed α** | `--experiment q3_fixed` | 100K (fixed) | 8 |
| **Q3 Auto** | `--experiment q3_auto` | 100K (fixed) | 8 |

---

## Q1 & Q2: Default LunarLander with Auto Tuning

**Single seed:**
```bash
python train_sac.py --experiment q1 --seed 0
```

**All 8 seeds (parallel):**
```bash
# PowerShell
0..7 | ForEach-Object { Start-Process python -ArgumentList "train_sac.py --experiment q1 --seed $_" -NoNewWindow }

# Bash
for seed in {0..7}; do
    python train_sac.py --experiment q1 --seed $seed &
done
```

**Configuration:**
- Hover reward: 0 (default environment)
- Auto entropy tuning: Yes
- Buffer size: 1M (infinite-like, per TA)
- Random exploration: 10K steps (per TA)
- Total timesteps: 500K

**Use for:**
- Q1: Proving SAC implementation works
- Q2: Analyzing behavior at different training stages

---

## Q3: Reward Switching Experiments

### Q3 - Fixed Alpha (α = 0.01)

**Single seed:**
```bash
python train_sac.py --experiment q3_fixed --seed 0
```

**All 8 seeds (parallel):**
```bash
# PowerShell
0..7 | ForEach-Object { Start-Process python -ArgumentList "train_sac.py --experiment q3_fixed --seed $_" -NoNewWindow }

# Bash
for seed in {0..7}; do
    python train_sac.py --experiment q3_fixed --seed $seed &
done
```

**Configuration:**
- Initial hover reward: +200
- Final hover reward: -100
- Switch timestep: 250K
- Auto entropy tuning: No
- Manual alpha: 0.01 (fixed)
- Buffer size: 100K (fixed, per TA)
- Random exploration: 10K steps
- Total timesteps: 500K

---

### Q3 - Auto Tuning

**Single seed:**
```bash
python train_sac.py --experiment q3_auto --seed 0
```

**All 8 seeds (parallel):**
```bash
# PowerShell
0..7 | ForEach-Object { Start-Process python -ArgumentList "train_sac.py --experiment q3_auto --seed $_" -NoNewWindow }

# Bash
for seed in {0..7}; do
    python train_sac.py --experiment q3_auto --seed $seed &
done
```

**Configuration:**
- Initial hover reward: +200
- Final hover reward: -100
- Switch timestep: 250K
- Auto entropy tuning: Yes
- Buffer size: 100K (fixed, per TA)
- Random exploration: 10K steps
- Total timesteps: 500K

---

## Full Experiment Pipeline

### Step 1: Q1/Q2 (8 seeds)
```bash
# Run all 8 seeds in parallel
0..7 | ForEach-Object { Start-Process python -ArgumentList "train_sac.py --experiment q1 --seed $_" -NoNewWindow }
```
**Time:** ~2-3 hours (with 8 workers, all run simultaneously)

### Step 2: Q3 Fixed Alpha (8 seeds)
```bash
# Run all 8 seeds in parallel
0..7 | ForEach-Object { Start-Process python -ArgumentList "train_sac.py --experiment q3_fixed --seed $_" -NoNewWindow }
```
**Time:** ~2-3 hours

### Step 3: Q3 Auto Tuning (8 seeds)
```bash
# Run all 8 seeds in parallel
0..7 | ForEach-Object { Start-Process python -ArgumentList "train_sac.py --experiment q3_auto --seed $_" -NoNewWindow }
```
**Time:** ~2-3 hours

**Total time:** ~6-9 hours for all LunarLander experiments

---

## Monitoring Progress

Check training progress:
```bash
# View latest results
Get-ChildItem outputs -Recurse -Filter "results.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content | ConvertFrom-Json

# Count completed runs
(Get-ChildItem outputs -Recurse -Filter "results.json").Count
```

---

## Output Structure

```
outputs/
├── SAC_default_auto_seed0_20260503_011234/
│   ├── run_config.json          # Experiment configuration
│   ├── results.json              # Training results (rewards, alpha)
│   ├── best_model.pt             # Best checkpoint
│   ├── final_model.pt            # Final checkpoint
│   └── checkpoint_*.pt           # Periodic checkpoints
├── SAC_Q3_hover+200to-100_alpha0.01_seed0_.../
└── SAC_Q3_hover+200to-100_auto_seed0_.../
```

---

## TA Requirements ✅

All implemented in `train_sac.py`:

1. **Official SAC**: Uses `SACAgentForAssignment` from pytorch_sac repo
2. **Random exploration**: 10K steps for LunarLander (500 for Pendulum)
3. **Replay buffer**:
   - Q1/Q2: 1M (infinite-like)
   - Q3: 100K (fixed, stores latest samples)
4. **Reward switching**: At 250K timesteps for Q3

---

## Plotting Results

After experiments complete:

```bash
# Single experiment
python plot_results.py --results outputs/SAC_default_auto_seed0_.../results.json --output plots/q1_learning_curve.png

# Q3 comparison
python plot_results.py --type q3 \
    --results outputs/SAC_Q3_hover+200to-100_auto_seed0_.../results.json \
    --manual-results outputs/SAC_Q3_hover+200to-100_alpha0.01_seed0_.../results.json \
    --output plots/q3_comparison.png
```
