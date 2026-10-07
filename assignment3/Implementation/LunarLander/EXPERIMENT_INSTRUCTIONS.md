# LunarLander SAC Experiments - Running Instructions

Complete instructions for running all LunarLander-v3 SAC experiments (Questions 1, 2, and 3).

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Question 1: SAC with Automated Temperature Tuning](#question-1-sac-with-automated-temperature-tuning)
3. [Question 2: Training Stage Analysis](#question-2-training-stage-analysis)
4. [Question 3: Hover Reward Experiments](#question-3-hover-reward-experiments)
5. [Generating Videos](#generating-videos)
6. [Analysis and Plotting](#analysis-and-plotting)

---

## Prerequisites

### Required Python Packages

**Install dependencies:**
```bash
pip install -r requirements.txt
```

### Required External Repository

**IMPORTANT:** This code requires the official `pytorch_sac` repository.

**Expected directory structure:**
```
D:\RL\A3\Implementation\
├── pytorch_sac/                 # Official SAC implementation (REQUIRED - shared by both)
│   ├── agent/
│   │   ├── actor.py
│   │   └── critic.py
│   ├── replay_buffer.py
│   └── utils.py
├── Pendulum/                    # Pendulum experiments
│   ├── official_sac_adapter.py
│   └── ...
└── LunarLander/                 # LunarLander experiments
    ├── official_sac_adapter.py  # SAC adapter (included)
    ├── train_sac.py
    ├── train_q3.py
    └── ...
```

**Clone pytorch_sac:**
```bash
cd D:\RL\A3\Implementation
git clone https://github.com/denisyarats/pytorch_sac.git
```

**Note:** Both Pendulum and LunarLander **share the same pytorch_sac** repository at `D:\RL\A3\Implementation\pytorch_sac`.

**Setup verification:**
```bash
cd D:\RL\A3\Pendulum\LunarLander
python test_setup.py
```

**Expected output:**
```
✓ ALL TESTS PASSED - Setup is correct!
```

---

## Question 1: SAC with Automated Temperature Tuning

**Objective:** Implement SAC with automated temperature tuning on continuous action LunarLander-v3.

### Step-by-Step Instructions

#### Step 1: Understand the Configuration

The `config.json` file controls all experiment settings:

```json
{
  "environment": {
    "name": "LunarLander-v3",
    "continuous": true,
    "hover_reward": 0,        // 0 = default, 200 = hover bonus, -100 = hover penalty
    "hover_box": {
      "x_range": [-0.1, 0.1],
      "y_range": [0.4, 0.6]
    }
  },
  "sac": {
    "auto_entropy_tuning": true,  // Automated temperature tuning
    "alpha": 0.2,                  // Initial alpha (or fixed if auto_entropy_tuning=false)
    "gamma": 0.99,
    "tau": 0.005,
    "lr": 0.0003,
    "hidden_dim": 256,
    "buffer_size": 1000000,
    "batch_size": 256
  },
  "training": {
    "total_timesteps": 500000,
    "eval_frequency": 10000,
    "num_eval_episodes": 20,
    "random_exploration_steps": 10000,
    "output_dir": "outputs/Q1_Q2_Experiment"
  }
}
```

#### Step 2: Run Single Experiment (Test)

Test with a quick run:

```bash
# Quick test: 50K steps, seed 0
python train_sac.py --config config.json --seed 0
```

This should complete in ~10-15 minutes and create output in `outputs/Q1_Q2_Experiment/`.

#### Step 3: Run Full Q1/Q2 Experiments

**Using the provided script (15 seeds):**

```powershell
.\run_q1_q2.ps1
```

**What it does:**
- Runs 15 seeds (0-14)
- Each seed: 500K timesteps
- Evaluations every 10K steps
- Saves results to `outputs/Q1_Q2_Experiment/`
- Uses automated temperature tuning

**Expected runtime:** ~3-5 hours (depending on hardware)

**Monitor progress:**
```powershell
# Check how many experiments completed
Get-ChildItem outputs\Q1_Q2_Experiment -Recurse -Filter "results.json" | Measure-Object

# Monitor running experiments
.\monitor_progress.ps1
```

#### Step 4: Verify Results

After completion, check the results:

```bash
# List all experiment directories
Get-ChildItem outputs\Q1_Q2_Experiment -Directory

# Check a specific result
cat outputs\Q1_Q2_Experiment\SAC_default_auto_seed0_*\results.json
```

Each experiment directory contains:
- `results.json` - Training metrics and evaluation rewards
- `training_progress.log` - Detailed training logs
- `run_config.json` - Experiment configuration
- `checkpoints/` - Saved model checkpoints

---

## Question 2: Training Stage Analysis

**Objective:** Explain SAC agent behavior in initial, intermediate, and final training stages. Identify local optima.

### Step-by-Step Instructions

#### Step 1: Use Q1 Results

The Q1/Q2 experiments provide data for all training stages:
- **Initial stage**: 0-50K timesteps
- **Intermediate stages**: 50K-300K timesteps
- **Final stage**: 300K-500K timesteps

#### Step 2: Analyze Training Stages

**Run the analysis script:**

```bash
python analyze_lunarlander.py
```

**What it generates:**
- Learning curves showing all training stages
- Statistical analysis of each stage
- Identification of local optima
- Plots saved to `lunarlander_analysis/`

#### Step 3: Generate Videos for Each Stage

Visualize agent behavior at different stages:

```bash
# Generate videos from checkpoints at different timesteps
python generate_videos_official_sac.py
```

**Manual video generation for specific checkpoints:**

```bash
# Initial stage (e.g., 50K steps)
python -c "
import gymnasium as gym
from generate_videos_official_sac import load_checkpoint_and_generate_video
load_checkpoint_and_generate_video(
    'outputs/Q1_Q2_Experiment/SAC_default_auto_seed0_*/checkpoints/checkpoint_t50000_*.pt',
    'videos/initial_stage',
    'initial_50k',
    num_episodes=5
)
"

# Intermediate stage (e.g., 200K steps)
# Final stage (e.g., 500K steps - best model)
```

#### Step 4: Identify Training Stages

Based on the analysis, document:

**Initial Stage (0-50K steps):**
- Random exploration
- Learning basic physics
- High variance in rewards

**Intermediate Stages (50K-300K steps):**
- Learning to land
- Possible local optima (e.g., hovering without landing)
- Gradual improvement

**Final Stage (300K-500K steps):**
- Near-optimal landing policy
- Consistent high rewards
- Stable performance

**Local Optima:**
- Hovering in place (gets positive reward but doesn't land)
- Landing with high velocity (lands but crashes)
- Oscillating near landing pad

---

## Question 3: Hover Reward Experiments

**Objective:** Compare fixed α vs automated α when reward changes from +200 hover bonus to -100 hover penalty.

### Overview

This question requires 4 phases:
1. **Phase 1**: Train with +200 hover reward (both fixed and auto α)
2. **Phase 2**: Continue training with -100 hover penalty (both versions)
3. **Analysis**: Compare adaptation before and after reward change

### Step-by-Step Instructions

#### Step 1: Run Fixed Alpha Experiments (α = 0.01)

**Script:** `run_q3_fixed.ps1`

```powershell
.\run_q3_fixed.ps1
```

**What it does:**
- Runs 15 seeds with fixed α = 0.01
- Phase 1: 250K steps with +200 hover reward
- Phase 2: 250K more steps with -100 hover penalty
- Total: 500K steps per seed
- Saves to `outputs/Q3_Fixed/`

**Configuration used:**
```json
{
  "sac": {
    "auto_entropy_tuning": false,
    "alpha": 0.01
  },
  "environment": {
    "hover_reward": 200  // Changes to -100 at 250K steps
  }
}
```

#### Step 2: Run Automated Alpha Experiments

**Script:** `run_q3_auto.ps1`

```powershell
.\run_q3_auto.ps1
```

**What it does:**
- Runs 15 seeds with automated α tuning
- Same phase structure as fixed α
- Saves to `outputs/Q3_Auto_Output/`

**Configuration used:**
```json
{
  "sac": {
    "auto_entropy_tuning": true
  },
  "environment": {
    "hover_reward": 200  // Changes to -100 at 250K steps
  }
}
```

#### Step 3: Monitor Progress

```powershell
# Check Q3 Fixed progress
Get-ChildItem outputs\Q3_Fixed -Recurse -Filter "results.json" | Measure-Object

# Check Q3 Auto progress
Get-ChildItem outputs\Q3_Auto_Output -Recurse -Filter "results.json" | Measure-Object

# Use monitor script
.\monitor_progress.ps1
```

#### Step 4: Analyze Results

**Run the analysis:**

```bash
cd D:\RL\A3\LunarLander_Results
python analyze_pendulum_results.py  # This also analyzes LunarLander results
python plot_learning_curves.py
```

**Or manually analyze:**

```bash
cd D:\RL\A3\Pendulum\LunarLander
python analyze_lunarlander.py
```

**What to compare:**

**Before Reward Change (0-250K steps):**
- Both versions learn to hover (get +200 bonus)
- Compare learning speed
- Compare final performance with hover bonus

**After Reward Change (250K-500K steps):**
- Fixed α: How quickly does it adapt to penalty?
- Auto α: How quickly does it adapt to penalty?
- Which version adapts better?

**Key Metrics:**
- Reward drop magnitude when penalty introduced
- Recovery speed after reward change
- Final performance with penalty
- Alpha evolution (for auto α version)

#### Step 5: Generate Comparison Videos

```bash
# Videos before reward change (e.g., 200K steps)
python generate_videos_official_sac.py

# Videos after reward change (e.g., 400K steps)
# Compare behavior: Does agent still hover or avoid hover box?
```

---

## Generating Videos

### Using the Video Generation Script

**Script:** `generate_videos_official_sac.py`

**Basic usage:**
```bash
python generate_videos_official_sac.py
```

This automatically generates videos for all experiments.

**Manual video generation:**

```python
import gymnasium as gym
from generate_videos_official_sac import load_checkpoint_and_generate_video

# Generate video from specific checkpoint
load_checkpoint_and_generate_video(
    checkpoint_path='outputs/Q1_Q2_Experiment/SAC_default_auto_seed0_*/checkpoints/best_*.pt',
    output_dir='videos/q1_q2',
    experiment_name='q1_q2_best',
    num_episodes=5
)
```

**Output:**
- Videos saved as MP4 files
- Shows agent behavior over multiple episodes
- Useful for qualitative analysis

---

## Analysis and Plotting

### Automated Analysis

**Run comprehensive analysis:**

```bash
python analyze_lunarlander.py
```

**Generates:**
- Learning curves for all experiments
- Training stage analysis (Q2)
- Hover reward comparison plots (Q3)
- Statistical summaries
- Saved to `lunarlander_analysis/`

### Manual Plotting

**Using plot_results.py:**

```bash
python plot_results.py --experiment outputs/Q1_Q2_Experiment/SAC_default_auto_seed0_*
```

**Custom analysis:**

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results from multiple seeds
results = []
for seed in range(15):
    path = f'outputs/Q1_Q2_Experiment/SAC_default_auto_seed{seed}_*/results.json'
    with open(path) as f:
        results.append(json.load(f))

# Plot learning curves
rewards = [r['eval_rewards'] for r in results]
mean_rewards = np.mean(rewards, axis=0)
std_rewards = np.std(rewards, axis=0)
timesteps = results[0]['eval_timesteps']

plt.plot(timesteps, mean_rewards)
plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
plt.xlabel('Timesteps')
plt.ylabel('Average Return')
plt.title('SAC Learning Curve (15 seeds)')
plt.savefig('learning_curve.png')
```

---

## Expected Outputs

### Directory Structure After All Experiments

```
outputs/
├── Q1_Q2_Experiment/
│   ├── SAC_default_auto_seed0_*/
│   │   ├── results.json
│   │   ├── training_progress.log
│   │   ├── run_config.json
│   │   └── checkpoints/
│   ├── SAC_default_auto_seed1_*/
│   └── ... (15 seeds total)
├── Q3_Fixed/
│   ├── SAC_hover_fixed_seed0_*/
│   └── ... (15 seeds total)
└── Q3_Auto_Output/
    ├── SAC_hover_auto_seed0_*/
    └── ... (15 seeds total)

videos/
├── q1_q2/
├── q3_fixed/
└── q3_auto/

lunarlander_analysis/
├── q2_training_stages.png
├── q3_hover_comparison.png
├── learning_curves.png
└── analysis_report.txt
```

---

## Quick Start Commands

### Complete Workflow

```powershell
# 1. Test setup
cd D:\RL\A3\Pendulum\LunarLander
python test_setup.py

# 2. Run Q1/Q2 experiments (15 seeds)
.\run_q1_q2.ps1

# 3. Run Q3 experiments
.\run_q3_fixed.ps1
.\run_q3_auto.ps1

# 4. Generate analysis
python analyze_lunarlander.py

# 5. Generate videos
python generate_videos_official_sac.py
```

### Quick Test (Single Seed)

```bash
# Test Q1/Q2
python train_sac.py --config config.json --seed 0

# Test Q3 Fixed
python train_q3.py --config config.json --alpha 0.01 --seed 0

# Test Q3 Auto
python train_q3.py --config config.json --seed 0
```

---

## Troubleshooting

### Common Issues

1. **"Module not found: official_sac_adapter"**
   - Ensure Pendulum folder exists at `D:\RL\A3\Pendulum\Pendulum\`
   - Run `python test_setup.py` to verify

2. **"pytorch_sac not found"**
   - Clone pytorch_sac in parent directory:
   ```bash
   cd D:\RL\A3
   git clone https://github.com/denisyarats/pytorch_sac.git
   ```

3. **Experiments not starting**
   - Check if Python processes are running: `Get-Process python`
   - Scripts wait for existing processes to finish

4. **Out of memory**
   - Reduce batch_size in config.json
   - Reduce buffer_size in config.json

### Monitoring

```powershell
# Check experiment progress
.\monitor_progress.ps1

# View latest training log
Get-ChildItem outputs -Recurse -Filter "training_progress.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 20
```

---

## Summary of Experiments

| Question | Script | Seeds | Timesteps | Config | Output |
|----------|--------|-------|-----------|--------|--------|
| **Q1** | `run_q1_q2.ps1` | 15 | 500K | Auto α | `outputs/Q1_Q2_Experiment/` |
| **Q2** | (uses Q1 data) | 15 | 500K | Auto α | Analysis from Q1 |
| **Q3 Fixed** | `run_q3_fixed.ps1` | 15 | 500K | α=0.01 | `outputs/Q3_Fixed/` |
| **Q3 Auto** | `run_q3_auto.ps1` | 15 | 500K | Auto α | `outputs/Q3_Auto_Output/` |

**Total experiments:** 45 (15 + 15 + 15)  
**Total runtime:** ~10-15 hours (with parallel execution)

---

## Key Files

- `train_sac.py` - Main training script for Q1/Q2
- `train_q3.py` - Training script for Q3 (hover reward experiments)
- `lunar_lander_env.py` - Custom environment wrapper
- `sac_agent.py` - SAC agent implementation (backup)
- `config.json` - Main configuration file
- `run_q1_q2.ps1` - Script to run Q1/Q2 experiments
- `run_q3_fixed.ps1` - Script to run Q3 with fixed α
- `run_q3_auto.ps1` - Script to run Q3 with auto α
- `analyze_lunarlander.py` - Analysis script
- `generate_videos_official_sac.py` - Video generation
- `test_setup.py` - Setup verification

---

**All experiments ready to run!**
