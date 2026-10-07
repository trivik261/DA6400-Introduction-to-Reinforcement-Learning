# Pendulum SAC Experiments - Running Instructions

This document provides step-by-step instructions for running all experiments related to Questions 3, 4, and 5.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Question 3: Comparing Different Target Angles](#question-3-comparing-different-target-angles)
3. [Question 4: Optimal Behavior Analysis](#question-4-optimal-behavior-analysis)
4. [Question 5a: Manual vs Automated Alpha Tuning](#question-5a-manual-vs-automated-alpha-tuning)
5. [Question 5b: Reward Scaling Experiments](#question-5b-reward-scaling-experiments)
6. [Analysis and Plotting](#analysis-and-plotting)
7. [Expected Outputs](#expected-outputs)

---

## Prerequisites

### Required Python Packages

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

**Or install manually:**
```bash
pip install numpy matplotlib torch gymnasium scipy pandas
```

### Required External Repository

**IMPORTANT:** This code requires the official `pytorch_sac` repository to be cloned in the parent directory.

```bash
# Navigate to parent directory of Pendulum_V1_final
cd D:\RL\A3

# Clone the official pytorch_sac repository
git clone https://github.com/denisyarats/pytorch_sac.git
```

**Expected directory structure:**
```
D:\RL\A3\
├── pytorch_sac/              # Official SAC implementation (REQUIRED)
│   ├── agent/
│   │   ├── actor.py
│   │   └── critic.py
│   ├── replay_buffer.py
│   └── utils.py
└── Pendulum_V1_final/        # This project
    ├── train.py
    ├── official_sac_adapter.py  # Imports from pytorch_sac
    └── ...
```

**Why this is needed:**
- `official_sac_adapter.py` imports `DiagGaussianActor`, `DoubleQCritic`, and `ReplayBuffer` from the official pytorch_sac repo
- The adapter provides a clean interface while using the official implementation
- This ensures consistency with the assignment requirements

**Verification:**
```bash
# Check if pytorch_sac exists
ls D:\RL\A3\pytorch_sac
```

If the folder doesn't exist, the code will raise an error:
```
ImportError: pytorch_sac repo not found at D:\RL\A3\pytorch_sac
Please clone it with: git clone https://github.com/denisyarats/pytorch_sac.git
```

### File Structure
```
Pendulum_V1_final/
├── train.py                          # Main training script
├── config.json                       # Configuration file
├── official_sac_adapter.py           # SAC agent implementation
├── checkpoint_manager.py             # Checkpoint management
├── early_stopping.py                 # Early stopping logic
├── run_manual_alpha.ps1              # Script for manual alpha experiments
├── run_automatic_alpha.ps1           # Script for automated alpha experiments
├── 5a_manual_angle_adaptive_reward_12Torque.ps1    # Q5a experiments
├── 5b_manual_90_10x_adaptive_reward.ps1            # Q5b manual 10x
├── 5b_manual_90_0.1x_adaptive_reward_12Torque.ps1  # Q5b manual 0.1x
├── 5b_automatic_90_10x_adaptive_reward_12Torque.ps1 # Q5b auto 10x
├── 5b_automatic_90_0.1x_adaptive_reward_12Torque.ps1 # Q5b auto 0.1x
└── outputs/                          # Results directory
```

---

## Question 3: Comparing Different Target Angles

**Objective:** Compare learning curves for different θ_target values in terms of learning speed and behavior achievement.

### Step-by-Step Instructions

#### Option 1: Using Existing Results (Recommended)

The experiments for different target angles have already been run as part of Q5a. The results are in:
- `outputs/5a_manual_angle_adaptive_reward_12Torque/`

This includes experiments for θ_target = {-60°, 90°, 120°, -150°} with optimal α for each angle.

**To analyze existing results:**
```bash
cd D:\RL\A3\Pendulum_Results
python analyze_pendulum_results.py
python plot_learning_curves.py
```

#### Option 2: Run New Experiments for Specific Angles

**Step 1: Edit config.json**

Open `config.json` and set:
```json
{
  "environment": {
    "target_angle_deg": 90,
    "max_torque": 12.0,
    "reward_type": "adaptive",
    "reward_scale": 1.0
  },
  "sac": {
    "type": "automatic",
    "auto_entropy_tuning": true
  },
  "training": {
    "total_timesteps": 60000,
    "eval_frequency": 10000,
    "num_eval_episodes": 20,
    "output_dir": "outputs/Q3_angle_90"
  }
}
```

**Step 2: Run single experiment (test):**
```bash
python train.py --config config.json --target_angle 90 --seed 0
```

**Step 3: Run multiple seeds:**
```powershell
# Run 15 seeds for angle 90
for ($seed=0; $seed -lt 15; $seed++) {
    python train.py --config config.json --target_angle 90 --seed $seed
}
```

**Step 4: Repeat for other angles:**
```powershell
# Run for all angles
$angles = @(-60, 90, 120, -150)
foreach ($angle in $angles) {
    for ($seed=0; $seed -lt 15; $seed++) {
        python train.py --config config.json --target_angle $angle --seed $seed
    }
}
```

**Step 5: Generate videos (optional):**
```bash
# Generate video from best checkpoint for angle 90
python generate_videos.py --checkpoint outputs/Q3_angle_90/checkpoints/best_*.pt --num_episodes 5
```

### Analysis for Q3

Compare the learning curves by examining:
1. **Learning Speed**: How quickly does the agent reach near-optimal performance?
2. **Final Performance**: What is the final average return?
3. **Stability**: How consistent is the learning across seeds?

**Key Metrics:**
- Timesteps to convergence
- Final evaluation reward
- Variance across seeds

---

## Question 4: Optimal Behavior Analysis

**Objective:** Analyze the optimal behavior and identify its two key aspects.

### Step-by-Step Instructions

#### Step 1: Use Trained Models from Q3

Use the same experiments from Q3. The optimal behavior can be analyzed from:
1. The trained policy checkpoints
2. The evaluation rewards during training
3. Visual inspection of the pendulum behavior

#### Step 2: Identify Two Aspects of Optimal Behavior

**Aspect 1: Reaching the Target Angle**
- The agent must swing up and stabilize at θ_target
- Measured by: Final angle error, settling time

**Aspect 2: Minimizing Energy/Torque**
- The agent should use minimal control effort
- Measured by: Cumulative torque magnitude, smoothness

#### Step 3: Generate Videos for Visual Analysis

```bash
# Generate videos for different angles to compare behavior
python generate_videos.py --checkpoint outputs/Q3_angle_90/checkpoints/best_*.pt --num_episodes 5
python generate_videos.py --checkpoint outputs/Q3_angle_-60/checkpoints/best_*.pt --num_episodes 5
python generate_videos.py --checkpoint outputs/Q3_angle_120/checkpoints/best_*.pt --num_episodes 5
```

#### Step 4: Quantitative Analysis (Optional)

Create a simple evaluation script:

**`evaluate_behavior.py`:**
```python
import torch
import numpy as np
from official_sac_adapter import SACAgentForAssignment
import gymnasium as gym

def evaluate_checkpoint(checkpoint_path, num_episodes=10):
    # Load agent
    agent = SACAgentForAssignment(state_dim=3, action_dim=1, 
                                  action_scale=2.0, action_bias=0.0,
                                  auto_entropy_tuning=True, device='cpu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    agent.actor.load_state_dict(checkpoint['actor'])
    
    # Evaluate
    env = gym.make('Pendulum-v1')
    rewards = []
    final_angles = []
    torques = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_torques = []
        
        for step in range(200):
            action = agent.select_action(state, evaluate=True)
            episode_torques.append(abs(action[0]))
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        rewards.append(episode_reward)
        final_angles.append(state[0])  # Final angle
        torques.append(np.mean(episode_torques))
    
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Final Angle Error: {np.mean(np.abs(final_angles)):.3f}")
    print(f"Average Torque Used: {np.mean(torques):.3f}")
    
    env.close()

# Run evaluation
evaluate_checkpoint('outputs/Q3_angle_90/checkpoints/best_*.pt')
```

**Run the evaluation:**
```bash
python evaluate_behavior.py
```

#### Step 5: Compare Results

Compare the two aspects across different target angles:
- **Learning speed**: How many timesteps to reach good performance?
- **Final performance**: What is the best reward achieved?
- **Behavior quality**: How smooth and efficient is the control?

---

## Question 5a: Manual vs Automated Alpha Tuning

**Objective:** Compare manual and automated alpha tuning for θ_target = {-60°, 90°, 120°, -150°}.

### Step-by-Step Instructions

#### Step 1: Run Manual Alpha Tuning Experiments

**Script:** `5a_manual_angle_adaptive_reward_12Torque.ps1`

This script runs experiments for:
- 4 target angles: {-60°, 90°, 120°, -150°}
- 5 alpha values: {0.01, 0.025, 0.05, 0.1, 0.2}
- 15 seeds per configuration
- Total: 300 experiments

**Run the script:**
```powershell
cd D:\RL\A3\Pendulum_V1_final
.\5a_manual_angle_adaptive_reward_12Torque.ps1
```

**What it does:**
1. Waits for any existing Python processes to finish
2. Runs experiments with max 8 concurrent processes
3. Saves results to `outputs/5a_manual_angle_adaptive_reward_12Torque/`
4. Each experiment runs for 60,000 timesteps with evaluations every 10,000 steps

**Expected runtime:** ~10-15 hours (depending on hardware)

**Monitor progress:**
```powershell
# Check how many experiments completed
Get-ChildItem outputs\5a_manual_angle_adaptive_reward_12Torque -Recurse -Filter "results.json" | Measure-Object
```

#### Step 2: Run Single Test Experiment (Optional)

Before running all 300 experiments, test with a single configuration:

```bash
# Test manual alpha for angle 90, alpha 0.01, seed 0
python train.py --config config.json --target_angle 90 --alpha 0.01 --seed 0
```

This should complete in ~5-10 minutes.

#### Step 3: Generate Videos from Best Checkpoints

After experiments complete, generate videos to visualize behavior:

```bash
# Generate video for best performing configuration
python generate_videos.py --checkpoint outputs/5a_manual_angle_adaptive_reward_12Torque/SAC_Manual_alpha0.01_theta90_seed0_*/checkpoints/best_*.pt --num_episodes 5
```

#### Step 4: Run Automated Alpha Tuning for Comparison

For automated tuning, run experiments with the same target angles:

**Quick approach (single angle):**
```bash
# Run automated alpha for angle 90, 15 seeds
for ($seed=0; $seed -lt 15; $seed++) {
    python train.py --config config.json --target_angle 90 --seed $seed
}
```

**Full approach (all angles):**
```powershell
# Run automated alpha for all 4 angles
$angles = @(-60, 90, 120, -150)
foreach ($angle in $angles) {
    for ($seed=0; $seed -lt 15; $seed++) {
        python train.py --config config.json --target_angle $angle --seed $seed
    }
}
```

**Note:** Make sure `config.json` has:
```json
{
  "sac": {
    "type": "automatic",
    "auto_entropy_tuning": true
  }
}
```

#### Step 5: Analyze Results

```bash
cd D:\RL\A3\Pendulum_Results
python analyze_pendulum_results.py
python plot_learning_curves.py
```

This generates:
- Manual alpha sensitivity plots
- Optimal alpha summary (α_mnl for each angle)
- Learning curves with confidence intervals

### Analysis for Q5a

1. **Identify optimal α_mnl for each angle** from manual tuning results
2. **Compare performance** between manual (with optimal α) and automated tuning
3. **Assess difficulty** of manual tuning based on:
   - Performance variance across alpha values
   - Sensitivity to alpha choice
   - Consistency across seeds

**Results location:** `D:\RL\A3\Pendulum_Results\Q5a_Manual_vs_Auto_Alpha\`

---

## Question 5b: Reward Scaling Experiments

**Objective:** Test adaptation to reward scaling (10× and 0.1×) for θ_target = 90° with manual vs automated alpha.

### Experiments Required

For θ_target = 90°, run 4 experiment types:

1. **Manual α (α_mnl = 0.01), 10× reward scaling**
2. **Manual α (α_mnl = 0.01), 0.1× reward scaling**
3. **Automated α, 10× reward scaling**
4. **Automated α, 0.1× reward scaling**

### Running the Experiments

#### 1. Manual α, 10× Reward Scaling (2T torque)

**Script:** `5b_manual_90_10x_adaptive_reward.ps1`

```powershell
.\5b_manual_90_10x_adaptive_reward.ps1
```

**Configuration:**
- θ_target = 90°
- α = 0.01 (from Q5a)
- reward_scale = 10.0
- max_torque = 2.0
- 15 seeds

#### 2. Manual α, 0.1× Reward Scaling (12T torque)

**Script:** `5b_manual_90_0.1x_adaptive_reward_12Torque.ps1`

```powershell
.\5b_manual_90_0.1x_adaptive_reward_12Torque.ps1
```

**Configuration:**
- θ_target = 90°
- α = 0.01
- reward_scale = 0.1
- max_torque = 12.0
- 15 seeds

#### 3. Automated α, 10× Reward Scaling (12T torque)

**Script:** `5b_automatic_90_10x_adaptive_reward_12Torque.ps1`

```powershell
.\5b_automatic_90_10x_adaptive_reward_12Torque.ps1
```

**Configuration:**
- θ_target = 90°
- auto_entropy_tuning = true
- reward_scale = 10.0
- max_torque = 12.0
- 15 seeds

#### 4. Automated α, 0.1× Reward Scaling (12T torque)

**Script:** `5b_automatic_90_0.1x_adaptive_reward_12Torque.ps1`

```powershell
.\5b_automatic_90_0.1x_adaptive_reward_12Torque.ps1
```

**Configuration:**
- θ_target = 90°
- auto_entropy_tuning = true
- reward_scale = 0.1
- max_torque = 12.0
- 15 seeds

### Running All Q5b Experiments

**Run all scripts sequentially:**
```powershell
# Run all Q5b experiments
.\5b_manual_90_10x_adaptive_reward.ps1
.\5b_manual_90_0.1x_adaptive_reward_12Torque.ps1
.\5b_automatic_90_10x_adaptive_reward_12Torque.ps1
.\5b_automatic_90_0.1x_adaptive_reward_12Torque.ps1
```

**Expected runtime:** ~8-12 hours total

### Analysis for Q5b

Compare:
1. **Learning curves** for manual vs automated alpha
2. **Final performance** under different reward scales
3. **Adaptation capability**: Does automated alpha adjust better to scaled rewards?
4. **Variance**: Which method is more stable?

**Results location:** `D:\RL\A3\Pendulum_Results\Q5b_Reward_Scaling\`

---

## Analysis and Plotting

### Using the Analysis Scripts

After running experiments, use the provided analysis scripts in `D:\RL\A3\Pendulum_Results\`:

#### 1. Generate All Plots

```bash
cd D:\RL\A3\Pendulum_Results
python analyze_pendulum_results.py
```

**Generates:**
- Manual alpha tuning sensitivity plots (Q5a)
- Reward scaling comparison plots (Q5b)
- Alpha evolution plots (Q5b)
- Summary CSV files

#### 2. Generate Learning Curves

```bash
python plot_learning_curves.py
```

**Generates:**
- Learning curves with 95% confidence intervals
- Average undiscounted return vs timesteps
- Separate plots for Q5a (4 angles) and Q5b (reward scaling)

### Plot Specifications

All plots follow assignment requirements:
- **X-axis:** Environment Timesteps (NOT episodes)
- **Y-axis:** Average Undiscounted Return
- **Evaluations:** Every 10,000 steps
- **Confidence Intervals:** 95% CI (1.96 × SE)
- **Seeds:** 15 per configuration (or 5 for Q5a manual)

---

## Expected Outputs

### Directory Structure After Running All Experiments

```
Pendulum_V1_final/outputs/
├── 5a_manual_angle_adaptive_reward_12Torque/
│   ├── SAC_Manual_alpha0.01_theta-60_seed0_*/
│   ├── SAC_Manual_alpha0.01_theta90_seed0_*/
│   ├── ... (300 experiment directories)
│   └── results.json (in each directory)
├── 5b_manual_90_10x_adaptive_reward/
│   └── SAC_Manual_alpha0.01_scale10.0_theta90_seed*_*/
├── 5b_manual_90_0.1x_adaptive_reward_12Torque/
│   └── SAC_Manual_alpha0.01_scale0.1_theta90_seed*_*/
├── 5b_automatic_90_10x_adaptive_reward_12Torque/
│   └── SAC_automatic_theta90.0_alphaauto_seed*_*/
└── 5b_automatic_90_0.1x_adaptive_reward_12Torque/
    └── SAC_automatic_theta90.0_alphaauto_seed*_*/
```

### Analysis Results

```
D:\RL\A3\Pendulum_Results/
├── Q5a_Manual_vs_Auto_Alpha/
│   ├── plots/
│   │   ├── manual_alpha_tuning.png
│   │   └── learning_curves_manual_alpha.png
│   └── data/
│       └── optimal_alphas.csv
├── Q5b_Reward_Scaling/
│   ├── plots/
│   │   ├── learning_curves_reward_scaling.png
│   │   ├── reward_scaling_comparison.png
│   │   └── alpha_evolution.png
│   └── data/
│       └── reward_scaling_summary.csv
└── COMPREHENSIVE_ANALYSIS.txt
```

---

## Testing Your Setup

Before running experiments, verify your setup is correct:

```bash
cd D:\RL\A3\Pendulum_V1_final
python test_setup.py
```

**Expected output:**
```
✓ ALL TESTS PASSED - Setup is correct!
```

If any test fails, follow the error messages to fix the issue.

### Quick Test Run

Run a short test experiment to verify everything works:

```bash
# Quick test: 10K steps, angle 0° (easy)
python train.py --config config.json --target_angle 0 --seed 0
```

This should complete in ~2-3 minutes and create output in `outputs/`.

---

## Generating Videos from Checkpoints

After training, you can generate videos from saved checkpoints to visualize agent behavior.

### Using the Video Generation Script

**Script:** `generate_videos.py`

**Basic usage:**
```bash
python generate_videos.py --checkpoint outputs/EXPERIMENT_NAME/checkpoints/best_*.pt
```

**Full options:**
```bash
python generate_videos.py \
    --checkpoint outputs/SAC_manual_theta90_alpha0.01_seed0_*/checkpoints/best_*.pt \
    --output videos/my_experiment \
    --num_episodes 5 \
    --max_steps 200
```

**Parameters:**
- `--checkpoint`: Path to checkpoint file (required)
- `--output`: Output directory for videos (default: experiment_dir/videos)
- `--num_episodes`: Number of episodes to record (default: 3)
- `--max_steps`: Maximum steps per episode (default: 200)

**Example - Generate video from best checkpoint:**
```bash
# Find the best checkpoint
$best_checkpoint = Get-ChildItem "outputs/SAC_manual_theta90_alpha0.01_seed0_*" -Recurse -Filter "best_*.pt" | Select-Object -First 1

# Generate video
python generate_videos.py --checkpoint $best_checkpoint.FullName --num_episodes 5
```

**Output:**
- Videos saved as MP4 files
- Console shows average reward across episodes
- Useful for comparing different checkpoints or experiments

---

## Quick Start Guide

### For Complete Reproduction

```powershell
# 1. Run Q5a: Manual alpha tuning (all angles)
cd D:\RL\A3\Pendulum_V1_final
.\5a_manual_angle_adaptive_reward_12Torque.ps1

# 2. Run Q5b: Reward scaling experiments
.\5b_manual_90_10x_adaptive_reward.ps1
.\5b_manual_90_0.1x_adaptive_reward_12Torque.ps1
.\5b_automatic_90_10x_adaptive_reward_12Torque.ps1
.\5b_automatic_90_0.1x_adaptive_reward_12Torque.ps1

# 3. Generate analysis and plots
cd D:\RL\A3\Pendulum_Results
python analyze_pendulum_results.py
python plot_learning_curves.py
```

### For Quick Testing (Single Seed)

```bash
# Test Q5a: Manual alpha for one angle
python train.py --config config.json --seed 0

# Test Q5b: One reward scaling experiment
python train.py --config config_5b_auto_10x_temp.json --seed 0
```

---

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **PowerShell execution policy**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Out of memory**
   - Reduce concurrent processes in scripts (change `$maxConcurrent` from 8 to 4)
   - Reduce buffer size in config.json

4. **Experiments not starting**
   - Check if Python processes are already running
   - Scripts wait for existing processes to finish

### Monitoring Progress

Check experiment progress:
```powershell
# Count completed experiments
Get-ChildItem outputs\5a_manual_angle_adaptive_reward_12Torque -Recurse -Filter "results.json" | Measure-Object
```

View training logs:
```bash
# View latest training log
Get-ChildItem outputs -Recurse -Filter "training.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 20
```

---

## Contact and Support

For questions about:
- **Experiment setup**: Check config.json and train.py
- **SAC implementation**: See official_sac_adapter.py
- **Results analysis**: See Pendulum_Results/analyze_pendulum_results.py

**Key Files:**
- `train.py` - Main training loop
- `config.json` - Experiment configuration
- `official_sac_adapter.py` - SAC agent
- PowerShell scripts - Batch experiment runners
