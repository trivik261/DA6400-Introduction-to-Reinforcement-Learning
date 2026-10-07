# LunarLander-v3 SAC Experiments

This folder contains SAC implementation for LunarLander-v3 continuous action space.

## Project Structure

```
LunarLander_V3_SAC/
├── config.json              # Training configuration
├── sac_agent.py            # SAC agent implementation
├── lunar_lander_env.py     # Modified environment with hover reward
├── train.py                # Main training script
├── train_q3.py             # Q3 experiment (reward switching)
├── plot_results.py         # Plotting utilities
├── outputs/                # Training results
├── checkpoints/            # Saved models
├── videos/                 # Recorded episodes
└── plots/                  # Generated plots
```

## Questions Covered

### Q1: SAC Implementation [4M]
- Implemented SAC with automated temperature tuning
- Continuous action space for LunarLander-v3
- Target entropy = -action_dim

### Q2: Training Behavior Analysis [4M]
- Train agent and analyze behavior at different stages:
  - Initial: Random exploration
  - Intermediate: Learning to hover, local optima
  - Final: Near-optimal landing policy

### Q3: Reward Switching Experiment [14M]
- Modified environment with hover reward (+200 in box)
- Two SAC versions:
  - Fixed α = 0.01
  - Automated temperature tuning
- After training, switch hover reward to -100
- Compare adaptation between fixed and auto versions

## Usage

### Q1 & Q2: Basic Training

```bash
# Train with automated temperature tuning (default)
python train.py --seed 0

# Train with manual alpha
python train.py --seed 0 --manual-alpha 0.01
```

### Q3: Hover Reward Experiments

```bash
# Phase 1: Train with +200 hover reward
python train.py --seed 0 --hover-reward 200 --auto-entropy
python train.py --seed 0 --hover-reward 200 --manual-alpha 0.01

# Phase 2: Continue training with -100 hover reward (use train_q3.py)
python train_q3.py --seed 0 --initial-reward 200 --final-reward -100
```

## Configuration

Key parameters in `config.json`:

- `total_timesteps`: 500,000 (adjust as needed)
- `eval_freq`: 10,000 (evaluate every 10K steps)
- `num_eval_episodes`: 20
- `auto_entropy_tuning`: true/false
- `manual_alpha`: 0.01 (for fixed alpha experiments)

## Environment Details

**State Space (8 dimensions):**
- x, y position
- x, y velocity
- angle, angular velocity
- left leg contact, right leg contact

**Action Space (2 dimensions, continuous):**
- Main engine thrust: [-1, 1]
- Side engine thrust: [-1, 1]

**Hover Box:**
- x: [-0.1, 0.1]
- y: [0.4, 0.6]
- Reward given once per episode when entering box

## Expected Results

### Q2: Behavior Analysis
- **Initial stage**: Random actions, crashes
- **Intermediate stage**: Learns to hover, may get stuck in local optima (hovering without landing)
- **Final stage**: Smooth landing between flags

### Q3: Reward Switching
- **Fixed α = 0.01**: Struggles to adapt to reward change
- **Auto tuning**: Adapts better by adjusting exploration (alpha increases)

## Analysis Questions

**Q3(a)**: Compare learning curves before/after reward switch
- Fixed alpha: Slower adaptation, may not recover
- Auto tuning: Faster adaptation, alpha increases to explore

**Q3(b)**: Maximum entropy formulation
- Entropy bonus encourages exploration
- Auto tuning adjusts exploration based on learning progress
- When reward changes, auto tuning increases alpha to re-explore
- Fixed alpha cannot adapt exploration level
