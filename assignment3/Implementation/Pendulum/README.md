# Pendulum V1 Final - Unified SAC Training Framework

A unified, configuration-driven SAC training framework for the Pendulum environment. All experiments are controlled via `config.json` - no need to create new Python scripts for different experiments.

## Features

✅ **Configuration-Driven**: All settings controlled via `config.json`
✅ **Reward Types**: 
  - `default`: Fixed velocity penalty (0.1 for all angles)
  - `adaptive`: Angle-dependent velocity penalty (0.02 for ±90°, 0.05 for moderate, 0.1 for easy)
✅ **SAC Types**:
  - Manual alpha: Specify `--alpha <value>`
  - Automatic alpha: Omit `--alpha` flag
✅ **High Torque Support**: Configurable `max_torque` (default 2.0, can increase to 12.0 for ±90°)
✅ **Reward Scaling**: Configurable `reward_scale` multiplier
✅ **Full Features**: Checkpointing, early stopping, evaluation, plotting
✅ **Parallel Execution**: PowerShell scripts for running multiple experiments

## Quick Start

### 1. Single Experiment

**Automatic alpha tuning:**
```powershell
python train.py --config config.json --target_angle 90 --seed 0
```

**Manual alpha:**
```powershell
python train.py --config config.json --target_angle 90 --alpha 0.01 --seed 0
```

### 2. Parallel Experiments

**Manual alpha (multiple alpha values and seeds):**
```powershell
.\run_manual_alpha.ps1 -Angle 90 -AlphaValues "0.01,0.025,0.05,0.1,0.2" -Seeds "0,1,2,3,4"
```

**Automatic alpha (multiple angles and seeds):**
```powershell
.\run_automatic_alpha.ps1 -Angles "0,30,60,90" -Seeds "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
```

## Configuration

### config.json Structure

```json
{
  "training": {
    "target_angle": 0,
    "max_episode_steps": 1000,
    "total_timesteps": 200000,
    "random_exploration_steps": 500,
    "eval_freq": 10000,
    "num_eval_episodes": 20,
    "device": "auto",
    "output_dir": "outputs"
  },
  "environment": {
    "reward_type": "default",     // "default" or "adaptive"
    "reward_scale": 1.0,           // Reward multiplier
    "max_torque": 2.0              // Control torque limit (12.0 for high torque)
  },
  "sac": {
    "gamma": 0.99,
    "tau": 0.005,
    "learning_rate": 0.0003,
    "hidden_dim": 256,
    "buffer_size": 1000000,
    "batch_size": 256
  },
  "features": {
    "enable_checkpoints": true,
    "checkpoint_freq": 10000,
    "keep_best_n": 3,
    "keep_last_n": 2,
    "enable_early_stopping": true,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 5.0
  }
}
```

### Key Configuration Options

**Reward Type:**
- `"default"`: Fixed velocity penalty = 0.1 (same for all angles)
- `"adaptive"`: Angle-dependent velocity penalty
  - 0.02 for ±90° (allows oscillation)
  - 0.05 for moderate angles (60-80°, 100-120°)
  - 0.1 for easy angles

**SAC Type:**
- Automatic: Don't specify `--alpha` flag (uses entropy tuning)
- Manual: Specify `--alpha <value>` (fixed temperature)

**Device:**
- `"auto"`: Automatically use CUDA if available, else CPU
- `"cpu"`: Force CPU
- `"cuda"`: Force CUDA

## Example Configurations

Just edit `config.json` for different experiments:

### Example 1: High Torque + Adaptive Reward + Manual Alpha (for 90°)
Edit config.json:
```json
{
  "environment": {
    "reward_type": "adaptive",
    "reward_scale": 1.0,
    "max_torque": 12.0
  },
  "sac": {
    "type": "manual",
    "manual_alpha": 0.01,
    ...
  }
}
```
Run: `python train.py --config config.json --target_angle 90 --seed 0`

### Example 2: Default Reward + 10x Scaling + Automatic Alpha
Edit config.json:
```json
{
  "environment": {
    "reward_type": "default",
    "reward_scale": 10.0,
    "max_torque": 2.0
  },
  "sac": {
    "type": "automatic",
    ...
  }
}
```
Run: `python train.py --config config.json --target_angle 90 --seed 0`

### Example 3: Manual Alpha with Command-Line Override
Keep config.json as-is, override alpha via command line:
```powershell
python train.py --config config.json --target_angle 90 --alpha 0.05 --seed 0
```

## Output Structure

Each experiment creates a folder in `outputs/`:
```
outputs/
  SAC_manual_theta90_alpha0.01_seed0_20260511_013000/
    config.json              # Full experiment configuration
    results.json             # Training results and metrics
    training_results.png     # Comprehensive plots
    checkpoints/             # Model checkpoints
      best_t50000_r-4.28_*.pt
      checkpoint_t100000_*.pt
```

## Parallel Execution Scripts

### run_manual_alpha.ps1
Runs multiple manual alpha experiments in parallel.

**Parameters:**
- `-Angle`: Target angle (single value)
- `-AlphaValues`: Comma-separated alpha values
- `-Seeds`: Comma-separated seed numbers
- `-Config`: Config file path (optional, default: config.json)

**Example:**
```powershell
.\run_manual_alpha.ps1 -Angle 90 -AlphaValues "0.01,0.05,0.1" -Seeds "0,1,2,3,4"
```

### run_automatic_alpha.ps1
Runs multiple automatic alpha tuning experiments in parallel.

**Parameters:**
- `-Angles`: Comma-separated target angles
- `-Seeds`: Comma-separated seed numbers
- `-Config`: Config file path (optional, default: config.json)

**Example:**
```powershell
.\run_automatic_alpha.ps1 -Angles "0,30,60,90" -Seeds "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
```

## Files

- `train.py`: Main training script
- `config.json`: Configuration file (edit this for different experiments)
- `run_manual_alpha.ps1`: Parallel execution for manual alpha
- `run_automatic_alpha.ps1`: Parallel execution for automatic alpha
- `checkpoint_manager.py`: Checkpoint management
- `early_stopping.py`: Early stopping logic
- `official_sac_adapter.py`: SAC agent wrapper

## Tips

1. **For difficult angles (±90°)**: Use `max_torque: 12.0` and `reward_type: "adaptive"`
2. **For reward scaling experiments**: Adjust `reward_scale` (e.g., 0.1, 1.0, 10.0)
3. **For alpha tuning**: Use manual alpha with multiple values to find optimal
4. **Monitor jobs**: Use `Get-Job` in PowerShell to check running experiments
5. **Device selection**: Use `"auto"` for automatic GPU detection

## Requirements

- Python 3.9+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- Official pytorch_sac implementation

## Workflow

1. Edit `config.json` for your experiment settings
2. Run single experiment to test: `python train.py --config config.json --target_angle 90 --seed 0`
3. Run parallel experiments: `.\run_manual_alpha.ps1 -Angle 90 -AlphaValues "..." -Seeds "..."`
4. Analyze results in `outputs/` folder
5. Modify config and repeat as needed

**No need to create new Python scripts - just change the config!**
