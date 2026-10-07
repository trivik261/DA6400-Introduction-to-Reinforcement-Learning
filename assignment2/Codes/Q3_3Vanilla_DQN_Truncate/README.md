# RL MountainCar DQN Framework

A complete, modular reinforcement learning framework for running DQN experiments on MountainCar-v0, designed for Q3.1-Q3.4(a) assignments.

## 🔬 Truncation Usage

### **For Q3.3 Experiments:**
```bash
# Run truncation comparison automatically - Use this for 3.3. The values are hardcoded for 200, 1000 and 2000
python main.py --mode truncation

# Results in: experiments/Experiment_Q3.3_YYYYMMDD_HHMMSS/
├── data/
│   ├── truncation_stats.json    # Statistical analysis
│   └── comparison_plots.png   # Visual comparison
```

### **For Custom Analysis:**
```bash
# Compare specific truncation values
python main.py --mode plot --truncation-dirs exp1 exp2 exp3 --truncation-values 200 1000 2000
```


#### **To Modify Truncation:**
```json
{
  "environment": {
    "max_episode_steps": 1000  // Change here
  }
}
```

## ⚠️ Mode Exclusions for Truncation Experiments

### **Disabled Features in Truncation Mode:**
- **Replay Factor**: `replay_factor` parameter ignored
- **Replay Factors**: `replay_factors` array ignored  
- **Multi-factor Analysis**: Only single `replay_factor = 1` used
- **Factor Comparison**: No replay factor comparisons performed

### **Active Features in Truncation Mode:**
- **Episode Length**: `max_episode_steps` controls truncation
- **Multiple Values**: `truncation_values` array processed
- **Statistical Analysis**: Truncation rates and performance
- **Comparison Plots**: Side-by-side truncation comparison
- **Video Recording**: Standard video generation enabled

### **Mode-Specific Behavior:**
```json
{
  "training": {
    "replay_factor": 1,        // Fixed, ignored from config
    "update_every": 4           // Standard update frequency
  },
  "experiment": {
    "truncation_values": [200, 1000, 2000],  // Active
    "replay_factors": [1, 2, 4, 8]           // Ignored
  }
}
```

### **Why These Exclusions?**
- **Focus**: Pure truncation length analysis
- **Isolation**: Remove replay factor confounding variables
- **Fair Comparison**: Standard replay buffer behavior
- **Assignment Requirements**: Q3.3 focuses on truncation only
- **Clean Results**: Clear truncation impact analysis

### **If Replay Factors Needed:**
```bash
# Use standard training mode instead
python main.py --mode train

# Or use replay factor mode
python main.py --mode replay_factor
```




## 🚀 Features

- **Vanilla DQN Implementation** - Strict assignment-compliant DQN with no forbidden variants
- **Modular Design** - Independent modules for training, evaluation, plotting, and video recording
- **Multi-seed Training** - Support for exactly 15 random seeds as required
- **Automatic Checkpointing** - Save and resume training capabilities
- **Comprehensive Logging** - Detailed experiment tracking and statistics
- **Video Generation** - Record and visualize agent performance
- **Statistical Analysis** - Learning curves with 95% confidence intervals
- **Truncation Comparison** - Built-in support for Q3.3 experiments

## 📁 Project Structure

```
RL_MountainCar/
├── main.py                 # Main CLI interface
├── config.json            # Default configuration
├── config_loader.py       # Configuration management
├── dqn_model.py          # DQN agent implementation
├── replay_buffer.py      # Uniform replay buffer
├── trainer.py            # Training logic
├── evaluator.py          # Model evaluation
├── plotter.py            # Result visualization
├── video_recorder.py     # Video generation
├── experiment_manager.py # Experiment orchestration
└── README.md             # This file
```

## 🛠️ Installation

Install required dependencies:

```bash
pip install torch torchvision
pip install gymnasium[classic_control]
pip install numpy matplotlib seaborn tqdm imageio scipy pandas
```

## ⚙️ Configuration

The framework uses JSON configuration files. Key parameters:

```json
{
  "environment": {
    "name": "MountainCar-v0",
    "max_episode_steps": 2000
  },
  "training": {
    "num_episodes": 500,
    "num_seeds": 15,
    "batch_size": 64,
    "learning_rate": 0.001,
    "target_update_freq": 100,
    "replay_buffer_size": 10000,
    "update_every": 4
  },
  "exploration": {
    "eps_start": 1.0,
    "eps_end": 0.01,
    "eps_decay": 0.995
  },
  "experiment": {
    "experiment_type": "Q3.1",
    "truncation_values": [2000],
    "enable_video": true,
    "save_checkpoints": true
  },
  "system": {
    "device": "cpu",
    "fast_mode": false
  }
}
```

## 🎯 Usage Examples

### Basic Training

```bash
# Train with default config (Q3.1)
python main.py --mode train

# Train specific experiment type
python main.py --mode train --experiment Q3.2

# Use parallel training (default)
python main.py --mode train --parallel

# Use sequential training
python main.py --mode train --no-parallel
```

### Q3.3 Truncation Comparison

```bash
# Run all truncation experiments automatically
python main.py --mode truncation
```

### Evaluation

```bash
# Evaluate trained model
python main.py --mode eval --model experiments/Experiment_Q3.1_20231201_120000/models/best_model.pth

# Evaluate for specific number of episodes
python main.py --mode eval --model best_model.pth --episodes 50
```

### Plotting

```bash
# Plot learning curve from data
python main.py --mode plot --data experiments/Experiment_Q3.1_20231201_120000/data/all_returns.npy

# Compare multiple experiments
python main.py --mode plot --data exp1/data/all_returns.npy --compare exp2/data/all_returns.npy exp3/data/all_returns.npy

# Plot truncation comparison
python main.py --mode plot --truncation-dirs exp1 exp2 exp3 --truncation-values 200 1000 2000
```

### Video Generation

```bash
# Generate videos of trained model
python main.py --mode video --model experiments/Experiment_Q3.1_20231201_120000/models/best_model.pth

# Generate specific number of episodes
python main.py --mode video --model best_model.pth --episodes 5 --video-dir my_videos
```

## 📊 Experiment Results

Each experiment creates a structured directory:

```
Experiment_Q3.1_20231201_120000/
├── config/
│   └── config.json              # Experiment configuration
├── models/
│   ├── best_model.pth           # Best performing model
│   ├── final_model.pth          # Final model
│   └── checkpoints/            # Training checkpoints
├── data/
│   ├── all_returns.npy          # All returns (seeds × episodes)
│   ├── seed_summary.json       # Per-seed statistics
│   └── truncation_stats.json   # Truncation statistics
├── plots/
│   ├── learning_curve.png       # Learning curve with CI
│   ├── seed_performance.png     # Individual seed performance
│   └── statistics.json         # Detailed statistics
├── videos/
│   ├── episode_001.mp4          # Recorded episodes
│   └── ...
├── logs/
│   └── experiment.log           # Experiment logs
└── experiment_info.json        # Complete experiment summary
```

## 🔧 Module Usage

### Independent Module Execution

All modules can be run independently:

```bash
# Plot results directly
python plotter.py --data path/to/all_returns.npy

# Evaluate model directly
python evaluator.py --model best_model.pth --episodes 100

# Record videos directly
python video_recorder.py --model best_model.pth --episodes 3
```

### Checkpoint Management

```bash
# Resume training from checkpoint
python main.py --mode train --resume checkpoints/model_ep_200.pth

# Load and evaluate specific checkpoint
python main.py --mode eval --model checkpoints/model_ep_200.pth
```

## 📈 Assignment Compliance

✅ **Q3.1**: Basic DQN experiment with 2000-step truncation  
✅ **Q3.2**: Extended DQN experiment  
✅ **Q3.3**: Truncation comparison (200, 1000, 2000 steps)  
✅ **Q3.4(a)**: Additional analysis  

### Strict Constraints Enforced

- **Environment**: MountainCar-v0 with discrete actions only
- **Observations**: Raw continuous observations (no discretization)
- **Rewards**: Default Gymnasium rewards (no shaping)
- **Discount**: Fixed γ = 0.99
- **Algorithm**: Vanilla DQN only (no Double, Dueling, etc.)
- **Seeds**: Exactly 15 random seeds
- **Network**: Reasonable architecture with Kaiming initialization

## 🎯 Expected Performance

The framework is designed to achieve near-optimal performance:

- **Target Return**: -100 to -120
- **Convergence**: Typically within 200-400 episodes
- **Final Performance**: Mean return ≈ -110 ± 20

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use `--device cpu` in config
2. **Slow Training**: Use `--fast_mode true` to skip optimizer saving
3. **Missing Dependencies**: Install all required packages
4. **Permission Errors**: Ensure write access to experiment directory

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 API Reference

### ConfigLoader

```python
from config_loader import ConfigLoader, ExperimentConfig

config = ConfigLoader.load_config("config.json")
ConfigLoader.validate_config_for_experiment(config)
```

### DQNTrainer

```python
from trainer import DQNTrainer

trainer = DQNTrainer(config, seed=0, experiment_dir="exp")
results = trainer.train()
```

### ExperimentManager

```python
from experiment_manager import ExperimentManager

manager = ExperimentManager(config)
results = manager.run_complete_experiment()
```

## 🤝 Contributing

1. Follow the existing code style
2. Add type hints to all functions
3. Include docstrings for new modules
4. Test with all experiment types

## 📄 License

This framework is designed for educational purposes and assignment compliance.

## 🙏 Acknowledgments

- OpenAI Gym for the MountainCar environment
- PyTorch for deep learning implementation
- Assignment requirements for guiding the design

---

**🚀 Ready to start your RL experiments!**
