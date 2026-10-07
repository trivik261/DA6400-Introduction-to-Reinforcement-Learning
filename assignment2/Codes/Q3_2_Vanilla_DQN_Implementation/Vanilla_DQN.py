# -*- coding: utf-8 -*-
"""
@author: Sharan Sharmeela Karuppannan
Simplified DQN script for ONLY:
Q3.1 (Vanilla DQN)
Q3.2 (a) Plot mean + CI
Q3.2 (b) Behavior analysis (handled in report)
"""

import gymnasium as gym
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
from scipy.stats import t
import sys
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# =====================
# HYPERPARAMETERS
# =====================
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 3000
TARGET_UPDATE = 10
LR = 1e-3
MEMORY_SIZE = 50000
NUM_SEEDS = 15
NUM_EPISODES = 1200

# SPEED OPTIMIZATION SETTINGS
FAST_MODE = False  # False for submission (2000 steps), True for development (1000 steps)

if FAST_MODE:
    UPDATE_EVERY = 4  # Optimize every 4 steps
    CHECKPOINT_FREQ = 200  # Checkpoint every 200 episodes
    PRINT_FREQ = 100  # Print every 100 episodes
    DEVICE = "cpu"  # CPU is faster for tiny networks
    BATCH_SIZE = 128  # Larger batch for fewer updates
    MAX_EPISODE_STEPS = 1000  # Faster early training
else:
    UPDATE_EVERY = 1  # Optimize every step
    CHECKPOINT_FREQ = 50  # Checkpoint every 50 episodes
    PRINT_FREQ = 50  # Print every 50 episodes
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64  # Original batch size
    MAX_EPISODE_STEPS = 2000  # Full training steps

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# =====================
# REPLAY BUFFER
# =====================
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# =====================
# DQN NETWORK
# =====================
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# =====================
# UTILITIES
# =====================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def verify_device():
    """Verify and display actual device being used"""
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    device = torch.device(DEVICE)
    print(f"Configured Device: {DEVICE}")
    print(f"Actual Device: {device}")
    
    # Test with small tensor
    test_tensor = torch.randn(1000, 1000).to(device)
    if device.type == 'cuda':
        print(f"Test tensor on GPU: {test_tensor.device}")
        print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    else:
        print(f"Test tensor on CPU: {test_tensor.device}")
    
    return device


def select_action(state, policy_net, env, device, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1), steps_done
    else:
        return torch.tensor([[env.action_space.sample()]], device=device), steps_done


# =====================
# OPTIMIZATION
# =====================
def optimize_model(memory, policy_net, target_net, optimizer, device):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) if any(non_final_mask) else None

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        if non_final_next_states is not None:
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected = reward_batch + GAMMA * next_state_values

    loss = nn.SmoothL1Loss()(state_action_values, expected.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# =====================
# TRAINING LOOP
# =====================
def train_one_seed(seed):
    set_seed(seed)

    env = gym.make("MountainCar-v0", max_episode_steps=MAX_EPISODE_STEPS)
    device = torch.device(DEVICE)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    n_actions = env.action_space.n
    
    # Get initial state for observation space size
    state, _ = env.reset()
    n_obs = len(state)

    policy_net = DQN(n_obs, n_actions).to(device)
    target_net = DQN(n_obs, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0
    returns = []
    episode_times = []

    # Create checkpoint folder for this seed
    checkpoint_folder = f"checkpoints/seed_{seed+1:02d}"
    os.makedirs(checkpoint_folder, exist_ok=True)
    
    print(f"  Training {NUM_EPISODES} episodes...")
    print(f"  Checkpoints saved in: {checkpoint_folder}/")
    if FAST_MODE:
        print(f"  FAST MODE: {DEVICE.upper()}, {UPDATE_EVERY}x updates, {CHECKPOINT_FREQ}x checkpoints")
    sys.stdout.flush()

    for ep in range(NUM_EPISODES):
        episode_start_time = time.time()
        
        # Reset environment for each episode!
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        ep_return = 0

        for t in count():
            action, steps_done = select_action(state, policy_net, env, device, steps_done)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            reward = torch.tensor([reward], device=device)
            ep_return += reward.item()

            next_state = None if done else torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            state = next_state

            if steps_done % UPDATE_EVERY == 0:
                optimize_model(memory, policy_net, target_net, optimizer, device)

            if done:
                returns.append(ep_return)
                break

        episode_end_time = time.time()
        episode_time = episode_end_time - episode_start_time
        episode_times.append(episode_time)

        if (ep + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (ep + 1) % PRINT_FREQ == 0:
            avg_return = np.mean(returns[-50:]) if len(returns) >= 50 else np.mean(returns)
            avg_time = np.mean(episode_times[-50:]) if len(episode_times) >= 50 else np.mean(episode_times)
            remaining_episodes = NUM_EPISODES - (ep + 1)
            eta_minutes = (remaining_episodes * avg_time) / 60
            
            print(f"  Episode {ep+1}/{NUM_EPISODES}: Avg 50-ep return = {avg_return:.2f}, "
                  f"Avg time/ep = {avg_time:.2f}s, ETA = {eta_minutes:.1f}min")
            sys.stdout.flush()

        if (ep + 1) % CHECKPOINT_FREQ == 0:
            # Save model checkpoint (smaller in fast mode)
            checkpoint_data = {
                'episode': ep + 1,
                'model_state_dict': policy_net.state_dict(),
                'returns': returns,
                'episode_times': episode_times,
                'seed': seed
            }
            
            # Skip optimizer state in fast mode to reduce file size
            if not FAST_MODE:
                checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
            
            checkpoint_path = os.path.join(checkpoint_folder, f"model_ep_{ep+1:04d}.pth")
            torch.save(checkpoint_data, checkpoint_path)
            
            # Save raw returns for this seed
            returns_path = os.path.join(checkpoint_folder, f"returns_ep_{ep+1:04d}.npy")
            np.save(returns_path, returns)
            
            print(f"    Checkpoint saved: {checkpoint_path}")
            sys.stdout.flush()

    total_time = sum(episode_times)
    avg_episode_time = np.mean(episode_times)
    
    # Final checkpoint (all episodes complete)
    final_checkpoint_data = {
        'episode': NUM_EPISODES,
        'model_state_dict': policy_net.state_dict(),
        'returns': returns,
        'episode_times': episode_times,
        'seed': seed
    }
    
    if not FAST_MODE:
        final_checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
    
    final_checkpoint_path = os.path.join(checkpoint_folder, "model_final.pth")
    torch.save(final_checkpoint_data, final_checkpoint_path)
    
    # Save final returns
    final_returns_path = os.path.join(checkpoint_folder, "returns_final.npy")
    np.save(final_returns_path, returns)
    
    print(f"  Seed completed: Total time = {total_time/60:.1f}min, "
          f"Avg time/episode = {avg_episode_time:.2f}s")
    print(f"    Final checkpoint: {final_checkpoint_path}")
    sys.stdout.flush()

    env.close()
    return returns, policy_net  # Return both returns and trained model


# =====================
# PLOTTING
# =====================
def plot_results(all_returns):
    all_returns = np.array(all_returns)

    mean = np.mean(all_returns, axis=0)
    std = np.std(all_returns, axis=0, ddof=1)

    t_val = t.ppf(0.975, df=len(all_returns) - 1)
    ci = t_val * std / np.sqrt(len(all_returns))

    episodes = np.arange(1, len(mean) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean, label="Vanilla DQN")
    plt.fill_between(episodes, mean - ci, mean + ci, alpha=0.2)

    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.title("DQN on MountainCar-v0")
    plt.grid()
    plt.legend()

    plt.savefig("q3_dqn_plot.png", dpi=300)
    plt.show()


# =====================
# UTILITY FUNCTIONS
# =====================
def load_all_returns(filepath="q3_all_returns.npy"):
    """Load all returns for replotting"""
    try:
        all_returns = np.load(filepath)
        print(f"[OK] Loaded returns from {filepath}")
        print(f"  Shape: {all_returns.shape} (seeds × episodes)")
        return all_returns
    except FileNotFoundError:
        print(f"[ERROR] File {filepath} not found!")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading {filepath}: {e}")
        return None


def replot_from_saved(filepath="q3_all_returns.npy", save_path="q3_dqn_plot_reloaded.png"):
    """Replot results from saved returns file"""
    all_returns = load_all_returns(filepath)
    if all_returns is None:
        return
    
    print("Replotting from saved data...")
    plot_results(all_returns)
    
    # Save with new name
    plt.savefig(save_path, dpi=300)
    print(f"Replotted saved as: {save_path}")
    plt.show()


def load_checkpoint(checkpoint_path):
    """Load a specific checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"[OK] Loaded checkpoint: {checkpoint_path}")
        print(f"  Episode: {checkpoint['episode']}")
        print(f"  Seed: {checkpoint['seed']}")
        print(f"  Returns shape: {len(checkpoint['returns'])}")
        return checkpoint
    except FileNotFoundError:
        print(f"[ERROR] Checkpoint {checkpoint_path} not found!")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading checkpoint: {e}")
        return None


def list_checkpoints():
    """List all available checkpoints"""
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory {checkpoints_dir} not found!")
        return
    
    print("[INFO] Available checkpoints:")
    for seed_dir in sorted(os.listdir(checkpoints_dir)):
        seed_path = os.path.join(checkpoints_dir, seed_dir)
        if os.path.isdir(seed_path):
            files = sorted(os.listdir(seed_path))
            print(f"  {seed_dir}/: {len(files)} files")
            for file in files[:3]:  # Show first 3 files
                print(f"    - {file}")
            if len(files) > 3:
                print(f"    ... and {len(files)-3} more files")


# =====================
# VIDEO RECORDING (Q3.2b)
# =====================
def record_trained_agent(model_path="q3_best_dqn.pth", num_episodes=3):
    """Record videos of the trained agent behavior for Q3.2(b)"""
    
    video_folder = "mountaincar_videos"
    os.makedirs(video_folder, exist_ok=True)
    
    env = gym.make("MountainCar-v0", max_episode_steps=2000, render_mode="rgb_array")
    
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda x: True, 
        name_prefix="mountaincar_dqn"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_actions = env.action_space.n
    state, _ = env.reset()
    n_obs = len(state)
    
    policy_net = DQN(n_obs, n_actions).to(device)
    
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[OK] Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"[ERROR] Model file {model_path} not found!")
        print("Please complete training first to generate the model.")
        return
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return
    
    print(f"\n=== Recording MountainCar-v0 Videos (Q3.2b) ===")
    print(f"Videos will be saved in: ./{video_folder}/")
    print(f"Recording {num_episodes} episodes...")
    print("Watch for the valley momentum strategy!\n")
    
    episode_results = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_return = 0
        
        print(f"Recording Episode {episode + 1}/{num_episodes}...")
        
        for step in range(2000):
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(1, 1)
            
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            episode_return += reward
            
            if terminated:
                print(f"  [GOAL] REACHED! Steps: {step + 1}, Return: {episode_return:.2f}")
                break
            elif truncated:
                print(f"  [TIME] Episode truncated. Steps: {step + 1}, Return: {episode_return:.2f}")
                break
            
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        episode_results.append({
            'episode': episode + 1,
            'return': episode_return,
            'steps': step + 1,
            'success': terminated
        })
        
        print(f"  Video saved in folder: {video_folder}/")
    
    env.close()
    
    print(f"\n=== Recording Summary (Q3.2b) ===")
    success_count = sum(1 for r in episode_results if r['success'])
    avg_return = np.mean([r['return'] for r in episode_results])
    avg_steps = np.mean([r['steps'] for r in episode_results])
    
    print(f"Episodes recorded: {num_episodes}")
    print(f"Successful episodes: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"Average return: {avg_return:.2f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Video folder: ./{video_folder}/")
    
    summary_text = f"""
MountainCar-v0 DQN Performance Summary (Q3.2b)
============================================

Episodes Analyzed: {len(episode_results)}

Performance Metrics:
- Success Rate: {success_count}/{len(episode_results)} ({success_count/len(episode_results)*100:.1f}%)
- Average Return: {avg_return:.2f}
- Average Steps: {avg_steps:.1f}

Strategy Analysis:
The car demonstrates the valley momentum strategy:
1. Drives LEFT to build potential energy
2. Uses gravity to accelerate through valley
3. Momentum carries car up RIGHT hill to goal

This is why the car doesn't drive directly to the target!

Episode Details:
"""
    
    for result in episode_results:
        status = "[SUCCESS]" if result['success'] else "[FAILED]"
        summary_text += f"\nEpisode {result['episode']}: Return = {result['return']:.2f}, Steps = {result['steps']}, {status}"
    
    # Save summary
    with open("mountaincar_summary.txt", "w") as f:
        f.write(summary_text)
    
    print("Summary saved as: mountaincar_summary.txt")
    return episode_results


def visualize_live_agent(model_path="q3_best_dqn.pth", num_episodes=3):
    """Live visualization of trained agent (alternative to video recording)"""
    
    env = gym.make("MountainCar-v0", max_episode_steps=2000, render_mode="human")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_actions = env.action_space.n
    state, _ = env.reset()
    n_obs = len(state)
    
    policy_net = DQN(n_obs, n_actions).to(device)
    
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[OK] Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"[ERROR] Model file {model_path} not found!")
        print("Please complete training first to generate the model.")
        return
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return
    
    print("\n=== Live MountainCar-v0 Visualization (Q3.2b) ===")
    print("Watch how the car builds momentum to reach the goal!")
    print("The car will NOT drive directly to the target.")
    print("Instead, it will build potential energy by going up the opposite hill first.\n")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_return = 0
        
        print(f"Episode {episode + 1}:")
        print("Position: [LEFT] --- [MIDDLE] --- [RIGHT GOAL]")
        
        for step in range(2000):
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(1, 1)
            
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            episode_return += reward
            
            position = obs[0]
            action_name = ["LEFT", "NEUTRAL", "RIGHT"][action.item()]
            
            if step % 50 == 0: 
                if position < -0.5:
                    pos_desc = "LEFT HILL"
                elif position > 0.5:
                    pos_desc = "RIGHT HILL (GOAL!)"
                else:
                    pos_desc = "VALLEY"
                
                print(f"  Step {step:4d}: Position={position:6.3f} ({pos_desc}), Action={action_name}")
            
            if terminated:
                print(f"  [GOAL] REACHED! Final position: {position:.3f}")
                print(f"  [DATA] Episode return: {episode_return:.2f}")
                break
            elif truncated:
                print(f"  [TIME] Episode truncated. Final position: {position:.3f}")
                print(f"  [DATA] Episode return: {episode_return:.2f}")
                break
            
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        print("-" * 50)
        time.sleep(2) 
    
    env.close()


# =====================
# AUTO-RUN PIPELINE
# =====================
def run_complete_q3():
    """Execute complete Q3 pipeline automatically"""
    print("\n" + "=" * 60)
    print("COMPLETE Q3 PIPELINE")
    print("=" * 60)
    
    # Check if training already completed
    if os.path.exists("q3_best_dqn.pth") and os.path.exists("q3_all_returns.npy"):
        print("\nFound existing trained model and results!")
        print("Skipping training step...")
    else:
        # Step 1: Train DQN (Q3.1 + Q3.2a)
        print("\nSTEP 1: Training DQN (Q3.1 + Q3.2a)")
        print("-" * 40)
        train_dqn()
    
    # Step 2: Record videos (Q3.2b)
    print("\nSTEP 2: Recording Videos (Q3.2b)")
    print("-" * 40)
    record_trained_agent("q3_best_dqn.pth", num_episodes=5)
    
    # Step 3: Final summary
    print("\n" + "=" * 60)
    print("Q3 COMPLETED SUCCESSFULLY!")
    print("Q3 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Files generated:")
    print("  q3_dqn_plot.png (performance plot with CI)")
    print("  q3_all_returns.npy (raw results - 15×1200)")
    print("  q3_trained_dqn.pth (last seed model)")
    print("  q3_best_dqn.pth (best performing model)")
    print("  checkpoints/ (detailed checkpoints per seed)")
    print("=" * 60)


# =====================
# MAIN
# =====================
def main():
    print("MountainCar-v0 DQN - Complete Q3 Implementation")
    print("=" * 50)
    print(f"FAST MODE: {'ON' if FAST_MODE else 'OFF'}")
    if FAST_MODE:
        print(f"   Device: {DEVICE.upper()}")
        print(f"   Updates: Every {UPDATE_EVERY} steps")
        print(f"   Checkpoints: Every {CHECKPOINT_FREQ} episodes")
        print(f"   Max steps: {MAX_EPISODE_STEPS}")
    print("=" * 50)
    print(f"FAST MODE: {'ON' if FAST_MODE else 'OFF'}")
    if FAST_MODE:
        print(f"   Device: {DEVICE.upper()}")
        print(f"   Updates: Every {UPDATE_EVERY} steps")
        print(f"   Checkpoints: Every {CHECKPOINT_FREQ} episodes")
        print(f"   Max steps: {MAX_EPISODE_STEPS}")
    print("=" * 50)
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        print("AUTO-RUN MODE: Executing complete Q3 pipeline...")
        run_complete_q3()
        return
    
    print("Options:")
    print("1. Train DQN (Q3.1 + Q3.2a)")
    print("2. Record videos of trained agent (Q3.2b)")
    print("3. Live visualization of trained agent (Q3.2b)")
    print("4. Train + Record videos (Complete Q3)")
    print("5. Load and replot from saved data")
    print("6. List available checkpoints")
    print("7. Load specific checkpoint")
    print("8. Toggle Fast Mode (current: " + ("ON" if FAST_MODE else "OFF") + ")")
    print("9. Auto-run complete Q3 pipeline")
    print("10. Custom video recording (specify episode count)")
    
    choice = input("\nEnter choice (1-10): ").strip()
    
    if choice == "9":
        run_complete_q3()
        return
    
    if choice == "8":
        print(f"\n FAST MODE toggle requires script restart")
        print("Current settings:")
        print(f"  FAST_MODE: {'ON' if FAST_MODE else 'OFF'}")
        print(f"  Device: {DEVICE.upper()}")
        print(f"  Updates: Every {UPDATE_EVERY} steps")
        print(f"  Checkpoints: Every {CHECKPOINT_FREQ} episodes")
        print(f"  Max steps: {MAX_EPISODE_STEPS}")
        print("\nTo toggle FAST_MODE, change line 45 in Q3.py:")
        print("  FAST_MODE = True  # For development (fast)")
        print("  FAST_MODE = False # For submission (compliant)")
        print("\nThen restart the script.")
        return
    
    if choice == "1" or choice == "4":
        print("\n" + "=" * 50)
        print("TRAINING DQN (Q3.1 + Q3.2a)")
        print("=" * 50)
        train_dqn()
        
        if choice == "4":
            print("\n" + "=" * 50)
            print("RECORDING VIDEOS (Q3.2b)")
            print("=" * 50)
            record_trained_agent("q3_best_dqn.pth", num_episodes=5)
            
    elif choice == "2":
        print("\n" + "=" * 50)
        print("RECORDING VIDEOS (Q3.2b)")
        print("=" * 50)
        record_trained_agent("q3_best_dqn.pth", num_episodes=5)
        
    elif choice == "3":
        print("\n" + "=" * 50)
        print("LIVE VISUALIZATION (Q3.2b)")
        print("=" * 50)
        visualize_live_agent("q3_best_dqn.pth", num_episodes=3)
        
    elif choice == "5":
        print("\n" + "=" * 50)
        print("LOAD AND REPLOT FROM SAVED DATA")
        print("=" * 50)
        replot_from_saved()
        
    elif choice == "6":
        print("\n" + "=" * 50)
        print("LIST AVAILABLE CHECKPOINTS")
        print("=" * 50)
        list_checkpoints()
        
    elif choice == "7":
        print("\n" + "=" * 50)
        print("LOAD SPECIFIC CHECKPOINT")
        print("=" * 50)
        checkpoint_path = input("Enter checkpoint path (e.g., checkpoints/seed_01/model_ep_0600.pth): ").strip()
        load_checkpoint(checkpoint_path)
        
    elif choice == "10":
        print("\n" + "=" * 50)
        print("CUSTOM VIDEO RECORDING (Q3.2b)")
        print("=" * 50)
        try:
            num_episodes = int(input("Enter number of episodes to record (1-20): "))
            if num_episodes < 1 or num_episodes > 20:
                print("Please enter a number between 1 and 20.")
                return
        except ValueError:
            print("Invalid input. Please enter a number.")
            return
        
        record_trained_agent("q3_best_dqn.pth", num_episodes=num_episodes)
    
    else:
        print("Invalid choice! Please enter 1-10.")
        return
        
    print("\n" + "=" * 50)
    print("Q3 COMPLETED!")
    print("Files generated:")
    print("- q3_dqn_plot.png (performance plot)")
    print("- q3_all_returns.npy (raw results - 15×1200)")
    print("- q3_trained_dqn.pth (last seed model)")
    print("- q3_best_dqn.pth (best performing model)")
    print("- checkpoints/ (detailed checkpoints per seed)")
    
    if choice in ["2", "4"]:
        print("  mountaincar_videos/ (MP4 videos)")
        print("  mountaincar_summary.txt (performance analysis)")
    
    print("=" * 50)


def train_dqn():
    """Main training function for Q3.1 and Q3.2a"""
    print("Running ONLY Q3 experiments (Vanilla DQN)...")
    
    print(f"Device: {DEVICE.upper()}")
    if torch.cuda.is_available() and DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"Configuration: {NUM_SEEDS} seeds, {NUM_EPISODES} episodes per seed")
    print("=" * 50)
    sys.stdout.flush()
    
    all_returns = []
    overall_start_time = time.time()
    best_return = -float('inf')
    best_model = None
    best_seed = 0

    for seed in range(NUM_SEEDS):
        seed_start_time = time.time()
        print(f"Seed {seed+1}/{NUM_SEEDS}")
        sys.stdout.flush()
        
        returns, trained_model = train_one_seed(seed)
        all_returns.append(returns)
        
        final_performance = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
        if final_performance > best_return:
            best_return = final_performance
            best_model = trained_model
            best_seed = seed + 1
        print(f"  New best model! Seed {seed+1}: {final_performance:.2f}")
        
        if seed == NUM_SEEDS - 1:
            final_model_path = "q3_trained_dqn.pth"
            torch.save(trained_model.state_dict(), final_model_path)
            print(f"Final trained model saved as {final_model_path}")
        
        seed_time = time.time() - seed_start_time
        remaining_seeds = NUM_SEEDS - (seed + 1)
        eta_minutes = (remaining_seeds * seed_time) / 60
        
        print(f"  Seed time: {seed_time/60:.1f}min, Overall ETA: {eta_minutes:.1f}min")
        sys.stdout.flush()

    overall_time = time.time() - overall_start_time
    plot_results(all_returns)
    
    np.save("q3_all_returns.npy", all_returns)
    print("All returns saved as: q3_all_returns.npy")
    
    if best_model is not None:
        best_model_path = "q3_best_dqn.pth"
        torch.save(best_model.state_dict(), best_model_path)
        print(f"Best model saved as {best_model_path} (Seed {best_seed}, Return: {best_return:.2f})")

    print(f"\nAll experiments completed in {overall_time/60:.1f} minutes!")
    print("Files saved:")
    print("  - q3_dqn_plot.png (performance plot)")
    print("  - q3_all_returns.npy (raw results - 15×1200)")
    print("  - q3_trained_dqn.pth (last seed model)")
    print("  - q3_best_dqn.pth (best performing model)")
    print("  - checkpoints/ (detailed checkpoints per seed)")


if __name__ == "__main__":
    main()
