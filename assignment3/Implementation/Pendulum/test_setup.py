"""
Quick test to verify the setup is correct before running experiments
"""
import sys
import os
print("="*80)
print("TESTING PENDULUM SAC SETUP")
print("="*80)
print("\n1. Checking Python packages...")
try:
    import numpy
    import torch
    import gymnasium
    import matplotlib
    import scipy
    import pandas
    print("   ✓ All required packages installed")
except ImportError as e:
    print(f"   ✗ Missing package: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)
print("\n2. Checking pytorch_sac repository...")
pytorch_sac_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pytorch_sac'))
if not os.path.exists(pytorch_sac_path):
    print(f"   ✗ pytorch_sac not found at {pytorch_sac_path}")
    print("   Run: cd .. && git clone https://github.com/denisyarats/pytorch_sac.git")
    sys.exit(1)
else:
    print(f"   ✓ pytorch_sac found at {pytorch_sac_path}")
print("\n3. Testing pytorch_sac imports...")
try:
    sys.path.insert(0, pytorch_sac_path)
    from agent.actor import DiagGaussianActor
    from agent.critic import DoubleQCritic
    from replay_buffer import ReplayBuffer
    print("   ✓ Successfully imported from pytorch_sac")
except ImportError as e:
    print(f"   ✗ Failed to import from pytorch_sac: {e}")
    sys.exit(1)
print("\n4. Testing official_sac_adapter...")
try:
    from official_sac_adapter import SACAgentForAssignment
    print("   ✓ Successfully imported SACAgentForAssignment")
except ImportError as e:
    print(f"   ✗ Failed to import SACAgentForAssignment: {e}")
    sys.exit(1)
print("\n5. Checking config.json...")
if os.path.exists('config.json'):
    print("   ✓ config.json found")
else:
    print("   ✗ config.json not found")
    sys.exit(1)
print("\n6. Testing Gymnasium environment...")
try:
    import gymnasium as gym
    env = gym.make('Pendulum-v1')
    state, _ = env.reset()
    print(f"   ✓ Environment created successfully")
    print(f"   State shape: {state.shape}")
    print(f"   Action space: {env.action_space}")
    env.close()
except Exception as e:
    print(f"   ✗ Environment test failed: {e}")
    sys.exit(1)
print("\n7. Testing SAC agent initialization...")
try:
    agent = SACAgentForAssignment(
        state_dim=3,
        action_dim=1,
        action_scale=2.0,
        action_bias=0.0,
        gamma=0.99,
        tau=0.005,
        lr=0.0003,
        hidden_dim=256,
        auto_entropy_tuning=True,
        device='cpu'
    )
    print("   ✓ SAC agent initialized successfully")
except Exception as e:
    print(f"   ✗ Agent initialization failed: {e}")
    sys.exit(1)
print("\n" + "="*80)
print("✓ ALL TESTS PASSED - Setup is correct!")
print("="*80)
print("\nYou can now run experiments:")
print("  python train.py --config config.json --target_angle 90 --seed 0")
print("\nOr run the test experiment:")
print("  python train.py --config config.json --target_angle 0 --seed 0")
print("="*80)
