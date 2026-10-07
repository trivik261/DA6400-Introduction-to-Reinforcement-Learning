"""
Quick test to verify LunarLander SAC setup is correct
"""
import sys
import os
print("="*80)
print("TESTING LUNARLANDER SAC SETUP")
print("="*80)
print("\n1. Checking Python packages...")
try:
    import numpy
    import torch
    import gymnasium
    import matplotlib
    print("   ✓ All required packages installed")
except ImportError as e:
    print(f"   ✗ Missing package: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)
print("\n2. Checking pytorch_sac repository...")
pytorch_sac_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pytorch_sac'))
if not os.path.exists(pytorch_sac_path):
    print(f"   ✗ pytorch_sac not found at {pytorch_sac_path}")
    print("   Run: cd D:\\RL\\A3\\Implementation && git clone https://github.com/denisyarats/pytorch_sac.git")
    sys.exit(1)
else:
    print(f"   ✓ pytorch_sac found at {pytorch_sac_path}")
print("\n3. Testing official_sac_adapter import...")
try:
    from official_sac_adapter import SACAgentForAssignment
    print("   ✓ Successfully imported SACAgentForAssignment")
except ImportError as e:
    print(f"   ✗ Failed to import SACAgentForAssignment: {e}")
    print("   Make sure official_sac_adapter.py exists in LunarLander folder")
    print("   And pytorch_sac repo is cloned in D:\\RL\\A3\\pytorch_sac")
    sys.exit(1)
print("\n4. Checking config.json...")
if os.path.exists('config.json'):
    print("   ✓ config.json found")
else:
    print("   ✗ config.json not found")
    sys.exit(1)
print("\n5. Testing lunar_lander_env...")
try:
    from lunar_lander_env import make_lunar_lander_env
    print("   ✓ Successfully imported make_lunar_lander_env")
except ImportError as e:
    print(f"   ✗ Failed to import: {e}")
    sys.exit(1)
print("\n6. Testing LunarLander-v3 environment...")
try:
    import gymnasium as gym
    env = gym.make('LunarLander-v3', continuous=True)
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
        state_dim=8,
        action_dim=2,
        action_scale=1.0,
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
print("  python train_sac.py --config config.json --seed 0")
print("\nOr run Q1/Q2 experiments:")
print("  .\\run_q1_q2.ps1")
print("="*80)
