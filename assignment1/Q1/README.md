# GridWorld MDP: Value Iteration

This implementation models a 5×5 firefighting drone environment and computes an optimal value function and policy using value iteration.

## Setup

Python 3.10 was used for the original implementation.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python GridWorld.py
```

The script displays value-function and policy plots for both water states: empty-handed (target: lake) and carrying water (target: fire).

## Environment

- Grid size: 5×5
- Discount factor: γ = 0.95
- Convergence threshold: θ = 1e-8
- Lake: (0, 0); fire: (4, 4)
- Smoke: (1, 2), (3, 2); boulders: (2, 4), (3, 4)
