# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:29:43 2026

@author: Sharan Sharmeela Karuppannan - Gridworld Q2
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Environment Specification
# ==========================

grid_size = 5
gamma = 0.95 
theta = 1e-8

lake = (0, 0) 
fire = (4, 4) 
smoke = [(1, 2), (3, 2)] 
boulders = [(2, 4), (3, 4)] 

# Actions: 0:N, 1:S, 2:W, 3:E, 4:Hover
actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

# State space: (row, col, water_status)
states = [(i, j, w) for i in range(grid_size) for j in range(grid_size) for w in [0, 1]]

# ==========================
# Transition Function
# ==========================

def move(state, action):
    i, j, water = state
    
    # Terminal check
    if (i, j) in boulders or ((i, j) == fire and water == 1):
        return []
    
    # Hover is deterministic and unaffected by wind 
    if action == 4:
        reward = -1
        if (i, j) in smoke:
            reward -= 10
        return [(1.0, (i, j, water), reward, False)]
    
    # Wind probabilities based on current location
    if (i, j) in smoke:
        # 40% intended, 10% each perpendicular, 40% stay (backdraft) 
        probs = [0.4, 0.1, 0.1, 0.4]
    else:
        # 70% intended, 10% each perpendicular, 10% stay (motor breaks) [cite: 22]
        probs = [0.7, 0.1, 0.1, 0.1]
    
    intended = actions[action]

    perp1 = (-intended[1], intended[0])
    perp2 = (intended[1], -intended[0])
    stay = (0, 0)
    
    directions = [intended, perp1, perp2, stay]
    outcomes = []
    
    for p, d in zip(probs, directions):
        ni, nj = i + d[0], j + d[1]
        
        if not (0 <= ni < grid_size and 0 <= nj < grid_size):
            ni, nj = i, j
            
        new_water = water
        reward = -1 # Standard per-step penalty 
        done = False
        
        # Additional penalty for hazardous smoke entry 
        if (ni, nj) in smoke:
            reward -= 10
            
        # Entering lake empty-handed
        if (ni, nj) == lake and water == 0:
            new_water = 1
            
        # Terminal: Boulder crash 
        if (ni, nj) in boulders:
            reward = -100
            done = True
            
        # Terminal: Successful water delivery to fire
        if (ni, nj) == fire and new_water == 1:
            reward = 100
            done = True
            
        outcomes.append((p, (ni, nj, new_water), reward, done))
        
    return outcomes

# ==========================
# Value Iteration
# ==========================

V = {s: 0.0 for s in states}
policy = {s: 4 for s in states}

while True:
    delta = 0
    new_V = V.copy()
    for s in states:
        i, j, w = s
        # Skip terminal states 
        if (i, j) in boulders or ((i, j) == fire and w == 1):
            continue
            
        action_values = []
        for a_idx in range(len(actions)):
            total = 0
            # Bellman equation: p * (r + gamma * V_next)
            for p, next_s, r, done in move(s, a_idx):
                total += p * (r + (gamma * V[next_s] if not done else 0))
            action_values.append(total)
        
        best_val = max(action_values)
        delta = max(delta, abs(V[s] - best_val))
        new_V[s] = best_val
    
    V = new_V
    if delta < theta:
        break

# Optimal Policy Extraction
for s in states:
    i, j, w = s
    if (i, j) in boulders or ((i, j) == fire and w == 1):
        continue
    action_values = []
    for a_idx in range(len(actions)):
        total = 0
        for p, next_s, r, done in move(s, a_idx):
            total += p * (r + (gamma * V[next_s] if not done else 0))
        action_values.append(total)
    policy[s] = np.argmax(action_values)

# ==========================
# Visualization
# ==========================

def plot_phase(water_level, title_suffix):
    v_grid = np.zeros((grid_size, grid_size))
    p_grid = np.zeros((grid_size, grid_size))
    
    symbols = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "H"}
    
    for i in range(grid_size):
        for j in range(grid_size):
            v_grid[i, j] = V[(i, j, water_level)]
            p_grid[i, j] = policy.get((i, j, water_level), 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Phase: {title_suffix}", fontsize=16)

    # Value Function Heatmap
    sns.heatmap(v_grid, annot=True, fmt=".1f", ax=ax1, cmap="YlGnBu", cbar_kws={'label': 'Value'})
    ax1.set_title("Optimal Value Function")

    # Policy Visualization (Sutton & Barto style reference) [cite: 38]
    ax2.set_xticks(np.arange(grid_size))
    ax2.set_yticks(np.arange(grid_size))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.grid(True)
    
    for i in range(grid_size):
        for j in range(grid_size):
            label = symbols[p_grid[i, j]]
            if (i, j) in boulders: label = "B"
            if (i, j) == fire and water_level == 1: label = "GOAL"
            if (i, j) == lake and water_level == 0: label = "LAKE"
            
            ax2.text(j, i, label, ha='center', va='center', 
                     fontsize=12, fontweight='bold', color='darkred')
    
    ax2.set_xlim(-0.5, grid_size-0.5)
    ax2.set_ylim(grid_size-0.5, -0.5)
    ax2.set_title("Optimal Policy")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Visualize results [cite: 37]
plot_phase(0, "Empty Handed (Target: Lake)")
plot_phase(1, "Water Filled (Target: Fire Zone)")