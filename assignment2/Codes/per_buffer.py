"""
Prioritized Experience Replay (PER) buffer using a Sum-Tree.
Used for Bonus Q5.
"""

import numpy as np


class SumTree:
    """Binary sum-tree for efficient priority-based sampling."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """PER buffer with proportional prioritization."""

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_anneal_steps=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.max_priority = 1.0
        self.step_count = 0

    def _get_beta(self):
        fraction = min(1.0, self.step_count / self.beta_anneal_steps)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def push(self, state, action, reward, next_state, done):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        self.step_count += 1
        beta = self._get_beta()

        indices = []
        priorities = []
        batch = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            if data is None:
                s = np.random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        priorities = np.array(priorities, dtype=np.float64) + 1e-8
        sampling_probs = priorities / self.tree.total()
        weights = (self.tree.size * sampling_probs) ** (-beta)
        weights = weights / weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(indices, dtype=np.int64),
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size
