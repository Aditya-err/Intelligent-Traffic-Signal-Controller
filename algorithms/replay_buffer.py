"""Fixed-capacity experience replay buffer for CPU-friendly DQN training."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Circular experience replay buffer backed by pre-allocated NumPy arrays."""

    def __init__(
        self,
        capacity: int = 20_000,
        state_dim: int = 18,
        seed: int = 42,
    ) -> None:
        if capacity < 1:
            raise ValueError("ReplayBuffer capacity must be >= 1.")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self._rng = np.random.default_rng(seed)

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.terminateds = np.zeros((self.capacity,), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store one transition; overwrites oldest entry when full."""
        self.states[self._ptr] = state
        self.actions[self._ptr] = int(action)
        self.rewards[self._ptr] = float(reward)
        self.next_states[self._ptr] = next_state
        self.terminateds[self._ptr] = 1.0 if terminated else 0.0

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a uniform mini-batch of transitions without replacement."""
        if batch_size < 1 or batch_size > self._size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from buffer size={self._size}."
            )
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminateds[indices],
        )
