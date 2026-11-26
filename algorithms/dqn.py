"""CPU-only Deep Q-Network (DQN) agent for traffic signal control."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from algorithms.base_agent import BaseRLAgent
from algorithms.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """Lightweight MLP mapping 18 traffic state features to 2 action Q-values."""

    def __init__(self, state_dim: int = 18, action_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(BaseRLAgent):
    """CPU-based Deep Q-Network agent with target network and experience replay."""

    def __init__(
        self,
        state_dim: int = 18,
        action_dim: int = 2,
        *,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        batch_size: int = 64,
        replay_capacity: int = 20_000,
        target_update_freq: int = 100,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.85,
        max_grad_norm: float = 1.0,
        min_replay_size: int = 128,
        seed: int = 42,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, seed=seed)
        self.device = torch.device("cpu")
        torch.manual_seed(self.seed)
        self._rng = np.random.default_rng(self.seed)

        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.replay_capacity = int(replay_capacity)
        self.target_update_freq = int(target_update_freq)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.min_replay_size = int(min_replay_size)

        self.q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.sync_target_network()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(
            capacity=self.replay_capacity,
            state_dim=self.state_dim,
            seed=self.seed,
        )
        self.learn_steps = 0

    def sync_target_network(self) -> None:
        """Copy online Q-network weights to target Q-network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.action_dim))
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def compute_bellman_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminateds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute r + gamma * max_a' Q_target(s', a') * (1 - terminated)."""
        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(dim=1).values
            return rewards + self.gamma * next_q_max * (1.0 - terminateds)

    def train_step(self) -> dict[str, float] | None:
        """Sample a mini-batch and perform one gradient step if buffer is ready."""
        required = max(self.batch_size, self.min_replay_size)
        if len(self.replay_buffer) < required:
            return None

        states_np, actions_np, rewards_np, next_states_np, terms_np = (
            self.replay_buffer.sample(self.batch_size)
        )
        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            actions_np, dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            next_states_np, dtype=torch.float32, device=self.device
        )
        terminateds = torch.as_tensor(
            terms_np, dtype=torch.float32, device=self.device
        )

        q_pred = self.q_net(states).gather(1, actions).squeeze(1)
        q_target = self.compute_bellman_targets(rewards, next_states, terminateds)
        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = float(
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        )
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.sync_target_network()

        return {"loss": float(loss.item()), "grad_norm": grad_norm}

    def observe_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> dict[str, float] | None:
        # Note: Only true episode termination zeroes future bootstrap value;
        # truncated episodes continue bootstrapping via terminated=False.
        self.replay_buffer.push(state, action, reward, next_state, terminated)
        return self.train_step()

    def on_episode_end(self, episode: int, metrics: dict[str, Any]) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.get_config(),
                "epsilon": self.epsilon,
                "learn_steps": self.learn_steps,
                "q_net_state_dict": self.q_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            out,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net_state_dict"])
        self.target_net.load_state_dict(ckpt["target_net_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon = float(ckpt.get("epsilon", self.epsilon_min))
        self.learn_steps = int(ckpt.get("learn_steps", 0))
        self.target_net.eval()

    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg.update(
            {
                "device": str(self.device),
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "batch_size": self.batch_size,
                "replay_capacity": self.replay_capacity,
                "target_update_freq": self.target_update_freq,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "max_grad_norm": self.max_grad_norm,
                "min_replay_size": self.min_replay_size,
            }
        )
        return cfg
