"""Modular Reinforcement Learning agent interface for Phase 5+ algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseRLAgent(ABC):
    """Abstract base class supporting Tabular (Q-Learning/SARSA) and Deep RL (DQN).

    All future RL controllers in ``algorithms/`` must implement this contract
    so evaluation, logging, and checkpointing remain algorithm-agnostic.
    """

    def __init__(self, state_dim: int, action_dim: int, seed: int = 42) -> None:
        if state_dim < 1 or action_dim < 2:
            raise ValueError("state_dim must be >= 1 and action_dim must be >= 2.")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.seed = int(seed)

    @abstractmethod
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Choose an action in ``{0, ..., action_dim - 1}`` given ``state``."""

    @abstractmethod
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
        """Record/learn from one transition; optionally return training metrics."""

    def on_episode_start(self, episode: int) -> None:
        """Hook called at the beginning of each training/evaluation episode."""

    def on_episode_end(self, episode: int, metrics: dict[str, Any]) -> None:
        """Hook called at the end of each episode (e.g., epsilon decay)."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist agent parameters/tables to disk (on E: drive)."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load agent parameters/tables from disk."""

    def get_config(self) -> dict[str, Any]:
        """Return serializable agent hyperparameters."""
        return {
            "agent_type": self.__class__.__name__,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "seed": self.seed,
        }
