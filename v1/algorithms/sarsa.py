"""Tabular SARSA agent for traffic signal control."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.base_agent import BaseRLAgent
from environment.rl_wrapper import discretize_state


class SARSAAgent(BaseRLAgent):
    """On-policy Tabular SARSA controller.

    Update rule:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
        where ``a'`` is the actual next action chosen by the policy from ``s'``
        (with future term = 0 when ``terminated`` is True).
    """

    def __init__(
        self,
        state_dim: int = 18,
        action_dim: int = 2,
        *,
        alpha: float = 0.15,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.85,
        queue_bins: int = 4,
        green_bins: int = 3,
        seed: int = 42,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, seed=seed)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.queue_bins = int(queue_bins)
        self.green_bins = int(green_bins)
        self._rng = np.random.default_rng(self.seed)
        self.q_table: dict[tuple[int, int, int, int], np.ndarray] = {}

    def _encode(self, state: np.ndarray) -> tuple[int, int, int, int]:
        return discretize_state(
            state, queue_bins=self.queue_bins, green_bins=self.green_bins
        )

    def _get_q(self, key: tuple[int, int, int, int]) -> np.ndarray:
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_dim, dtype=np.float64)
        return self.q_table[key]

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.action_dim))
        q_vals = self._get_q(self._encode(state))
        max_q = np.max(q_vals)
        best_actions = np.flatnonzero(np.isclose(q_vals, max_q))
        return int(self._rng.choice(best_actions))

    def observe_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        s_key = self._encode(state)
        ns_key = self._encode(next_state)
        q_s = self._get_q(s_key)

        if terminated:
            target = float(reward)
        else:
            if info is not None and "next_action" in info:
                next_action = int(info["next_action"])
            else:
                next_action = self.select_action(next_state, explore=True)
            q_next = float(self._get_q(ns_key)[next_action])
            target = float(reward) + self.gamma * q_next

        td_error = target - float(q_s[action])
        q_s[action] += self.alpha * td_error
        return {"td_error": float(td_error), "q_value": float(q_s[action])}

    def on_episode_end(self, episode: int, metrics: dict[str, Any]) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.get_config(),
            "epsilon": self.epsilon,
            "q_table": {repr(k): v.tolist() for k, v in self.q_table.items()},
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.epsilon = float(payload.get("epsilon", self.epsilon_min))
        self.q_table = {
            tuple(ast.literal_eval(k)): np.array(v, dtype=np.float64)
            for k, v in payload["q_table"].items()
        }

    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg.update(
            {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "queue_bins": self.queue_bins,
                "green_bins": self.green_bins,
                "max_discrete_states": (
                    self.queue_bins * self.queue_bins * 2 * self.green_bins
                ),
            }
        )
        return cfg
