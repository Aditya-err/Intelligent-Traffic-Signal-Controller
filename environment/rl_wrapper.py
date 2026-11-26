"""Phase 4 RL state-representation, reward-shaping, and discretization wrapper."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from environment.traffic_env import SumoTrafficEnv


class TrafficRLWrapper(gym.Wrapper):
    """Wraps ``SumoTrafficEnv`` with an 18-dim RL state and shaped reward.

    State Vector (18 dimensions, ``float32`` in ``[0.0, 1.0]``):
        - ``obs[0:8]``: Normalized halting queue lengths on the 8 incoming lanes.
        - ``obs[8:16]``: Normalized accumulated waiting times on the 8 incoming lanes.
        - ``obs[16]``: Current active signal phase (``0.0`` = NS, ``1.0`` = EW).
        - ``obs[17]``: Normalized current green duration (``green_elapsed / max_green_ref``).

    Reward Formulation:
        Combines normalized waiting-time penalty, queue-length penalty, and a
        small phase-switch penalty to discourage chattering:

        .. math::
            r_t = -\\left(
                w_{\\text{wait}} \\cdot \\tilde{W}_t
                + w_{\\text{queue}} \\cdot \\tilde{Q}_t
                + \\lambda_{\\text{switch}} \\cdot \\mathbb{I}[\\text{switch}_t]
            \\right)
    """

    def __init__(
        self,
        env: SumoTrafficEnv,
        *,
        queue_ref_per_lane: float = 20.0,
        wait_ref_per_lane: float = 600.0,
        max_green_ref: float = 60.0,
        wait_weight: float = 0.6,
        queue_weight: float = 0.4,
        switch_penalty: float = 0.02,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.unwrapped, SumoTrafficEnv):
            raise TypeError("TrafficRLWrapper requires a SumoTrafficEnv instance.")

        self.queue_ref_per_lane = float(queue_ref_per_lane)
        self.wait_ref_per_lane = float(wait_ref_per_lane)
        self.max_green_ref = float(max_green_ref)
        self.wait_weight = float(wait_weight)
        self.queue_weight = float(queue_weight)
        self.switch_penalty = float(switch_penalty)

        num_lanes = len(SumoTrafficEnv.INCOMING_LANES)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_lanes * 2 + 2,),
            dtype=np.float32,
        )
        self.action_space = env.action_space

    def _build_state(self) -> tuple[np.ndarray, dict[str, Any]]:
        base_env: SumoTrafficEnv = self.env.unwrapped  # type: ignore[assignment]
        conn = base_env._conn
        if conn is None:
            raise RuntimeError("Environment must be reset() before reading state.")

        queues = [
            min(
                conn.lane.getLastStepHaltingNumber(lane_id) / self.queue_ref_per_lane,
                1.0,
            )
            for lane_id in base_env.INCOMING_LANES
        ]
        raw_waits = [
            float(conn.lane.getWaitingTime(lane_id))
            for lane_id in base_env.INCOMING_LANES
        ]
        norm_waits = [min(w / self.wait_ref_per_lane, 1.0) for w in raw_waits]

        phase = float(base_env._current_group)
        norm_green = min(base_env._green_elapsed / self.max_green_ref, 1.0)

        state = np.array(
            queues + norm_waits + [phase, norm_green],
            dtype=np.float32,
        )
        return np.clip(state, 0.0, 1.0), {
            "lane_waiting_times": raw_waits,
            "total_lane_waiting_time": float(sum(raw_waits)),
            "green_duration_s": int(base_env._green_elapsed),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = self.env.reset(seed=seed, options=options)
        state, state_info = self._build_state()
        info.update(state_info)
        return state, info

    def step(
        self, action: int | np.integer
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        _, _, terminated, truncated, info = self.env.step(action)
        state, state_info = self._build_state()

        sim_seconds = max(len(info.get("signal_states", [])), 1)
        num_lanes = len(SumoTrafficEnv.INCOMING_LANES)
        denom = num_lanes * self.queue_ref_per_lane

        norm_wait = min(
            float(info["accumulated_waiting_time"]) / (sim_seconds * denom), 1.0
        )
        norm_queue = min(float(info["total_queue"]) / denom, 1.0)
        switch_cost = (
            self.switch_penalty if info.get("switch_executed", False) else 0.0
        )
        shaped_reward = -(
            self.wait_weight * norm_wait
            + self.queue_weight * norm_queue
            + switch_cost
        )

        info.update(state_info)
        info.update(
            {
                "shaped_norm_wait": float(norm_wait),
                "shaped_norm_queue": float(norm_queue),
                "switch_penalty": float(switch_cost),
                "shaped_reward": float(shaped_reward),
            }
        )
        return state, float(shaped_reward), terminated, truncated, info


def discretize_state(
    state: np.ndarray, queue_bins: int = 4, green_bins: int = 3
) -> tuple[int, int, int, int]:
    """Discretize 18-dim continuous state into a compact tabular RL key.

    Returns ``(ns_level, ew_level, phase, green_bin)`` for future Tabular
    Q-Learning / SARSA while keeping the state space small on CPU/RAM.
    """
    if state.shape != (18,):
        raise ValueError(f"Expected 18-dim state, got {state.shape}.")
    ns_pressure = float(np.mean(state[0:4]))
    ew_pressure = float(np.mean(state[4:8]))
    ns_level = min(int(ns_pressure * queue_bins), queue_bins - 1)
    ew_level = min(int(ew_pressure * queue_bins), queue_bins - 1)
    phase = int(round(float(state[16])))
    green_bin = min(int(float(state[17]) * green_bins), green_bins - 1)
    return (ns_level, ew_level, phase, green_bin)
