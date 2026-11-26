"""Traditional fixed-time traffic signal baseline controller."""

from __future__ import annotations

from typing import Any


class FixedTimeController:
    """Deterministic fixed-time signal controller alternating NS (0) and EW (1).

    The controller issues action ``0`` (North-South) for ``ns_green_steps``
    decision intervals, followed by action ``1`` (East-West) for
    ``ew_green_steps`` decision intervals. Safe yellow, all-red, minimum green,
    and protected right-turn clearances are enforced by ``SumoTrafficEnv``.
    """

    def __init__(self, ns_green_steps: int = 5, ew_green_steps: int = 4) -> None:
        if ns_green_steps < 1 or ew_green_steps < 1:
            raise ValueError("ns_green_steps and ew_green_steps must be >= 1.")
        self.ns_green_steps = int(ns_green_steps)
        self.ew_green_steps = int(ew_green_steps)
        self._cycle_length = self.ns_green_steps + self.ew_green_steps
        self._step_idx = 0

    def reset(self) -> None:
        """Reset internal cycle counter at the start of an episode."""
        self._step_idx = 0

    def select_action(self, observation: Any = None) -> int:
        """Return 0 (North-South) or 1 (East-West) according to fixed schedule."""
        pos = self._step_idx % self._cycle_length
        self._step_idx += 1
        return 0 if pos < self.ns_green_steps else 1
