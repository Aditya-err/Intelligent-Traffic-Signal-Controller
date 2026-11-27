"""
Shared DQN Controller & Hard Safety Layer for Intelligent-Traffic-RL v2.

Design (fresher-friendly, parameter sharing):
- ONE shared QNetwork (10 -> 128 -> 64 -> 2) is used by all 3 junctions (J1, J2, J3).
- Each junction builds its own 10-feature normalized state vector and queries the
  same network every `DQN_DECISION_INTERVAL_S` seconds.
- A hard safety layer outside the neural network enforces:
  10 s min green, 3 s yellow, 2 s all-red (and logs overrides if the agent asks
  to switch before 10 s).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
import traci

from v2 import config
from v2.evaluation.run_episode import JUNCTION_IN_EDGES


class QNetwork(nn.Module):
    """Small 2-hidden-layer MLP mapping 10 normalized features to 2 Q-values."""

    def __init__(
        self,
        state_dim: int = config.DQN_STATE_DIM,
        action_dim: int = config.DQN_ACTION_DIM,
    ) -> None:
        super().__init__()
        h1, h2 = config.DQN_HIDDEN_DIMS
        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_junction_state(jid: str, current_action: int, time_in_green_s: int) -> np.ndarray:
    """
    Build the 10-element normalized state vector for junction `jid`:
    [4 approach queues, 4 approach waits, current_phase (0/1), time_in_phase_norm].
    """
    edges = JUNCTION_IN_EDGES[jid]
    queues = [
        min(traci.edge.getLastStepHaltingNumber(eid) / config.DQN_MAX_QUEUE_NORM, 1.0)
        for eid in edges
    ]
    waits = [
        min(traci.edge.getWaitingTime(eid) / config.DQN_MAX_WAIT_NORM, 1.0)
        for eid in edges
    ]
    phase_feat = float(current_action)
    time_feat = min(time_in_green_s / float(config.MAX_GREEN_S), 1.0)
    return np.array(queues + waits + [phase_feat, time_feat], dtype=np.float32)


def compute_junction_reward(jid: str, switched: bool) -> float:
    """
    Compute per-junction reward:
        r = -(w_queue * sum_queues + w_wait * sum_waits_s + w_switch * switched)
    """
    edges = JUNCTION_IN_EDGES[jid]
    sum_q = sum(traci.edge.getLastStepHaltingNumber(eid) for eid in edges)
    sum_w = sum(traci.edge.getWaitingTime(eid) for eid in edges)
    return -(
        config.DQN_W_QUEUE * sum_q
        + config.DQN_W_WAIT * sum_w
        + config.DQN_W_SWITCH * (1.0 if switched else 0.0)
    )


class SafetySignalManager:
    """
    Hard safety state machine for one junction (outside the neural network).

    States: "GREEN", "YELLOW", "ALL_RED".
    Enforces MIN_GREEN_S (10s), YELLOW_S (3s), ALL_RED_S (2s), and MAX_GREEN_S (60s).
    """

    def __init__(self, jid: str) -> None:
        self.jid = jid
        logics = traci.trafficlight.getAllProgramLogics(jid)
        phases = logics[0].phases
        p0_is_ns = phases[0].state[0] in ("g", "G")
        ns_idx = 0 if p0_is_ns else 4
        ew_idx = 4 if p0_is_ns else 0

        self.green_phase = {0: ns_idx, 1: ew_idx}
        self.yellow_phase = {0: ns_idx + 2, 1: ew_idx + 2}
        self.allred_phase = {0: ns_idx + 3, 1: ew_idx + 3}

        self.state = "GREEN"
        self.current_action = 0  # Start in NS green
        self.target_action = 0
        self.time_in_green = 0
        self.transition_timer = 0
        self.safety_overrides = 0

        self._set_sumo_phase(self.green_phase[self.current_action])

    def _set_sumo_phase(self, phase_idx: int) -> None:
        traci.trafficlight.setPhase(self.jid, phase_idx)
        # Prevent SUMO's static timer from auto-advancing behind our safety layer
        traci.trafficlight.setPhaseDuration(self.jid, 9999.0)

    def apply_decision_and_step(self, requested_action: Optional[int]) -> bool:
        """
        Advance the signal state machine by 1 second.

        Returns True if a phase switch was initiated this second.
        """
        if self.state == "YELLOW":
            self.transition_timer -= 1
            if self.transition_timer <= 0:
                self.state = "ALL_RED"
                self.transition_timer = config.ALL_RED_S
                self._set_sumo_phase(self.allred_phase[self.current_action])
            return False

        if self.state == "ALL_RED":
            self.transition_timer -= 1
            if self.transition_timer <= 0:
                self.state = "GREEN"
                self.current_action = self.target_action
                self.time_in_green = 0
                self._set_sumo_phase(self.green_phase[self.current_action])
            return False

        # State == "GREEN"
        self.time_in_green += 1

        # Force switch if MAX_GREEN_S is reached to prevent starvation
        if self.time_in_green >= config.MAX_GREEN_S:
            requested_action = 1 - self.current_action

        if requested_action is not None and requested_action != self.current_action:
            if self.time_in_green < config.MIN_GREEN_S:
                # Hard safety rule: hold current green and log override
                self.safety_overrides += 1
                return False
            self.state = "YELLOW"
            self.target_action = requested_action
            self.transition_timer = config.YELLOW_S
            self._set_sumo_phase(self.yellow_phase[self.current_action])
            return True

        return False


class DQNController:
    """Inference-time shared DQN controller compatible with `run_episode()`."""

    name = "dqn"

    def __init__(self, checkpoint_path: Path | str | None = None) -> None:
        self.device = torch.device("cpu")
        self.q_net = QNetwork().to(self.device)
        ckpt = Path(checkpoint_path) if checkpoint_path else (config.MODELS_DIR / "dqn_best.pt")
        if not ckpt.exists():
            raise FileNotFoundError(f"DQN checkpoint not found: {ckpt}")
        state_dict = torch.load(ckpt, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(state_dict)
        self.q_net.eval()
        self.managers: Dict[str, SafetySignalManager] = {}

    def extra_sumo_args(self) -> List[str]:
        return []

    def on_simulation_start(self) -> None:
        self.managers = {jid: SafetySignalManager(jid) for jid in config.JUNCTION_COORDS}

    def step(self, sim_time_s: float) -> None:
        make_decision = int(sim_time_s) % config.DQN_DECISION_INTERVAL_S == 0
        for jid, mgr in self.managers.items():
            req_action: Optional[int] = None
            if make_decision and mgr.state == "GREEN":
                obs = get_junction_state(jid, mgr.current_action, mgr.time_in_green)
                with torch.no_grad():
                    q_vals = self.q_net(torch.from_numpy(obs).unsqueeze(0))
                    req_action = int(torch.argmax(q_vals, dim=1).item())
            mgr.apply_decision_and_step(req_action)
