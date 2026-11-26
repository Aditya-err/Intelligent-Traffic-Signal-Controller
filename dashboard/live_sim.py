"""Live 1-second SUMO + TraCI 2D simulation session manager for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from algorithms.dqn import DQNAgent
from algorithms.fixed_time import FixedTimeController
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from environment.rl_wrapper import TrafficRLWrapper
from environment.traffic_env import SumoTrafficEnv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SCENARIO_CONFIGS: dict[str, Path] = {
    "low": PROJECT_ROOT / "simulation" / "config" / "scenario_low.sumocfg",
    "medium": PROJECT_ROOT / "simulation" / "config" / "scenario_medium.sumocfg",
    "high": PROJECT_ROOT / "simulation" / "config" / "scenario_high.sumocfg",
}


@dataclass
class SimSnapshot:
    """Immutable snapshot of one 1-second live SUMO frame for 2D rendering."""

    sim_time: int
    scenario: str
    controller_name: str
    selected_action: int
    current_group: int
    green_elapsed: int
    tls_state: str
    waiting_time: float
    queue_length: int
    vehicle_count: int
    total_spawned: int
    completed_vehicles: int
    waiting_vehicles: int
    terminated: bool
    vehicles: list[dict[str, Any]]


def build_frozen_controller(controller_key: str) -> Any:
    """Load a frozen controller (epsilon=0, eval mode) for live demonstration."""
    if controller_key == "fixed_time":
        # With delta_time=1s, 25 steps NS / 20 steps EW matches Phase 3 timings
        return FixedTimeController(ns_green_steps=25, ew_green_steps=20)
    if controller_key == "q_learning":
        agent = QLearningAgent(seed=42)
        q_path = MODELS_DIR / "q_learning_qtable.json"
        if q_path.is_file():
            agent.load(q_path)
        agent.epsilon = 0.0
        return agent
    if controller_key == "sarsa":
        agent = SARSAAgent(seed=42)
        s_path = MODELS_DIR / "sarsa_qtable.json"
        if s_path.is_file():
            agent.load(s_path)
        agent.epsilon = 0.0
        return agent
    if controller_key == "dqn_medium_only":
        agent = DQNAgent(seed=42)
        ckpt = MODELS_DIR / "dqn_checkpoint.pt"
        if ckpt.is_file():
            agent.load(ckpt)
        agent.epsilon = 0.0
        agent.q_net.eval()
        return agent
    if controller_key == "dqn_curriculum":
        agent = DQNAgent(seed=42)
        ckpt = MODELS_DIR / "dqn_curriculum_best.pt"
        if ckpt.is_file():
            agent.load(ckpt)
        agent.epsilon = 0.0
        agent.q_net.eval()
        return agent
    raise ValueError(f"Unknown controller: {controller_key!r}")


class LiveSimSession:
    """Manages a single 1-second-resolution live SUMO + TraCI episode."""

    def __init__(
        self,
        scenario: str = "medium",
        controller_key: str = "dqn_curriculum",
        seed: int = 1001,
        max_steps: int = 1800,
        decision_interval: int = 5,
    ) -> None:
        if scenario not in SCENARIO_CONFIGS:
            raise ValueError(f"Invalid scenario: {scenario!r}")
        self.scenario = scenario
        self.controller_key = controller_key
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.decision_interval = int(decision_interval)

        # Use delta_time=1 so every step_once() advances 1 smooth SUMO second
        self.env = TrafficRLWrapper(
            SumoTrafficEnv(
                sumocfg_path=SCENARIO_CONFIGS[scenario],
                use_gui=False,
                max_steps=self.max_steps,
                delta_time=1,
                min_green=10,
                yellow_time=3,
                all_red_time=2,
                right_turn_green=5,
                sumo_seed=self.seed,
            )
        )
        self.controller = build_frozen_controller(controller_key)
        self.state: Any = None
        self.is_running: bool = False
        self.terminated: bool = False
        self.last_action: int = 0
        self._ticks_since_decision: int = 0
        self._seen_vehicle_ids: set[str] = set()
        self.snapshot: SimSnapshot | None = None

    def reset(self, warmup_steps: int = 0) -> SimSnapshot:
        """Start or restart the live SUMO simulation and return initial snapshot."""
        self.state, info = self.env.reset(seed=self.seed)
        if isinstance(self.controller, FixedTimeController):
            self.controller.reset()
        self.terminated = False
        self.last_action = 0
        self._ticks_since_decision = 0
        self._seen_vehicle_ids = set()
        self.snapshot = self._capture_snapshot(
            selected_action=0,
            waiting_time=0.0,
            queue_length=0,
            info=info,
        )
        for _ in range(warmup_steps):
            if self.terminated:
                break
            self.step_once()
        assert self.snapshot is not None
        return self.snapshot

    def step_once(self) -> SimSnapshot:
        """Advance the live SUMO simulation by 1 second."""
        if self.state is None:
            return self.reset()
        if self.terminated:
            self.is_running = False
            assert self.snapshot is not None
            return self.snapshot

        if isinstance(self.controller, FixedTimeController):
            action = int(self.controller.select_action(self.state))
        else:
            if self._ticks_since_decision % self.decision_interval == 0:
                action = int(self.controller.select_action(self.state, explore=False))
            else:
                action = self.last_action
        self._ticks_since_decision += 1

        next_state, _, terminated, truncated, info = self.env.step(action)
        self.state = next_state
        self.last_action = action
        self.terminated = bool(terminated or truncated)
        if self.terminated:
            self.is_running = False

        self.snapshot = self._capture_snapshot(
            selected_action=action,
            waiting_time=float(info.get("accumulated_waiting_time", 0.0)),
            queue_length=int(info.get("total_queue", 0)),
            info=info,
        )
        return self.snapshot

    def _capture_snapshot(
        self,
        *,
        selected_action: int,
        waiting_time: float,
        queue_length: int,
        info: dict[str, Any],
    ) -> SimSnapshot:
        base_env: SumoTrafficEnv = self.env.unwrapped  # type: ignore[assignment]
        conn = base_env._conn
        vehicles: list[dict[str, Any]] = []
        tls_state = base_env._current_state_str
        waiting_vehicles = 0

        if conn is not None:
            tls_state = conn.trafficlight.getRedYellowGreenState(base_env.TLS_ID)
            cx, cy = conn.junction.getPosition(base_env.TLS_ID)
            active_ids = conn.vehicle.getIDList()
            for vid in active_ids:
                self._seen_vehicle_ids.add(vid)
                x, y = conn.vehicle.getPosition(vid)
                angle = conn.vehicle.getAngle(vid)
                vtype = conn.vehicle.getTypeID(vid)
                lane_id = conn.vehicle.getLaneID(vid)
                speed = float(conn.vehicle.getSpeed(vid))
                if speed < 0.1:
                    waiting_vehicles += 1
                vehicles.append(
                    {
                        "id": vid,
                        "x": float(x - cx),
                        "y": float(y - cy),
                        "angle": float(angle),
                        "vtype": vtype,
                        "lane_id": lane_id,
                        "speed": speed,
                    }
                )

        total_spawned = len(self._seen_vehicle_ids)
        completed_vehicles = max(0, total_spawned - len(vehicles))

        return SimSnapshot(
            sim_time=int(info.get("sim_time", base_env._sim_time)),
            scenario=self.scenario,
            controller_name=self.controller_key,
            selected_action=selected_action,
            current_group=int(info.get("current_group", base_env._current_group)),
            green_elapsed=int(info.get("green_elapsed", base_env._green_elapsed)),
            tls_state=tls_state,
            waiting_time=waiting_time,
            queue_length=queue_length,
            vehicle_count=len(vehicles),
            total_spawned=total_spawned,
            completed_vehicles=completed_vehicles,
            waiting_vehicles=waiting_vehicles,
            terminated=self.terminated,
            vehicles=vehicles,
        )

    def close(self) -> None:
        """Cleanly close the underlying TraCI connection."""
        self.is_running = False
        self.env.close()
