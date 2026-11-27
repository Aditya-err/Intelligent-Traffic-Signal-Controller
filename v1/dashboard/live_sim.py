"""Live 1-second SUMO + TraCI 2D simulation session manager for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    ns_countdown: int
    ew_countdown: int
    next_decision_in: int
    min_green_remaining: int
    waiting_time: float
    queue_length: int
    vehicle_count: int
    total_spawned: int
    completed_vehicles: int
    waiting_vehicles: int
    terminated: bool
    approach_queues: dict[str, int] = field(default_factory=dict)
    approach_waits: dict[str, float] = field(default_factory=dict)
    decision_reason: str = ""
    vehicles: list[dict[str, Any]] = field(default_factory=list)
    prev_vehicles: list[dict[str, Any]] = field(default_factory=list)
    interp_ms: int = 420


def build_frozen_controller(controller_key: str) -> Any:
    """Load a frozen controller (epsilon=0, eval mode) for live demonstration."""
    if controller_key == "fixed_time":
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
                random_depart_offset=4,
            )
        )
        self.controller = build_frozen_controller(controller_key)
        self.state: Any = None
        self.is_running: bool = False
        self.terminated: bool = False
        self.last_action: int = 0
        self._ticks_since_decision: int = 0
        self._seen_vehicle_ids: set[str] = set()
        self._prev_vehicles: list[dict[str, Any]] = []
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
        self._prev_vehicles = []
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

        if self.snapshot is not None:
            self._prev_vehicles = list(self.snapshot.vehicles)

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

    def _compute_countdowns(
        self, current_group: int, green_elapsed: int
    ) -> tuple[int, int, int, int]:
        """Return (ns_countdown, ew_countdown, next_decision_in, min_green_rem)."""
        base_env: SumoTrafficEnv = self.env.unwrapped  # type: ignore[assignment]
        min_green_rem = max(0, base_env.min_green - green_elapsed)

        if isinstance(self.controller, FixedTimeController):
            pos = self.controller._step_idx % self.controller._cycle_length
            if current_group == 0:
                rem = max(1, self.controller.ns_green_steps - pos)
                return rem, rem + base_env.yellow_time + base_env.all_red_time, rem, min_green_rem
            ew_pos = max(0, pos - self.controller.ns_green_steps)
            rem = max(1, self.controller.ew_green_steps - ew_pos)
            return rem + base_env.yellow_time + base_env.all_red_time, rem, rem, min_green_rem

        mod = self._ticks_since_decision % self.decision_interval
        next_dec = 0 if mod == 0 else (self.decision_interval - mod)
        active_rem = max(min_green_rem, next_dec)
        red_rem = active_rem + base_env.yellow_time + base_env.all_red_time
        if current_group == 0:
            return active_rem, red_rem, next_dec, min_green_rem
        return red_rem, active_rem, next_dec, min_green_rem

    def _explain_decision(
        self,
        selected_action: int,
        current_group: int,
        min_green_rem: int,
        approach_queues: dict[str, int],
        approach_waits: dict[str, float],
    ) -> str:
        ns_q = approach_queues.get("N", 0) + approach_queues.get("S", 0)
        ew_q = approach_queues.get("E", 0) + approach_queues.get("W", 0)
        ns_w = approach_waits.get("N", 0.0) + approach_waits.get("S", 0.0)
        ew_w = approach_waits.get("E", 0.0) + approach_waits.get("W", 0.0)

        if isinstance(self.controller, FixedTimeController):
            return (
                f"Fixed-Time follows a preset 25s NS / 20s EW timer "
                f"(currently {'North-South' if current_group == 0 else 'East-West'}; "
                f"NS queue={ns_q}, EW queue={ew_q})."
            )
        if min_green_rem > 0 and selected_action != current_group:
            return (
                f"Controller requested {'East-West' if selected_action == 1 else 'North-South'}, "
                f"but safety interlock holds current green for {min_green_rem}s more (10s minimum green)."
            )
        chosen = "North-South (0)" if selected_action == 0 else "East-West (1)"
        return (
            f"Controller selected {chosen} based on live queues "
            f"(NS: {ns_q} veh / {ns_w:.0f}s wait vs EW: {ew_q} veh / {ew_w:.0f}s wait)."
        )

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
        approach_queues = {"N": 0, "S": 0, "E": 0, "W": 0}
        approach_waits = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}

        if conn is not None:
            tls_state = conn.trafficlight.getRedYellowGreenState(base_env.TLS_ID)
            cx, cy = conn.junction.getPosition(base_env.TLS_ID)

            for lane_id in base_env.INCOMING_LANES:
                arm = lane_id[0]
                approach_queues[arm] += int(conn.lane.getLastStepHaltingNumber(lane_id))
                approach_waits[arm] += float(conn.lane.getWaitingTime(lane_id))

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

        current_group = int(info.get("current_group", base_env._current_group))
        green_elapsed = int(info.get("green_elapsed", base_env._green_elapsed))
        ns_cd, ew_cd, next_dec, min_rem = self._compute_countdowns(
            current_group, green_elapsed
        )
        reason = self._explain_decision(
            selected_action, current_group, min_rem, approach_queues, approach_waits
        )

        total_spawned = len(self._seen_vehicle_ids)
        completed_vehicles = max(0, total_spawned - len(vehicles))

        return SimSnapshot(
            sim_time=int(info.get("sim_time", base_env._sim_time)),
            scenario=self.scenario,
            controller_name=self.controller_key,
            selected_action=selected_action,
            current_group=current_group,
            green_elapsed=green_elapsed,
            tls_state=tls_state,
            ns_countdown=ns_cd,
            ew_countdown=ew_cd,
            next_decision_in=next_dec,
            min_green_remaining=min_rem,
            waiting_time=waiting_time,
            queue_length=queue_length,
            vehicle_count=len(vehicles),
            total_spawned=total_spawned,
            completed_vehicles=completed_vehicles,
            waiting_vehicles=waiting_vehicles,
            terminated=self.terminated,
            approach_queues=approach_queues,
            approach_waits=approach_waits,
            decision_reason=reason,
            vehicles=vehicles,
            prev_vehicles=self._prev_vehicles,
        )

    def close(self) -> None:
        """Cleanly close the underlying TraCI connection."""
        self.is_running = False
        self.env.close()
