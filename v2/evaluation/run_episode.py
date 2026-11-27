"""
Reusable TraCI Episode Runner for Intelligent-Traffic-RL v2.

Runs one simulation episode for any controller implementing:
- `name: str`
- `extra_sumo_args() -> list[str]`
- `on_simulation_start() -> None`
- `step(sim_time_s: float) -> None`

Used by Phase 3 (Baselines), Phase 4 (RL Controllers), and Phase 6 (Final Evaluation).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Protocol

from v2 import config
from v2.traffic.generate_traffic import generate_route_file

_SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")
_SUMO_TOOLS = Path(_SUMO_HOME) / "tools"
if str(_SUMO_TOOLS) not in sys.path:
    sys.path.insert(0, str(_SUMO_TOOLS))

import traci  # noqa: E402


# Incoming edges per junction (for queue length and per-junction waiting time)
JUNCTION_IN_EDGES: Dict[str, List[str]] = {
    "J1": ["W1_J1", "N1_J1", "S1_J1", "J2_J1"],
    "J2": ["J1_J2", "N2_J2", "S2_J2", "J3_J2"],
    "J3": ["J2_J3", "N3_J3", "S3_J3", "E3_J3"],
}
ALL_IN_EDGES: List[str] = [e for edges in JUNCTION_IN_EDGES.values() for e in edges]


class SignalController(Protocol):
    """Minimal controller interface shared by baselines and RL agents."""

    name: str

    def extra_sumo_args(self) -> List[str]: ...
    def on_simulation_start(self) -> None: ...
    def step(self, sim_time_s: float) -> None: ...


def run_episode(
    controller: SignalController,
    demand: str,
    seed: int,
    duration_s: int = config.EPISODE_DURATION_S,
) -> Dict[str, Any]:
    """Run one TraCI episode and return traffic + safety metrics."""
    sumo_exe = Path(_SUMO_HOME) / "bin" / "sumo.exe"
    if not sumo_exe.exists():
        raise FileNotFoundError(f"SUMO binary not found: {sumo_exe}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        rou_path = tmp_path / f"{demand}_seed{seed}.rou.xml"
        log_path = tmp_path / "sumo.log"

        generate_route_file(
            demand_level=demand,
            seed=seed,
            duration_s=duration_s,
            output_path=rou_path,
        )

        cmd = [
            str(sumo_exe),
            "-n",
            str(config.NET_FILE),
            "-r",
            str(rou_path),
            *controller.extra_sumo_args(),
            "--seed",
            str(seed),
            "--step-length",
            str(config.STEP_LENGTH_S),
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
            "--collision.action",
            "warn",
            "--log",
            str(log_path),
        ]

        traci.start(cmd)
        try:
            controller.on_simulation_start()

            depart_time: Dict[str, float] = {}
            veh_wait: Dict[str, float] = {}
            travel_times: List[float] = []
            queue_lengths: List[int] = []

            j_wait_steps: Dict[str, int] = {jid: 0 for jid in JUNCTION_IN_EDGES}
            j_seen_vehs: Dict[str, set[str]] = {jid: set() for jid in JUNCTION_IN_EDGES}

            teleports = 0
            collisions = 0
            steps = int(duration_s / config.STEP_LENGTH_S)

            for step_idx in range(steps):
                sim_time = step_idx * config.STEP_LENGTH_S
                controller.step(sim_time)
                traci.simulationStep()

                now = traci.simulation.getTime()
                for vid in traci.simulation.getDepartedIDList():
                    depart_time[vid] = now

                for vid in traci.simulation.getArrivedIDList():
                    if vid in depart_time:
                        travel_times.append(now - depart_time[vid])

                for vid in traci.vehicle.getIDList():
                    veh_wait[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)

                total_halting = sum(
                    traci.edge.getLastStepHaltingNumber(eid) for eid in ALL_IN_EDGES
                )
                queue_lengths.append(total_halting)

                for jid, edges in JUNCTION_IN_EDGES.items():
                    for eid in edges:
                        for vid in traci.edge.getLastStepVehicleIDs(eid):
                            j_seen_vehs[jid].add(vid)
                            if traci.vehicle.getSpeed(vid) < 0.1:
                                j_wait_steps[jid] += 1

                teleports += traci.simulation.getStartingTeleportNumber()
                collisions += traci.simulation.getCollidingVehiclesNumber()

            vehicles_remaining = len(traci.vehicle.getIDList())
        finally:
            traci.close()

        log_text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
        log_warning_lines = [
            line.strip()
            for line in log_text.splitlines()
            if "teleport" in line or "collision" in line
        ]

    vehicles_arrived = len(travel_times)
    avg_waiting_time = sum(veh_wait.values()) / len(veh_wait) if veh_wait else 0.0
    avg_queue_length = sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0.0
    avg_travel_time = sum(travel_times) / vehicles_arrived if vehicles_arrived else 0.0
    throughput_vph = vehicles_arrived / (duration_s / 3600.0)
    max_vehicle_wait = max(veh_wait.values(), default=0.0)

    per_junction_wait = {
        jid: round(
            (j_wait_steps[jid] * config.STEP_LENGTH_S) / len(j_seen_vehs[jid]), 2
        )
        if j_seen_vehs[jid]
        else 0.0
        for jid in JUNCTION_IN_EDGES
    }

    return {
        "controller": controller.name,
        "demand": demand,
        "seed": seed,
        "duration_s": duration_s,
        "avg_waiting_time_s": round(avg_waiting_time, 2),
        "avg_queue_length": round(avg_queue_length, 2),
        "avg_travel_time_s": round(avg_travel_time, 2),
        "throughput_vph": round(throughput_vph, 2),
        "vehicles_arrived": vehicles_arrived,
        "vehicles_remaining": vehicles_remaining,
        "j1_avg_wait_s": per_junction_wait["J1"],
        "j2_avg_wait_s": per_junction_wait["J2"],
        "j3_avg_wait_s": per_junction_wait["J3"],
        "teleports": teleports,
        "collisions": collisions,
        "max_vehicle_wait_s": round(max_vehicle_wait, 2),
        "gridlock_warning": max_vehicle_wait >= config.GRIDLOCK_WAIT_THRESHOLD_S,
        "log_warnings": len(log_warning_lines),
    }
