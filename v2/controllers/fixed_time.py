"""
Fixed-Time Baseline Controller for Intelligent-Traffic-RL v2.

Applies a simple repeating signal cycle at each of J1, J2, and J3 using
the green durations and safety timings defined in `v2/config.py`:
- N-S Green: 25 s (20 s with pedestrian crossing + 5 s pedestrian clearance)
- E-W Green: 20 s (15 s with pedestrian crossing + 5 s pedestrian clearance)
- Yellow: 3 s
- All-Red: 2 s
"""

from __future__ import annotations

import traci

from v2 import config


class FixedTimeController:
    """Repeating fixed-time signal controller for J1, J2, J3."""

    name = "fixed_time"

    def extra_sumo_args(self) -> list[str]:
        """No extra SUMO CLI flags needed; program is configured via TraCI."""
        return []

    def on_simulation_start(self) -> None:
        """Configure fixed-time phase durations on J1, J2, J3 via TraCI."""
        for jid in config.JUNCTION_COORDS:
            logics = traci.trafficlight.getAllProgramLogics(jid)
            if not logics:
                continue
            logic = logics[0]
            phases = list(logic.phases)
            if len(phases) != 8:
                continue

            # Detect whether phase 0 serves N-S (link 0 green) or E-W (link 4 green)
            p0_is_ns = phases[0].state[0] in ("g", "G")
            p0_total_green = (
                config.FIXED_TIME_NS_GREEN_S if p0_is_ns else config.FIXED_TIME_EW_GREEN_S
            )
            p4_total_green = (
                config.FIXED_TIME_EW_GREEN_S if p0_is_ns else config.FIXED_TIME_NS_GREEN_S
            )

            # Enforce minimum green safety rule
            p0_total_green = max(p0_total_green, config.MIN_GREEN_S)
            p4_total_green = max(p4_total_green, config.MIN_GREEN_S)

            phases[0].duration = float(p0_total_green - config.PED_CLEARANCE_S)
            phases[1].duration = float(config.PED_CLEARANCE_S)
            phases[2].duration = float(config.YELLOW_S)
            phases[3].duration = float(config.ALL_RED_S)

            phases[4].duration = float(p4_total_green - config.PED_CLEARANCE_S)
            phases[5].duration = float(config.PED_CLEARANCE_S)
            phases[6].duration = float(config.YELLOW_S)
            phases[7].duration = float(config.ALL_RED_S)

            logic.phases = tuple(phases)
            traci.trafficlight.setProgramLogic(jid, logic)
            traci.trafficlight.setProgram(jid, logic.programID)

    def step(self, sim_time_s: float) -> None:
        """SUMO executes the repeating fixed-time schedule automatically."""
        return None
