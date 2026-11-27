"""Gymnasium + TraCI environment for an Indian Left-Hand-Traffic 4-way intersection.

Mathematical Reward Formulation & Normalization
-----------------------------------------------
At decision step :math:`t` (advancing :math:`K_t` simulation seconds):

1. **Accumulated Waiting Time** (:math:`W_t`, vehicle-seconds):
   Summed over the 8 incoming lanes across every 1-second simulation tick in step :math:`t`:

   .. math::
       W_t = \\sum_{\\tau=1}^{K_t} \\sum_{l \\in \\mathcal{L}_{\\text{in}}} h_l(\\tau)

   where :math:`h_l(\\tau)` is the halting vehicle count on lane :math:`l` at second :math:`\\tau`.

2. **Queue-Length Penalty** (:math:`Q_t`, vehicles):
   Total halting vehicles across the 8 incoming lanes at the end of step :math:`t`:

   .. math::
       Q_t = \\sum_{l \\in \\mathcal{L}_{\\text{in}}} h_l(K_t)

3. **Normalized Reward** (:math:`r_t`):

   .. math::
       r_t = -\\left(
           w_{\\text{wait}} \\cdot \\frac{W_t}{K_t \\cdot N_{\\text{lanes}} \\cdot Q_{\\max}}
           + w_{\\text{queue}} \\cdot \\frac{Q_t}{N_{\\text{lanes}} \\cdot Q_{\\max}}
       \\right)

   where :math:`N_{\\text{lanes}} = 8` and :math:`Q_{\\max}` is ``max_queue_per_lane``
   (default ``40.0``). Dividing :math:`W_t` by :math:`K_t \\cdot N_{\\text{lanes}} \\cdot Q_{\\max}`
   keeps both terms bounded in :math:`[0, 1]` (or clipped) regardless of step duration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import uuid

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sumolib
import traci


DEFAULT_SUMOCFG = (
    Path(__file__).resolve().parent.parent
    / "simulation"
    / "config"
    / "intersection.sumocfg"
)
DEFAULT_SUMO_HOME = Path(r"C:\Program Files (x86)\SUMO")


class SumoTrafficEnv(gym.Env):
    """Gymnasium environment wrapping an Indian LHT 4-way SUMO intersection.

    Observation Space:
        ``Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)``
        - ``obs[0:8]``: Normalized halting queue lengths on the 8 incoming lanes:
          ``N_to_C_0, N_to_C_1, S_to_C_0, S_to_C_1, E_to_C_0, E_to_C_1, W_to_C_0, W_to_C_1``.
        - ``obs[8]``: Current active signal group (``0.0`` = North-South, ``1.0`` = East-West).

    Action Space:
        ``Discrete(2)``
        - ``0``: North-South signal group.
        - ``1``: East-West signal group.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    INCOMING_LANES: tuple[str, ...] = (
        "N_to_C_0",
        "N_to_C_1",
        "S_to_C_0",
        "S_to_C_1",
        "E_to_C_0",
        "E_to_C_1",
        "W_to_C_0",
        "W_to_C_1",
    )

    # Right-turn median lanes under Indian left-hand traffic
    NS_RIGHT_LANES: tuple[str, ...] = ("N_to_C_1", "S_to_C_1")
    EW_RIGHT_LANES: tuple[str, ...] = ("E_to_C_1", "W_to_C_1")

    TLS_ID: str = "center"

    # 12-link signal states verified against intersection.net.xml
    NS_GREEN_SL = "GGrrrrGGrrrr"
    NS_YELLOW_SL = "yyrrrryyrrrr"
    NS_GREEN_R = "rrGrrrrrGrrr"
    NS_YELLOW_R = "rryrrrrryrrr"

    EW_GREEN_SL = "rrrGGrrrrGGr"
    EW_YELLOW_SL = "rrryyrrrryyr"
    EW_GREEN_R = "rrrrrGrrrrrG"
    EW_YELLOW_R = "rrrrryrrrrry"

    ALL_RED = "rrrrrrrrrrrr"

    def __init__(
        self,
        sumocfg_path: str | Path | None = None,
        use_gui: bool = False,
        render_mode: str | None = None,
        max_steps: int = 1800,
        delta_time: int = 5,
        min_green: int = 10,
        yellow_time: int = 3,
        all_red_time: int = 2,
        right_turn_green: int = 5,
        max_queue_per_lane: float = 40.0,
        wait_weight: float = 0.6,
        queue_weight: float = 0.4,
        sumo_seed: int = 42,
        random_depart_offset: int = 0,
    ) -> None:
        super().__init__()
        self.sumocfg_path = Path(sumocfg_path or DEFAULT_SUMOCFG).resolve()
        if not self.sumocfg_path.is_file():
            raise FileNotFoundError(f"SUMO config not found: {self.sumocfg_path}")

        self.use_gui = use_gui or (render_mode == "human")
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.delta_time = int(delta_time)
        self.min_green = int(min_green)
        self.yellow_time = int(yellow_time)
        self.all_red_time = int(all_red_time)
        self.right_turn_green = int(right_turn_green)
        self.max_queue_per_lane = float(max_queue_per_lane)
        self.wait_weight = float(wait_weight)
        self.queue_weight = float(queue_weight)
        self.sumo_seed = int(sumo_seed)
        self.random_depart_offset = int(random_depart_offset)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.INCOMING_LANES) + 1,),
            dtype=np.float32,
        )

        self._label: str = f"traffic_rl_{uuid.uuid4().hex[:8]}"
        self._conn: traci.Connection | None = None
        self._sim_time: int = 0
        self._current_group: int = 0  # 0 = NS, 1 = EW
        self._green_elapsed: int = 0
        self._current_state_str: str = self.NS_GREEN_SL
        self._step_signal_history: list[str] = []

    def _resolve_sumo_binary(self) -> str:
        """Locate sumo or sumo-gui binary using SUMO_HOME or default install path."""
        if not os.environ.get("SUMO_HOME") and DEFAULT_SUMO_HOME.is_dir():
            os.environ["SUMO_HOME"] = str(DEFAULT_SUMO_HOME)
        binary_name = "sumo-gui" if self.use_gui else "sumo"
        binary_path = sumolib.checkBinary(binary_name)
        if not Path(binary_path).is_file():
            fallback = DEFAULT_SUMO_HOME / "bin" / f"{binary_name}.exe"
            if fallback.is_file():
                return str(fallback)
        return binary_path

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.close()

        sim_seed = int(seed) if seed is not None else self.sumo_seed
        binary = self._resolve_sumo_binary()
        cmd = [
            binary,
            "-c",
            str(self.sumocfg_path),
            "--seed",
            str(sim_seed),
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--quit-on-end",
            "true",
        ]
        if self.random_depart_offset > 0:
            cmd.extend(["--random-depart-offset", str(self.random_depart_offset)])
        traci.start(cmd, label=self._label)
        self._conn = traci.getConnection(self._label)

        self._sim_time = 0
        self._current_group = 0
        self._green_elapsed = 0
        self._step_signal_history = []
        self._set_signal_state(self.NS_GREEN_SL)

        obs = self._get_obs()
        info = {
            "sim_time": self._sim_time,
            "current_group": self._current_group,
            "green_elapsed": self._green_elapsed,
            "signal_state": self._current_state_str,
        }
        return obs, info

    def step(
        self, action: int | np.integer
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._conn is None:
            raise RuntimeError("Environment must be reset() before calling step().")

        action_int = int(action) if isinstance(action, (int, np.integer)) else action
        if not self.action_space.contains(action_int):
            raise ValueError(f"Invalid action {action!r}; expected 0 (NS) or 1 (EW).")

        self._step_signal_history = []
        step_waiting_time = 0.0
        switch_requested = action_int != self._current_group
        switch_executed = False
        min_green_blocked = False

        if switch_requested:
            if self._green_elapsed < self.min_green:
                # Enforce minimum green independently of agent decision
                min_green_blocked = True
                wait, green_secs = self._advance_seconds(self.delta_time)
                step_waiting_time += wait
                self._green_elapsed += green_secs
            else:
                # Serve protected right turn if leaving group has right-lane queue;
                # otherwise run Straight+Left Yellow -> All-Red clearance.
                if self._has_right_queue(self._current_group):
                    step_waiting_time += self._maybe_serve_right_turn(
                        self._current_group
                    )
                else:
                    yellow_str = (
                        self.NS_YELLOW_SL
                        if self._current_group == 0
                        else self.EW_YELLOW_SL
                    )
                    step_waiting_time += self._run_clearance(yellow_str)

                # Switch to target group's Straight + Left green
                self._current_group = action_int
                target_green = self.NS_GREEN_SL if action_int == 0 else self.EW_GREEN_SL
                self._set_signal_state(target_green)
                self._green_elapsed = 0
                switch_executed = True

                wait, green_secs = self._advance_seconds(self.delta_time)
                step_waiting_time += wait
                self._green_elapsed += green_secs
        else:
            # Holding current group: serve protected right turn if right-lane queue exists
            if self._green_elapsed >= self.min_green and self._has_right_queue(
                self._current_group
            ):
                step_waiting_time += self._serve_right_turn_and_return(
                    self._current_group
                )
            else:
                wait, green_secs = self._advance_seconds(self.delta_time)
                step_waiting_time += wait
                self._green_elapsed += green_secs

        obs = self._get_obs()
        sim_seconds = max(len(self._step_signal_history), 1)
        reward, reward_info = self._compute_reward(step_waiting_time, sim_seconds)

        terminated = (
            self._sim_time >= self.max_steps
            or self._conn.simulation.getMinExpectedNumber() <= 0
        )
        truncated = False
        info = {
            "sim_time": self._sim_time,
            "current_group": self._current_group,
            "green_elapsed": self._green_elapsed,
            "switch_requested": switch_requested,
            "switch_executed": switch_executed,
            "min_green_blocked": min_green_blocked,
            "signal_states": list(self._step_signal_history),
            **reward_info,
        }
        return obs, reward, terminated, truncated, info

    def _set_signal_state(self, state_str: str) -> None:
        assert self._conn is not None
        self._conn.trafficlight.setRedYellowGreenState(self.TLS_ID, state_str)
        self._current_state_str = state_str

    def _advance_seconds(self, seconds: int) -> tuple[float, int]:
        """Advance SUMO by up to `seconds` ticks, stopping at max_steps."""
        assert self._conn is not None
        accumulated_wait = 0.0
        elapsed = 0
        for _ in range(seconds):
            if self._sim_time >= self.max_steps:
                break
            self._conn.simulationStep()
            self._sim_time += 1
            elapsed += 1
            self._step_signal_history.append(self._current_state_str)
            accumulated_wait += sum(
                self._conn.lane.getLastStepHaltingNumber(lane_id)
                for lane_id in self.INCOMING_LANES
            )
        return accumulated_wait, elapsed

    def _run_clearance(self, yellow_state: str) -> float:
        """Execute Yellow followed by All-Red clearance phases."""
        wait_total = 0.0
        self._set_signal_state(yellow_state)
        w, _ = self._advance_seconds(self.yellow_time)
        wait_total += w

        self._set_signal_state(self.ALL_RED)
        w, _ = self._advance_seconds(self.all_red_time)
        wait_total += w
        return wait_total

    def _has_right_queue(self, group: int) -> bool:
        assert self._conn is not None
        lanes = self.NS_RIGHT_LANES if group == 0 else self.EW_RIGHT_LANES
        return any(
            self._conn.lane.getLastStepHaltingNumber(lane_id) > 0 for lane_id in lanes
        )

    def _maybe_serve_right_turn(self, leaving_group: int) -> float:
        """Before switching away, serve protected right turn if lane 1 has queue."""
        if not self._has_right_queue(leaving_group) or self._sim_time >= self.max_steps:
            return 0.0
        wait_total = 0.0
        yellow_sl = self.NS_YELLOW_SL if leaving_group == 0 else self.EW_YELLOW_SL
        green_r = self.NS_GREEN_R if leaving_group == 0 else self.EW_GREEN_R
        yellow_r = self.NS_YELLOW_R if leaving_group == 0 else self.EW_YELLOW_R

        wait_total += self._run_clearance(yellow_sl)
        self._set_signal_state(green_r)
        w, _ = self._advance_seconds(self.right_turn_green)
        wait_total += w
        # Note: caller will run yellow_sl; instead we run yellow_r + all-red and skip?
        # Keep clean: finish right-turn yellow+all-red, then set_signal_state is safe.
        wait_total += self._run_clearance(yellow_r)
        return wait_total

    def _serve_right_turn_and_return(self, group: int) -> float:
        """Serve protected right turn for `group` and return to Straight+Left green."""
        wait_total = 0.0
        yellow_sl = self.NS_YELLOW_SL if group == 0 else self.EW_YELLOW_SL
        green_r = self.NS_GREEN_R if group == 0 else self.EW_GREEN_R
        yellow_r = self.NS_YELLOW_R if group == 0 else self.EW_YELLOW_R
        green_sl = self.NS_GREEN_SL if group == 0 else self.EW_GREEN_SL

        wait_total += self._run_clearance(yellow_sl)
        self._set_signal_state(green_r)
        w, _ = self._advance_seconds(self.right_turn_green)
        wait_total += w
        wait_total += self._run_clearance(yellow_r)

        self._set_signal_state(green_sl)
        self._green_elapsed = 0
        w, elapsed = self._advance_seconds(self.delta_time)
        wait_total += w
        self._green_elapsed += elapsed
        return wait_total

    def _get_obs(self) -> np.ndarray:
        assert self._conn is not None
        queues = [
            min(
                self._conn.lane.getLastStepHaltingNumber(lane_id)
                / self.max_queue_per_lane,
                1.0,
            )
            for lane_id in self.INCOMING_LANES
        ]
        obs = np.array(queues + [float(self._current_group)], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _compute_reward(
        self, step_waiting_time: float, sim_seconds: int
    ) -> tuple[float, dict[str, float]]:
        assert self._conn is not None
        total_queue = float(
            sum(
                self._conn.lane.getLastStepHaltingNumber(lane_id)
                for lane_id in self.INCOMING_LANES
            )
        )
        norm_denom = len(self.INCOMING_LANES) * self.max_queue_per_lane
        norm_wait = min(step_waiting_time / (sim_seconds * norm_denom), 1.0)
        norm_queue = min(total_queue / norm_denom, 1.0)
        reward = -(self.wait_weight * norm_wait + self.queue_weight * norm_queue)
        return float(reward), {
            "accumulated_waiting_time": float(step_waiting_time),
            "total_queue": total_queue,
            "norm_wait": float(norm_wait),
            "norm_queue": float(norm_queue),
        }

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            finally:
                self._conn = None

    def __del__(self) -> None:
        self.close()
