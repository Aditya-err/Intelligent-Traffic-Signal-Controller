"""
TraCI Simulation Recorder for v2 2D HTML5 Canvas Replay.

Records a 10-minute (600 s) clip every 0.5 s to a compact JSON file in `v2/replays/`:
- Vehicle id, type code, x, y, angle
- Pedestrian x, y
- Traffic light states (NS & EW lamps) and live countdown timer for J1, J2, J3
- Live stats (spawned, active, completed, total waiting, per-junction queues)
- For DQN: per-junction approach queues, Q-values, chosen action, reward,
  epsilon, and safety-layer override status.

Usage:
    python -m v2.replays.record_replay --controller dqn --demand medium --seed 1001
    python -m v2.replays.record_replay --all
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List

import torch

_THIS_DIR = Path(__file__).resolve().parent
_V2_ROOT = _THIS_DIR.parent
if str(_V2_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT.parent))

from v2 import config  # noqa: E402
from v2.controllers.actuated import ActuatedController  # noqa: E402
from v2.controllers.dqn import (  # noqa: E402
    DQNController,
    compute_junction_reward,
    get_junction_state,
)
from v2.controllers.fixed_time import FixedTimeController  # noqa: E402
from v2.evaluation.run_episode import JUNCTION_IN_EDGES  # noqa: E402
from v2.replays.network_geometry import extract_network_geometry  # noqa: E402
from v2.traffic.generate_traffic import generate_route_file  # noqa: E402

_SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")
_SUMO_TOOLS = Path(_SUMO_HOME) / "tools"
if str(_SUMO_TOOLS) not in sys.path:
    sys.path.insert(0, str(_SUMO_TOOLS))

import traci  # noqa: E402


_TYPE_CODE = {
    "motorcycle": "m",
    "car": "c",
    "auto": "a",
    "bus": "b",
    "truck": "t",
}


def _lamp_from_char(ch: str) -> str:
    c = ch.upper()
    if c == "G":
        return "G"
    if c == "Y":
        return "Y"
    return "R"


def _make_controller(name: str):
    if name == "fixed_time":
        return FixedTimeController()
    if name == "actuated":
        return ActuatedController()
    if name == "dqn":
        return DQNController(config.MODELS_DIR / "dqn_best.pt")
    raise ValueError(f"Unknown controller: {name}")


def record_replay(
    controller_name: str,
    demand: str,
    seed: int = 1001,
    duration_s: int = 600,
    step_s: float = 0.5,
    output_path: Path | None = None,
) -> Path:
    """Run a TraCI clip and save compact replay JSON."""
    ctrl = _make_controller(controller_name)
    sumo_exe = Path(_SUMO_HOME) / "bin" / "sumo.exe"

    if output_path is None:
        config.REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = config.REPLAYS_DIR / f"replay_{controller_name}_{demand}_seed{seed}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        rou_path = Path(tmpdir) / f"{demand}_seed{seed}.rou.xml"
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
            *ctrl.extra_sumo_args(),
            "--seed",
            str(seed),
            "--step-length",
            str(step_s),
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
        ]

        traci.start(cmd)
        try:
            ctrl.on_simulation_start()

            spawned = 0
            completed = 0
            n_steps = int(round(duration_s / step_s))

            # Track last RL decision info per junction for DQN replays
            rl_cache: Dict[str, Dict[str, Any]] = {}
            prev_overrides: Dict[str, int] = {jid: 0 for jid in config.JUNCTION_COORDS}

            for step_idx in range(n_steps):
                sim_time = round(step_idx * step_s, 2)
                is_int_sec = abs(sim_time - round(sim_time)) < 1e-6

                if is_int_sec:
                    if controller_name == "dqn":
                        make_dec = int(round(sim_time)) % config.DQN_DECISION_INTERVAL_S == 0
                        for jid, mgr in ctrl.managers.items():
                            req_act = None
                            if make_dec and mgr.state == "GREEN":
                                obs = get_junction_state(
                                    jid, mgr.current_action, mgr.time_in_green
                                )
                                with torch.no_grad():
                                    qv = ctrl.q_net(
                                        torch.from_numpy(obs).unsqueeze(0)
                                    )[0].tolist()
                                req_act = int(qv[1] > qv[0])
                                raw_q = [
                                    int(traci.edge.getLastStepHaltingNumber(e))
                                    for e in JUNCTION_IN_EDGES[jid]
                                ]
                                before_ov = mgr.safety_overrides
                                did_sw = mgr.apply_decision_and_step(req_act)
                                ov_now = mgr.safety_overrides > before_ov
                                rew = compute_junction_reward(jid, did_sw)
                                rl_cache[jid] = {
                                    "q": raw_q,
                                    "qv": [round(qv[0], 2), round(qv[1], 2)],
                                    "act": req_act,
                                    "rew": round(rew, 1),
                                    "eps": 0.0,
                                    "ov": ov_now,
                                    "hold": max(0, config.MIN_GREEN_S - mgr.time_in_green),
                                }
                            else:
                                mgr.apply_decision_and_step(None)
                                if jid in rl_cache:
                                    rl_cache[jid]["hold"] = max(
                                        0, config.MIN_GREEN_S - mgr.time_in_green
                                    )
                    else:
                        ctrl.step(sim_time)

                traci.simulationStep()
                spawned += traci.simulation.getDepartedNumber()
                completed += traci.simulation.getArrivedNumber()
                now = traci.simulation.getTime()

                # Collect vehicles
                vehs = []
                total_waiting = 0
                for vid in traci.vehicle.getIDList():
                    x, y = traci.vehicle.getPosition(vid)
                    ang = traci.vehicle.getAngle(vid)
                    vtype = _TYPE_CODE.get(traci.vehicle.getTypeID(vid), "c")
                    if traci.vehicle.getSpeed(vid) < 0.1:
                        total_waiting += 1
                    vehs.append([vid, vtype, round(x, 1), round(y, 1), round(ang, 1)])

                # Collect pedestrians
                peds = []
                for pid in traci.person.getIDList():
                    px, py = traci.person.getPosition(pid)
                    peds.append([round(px, 1), round(py, 1)])

                # Collect traffic lights & countdowns
                tls_data: Dict[str, Dict[str, Any]] = {}
                j_queues: List[int] = []
                for jid in ("J1", "J2", "J3"):
                    ryg = traci.trafficlight.getRedYellowGreenState(jid)
                    ns_lamp = _lamp_from_char(ryg[0]) if len(ryg) > 4 else "R"
                    ew_lamp = _lamp_from_char(ryg[4]) if len(ryg) > 4 else "R"

                    tls_data[jid] = {"ns": ns_lamp, "ew": ew_lamp}
                    jq = sum(
                        traci.edge.getLastStepHaltingNumber(e)
                        for e in JUNCTION_IN_EDGES[jid]
                    )
                    j_queues.append(int(jq))

                frame: Dict[str, Any] = {
                    "t": round(now, 1),
                    "v": vehs,
                    "p": peds,
                    "tls": tls_data,
                    "st": {
                        "sp": spawned,
                        "ac": len(vehs),
                        "co": completed,
                        "wa": total_waiting,
                        "jq": j_queues,
                    },
                }
                if controller_name == "dqn" and rl_cache:
                    frame["rl"] = {
                        jid: dict(rl_cache[jid]) for jid in rl_cache
                    }
                frames.append(frame)
        finally:
            traci.close()

    # Post-process exact per-arm countdowns:
    # - When arm is Green: seconds until Green ends.
    # - When arm is Red/Yellow: seconds until Green starts (includes yellow + all-red).
    n_frames = len(frames)
    for jid in ("J1", "J2", "J3"):
        for arm in ("ns", "ew"):
            cd_key = f"cd_{arm}"
            for i in range(n_frames):
                cur_state = frames[i]["tls"][jid][arm]
                t_now = frames[i]["t"]
                target_found = None
                if cur_state == "G":
                    for j in range(i + 1, n_frames):
                        if frames[j]["tls"][jid][arm] != "G":
                            target_found = frames[j]["t"] - t_now
                            break
                else:
                    for j in range(i + 1, n_frames):
                        if frames[j]["tls"][jid][arm] == "G":
                            target_found = frames[j]["t"] - t_now
                            break
                if target_found is None:
                    target_found = max(0.0, duration_s - t_now)
                frames[i]["tls"][jid][cd_key] = max(1, int(round(target_found)))

    payload = {
        "meta": {
            "controller": controller_name,
            "demand": demand,
            "seed": seed,
            "duration_s": duration_s,
            "step_s": step_s,
        },
        "geometry": extract_network_geometry(config.NET_FILE),
        "frames": frames,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record v2 2D replay JSON.")
    parser.add_argument(
        "--controller",
        choices=["fixed_time", "actuated", "dqn"],
        default="fixed_time",
    )
    parser.add_argument(
        "--demand",
        choices=list(config.DEMAND_LEVELS_VPH.keys()),
        default="medium",
    )
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--all", action="store_true", help="Record all 3x3 replays.")
    args = parser.parse_args()

    if args.all:
        for cname in ("fixed_time", "actuated", "dqn"):
            for dname in ("low", "medium", "high"):
                out = record_replay(
                    cname, dname, seed=args.seed, duration_s=args.duration
                )
                size_kb = out.stat().st_size / 1024.0
                print(f"[REPLAY] {cname:10s} | {dname:6s} -> {out.name} ({size_kb:.1f} KB)")
    else:
        out = record_replay(
            args.controller, args.demand, seed=args.seed, duration_s=args.duration
        )
        size_kb = out.stat().st_size / 1024.0
        print(f"[REPLAY] {args.controller} | {args.demand} -> {out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
