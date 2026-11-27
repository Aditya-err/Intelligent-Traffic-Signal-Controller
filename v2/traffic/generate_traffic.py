"""
Random Realistic Traffic & Pedestrian Route Generator for Intelligent-Traffic-RL v2.

Generates a SUMO route file (`rou.xml`) from `v2/config.py`:
- 5 Indian vehicle types: motorcycle, car, auto, bus, truck
- Configurable vehicle mix (default: 45% motorcycle, 25% car, 10% auto, 10% bus, 10% truck)
- Poisson-style random arrival times using a seeded RNG
- Random entry & exit boundary edges (producing straight, left, and right turns)
- Three demand levels: low (~720 veh/hr), medium (~1330 veh/hr), high (~1660 veh/hr)
- Random pedestrians crossing at the zebra crossings of J1, J2, J3

Usage:
    python -m v2.traffic.generate_traffic --demand medium --seed 1001
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

# Allow running both as `python -m v2.traffic.generate_traffic` and directly
_THIS_DIR = Path(__file__).resolve().parent
_V2_ROOT = _THIS_DIR.parent
if str(_V2_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT.parent))

from v2 import config  # noqa: E402

# Ensure SUMO tools are importable for sumolib
_SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")
_SUMO_TOOLS = Path(_SUMO_HOME) / "tools"
if str(_SUMO_TOOLS) not in sys.path:
    sys.path.insert(0, str(_SUMO_TOOLS))

import sumolib  # noqa: E402


# Pairs of incoming/outgoing edges on the same road (avoid immediate U-turns)
_OPPOSITE_EXIT = {
    "W1_J1": "J1_W1",
    "N1_J1": "J1_N1",
    "S1_J1": "J1_S1",
    "N2_J2": "J2_N2",
    "S2_J2": "J2_S2",
    "N3_J3": "J3_N3",
    "S3_J3": "J3_S3",
    "E3_J3": "J3_E3",
}

# Junction-local incoming/outgoing edges for zebra-crossing pedestrians
_JUNCTION_CROSS_EDGES = {
    "J1": (["W1_J1", "N1_J1", "S1_J1", "J2_J1"], ["J1_W1", "J1_N1", "J1_S1", "J1_J2"]),
    "J2": (["J1_J2", "N2_J2", "S2_J2", "J3_J2"], ["J2_J1", "J2_N2", "J2_S2", "J2_J3"]),
    "J3": (["J2_J3", "N3_J3", "S3_J3", "E3_J3"], ["J3_J2", "J3_N3", "J3_S3", "J3_E3"]),
}


def _build_valid_routes(net_path: Path) -> Dict[str, List[List[str]]]:
    """Precompute valid shortest edge paths from each entry edge to non-U-turn exits."""
    net = sumolib.net.readNet(str(net_path))
    routes_by_entry: Dict[str, List[List[str]]] = {}
    for entry_id in config.ENTRY_EDGES:
        entry_edge = net.getEdge(entry_id)
        valid_paths: List[List[str]] = []
        for exit_id in config.EXIT_EDGES:
            if exit_id == _OPPOSITE_EXIT.get(entry_id):
                continue
            exit_edge = net.getEdge(exit_id)
            path, _ = net.getShortestPath(entry_edge, exit_edge)
            if path:
                valid_paths.append([e.getID() for e in path])
        routes_by_entry[entry_id] = valid_paths
    return routes_by_entry


def _edge_lengths(net_path: Path) -> Dict[str, float]:
    """Return length (m) for each normal edge in the network."""
    net = sumolib.net.readNet(str(net_path))
    return {e.getID(): e.getLength() for e in net.getEdges()}


def _sample_poisson_departures(
    rng: random.Random, rate_vph: float, duration_s: float
) -> List[float]:
    """Generate sorted Poisson-process departure timestamps in [0, duration_s)."""
    if rate_vph <= 0 or duration_s <= 0:
        return []
    rate_per_sec = rate_vph / 3600.0
    times: List[float] = []
    t = rng.expovariate(rate_per_sec)
    while t < duration_s:
        times.append(round(t, 2))
        t += rng.expovariate(rate_per_sec)
    return times


def generate_route_file(
    demand_level: str = "medium",
    seed: int = 1001,
    duration_s: int = config.EPISODE_DURATION_S,
    output_path: Path | None = None,
    net_path: Path = config.NET_FILE,
) -> Dict[str, object]:
    """
    Generate a SUMO `.rou.xml` file for the requested demand level and seed.

    Returns a summary dictionary with vehicle counts per type and pedestrian count.
    """
    if demand_level not in config.DEMAND_LEVELS_VPH:
        raise ValueError(
            f"Unknown demand_level '{demand_level}'. Choose from {list(config.DEMAND_LEVELS_VPH)}."
        )

    if output_path is None:
        config.TRAFFIC_DIR.mkdir(parents=True, exist_ok=True)
        output_path = config.TRAFFIC_DIR / f"corridor_{demand_level}_seed{seed}.rou.xml"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    routes_by_entry = _build_valid_routes(net_path)
    lengths = _edge_lengths(net_path)

    # 1. Vehicle departure times (Poisson arrivals)
    vph = config.DEMAND_LEVELS_VPH[demand_level]
    veh_departs = _sample_poisson_departures(rng, vph, float(duration_s))

    vtype_names = list(config.VEHICLE_MIX.keys())
    vtype_weights = [config.VEHICLE_MIX[k] for k in vtype_names]

    events: List[Tuple[float, str, dict]] = []
    type_counts: Counter[str] = Counter()

    for idx, depart in enumerate(veh_departs):
        vtype = rng.choices(vtype_names, weights=vtype_weights, k=1)[0]
        entry_id = rng.choice(config.ENTRY_EDGES)
        edge_path = rng.choice(routes_by_entry[entry_id])
        type_counts[vtype] += 1
        events.append(
            (
                depart,
                "vehicle",
                {
                    "id": f"veh_{idx}",
                    "type": vtype,
                    "depart": f"{depart:.2f}",
                    "departLane": "best",
                    "departSpeed": "max",
                    "edges": " ".join(edge_path),
                },
            )
        )

    # 2. Pedestrian departure times (crossing at J1, J2, J3 zebra crossings)
    ped_departs = _sample_poisson_departures(
        rng, config.PEDESTRIAN_RATE_VPH, float(duration_s)
    )
    junction_ids = list(_JUNCTION_CROSS_EDGES.keys())
    for idx, depart in enumerate(ped_departs):
        jid = rng.choice(junction_ids)
        in_edges, out_edges = _JUNCTION_CROSS_EDGES[jid]
        from_edge = rng.choice(in_edges)
        to_edge = rng.choice(out_edges)
        # Start near the junction end of `from_edge` so pedestrians cross promptly
        start_pos = max(
            1.0,
            lengths.get(from_edge, 100.0) - config.PEDESTRIAN_SPAWN_OFFSET_FROM_JUNCTION_M,
        )
        end_pos = min(
            config.PEDESTRIAN_SPAWN_OFFSET_FROM_JUNCTION_M,
            lengths.get(to_edge, 100.0) - 1.0,
        )
        events.append(
            (
                depart,
                "person",
                {
                    "id": f"ped_{idx}",
                    "depart": f"{depart:.2f}",
                    "departPos": f"{start_pos:.2f}",
                    "from_edge": from_edge,
                    "to_edge": to_edge,
                    "arrivalPos": f"{end_pos:.2f}",
                },
            )
        )

    # SUMO requires vehicles and persons in rou.xml to be sorted by depart time
    events.sort(key=lambda item: item[0])

    # 3. Build XML tree
    root = ET.Element("routes")
    for vtype_id, attrs in config.VEHICLE_TYPES.items():
        ET.SubElement(
            root,
            "vType",
            id=vtype_id,
            vClass=vtype_id if vtype_id in ("motorcycle", "bus", "truck") else "passenger",
            length=str(attrs["length"]),
            width=str(attrs["width"]),
            minGap=str(attrs["minGap"]),
            maxSpeed=str(attrs["maxSpeed"]),
            accel=str(attrs["accel"]),
            decel=str(attrs["decel"]),
            sigma=str(attrs["sigma"]),
            guiShape=str(attrs["guiShape"]),
            color=str(attrs["color"]),
        )
    ET.SubElement(
        root,
        "vType",
        id="pedestrian",
        vClass="pedestrian",
        speed=str(config.PEDESTRIAN_SPEED_MPS),
    )

    for _, kind, data in events:
        if kind == "vehicle":
            veh_el = ET.SubElement(
                root,
                "vehicle",
                id=data["id"],
                type=data["type"],
                depart=data["depart"],
                departLane=data["departLane"],
                departSpeed=data["departSpeed"],
            )
            ET.SubElement(veh_el, "route", edges=data["edges"])
        else:
            ped_el = ET.SubElement(
                root,
                "person",
                id=data["id"],
                type="pedestrian",
                depart=data["depart"],
                departPos=data["departPos"],
            )
            ET.SubElement(
                ped_el,
                "walk",
                **{
                    "from": data["from_edge"],
                    "to": data["to_edge"],
                    "arrivalPos": data["arrivalPos"],
                },
            )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return {
        "demand_level": demand_level,
        "seed": seed,
        "duration_s": duration_s,
        "output_path": str(output_path),
        "total_vehicles": len(veh_departs),
        "vehicle_counts": {k: type_counts[k] for k in vtype_names},
        "total_pedestrians": len(ped_departs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v2 SUMO traffic routes.")
    parser.add_argument(
        "--demand",
        choices=list(config.DEMAND_LEVELS_VPH.keys()),
        default="medium",
        help="Demand level (low, medium, high).",
    )
    parser.add_argument("--seed", type=int, default=config.EVAL_SEEDS[0], help="RNG seed.")
    parser.add_argument(
        "--duration",
        type=int,
        default=config.EPISODE_DURATION_S,
        help="Episode duration in seconds.",
    )
    parser.add_argument(
        "--all-demands",
        action="store_true",
        help="Generate routes for all 3 demand levels using --seed.",
    )
    args = parser.parse_args()

    levels = list(config.DEMAND_LEVELS_VPH.keys()) if args.all_demands else [args.demand]
    for lvl in levels:
        summary = generate_route_file(
            demand_level=lvl, seed=args.seed, duration_s=args.duration
        )
        print(
            f"[{lvl.upper()}] seed={summary['seed']} | "
            f"vehicles={summary['total_vehicles']} | "
            f"pedestrians={summary['total_pedestrians']} | "
            f"mix={summary['vehicle_counts']} -> {summary['output_path']}"
        )


if __name__ == "__main__":
    main()
