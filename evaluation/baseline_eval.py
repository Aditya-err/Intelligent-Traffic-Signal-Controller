"""Phase 3 fixed-time baseline evaluation across low, medium, and high traffic.

Runs ``FixedTimeController`` inside ``SumoTrafficEnv`` with identical network,
simulation duration (1800s), and random seed (42) for reproducible comparisons.
Saves raw metrics to ``reports/baseline_results.csv`` and generates comparison
charts at ``reports/baseline_comparison.png``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms.fixed_time import FixedTimeController
from environment.traffic_env import SumoTrafficEnv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

SCENARIO_CONFIGS: dict[str, Path] = {
    "low": PROJECT_ROOT / "simulation" / "config" / "scenario_low.sumocfg",
    "medium": PROJECT_ROOT / "simulation" / "config" / "scenario_medium.sumocfg",
    "high": PROJECT_ROOT / "simulation" / "config" / "scenario_high.sumocfg",
}


def _parse_sumo_outputs(
    tripinfo_path: Path, summary_path: Path
) -> dict[str, float | int]:
    """Extract trip-level waiting/travel time and arrived/remaining counts."""
    trip_root = ET.parse(tripinfo_path).getroot()
    trips = trip_root.findall("tripinfo")
    if trips:
        avg_wait = sum(float(t.attrib["waitingTime"]) for t in trips) / len(trips)
        avg_travel = sum(float(t.attrib["duration"]) for t in trips) / len(trips)
    else:
        avg_wait = 0.0
        avg_travel = 0.0

    sum_root = ET.parse(summary_path).getroot()
    steps = sum_root.findall("step")
    if steps:
        last = steps[-1]
        arrived = int(last.attrib.get("arrived", len(trips)))
        remaining = int(last.attrib.get("running", 0)) + int(
            last.attrib.get("waiting", 0)
        )
    else:
        arrived = len(trips)
        remaining = 0

    return {
        "avg_waiting_time_s": round(avg_wait, 2),
        "avg_travel_time_s": round(avg_travel, 2),
        "vehicles_arrived": arrived,
        "vehicles_remaining": remaining,
    }


def evaluate_scenario(
    scenario_name: str,
    sumocfg_path: Path,
    *,
    max_steps: int = 1800,
    seed: int = 42,
    ns_green_steps: int = 5,
    ew_green_steps: int = 4,
) -> dict[str, Any]:
    """Run one fixed-time baseline episode and return all 7 required metrics."""
    env = SumoTrafficEnv(
        sumocfg_path=sumocfg_path,
        use_gui=False,
        max_steps=max_steps,
        delta_time=5,
        min_green=10,
        yellow_time=3,
        all_red_time=2,
        right_turn_green=5,
        sumo_seed=seed,
    )
    controller = FixedTimeController(
        ns_green_steps=ns_green_steps,
        ew_green_steps=ew_green_steps,
    )

    queue_samples: list[float] = []
    sim_time = 0
    try:
        obs, _ = env.reset(seed=seed)
        controller.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = controller.select_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            queue_samples.append(float(info["total_queue"]))
            sim_time = int(info["sim_time"])
    finally:
        env.close()

    tripinfo_path = REPORTS_DIR / f"tripinfo_{scenario_name}.xml"
    summary_path = REPORTS_DIR / f"summary_{scenario_name}.xml"
    sumo_metrics = _parse_sumo_outputs(tripinfo_path, summary_path)

    avg_queue = sum(queue_samples) / len(queue_samples) if queue_samples else 0.0
    max_queue = max(queue_samples) if queue_samples else 0.0
    hours = max(sim_time, 1) / 3600.0
    throughput = float(sumo_metrics["vehicles_arrived"]) / hours

    return {
        "scenario": scenario_name,
        "controller": "fixed_time",
        "sim_duration_s": sim_time,
        "seed": seed,
        "avg_waiting_time_s": sumo_metrics["avg_waiting_time_s"],
        "avg_queue_length": round(avg_queue, 2),
        "avg_travel_time_s": sumo_metrics["avg_travel_time_s"],
        "throughput_veh_per_hr": round(throughput, 2),
        "max_queue_length": round(max_queue, 2),
        "vehicles_arrived": sumo_metrics["vehicles_arrived"],
        "vehicles_remaining": sumo_metrics["vehicles_remaining"],
    }


def save_results_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    """Persist baseline evaluation rows to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "controller",
        "sim_duration_s",
        "seed",
        "avg_waiting_time_s",
        "avg_queue_length",
        "avg_travel_time_s",
        "throughput_veh_per_hr",
        "max_queue_length",
        "vehicles_arrived",
        "vehicles_remaining",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_comparison_plots(rows: list[dict[str, Any]], plot_path: Path) -> None:
    """Create a 2x2 Matplotlib summary chart comparing Low/Medium/High scenarios."""
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = [r["scenario"].capitalize() for r in rows]
    avg_wait = [r["avg_waiting_time_s"] for r in rows]
    avg_travel = [r["avg_travel_time_s"] for r in rows]
    avg_queue = [r["avg_queue_length"] for r in rows]
    max_queue = [r["max_queue_length"] for r in rows]
    throughput = [r["throughput_veh_per_hr"] for r in rows]
    arrived = [r["vehicles_arrived"] for r in rows]
    remaining = [r["vehicles_remaining"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "Phase 3 Fixed-Time Baseline Performance (Indian LHT 4-Way Intersection)",
        fontsize=13,
        fontweight="bold",
    )

    # 1. Average Waiting Time & Travel Time
    x = range(len(scenarios))
    w = 0.35
    axes[0, 0].bar([i - w / 2 for i in x], avg_wait, width=w, label="Avg Waiting (s)")
    axes[0, 0].bar([i + w / 2 for i in x], avg_travel, width=w, label="Avg Travel (s)")
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(scenarios)
    axes[0, 0].set_ylabel("Seconds")
    axes[0, 0].set_title("Waiting & Travel Time")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    # 2. Average & Max Queue Length
    axes[0, 1].bar([i - w / 2 for i in x], avg_queue, width=w, label="Avg Queue (veh)")
    axes[0, 1].bar([i + w / 2 for i in x], max_queue, width=w, label="Max Queue (veh)")
    axes[0, 1].set_xticks(list(x))
    axes[0, 1].set_xticklabels(scenarios)
    axes[0, 1].set_ylabel("Vehicles")
    axes[0, 1].set_title("Queue Length (8 Incoming Lanes)")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.3)

    # 3. Throughput (veh/hr)
    axes[1, 0].bar(scenarios, throughput, color="#2a9d8f")
    axes[1, 0].set_ylabel("Vehicles / Hour")
    axes[1, 0].set_title("Vehicle Throughput")
    axes[1, 0].grid(axis="y", alpha=0.3)

    # 4. Vehicles Arrived vs Remaining
    axes[1, 1].bar([i - w / 2 for i in x], arrived, width=w, label="Arrived")
    axes[1, 1].bar([i + w / 2 for i in x], remaining, width=w, label="Remaining")
    axes[1, 1].set_xticks(list(x))
    axes[1, 1].set_xticklabels(scenarios)
    axes[1, 1].set_ylabel("Vehicles")
    axes[1, 1].set_title("Completed vs Remaining Vehicles")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def run_all_baselines(
    max_steps: int = 1800, seed: int = 42
) -> list[dict[str, Any]]:
    """Run all 3 scenarios, save CSV + comparison plot, and return results."""
    results = [
        evaluate_scenario(name, cfg, max_steps=max_steps, seed=seed)
        for name, cfg in SCENARIO_CONFIGS.items()
    ]
    save_results_csv(results, REPORTS_DIR / "baseline_results.csv")
    generate_comparison_plots(results, REPORTS_DIR / "baseline_comparison.png")
    return results


if __name__ == "__main__":
    rows = run_all_baselines()
    for row in rows:
        print(row)
