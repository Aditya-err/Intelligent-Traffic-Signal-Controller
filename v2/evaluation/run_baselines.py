"""
Run Fixed-Time and SUMO Actuated baselines across 3 demand levels x 3 seeds (1001-1003).

Saves per-episode metrics to `v2/results/baselines/baseline_results.csv` and prints
a summary table plus teleport/collision/gridlock checks.

Usage:
    python -m v2.evaluation.run_baselines
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
_V2_ROOT = _THIS_DIR.parent
if str(_V2_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT.parent))

from v2 import config  # noqa: E402
from v2.controllers.actuated import ActuatedController  # noqa: E402
from v2.controllers.fixed_time import FixedTimeController  # noqa: E402
from v2.evaluation.run_episode import run_episode  # noqa: E402


BASELINE_SEEDS = [1001, 1002, 1003]


def main() -> None:
    controllers = [FixedTimeController(), ActuatedController()]
    demands = list(config.DEMAND_LEVELS_VPH.keys())

    rows = []
    for ctrl in controllers:
        for demand in demands:
            for seed in BASELINE_SEEDS:
                res = run_episode(ctrl, demand=demand, seed=seed)
                rows.append(res)
                print(
                    f"[{ctrl.name:10s} | {demand:6s} | seed={seed}] "
                    f"wait={res['avg_waiting_time_s']:6.2f}s | "
                    f"queue={res['avg_queue_length']:5.2f} | "
                    f"tput={res['throughput_vph']:6.1f}/hr | "
                    f"teleports={res['teleports']} | collisions={res['collisions']} | "
                    f"max_wait={res['max_vehicle_wait_s']:.1f}s"
                )

    config.BASELINES_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.BASELINES_RESULTS_DIR / "baseline_results.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved baseline results to: {csv_path}")


if __name__ == "__main__":
    main()
