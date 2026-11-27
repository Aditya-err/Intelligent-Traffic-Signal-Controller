"""
Evaluate trained shared DQN with `run_episode` on Low/Medium/High (seeds 1001-1003)
and compare against Fixed-Time and SUMO Actuated baselines in one table.

Usage:
    python -m v2.evaluation.compare_dqn
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
from v2.controllers.dqn import DQNController  # noqa: E402
from v2.evaluation.run_baselines import BASELINE_SEEDS  # noqa: E402
from v2.evaluation.run_episode import run_episode  # noqa: E402


def main() -> None:
    ctrl = DQNController(config.MODELS_DIR / "dqn_best.pt")
    demands = list(config.DEMAND_LEVELS_VPH.keys())

    dqn_rows = []
    for demand in demands:
        for seed in BASELINE_SEEDS:
            res = run_episode(ctrl, demand=demand, seed=seed)
            dqn_rows.append(res)
            print(
                f"[dqn       | {demand:6s} | seed={seed}] "
                f"wait={res['avg_waiting_time_s']:6.2f}s | "
                f"queue={res['avg_queue_length']:5.2f} | "
                f"tput={res['throughput_vph']:6.1f}/hr | "
                f"teleports={res['teleports']} | collisions={res['collisions']}"
            )

    dqn_csv = config.RESULTS_DIR / "dqn_eval_results.csv"
    dqn_csv.parent.mkdir(parents=True, exist_ok=True)
    with dqn_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dqn_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dqn_rows)

    # Load baselines if available and print combined comparison table
    base_csv = config.BASELINES_RESULTS_DIR / "baseline_results.csv"
    all_rows = list(dqn_rows)
    if base_csv.exists():
        with base_csv.open("r", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f)) + dqn_rows

    print("\n=== CONTROLLER COMPARISON (mean over seeds 1001-1003) ===")
    print(f"{'Controller':12s} | {'Demand':6s} | {'Wait (s)':8s} | {'Queue':6s} | {'Tput (/hr)':10s}")
    print("-" * 55)
    for cname in ("fixed_time", "actuated", "dqn"):
        for demand in demands:
            sub = [
                r for r in all_rows if r["controller"] == cname and r["demand"] == demand
            ]
            if not sub:
                continue
            w = sum(float(r["avg_waiting_time_s"]) for r in sub) / len(sub)
            q = sum(float(r["avg_queue_length"]) for r in sub) / len(sub)
            t = sum(float(r["throughput_vph"]) for r in sub) / len(sub)
            print(f"{cname:12s} | {demand:6s} | {w:8.2f} | {q:6.2f} | {t:10.1f}")


if __name__ == "__main__":
    main()
