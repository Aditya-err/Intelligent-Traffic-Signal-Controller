"""
Final Evaluation Runner for Intelligent-Traffic-RL v2 (Phase 6).

Runs Fixed-Time, SUMO Actuated, and Shared DQN on all 3 demand levels x
10 held-out seeds (1001-1010), full 1-hour episodes, using `run_episode`.
Saves:
- `v2/results/final/final_results.csv`
- `v2/results/final/summary_table.csv`
- `v2/results/final/statistical_tests.csv`

Usage:
    python -m v2.evaluation.final_eval
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_V2_ROOT = _THIS_DIR.parent
if str(_V2_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT.parent))

from v2 import config  # noqa: E402
from v2.controllers.actuated import ActuatedController  # noqa: E402
from v2.controllers.dqn import DQNController  # noqa: E402
from v2.controllers.fixed_time import FixedTimeController  # noqa: E402
from v2.evaluation.run_episode import run_episode  # noqa: E402
from v2.evaluation.stats import (  # noqa: E402
    build_statistical_tests,
    build_summary_table,
    save_csv,
)


def _run_one_task(task: Tuple[str, str, int]) -> Dict[str, Any]:
    cname, demand, seed = task
    if cname == "fixed_time":
        ctrl = FixedTimeController()
    elif cname == "actuated":
        ctrl = ActuatedController()
    elif cname == "dqn":
        ctrl = DQNController(config.MODELS_DIR / "dqn_best.pt")
    else:
        raise ValueError(f"Unknown controller: {cname}")
    return run_episode(ctrl, demand=demand, seed=seed)


def _load_existing_cache() -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    candidate_files = [
        config.FINAL_RESULTS_DIR / "final_results.csv",
        config.BASELINES_RESULTS_DIR / "baseline_results.csv",
        config.RESULTS_DIR / "dqn_eval_results.csv",
    ]
    for path in candidate_files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["controller"], row["demand"], int(row["seed"]))
                if key not in cache:
                    cache[key] = dict(row)
    return cache


def main() -> None:
    controllers = ("fixed_time", "actuated", "dqn")
    demands = ("low", "medium", "high")
    seeds = config.EVAL_SEEDS  # 1001..1010

    # Ensure actuated_tls.add.xml exists before spawning parallel workers
    ActuatedController()

    cache = _load_existing_cache()
    tasks: List[Tuple[str, str, int]] = []
    rows: List[Dict[str, Any]] = []

    for c in controllers:
        for d in demands:
            for s in seeds:
                key = (c, d, s)
                if key in cache:
                    rows.append(cache[key])
                else:
                    tasks.append(key)

    print(
        f"[final_eval] Cached={len(rows)} episodes | Running={len(tasks)} new episodes (4 workers)..."
    )

    if tasks:
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_run_one_task, t): t for t in tasks}
            done_cnt = 0
            for fut in as_completed(futures):
                res = fut.result()
                rows.append(res)
                done_cnt += 1
                print(
                    f"  ({done_cnt}/{len(tasks)}) [{res['controller']:10s} | "
                    f"{res['demand']:6s} | seed={res['seed']}] "
                    f"wait={res['avg_waiting_time_s']:.2f}s | "
                    f"tput={res['throughput_vph']:.1f}/hr | "
                    f"teleports={res['teleports']} | collisions={res['collisions']}"
                )

    # Sort deterministically
    c_order = {"fixed_time": 0, "actuated": 1, "dqn": 2}
    d_order = {"low": 0, "medium": 1, "high": 2}
    rows.sort(
        key=lambda r: (c_order[r["controller"]], d_order[r["demand"]], int(r["seed"]))
    )

    final_csv = config.FINAL_RESULTS_DIR / "final_results.csv"
    summary_csv = config.FINAL_RESULTS_DIR / "summary_table.csv"
    stats_csv = config.FINAL_RESULTS_DIR / "statistical_tests.csv"

    save_csv(rows, final_csv)
    summary = build_summary_table(rows)
    save_csv(summary, summary_csv)
    stats = build_statistical_tests(rows)
    save_csv(stats, stats_csv)

    print("\n=== SUMMARY TABLE (mean +/- 95% CI, seeds 1001-1010) ===")
    for s in summary:
        print(
            f"{s['controller']:10s} | {s['demand']:6s} | "
            f"Wait={s['avg_waiting_time_s_mean']:5.2f}+/-{s['avg_waiting_time_s_ci95']:4.2f}s "
            f"({s['wait_improvement_vs_fixed_pct']:+5.1f}%) | "
            f"Queue={s['avg_queue_length_mean']:5.2f} | "
            f"Travel={s['avg_travel_time_s_mean']:5.1f}s | "
            f"Tput={s['throughput_vph_mean']:6.1f}/hr"
        )

    print("\n=== STATISTICAL TESTS (Holm-corrected, seeds 1001-1010) ===")
    for st_row in stats:
        print(
            f"{st_row['comparison']:22s} | {st_row['demand']:6s} | "
            f"diff={st_row['diff_s']:+5.2f}s | "
            f"p_t_holm={st_row['p_ttest_holm']:.4f} | "
            f"p_perm_holm={st_row['p_perm_holm']:.4f} | "
            f"Sig={st_row['significant']:3s} | {st_row['direction']}"
        )


if __name__ == "__main__":
    main()
