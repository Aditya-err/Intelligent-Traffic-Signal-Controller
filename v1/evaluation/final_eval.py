"""Phase 8: Final 10-seed Frozen Evaluation and Statistical Comparison."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithms.dqn import DQNAgent
from algorithms.fixed_time import FixedTimeController
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from evaluation.baseline_eval import PROJECT_ROOT, REPORTS_DIR, SCENARIO_CONFIGS
from evaluation.stats_utils import (
    compute_summary_stats,
    exact_permutation_test,
    paired_t_test,
    pct_improvement,
)
from evaluation.train_tabular_rl import evaluate_controller_on_scenario


MODELS_DIR = PROJECT_ROOT / "models"
FINAL_EVAL_SEEDS: tuple[int, ...] = tuple(range(1001, 1011))  # 10 held-out seeds

CONTROLLER_ORDER: tuple[str, ...] = (
    "fixed_time",
    "q_learning",
    "sarsa",
    "dqn_medium_only",
    "dqn_curriculum",
)

CONTROLLER_LABELS: dict[str, str] = {
    "fixed_time": "Fixed-Time",
    "q_learning": "Q-Learning",
    "sarsa": "SARSA",
    "dqn_medium_only": "DQN (Med)",
    "dqn_curriculum": "DQN (Curr)",
}

CONTROLLER_COLORS: dict[str, str] = {
    "fixed_time": "#6c757d",
    "q_learning": "#1f77b4",
    "sarsa": "#2a9d8f",
    "dqn_medium_only": "#e76f51",
    "dqn_curriculum": "#9b2226",
}


def load_frozen_controllers() -> list[tuple[str, Any]]:
    """Load all 5 controllers in strict frozen evaluation mode (epsilon=0)."""
    q_agent = QLearningAgent(seed=42)
    q_agent.load(MODELS_DIR / "q_learning_qtable.json")
    q_agent.epsilon = 0.0

    sarsa_agent = SARSAAgent(seed=42)
    sarsa_agent.load(MODELS_DIR / "sarsa_qtable.json")
    sarsa_agent.epsilon = 0.0

    dqn_med = DQNAgent(seed=42)
    dqn_med.load(MODELS_DIR / "dqn_checkpoint.pt")
    dqn_med.epsilon = 0.0
    dqn_med.q_net.eval()

    dqn_curr = DQNAgent(seed=42)
    dqn_curr.load(MODELS_DIR / "dqn_curriculum_best.pt")
    dqn_curr.epsilon = 0.0
    dqn_curr.q_net.eval()

    return [
        ("fixed_time", FixedTimeController(ns_green_steps=5, ew_green_steps=4)),
        ("q_learning", q_agent),
        ("sarsa", sarsa_agent),
        ("dqn_medium_only", dqn_med),
        ("dqn_curriculum", dqn_curr),
    ]


def summarize_and_test(
    raw_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute mean/std/95%-CI/%-improvement and paired statistical tests."""
    metric_specs = (
        ("avg_waiting_time_s", True),
        ("avg_queue_length", True),
        ("avg_travel_time_s", True),
        ("throughput_veh_per_hr", False),
        ("max_queue_length", True),
        ("vehicles_arrived", False),
        ("vehicles_remaining", True),
        ("episode_reward", False),
    )

    by_sc_ctrl: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (str(row["scenario"]), str(row["controller"]))
        by_sc_ctrl.setdefault(key, []).append(row)
    for rows in by_sc_ctrl.values():
        rows.sort(key=lambda r: int(r["seed"]))

    summary_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []

    for scenario in SCENARIO_CONFIGS:
        ft_rows = by_sc_ctrl[(scenario, "fixed_time")]
        ft_means = {
            m: compute_summary_stats([float(r[m]) for r in ft_rows])["mean"]
            for m, _ in metric_specs
        }

        for ctrl in CONTROLLER_ORDER:
            rows = by_sc_ctrl[(scenario, ctrl)]
            srow: dict[str, Any] = {
                "scenario": scenario,
                "controller": ctrl,
                "num_seeds": len(rows),
            }
            for metric, lower_better in metric_specs:
                st = compute_summary_stats([float(r[metric]) for r in rows])
                srow[f"{metric}_mean"] = round(st["mean"], 2)
                srow[f"{metric}_std"] = round(st["std"], 2)
                srow[f"{metric}_ci95_half"] = round(st["ci95_half"], 2)
                srow[f"{metric}_pct_vs_ft"] = round(
                    pct_improvement(
                        ft_means[metric], st["mean"], lower_is_better=lower_better
                    ),
                    2,
                )
            summary_rows.append(srow)

            if ctrl != "fixed_time":
                for metric in ("avg_waiting_time_s", "avg_queue_length", "episode_reward"):
                    a_vals = [float(r[metric]) for r in rows]
                    b_vals = [float(r[metric]) for r in ft_rows]
                    pt = paired_t_test(a_vals, b_vals)
                    perm_p = exact_permutation_test(a_vals, b_vals)
                    stats_rows.append(
                        {
                            "scenario": scenario,
                            "comparison": f"{ctrl}_vs_fixed_time",
                            "metric": metric,
                            "mean_diff": round(pt["mean_diff"], 4),
                            "t_stat": round(pt["t_stat"], 4),
                            "paired_t_p": round(pt["p_value"], 6),
                            "perm_p": round(perm_p, 6),
                            "significant_005": perm_p < 0.05 and pt["p_value"] < 0.05,
                        }
                    )

        # Direct comparison: dqn_curriculum vs dqn_medium_only
        curr_rows = by_sc_ctrl[(scenario, "dqn_curriculum")]
        med_rows = by_sc_ctrl[(scenario, "dqn_medium_only")]
        for metric in ("avg_waiting_time_s", "avg_queue_length", "episode_reward"):
            a_vals = [float(r[metric]) for r in curr_rows]
            b_vals = [float(r[metric]) for r in med_rows]
            pt = paired_t_test(a_vals, b_vals)
            perm_p = exact_permutation_test(a_vals, b_vals)
            stats_rows.append(
                {
                    "scenario": scenario,
                    "comparison": "dqn_curriculum_vs_dqn_medium_only",
                    "metric": metric,
                    "mean_diff": round(pt["mean_diff"], 4),
                    "t_stat": round(pt["t_stat"], 4),
                    "paired_t_p": round(pt["p_value"], 6),
                    "perm_p": round(perm_p, 6),
                    "significant_005": perm_p < 0.05 and pt["p_value"] < 0.05,
                }
            )

    return summary_rows, stats_rows


def _save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_final_plots(
    raw_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]
) -> None:
    """Generate all 4 Phase 8 evaluation charts on E:."""
    scenarios = ["low", "medium", "high"]
    lookup = {(r["scenario"], r["controller"]): r for r in summary_rows}

    # 1. Controller Comparison Chart (with 95% CI error bars)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Phase 8 Final Controller Comparison (Mean ± 95% CI, 10 Held-Out Seeds)",
        fontsize=13,
        fontweight="bold",
    )
    x = np.arange(len(scenarios))
    w = 0.15
    panels = [
        (axes[0, 0], "avg_waiting_time_s", "Seconds", "Average Waiting Time"),
        (axes[0, 1], "avg_queue_length", "Vehicles", "Average Queue Length"),
        (axes[1, 0], "throughput_veh_per_hr", "Vehicles / Hour", "Throughput"),
        (axes[1, 1], "episode_reward", "Total Reward", "Episode Reward"),
    ]
    for ax, key, ylabel, title in panels:
        for idx, ctrl in enumerate(CONTROLLER_ORDER):
            means = [float(lookup[(sc, ctrl)][f"{key}_mean"]) for sc in scenarios]
            cis = [float(lookup[(sc, ctrl)][f"{key}_ci95_half"]) for sc in scenarios]
            ax.bar(
                x + (idx - 2.0) * w,
                means,
                yerr=cis,
                width=w,
                capsize=2,
                label=CONTROLLER_LABELS[ctrl],
                color=CONTROLLER_COLORS[ctrl],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in scenarios])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(REPORTS_DIR / "final_controller_comparison.png", dpi=150)
    plt.close(fig)

    # 2. Scenario-wise Percentage Improvement vs Fixed-Time
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Scenario-Wise % Improvement in Avg Waiting Time vs Fixed-Time",
        fontsize=12,
        fontweight="bold",
    )
    rl_ctrls = [c for c in CONTROLLER_ORDER if c != "fixed_time"]
    for ax, sc in zip(axes, scenarios):
        vals = [float(lookup[(sc, c)]["avg_waiting_time_s_pct_vs_ft"]) for c in rl_ctrls]
        cols = [CONTROLLER_COLORS[c] for c in rl_ctrls]
        ax.bar([CONTROLLER_LABELS[c] for c in rl_ctrls], vals, color=cols)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{sc.capitalize()} Demand")
        ax.set_ylabel("% Improvement (+ is lower wait)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(REPORTS_DIR / "final_scenario_performance.png", dpi=150)
    plt.close(fig)

    # 3. Waiting-time Distribution Boxplots across 10 Seeds
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.suptitle(
        "Average Waiting Time Distribution Across 10 Held-Out Seeds",
        fontsize=12,
        fontweight="bold",
    )
    for ax, sc in zip(axes, scenarios):
        data = [
            [
                float(r["avg_waiting_time_s"])
                for r in raw_rows
                if r["scenario"] == sc and r["controller"] == ctrl
            ]
            for ctrl in CONTROLLER_ORDER
        ]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=[CONTROLLER_LABELS[c] for c in CONTROLLER_ORDER])
        for patch, ctrl in zip(bp["boxes"], CONTROLLER_ORDER):
            patch.set_facecolor(CONTROLLER_COLORS[ctrl])
            patch.set_alpha(0.75)
        ax.set_title(f"{sc.capitalize()} Demand")
        ax.set_ylabel("Avg Waiting Time (s)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(REPORTS_DIR / "final_waiting_distribution.png", dpi=150)
    plt.close(fig)

    # 4. Multi-seed Variability Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Per-Seed Average Waiting Time Variability (Seeds 1001–1010)",
        fontsize=12,
        fontweight="bold",
    )
    for ax, sc in zip(axes, scenarios):
        for ctrl in CONTROLLER_ORDER:
            sub = sorted(
                [r for r in raw_rows if r["scenario"] == sc and r["controller"] == ctrl],
                key=lambda r: int(r["seed"]),
            )
            ax.plot(
                [int(r["seed"]) for r in sub],
                [float(r["avg_waiting_time_s"]) for r in sub],
                marker="o",
                markersize=4,
                label=CONTROLLER_LABELS[ctrl],
                color=CONTROLLER_COLORS[ctrl],
            )
        ax.set_title(f"{sc.capitalize()} Demand")
        ax.set_xlabel("Evaluation Seed")
        ax.set_ylabel("Avg Waiting Time (s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(REPORTS_DIR / "final_seed_variability.png", dpi=150)
    plt.close(fig)


def run_final_evaluation(
    seeds: tuple[int, ...] = FINAL_EVAL_SEEDS, max_steps: int = 1800
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all 5 frozen controllers across all scenarios and held-out seeds."""
    controllers = load_frozen_controllers()
    raw_rows: list[dict[str, Any]] = []

    for scenario_name, cfg_path in SCENARIO_CONFIGS.items():
        for ctrl_name, ctrl in controllers:
            for seed in seeds:
                raw_rows.append(
                    evaluate_controller_on_scenario(
                        ctrl,
                        ctrl_name,
                        scenario_name,
                        cfg_path,
                        max_steps=max_steps,
                        seed=seed,
                    )
                )

    summary_rows, stats_rows = summarize_and_test(raw_rows)
    _save_csv(raw_rows, REPORTS_DIR / "final_eval_raw.csv")
    _save_csv(summary_rows, REPORTS_DIR / "final_eval_summary.csv")
    _save_csv(stats_rows, REPORTS_DIR / "final_eval_stats.csv")
    generate_final_plots(raw_rows, summary_rows)
    return raw_rows, summary_rows, stats_rows


if __name__ == "__main__":
    raw, summary, stats = run_final_evaluation()
    print(f"Total evaluated episodes: {len(raw)}")
    print("=== SUMMARY ===")
    for r in summary:
        print(r)
    print("=== STATISTICAL TESTS ===")
    for s in stats:
        print(s)
