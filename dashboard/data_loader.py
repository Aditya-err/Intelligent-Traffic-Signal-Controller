"""Resilient CSV data loading, formatting, and validation for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

CONTROLLER_DISPLAY_NAMES: dict[str, str] = {
    "fixed_time": "Fixed-Time",
    "q_learning": "Q-Learning",
    "sarsa": "SARSA",
    "dqn_medium_only": "DQN Medium-only",
    "dqn_curriculum": "DQN Curriculum",
}

CONTROLLER_ORDER: tuple[str, ...] = (
    "fixed_time",
    "q_learning",
    "sarsa",
    "dqn_medium_only",
    "dqn_curriculum",
)

CONTROLLER_COLORS: dict[str, str] = {
    "Fixed-Time": "#94a3b8",
    "Q-Learning": "#3b82f6",
    "SARSA": "#10b981",
    "DQN Medium-only": "#f59e0b",
    "DQN Curriculum": "#f43f5e",
}

SCENARIO_DISPLAY_NAMES: dict[str, str] = {
    "low": "Low (~720 veh/hr)",
    "medium": "Medium (~1,330 veh/hr)",
    "high": "High (~2,200 veh/hr)",
}

SUMMARY_COLUMN_LABELS: dict[str, str] = {
    "Scenario": "Scenario",
    "Controller": "Controller",
    "num_seeds": "Seeds",
    "avg_waiting_time_s_mean": "Avg Wait (s)",
    "avg_waiting_time_s_std": "Wait Std (s)",
    "avg_waiting_time_s_ci95_half": "Wait 95% CI (±s)",
    "avg_waiting_time_s_pct_vs_ft": "Wait % vs Fixed-Time",
    "avg_queue_length_mean": "Avg Queue (veh)",
    "avg_travel_time_s_mean": "Avg Travel (s)",
    "throughput_veh_per_hr_mean": "Throughput (veh/hr)",
    "max_queue_length_mean": "Max Queue (veh)",
    "vehicles_arrived_mean": "Arrived (veh)",
    "vehicles_remaining_mean": "Remaining (veh)",
    "episode_reward_mean": "Episode Reward",
}


def format_p_value(val: float) -> str:
    """Format p-values cleanly, displaying very small values as '< 0.0001'."""
    if val < 0.0001:
        return "< 0.0001"
    return f"{val:.4f}"


@dataclass(frozen=True)
class CsvLoadResult:
    """Container for a loaded CSV DataFrame and optional validation error."""

    df: pd.DataFrame | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.df is not None and self.error is None


def load_validated_csv(
    csv_path: Path, required_columns: Iterable[str]
) -> CsvLoadResult:
    """Load a CSV file and verify required columns and non-empty rows."""
    if not csv_path.is_file():
        return CsvLoadResult(df=None, error=f"Missing data file: {csv_path.name}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        return CsvLoadResult(
            df=None, error=f"Failed to read {csv_path.name}: {exc}"
        )

    if df.empty:
        return CsvLoadResult(df=None, error=f"Data file is empty: {csv_path.name}")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return CsvLoadResult(
            df=None,
            error=f"{csv_path.name} missing required columns: {', '.join(missing_cols)}",
        )

    return CsvLoadResult(df=df, error=None)


def load_dashboard_datasets(
    reports_dir: Path | None = None,
) -> dict[str, CsvLoadResult]:
    """Load all Phase 6–8 evaluation and training CSVs."""
    rdir = reports_dir or DEFAULT_REPORTS_DIR
    return {
        "final_summary": load_validated_csv(
            rdir / "final_eval_summary.csv",
            [
                "scenario",
                "controller",
                "num_seeds",
                "avg_waiting_time_s_mean",
                "avg_waiting_time_s_std",
                "avg_waiting_time_s_ci95_half",
                "avg_waiting_time_s_pct_vs_ft",
                "avg_queue_length_mean",
                "avg_travel_time_s_mean",
                "throughput_veh_per_hr_mean",
                "vehicles_arrived_mean",
                "vehicles_remaining_mean",
                "episode_reward_mean",
            ],
        ),
        "final_raw": load_validated_csv(
            rdir / "final_eval_raw.csv",
            [
                "scenario",
                "controller",
                "seed",
                "avg_waiting_time_s",
                "avg_queue_length",
                "avg_travel_time_s",
                "throughput_veh_per_hr",
                "episode_reward",
            ],
        ),
        "final_stats": load_validated_csv(
            rdir / "final_eval_stats.csv",
            [
                "scenario",
                "comparison",
                "metric",
                "mean_diff",
                "t_stat",
                "paired_t_p",
                "perm_p",
                "significant_005",
            ],
        ),
        "dqn_training": load_validated_csv(
            rdir / "dqn_training_history.csv",
            ["episode", "epsilon", "episode_reward", "avg_loss", "avg_queue_length"],
        ),
        "curriculum_training": load_validated_csv(
            rdir / "dqn_curriculum_hparam_experiments.csv",
            [
                "experiment",
                "episode",
                "stage",
                "scenario",
                "epsilon",
                "episode_reward",
                "avg_waiting_time_s",
                "avg_loss",
            ],
        ),
    }
