"""Phase 7: DQN Curriculum Learning, Hyperparameter Experiments, and Multi-Seed Evaluation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from algorithms.dqn import DQNAgent
from algorithms.fixed_time import FixedTimeController
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from environment.rl_wrapper import TrafficRLWrapper
from environment.traffic_env import SumoTrafficEnv
from evaluation.baseline_eval import (
    PROJECT_ROOT,
    REPORTS_DIR,
    SCENARIO_CONFIGS,
    _parse_sumo_outputs,
)
from evaluation.train_tabular_rl import evaluate_controller_on_scenario


MODELS_DIR = PROJECT_ROOT / "models"
EVAL_SEEDS: tuple[int, ...] = (101, 202, 303)


def build_curriculum_schedule(
    low_episodes: int = 4,
    medium_episodes: int = 4,
    high_episodes: int = 4,
    mixed_episodes: int = 3,
) -> list[tuple[str, str]]:
    """Build ordered (stage_name, scenario_name) pairs for curriculum training."""
    schedule: list[tuple[str, str]] = []
    schedule.extend([("stage1_low", "low")] * low_episodes)
    schedule.extend([("stage2_medium", "medium")] * medium_episodes)
    schedule.extend([("stage3_high", "high")] * high_episodes)
    mixed_cycle = ("low", "medium", "high")
    for idx in range(mixed_episodes):
        schedule.append(("stage4_mixed", mixed_cycle[idx % len(mixed_cycle)]))
    return schedule


def train_dqn_curriculum(
    agent: DQNAgent,
    schedule: list[tuple[str, str]],
    *,
    experiment_name: str = "curriculum_baseline",
    max_steps: int = 1800,
    base_seed: int = 42,
    best_ckpt_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Train DQNAgent across a multi-scenario curriculum with live SUMO.

    Best checkpoint selection uses ONLY training-time episode rewards during
    the final stage (never evaluation seeds).
    """
    history: list[dict[str, Any]] = []
    best_train_score = -float("inf")
    final_stage = schedule[-1][0] if schedule else ""

    for ep_idx, (stage_name, scenario_name) in enumerate(schedule, start=1):
        ep_seed = base_seed + ep_idx
        cfg_path = SCENARIO_CONFIGS[scenario_name]
        env = TrafficRLWrapper(
            SumoTrafficEnv(
                sumocfg_path=cfg_path,
                use_gui=False,
                max_steps=max_steps,
                delta_time=5,
                min_green=10,
                yellow_time=3,
                all_red_time=2,
                right_turn_green=5,
                sumo_seed=ep_seed,
            )
        )
        ep_reward = 0.0
        losses: list[float] = []
        queues: list[float] = []
        try:
            state, _ = env.reset(seed=ep_seed)
            agent.on_episode_start(ep_idx)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.select_action(state, explore=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                queues.append(float(info["total_queue"]))
                metrics = agent.observe_transition(
                    state, action, reward, next_state, terminated, truncated, info
                )
                if metrics is not None:
                    losses.append(float(metrics["loss"]))
                state = next_state
        finally:
            env.close()

        trip_metrics = _parse_sumo_outputs(
            REPORTS_DIR / f"tripinfo_{scenario_name}.xml",
            REPORTS_DIR / f"summary_{scenario_name}.xml",
        )

        eps_before_decay = agent.epsilon
        agent.on_episode_end(ep_idx, {"episode_reward": ep_reward})
        avg_q = sum(queues) / len(queues) if queues else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        # Select best checkpoint strictly from training-time performance
        if (
            best_ckpt_path is not None
            and stage_name == final_stage
            and ep_reward > best_train_score
        ):
            best_train_score = ep_reward
            agent.save(best_ckpt_path)

        history.append(
            {
                "experiment": experiment_name,
                "episode": ep_idx,
                "stage": stage_name,
                "scenario": scenario_name,
                "seed": ep_seed,
                "epsilon": round(eps_before_decay, 4),
                "episode_reward": round(ep_reward, 4),
                "avg_waiting_time_s": trip_metrics["avg_waiting_time_s"],
                "avg_queue_length": round(avg_q, 2),
                "avg_loss": round(avg_loss, 6),
                "learn_steps": agent.learn_steps,
            }
        )

    if best_ckpt_path is not None and not best_ckpt_path.is_file():
        agent.save(best_ckpt_path)

    return history


def run_hyperparameter_experiments(
    *,
    max_steps: int = 1800,
    base_seed: int = 42,
) -> tuple[list[dict[str, Any]], str, Path]:
    """Run controlled one-factor-at-a-time DQN curriculum experiments."""
    schedule = build_curriculum_schedule(
        low_episodes=4, medium_episodes=4, high_episodes=4, mixed_episodes=3
    )
    experiments: list[tuple[str, dict[str, Any]]] = [
        ("exp0_baseline", {}),
        ("exp1_lr_5e4", {"learning_rate": 5e-4}),
        ("exp2_eps_decay_090", {"epsilon_decay": 0.90}),
        ("exp3_batch_128", {"batch_size": 128}),
        ("exp4_target_freq_50", {"target_update_freq": 50}),
        ("exp5_replay_50k", {"replay_capacity": 50_000}),
    ]

    all_history: list[dict[str, Any]] = []
    best_exp_name = "exp0_baseline"
    best_exp_score = -float("inf")
    best_ckpt = MODELS_DIR / "dqn_curriculum_best.pt"

    for exp_name, overrides in experiments:
        agent = DQNAgent(seed=base_seed, **overrides)
        exp_ckpt = MODELS_DIR / f"dqn_{exp_name}.pt"
        hist = train_dqn_curriculum(
            agent,
            schedule,
            experiment_name=exp_name,
            max_steps=max_steps,
            base_seed=base_seed,
            best_ckpt_path=exp_ckpt,
        )
        all_history.extend(hist)

        # Score each experiment strictly by mean training reward in stage4_mixed
        mixed_rewards = [
            r["episode_reward"] for r in hist if r["stage"] == "stage4_mixed"
        ]
        mean_mixed = sum(mixed_rewards) / len(mixed_rewards)
        if mean_mixed > best_exp_score:
            best_exp_score = mean_mixed
            best_exp_name = exp_name
            # Copy winning checkpoint to canonical best path
            winner = DQNAgent(seed=base_seed, **overrides)
            winner.load(exp_ckpt)
            winner.save(best_ckpt)

    return all_history, best_exp_name, best_ckpt


def evaluate_multiseed(
    best_curriculum_ckpt: Path,
    *,
    eval_seeds: tuple[int, ...] = EVAL_SEEDS,
    max_steps: int = 1800,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate all 5 controllers greedily across 3 seeds per scenario."""
    q_agent = QLearningAgent(seed=42)
    q_path = MODELS_DIR / "q_learning_qtable.json"
    if q_path.is_file():
        q_agent.load(q_path)
    q_agent.epsilon = 0.0

    sarsa_agent = SARSAAgent(seed=42)
    s_path = MODELS_DIR / "sarsa_qtable.json"
    if s_path.is_file():
        sarsa_agent.load(s_path)
    sarsa_agent.epsilon = 0.0

    dqn_orig = DQNAgent(seed=42)
    orig_path = MODELS_DIR / "dqn_checkpoint.pt"
    if orig_path.is_file():
        dqn_orig.load(orig_path)
    dqn_orig.epsilon = 0.0

    dqn_curr = DQNAgent(seed=42)
    dqn_curr.load(best_curriculum_ckpt)
    dqn_curr.epsilon = 0.0

    controllers: list[tuple[str, Any]] = [
        ("fixed_time", FixedTimeController(ns_green_steps=5, ew_green_steps=4)),
        ("q_learning", q_agent),
        ("sarsa", sarsa_agent),
        ("dqn_medium_only", dqn_orig),
        ("dqn_curriculum", dqn_curr),
    ]

    raw_rows: list[dict[str, Any]] = []
    for scenario_name, cfg_path in SCENARIO_CONFIGS.items():
        for ctrl_name, ctrl in controllers:
            for seed in eval_seeds:
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

    metric_keys = (
        "avg_waiting_time_s",
        "avg_queue_length",
        "avg_travel_time_s",
        "throughput_veh_per_hr",
        "max_queue_length",
        "vehicles_arrived",
        "vehicles_remaining",
        "episode_reward",
    )
    summary_rows: list[dict[str, Any]] = []
    for scenario_name in SCENARIO_CONFIGS:
        for ctrl_name, _ in controllers:
            sub = [
                r
                for r in raw_rows
                if r["scenario"] == scenario_name and r["controller"] == ctrl_name
            ]
            row: dict[str, Any] = {
                "scenario": scenario_name,
                "controller": ctrl_name,
                "num_seeds": len(sub),
            }
            for k in metric_keys:
                vals = np.array([float(r[k]) for r in sub], dtype=np.float64)
                row[f"{k}_mean"] = round(float(np.mean(vals)), 2)
                row[f"{k}_std"] = round(float(np.std(vals, ddof=0)), 2)
            summary_rows.append(row)

    return raw_rows, summary_rows


def _save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_curriculum_learning_plot(
    history: list[dict[str, Any]], plot_path: Path
) -> None:
    """Plot learning curves across all hyperparameter curriculum experiments."""
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    exp_names = list(dict.fromkeys(r["experiment"] for r in history))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for exp in exp_names:
        sub = [r for r in history if r["experiment"] == exp]
        eps = [r["episode"] for r in sub]
        ax1.plot(eps, [r["episode_reward"] for r in sub], marker="o", label=exp)
        ax2.plot(eps, [r["avg_waiting_time_s"] for r in sub], marker="s", label=exp)

    ax1.set_title("Curriculum Training Episode Reward")
    ax1.set_xlabel("Episode (1-4 Low, 5-8 Med, 9-12 High, 13-15 Mixed)")
    ax1.set_ylabel("Episode Reward")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_title("Curriculum Training Avg Waiting Time")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Seconds")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def generate_multiseed_comparison_plot(
    summary_rows: list[dict[str, Any]], plot_path: Path
) -> None:
    """Plot 5-controller mean ± std comparison across Low, Medium, High."""
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = ["low", "medium", "high"]
    controllers = [
        ("fixed_time", "Fixed-Time", "#6c757d"),
        ("q_learning", "Q-Learning", "#1f77b4"),
        ("sarsa", "SARSA", "#2a9d8f"),
        ("dqn_medium_only", "DQN (Med)", "#e76f51"),
        ("dqn_curriculum", "DQN (Curr)", "#9b2226"),
    ]
    lookup = {(r["scenario"], r["controller"]): r for r in summary_rows}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Phase 7 Multi-Seed Comparison (Mean ± Std across 3 Seeds)",
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
        for idx, (ctrl, lbl, col) in enumerate(controllers):
            means = [float(lookup[(sc, ctrl)][f"{key}_mean"]) for sc in scenarios]
            stds = [float(lookup[(sc, ctrl)][f"{key}_std"]) for sc in scenarios]
            offsets = x + (idx - 2.0) * w
            ax.bar(offsets, means, yerr=stds, width=w, capsize=2, label=lbl, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in scenarios])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def run_phase7_pipeline() -> dict[str, Any]:
    """Execute full Phase 7 curriculum experiments and 3-seed evaluation."""
    history, best_exp, best_ckpt = run_hyperparameter_experiments()
    _save_csv(history, REPORTS_DIR / "dqn_curriculum_hparam_experiments.csv")
    generate_curriculum_learning_plot(
        history, REPORTS_DIR / "dqn_curriculum_learning_curves.png"
    )

    raw_rows, summary_rows = evaluate_multiseed(best_ckpt)
    _save_csv(raw_rows, REPORTS_DIR / "phase7_multiseed_raw.csv")
    _save_csv(summary_rows, REPORTS_DIR / "phase7_multiseed_summary.csv")
    generate_multiseed_comparison_plot(
        summary_rows, REPORTS_DIR / "phase7_multiseed_comparison.png"
    )
    return {
        "best_experiment": best_exp,
        "best_checkpoint": str(best_ckpt),
        "summary_rows": summary_rows,
    }


if __name__ == "__main__":
    out = run_phase7_pipeline()
    print("Best curriculum experiment:", out["best_experiment"])
    print("Best checkpoint:", out["best_checkpoint"])
    for r in out["summary_rows"]:
        print(r)
