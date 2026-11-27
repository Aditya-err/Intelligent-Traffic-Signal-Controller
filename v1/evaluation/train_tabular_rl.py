"""Phase 5 training and evaluation for Tabular Q-Learning and SARSA agents."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


MODELS_DIR = PROJECT_ROOT / "models"


def train_tabular_agent(
    agent: QLearningAgent | SARSAAgent,
    agent_name: str,
    *,
    episodes: int = 15,
    max_steps: int = 1800,
    base_seed: int = 42,
    sumocfg_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Train a tabular agent via live SUMO episodes on E: drive."""
    cfg = sumocfg_path or SCENARIO_CONFIGS["medium"]
    env = TrafficRLWrapper(
        SumoTrafficEnv(
            sumocfg_path=cfg,
            use_gui=False,
            max_steps=max_steps,
            delta_time=5,
            min_green=10,
            yellow_time=3,
            all_red_time=2,
            right_turn_green=5,
            sumo_seed=base_seed,
        )
    )
    history: list[dict[str, Any]] = []
    is_sarsa = isinstance(agent, SARSAAgent)

    try:
        for ep in range(1, episodes + 1):
            ep_seed = base_seed + ep
            state, _ = env.reset(seed=ep_seed)
            agent.on_episode_start(ep)
            ep_reward = 0.0
            queues: list[float] = []
            terminated = False
            truncated = False

            action = agent.select_action(state, explore=True) if is_sarsa else 0
            while not (terminated or truncated):
                if not is_sarsa:
                    action = agent.select_action(state, explore=True)

                next_state, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                queues.append(float(info["total_queue"]))

                if is_sarsa:
                    next_action = (
                        agent.select_action(next_state, explore=True)
                        if not (terminated or truncated)
                        else 0
                    )
                    trans_info = dict(info)
                    trans_info["next_action"] = next_action
                    agent.observe_transition(
                        state,
                        action,
                        reward,
                        next_state,
                        terminated,
                        truncated,
                        trans_info,
                    )
                    action = next_action
                else:
                    agent.observe_transition(
                        state, action, reward, next_state, terminated, truncated, info
                    )

                state = next_state

            eps_before_decay = agent.epsilon
            agent.on_episode_end(ep, {"episode_reward": ep_reward})
            avg_q = sum(queues) / len(queues) if queues else 0.0
            history.append(
                {
                    "agent": agent_name,
                    "episode": ep,
                    "seed": ep_seed,
                    "epsilon": round(eps_before_decay, 4),
                    "episode_reward": round(ep_reward, 4),
                    "avg_queue_length": round(avg_q, 2),
                    "states_visited": len(agent.q_table),
                }
            )
    finally:
        env.close()

    return history


def evaluate_controller_on_scenario(
    controller: FixedTimeController | QLearningAgent | SARSAAgent,
    controller_name: str,
    scenario_name: str,
    sumocfg_path: Path,
    *,
    max_steps: int = 1800,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate any controller under identical TrafficRLWrapper & SUMO settings."""
    env = TrafficRLWrapper(
        SumoTrafficEnv(
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
    )
    queue_samples: list[float] = []
    episode_reward = 0.0
    sim_time = 0

    try:
        state, _ = env.reset(seed=seed)
        if isinstance(controller, FixedTimeController):
            controller.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            if isinstance(controller, FixedTimeController):
                action = controller.select_action(state)
            else:
                action = controller.select_action(state, explore=False)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
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
        "controller": controller_name,
        "sim_duration_s": sim_time,
        "seed": seed,
        "avg_waiting_time_s": sumo_metrics["avg_waiting_time_s"],
        "avg_queue_length": round(avg_queue, 2),
        "avg_travel_time_s": sumo_metrics["avg_travel_time_s"],
        "throughput_veh_per_hr": round(throughput, 2),
        "max_queue_length": round(max_queue, 2),
        "vehicles_arrived": sumo_metrics["vehicles_arrived"],
        "vehicles_remaining": sumo_metrics["vehicles_remaining"],
        "episode_reward": round(episode_reward, 4),
    }


def _save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_training_plot(history: list[dict[str, Any]], plot_path: Path) -> None:
    """Plot episode reward curves for Q-Learning and SARSA."""
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, color in (("q_learning", "#1f77b4"), ("sarsa", "#d62728")):
        sub = [r for r in history if r["agent"] == name]
        ax.plot(
            [r["episode"] for r in sub],
            [r["episode_reward"] for r in sub],
            marker="o",
            label=name.replace("_", "-").upper(),
            color=color,
        )
    ax.set_title("Phase 5 Tabular RL Training Episode Rewards (Medium Demand)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Shaped Episode Reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def generate_controller_comparison_plot(
    rows: list[dict[str, Any]], plot_path: Path
) -> None:
    """Plot Fixed-Time vs Q-Learning vs SARSA across Low/Medium/High."""
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = ["low", "medium", "high"]
    controllers = ["fixed_time", "q_learning", "sarsa"]
    labels = ["Fixed-Time", "Q-Learning", "SARSA"]
    colors = ["#6c757d", "#1f77b4", "#2a9d8f"]

    def _vals(ctrl: str, key: str) -> list[float]:
        lookup = {(r["scenario"], r["controller"]): r[key] for r in rows}
        return [float(lookup[(sc, ctrl)]) for sc in scenarios]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Phase 5 Controller Comparison: Fixed-Time vs Tabular Q-Learning vs SARSA",
        fontsize=13,
        fontweight="bold",
    )
    x = list(range(len(scenarios)))
    w = 0.25
    metrics = [
        (axes[0, 0], "avg_waiting_time_s", "Seconds", "Average Waiting Time"),
        (axes[0, 1], "avg_queue_length", "Vehicles", "Average Queue Length"),
        (axes[1, 0], "throughput_veh_per_hr", "Vehicles / Hour", "Throughput"),
        (axes[1, 1], "episode_reward", "Total Reward", "Episode Reward"),
    ]
    for ax, key, ylabel, title in metrics:
        for idx, (ctrl, lbl, col) in enumerate(zip(controllers, labels, colors)):
            offsets = [xi + (idx - 1) * w for xi in x]
            ax.bar(offsets, _vals(ctrl, key), width=w, label=lbl, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in scenarios])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def run_phase5_pipeline(
    episodes: int = 15, max_steps: int = 1800, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Train Q-Learning & SARSA, save models, and evaluate against baseline."""
    q_agent = QLearningAgent(seed=seed)
    sarsa_agent = SARSAAgent(seed=seed)

    q_hist = train_tabular_agent(
        q_agent, "q_learning", episodes=episodes, max_steps=max_steps, base_seed=seed
    )
    sarsa_hist = train_tabular_agent(
        sarsa_agent, "sarsa", episodes=episodes, max_steps=max_steps, base_seed=seed
    )
    full_history = q_hist + sarsa_hist

    q_agent.save(MODELS_DIR / "q_learning_qtable.json")
    sarsa_agent.save(MODELS_DIR / "sarsa_qtable.json")
    _save_csv(full_history, REPORTS_DIR / "tabular_training_history.csv")
    generate_training_plot(full_history, REPORTS_DIR / "tabular_training_rewards.png")

    controllers: list[tuple[str, FixedTimeController | QLearningAgent | SARSAAgent]] = [
        ("fixed_time", FixedTimeController(ns_green_steps=5, ew_green_steps=4)),
        ("q_learning", q_agent),
        ("sarsa", sarsa_agent),
    ]
    comparison_rows: list[dict[str, Any]] = []
    for scenario_name, cfg_path in SCENARIO_CONFIGS.items():
        for ctrl_name, ctrl in controllers:
            comparison_rows.append(
                evaluate_controller_on_scenario(
                    ctrl,
                    ctrl_name,
                    scenario_name,
                    cfg_path,
                    max_steps=max_steps,
                    seed=seed,
                )
            )

    _save_csv(comparison_rows, REPORTS_DIR / "phase5_comparison.csv")
    generate_controller_comparison_plot(
        comparison_rows, REPORTS_DIR / "phase5_controller_comparison.png"
    )
    return full_history, comparison_rows


if __name__ == "__main__":
    history, comp = run_phase5_pipeline()
    print("=== TRAINING SUMMARY ===")
    for agent_name in ("q_learning", "sarsa"):
        sub = [r for r in history if r["agent"] == agent_name]
        print(agent_name, "first:", sub[0], "last:", sub[-1])
    print("=== COMPARISON RESULTS ===")
    for row in comp:
        print(row)
