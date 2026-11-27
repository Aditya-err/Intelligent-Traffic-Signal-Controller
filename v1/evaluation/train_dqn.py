"""Phase 6 DQN training and 4-controller evaluation pipeline (CPU-only)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithms.dqn import DQNAgent
from algorithms.fixed_time import FixedTimeController
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from environment.rl_wrapper import TrafficRLWrapper
from environment.traffic_env import SumoTrafficEnv
from evaluation.baseline_eval import PROJECT_ROOT, REPORTS_DIR, SCENARIO_CONFIGS
from evaluation.train_tabular_rl import evaluate_controller_on_scenario


MODELS_DIR = PROJECT_ROOT / "models"


def train_dqn_agent(
    agent: DQNAgent,
    *,
    episodes: int = 15,
    max_steps: int = 1800,
    base_seed: int = 42,
    sumocfg_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Train DQNAgent via live SUMO episodes on E: drive."""
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

    try:
        for ep in range(1, episodes + 1):
            ep_seed = base_seed + ep
            state, _ = env.reset(seed=ep_seed)
            agent.on_episode_start(ep)
            ep_reward = 0.0
            losses: list[float] = []
            queues: list[float] = []
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

            eps_before_decay = agent.epsilon
            agent.on_episode_end(ep, {"episode_reward": ep_reward})
            avg_q = sum(queues) / len(queues) if queues else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            history.append(
                {
                    "episode": ep,
                    "seed": ep_seed,
                    "epsilon": round(eps_before_decay, 4),
                    "episode_reward": round(ep_reward, 4),
                    "avg_loss": round(avg_loss, 6),
                    "avg_queue_length": round(avg_q, 2),
                    "replay_size": len(agent.replay_buffer),
                    "learn_steps": agent.learn_steps,
                }
            )
    finally:
        env.close()

    return history


def _save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_dqn_training_plot(
    history: list[dict[str, Any]], plot_path: Path
) -> None:
    """Plot DQN episode reward and training loss curves."""
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = [r["episode"] for r in history]
    rewards = [r["episode_reward"] for r in history]
    losses = [r["avg_loss"] for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(episodes, rewards, marker="o", color="#9b2226")
    ax1.set_title("DQN Training Episode Rewards (Medium Demand)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Shaped Reward")
    ax1.grid(alpha=0.3)

    ax2.plot(episodes, losses, marker="s", color="#005f73")
    ax2.set_title("DQN Mean Smooth L1 Loss per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def generate_four_controller_plot(
    rows: list[dict[str, Any]], plot_path: Path
) -> None:
    """Compare Fixed-Time, Q-Learning, SARSA, and DQN across all scenarios."""
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = ["low", "medium", "high"]
    controllers = ["fixed_time", "q_learning", "sarsa", "dqn"]
    labels = ["Fixed-Time", "Q-Learning", "SARSA", "DQN"]
    colors = ["#6c757d", "#1f77b4", "#2a9d8f", "#9b2226"]

    lookup = {(r["scenario"], r["controller"]): r for r in rows}

    def _vals(ctrl: str, key: str) -> list[float]:
        return [float(lookup[(sc, ctrl)][key]) for sc in scenarios]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(
        "Phase 6 Controller Comparison: Fixed-Time vs Q-Learning vs SARSA vs DQN",
        fontsize=13,
        fontweight="bold",
    )
    x = list(range(len(scenarios)))
    w = 0.19
    metrics = [
        (axes[0, 0], "avg_waiting_time_s", "Seconds", "Average Waiting Time"),
        (axes[0, 1], "avg_queue_length", "Vehicles", "Average Queue Length"),
        (axes[1, 0], "throughput_veh_per_hr", "Vehicles / Hour", "Throughput"),
        (axes[1, 1], "episode_reward", "Total Reward", "Episode Reward"),
    ]
    for ax, key, ylabel, title in metrics:
        for idx, (ctrl, lbl, col) in enumerate(zip(controllers, labels, colors)):
            offsets = [xi + (idx - 1.5) * w for xi in x]
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


def run_phase6_pipeline(
    episodes: int = 15, max_steps: int = 1800, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Train DQN, save checkpoint, and evaluate all 4 controllers fairly."""
    dqn_agent = DQNAgent(seed=seed)
    history = train_dqn_agent(
        dqn_agent, episodes=episodes, max_steps=max_steps, base_seed=seed
    )
    dqn_agent.save(MODELS_DIR / "dqn_checkpoint.pt")
    _save_csv(history, REPORTS_DIR / "dqn_training_history.csv")
    generate_dqn_training_plot(history, REPORTS_DIR / "dqn_training_rewards.png")

    # Load Phase 5 trained tabular agents if available
    q_agent = QLearningAgent(seed=seed)
    q_path = MODELS_DIR / "q_learning_qtable.json"
    if q_path.is_file():
        q_agent.load(q_path)

    sarsa_agent = SARSAAgent(seed=seed)
    s_path = MODELS_DIR / "sarsa_qtable.json"
    if s_path.is_file():
        sarsa_agent.load(s_path)

    # Ensure greedy evaluation (epsilon=0, no updates)
    dqn_agent.epsilon = 0.0
    q_agent.epsilon = 0.0
    sarsa_agent.epsilon = 0.0

    controllers: list[tuple[str, Any]] = [
        ("fixed_time", FixedTimeController(ns_green_steps=5, ew_green_steps=4)),
        ("q_learning", q_agent),
        ("sarsa", sarsa_agent),
        ("dqn", dqn_agent),
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

    _save_csv(comparison_rows, REPORTS_DIR / "phase6_comparison.csv")
    generate_four_controller_plot(
        comparison_rows, REPORTS_DIR / "phase6_controller_comparison.png"
    )
    return history, comparison_rows


if __name__ == "__main__":
    hist, comp = run_phase6_pipeline()
    print("=== DQN TRAINING SUMMARY ===")
    print("first:", hist[0])
    print("last:", hist[-1])
    print("=== 4-CONTROLLER COMPARISON ===")
    for row in comp:
        print(row)
