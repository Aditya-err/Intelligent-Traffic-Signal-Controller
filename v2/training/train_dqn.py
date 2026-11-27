"""
Train ONE shared DQN controller across J1, J2, and J3 (parameter sharing).

Features:
- Curriculum over Low -> Medium -> High -> Random Mix using TRAIN_SEEDS only
- Exact linear epsilon decay from 1.0 down to 0.05
- Hard safety layer (10 s min green, 3 s yellow, 2 s all-red) outside the network
- Logs CSV (`dqn_training_log.csv`), saves `dqn_best.pt` & `dqn_last.pt`,
  and plots reward, loss, and epsilon curves (`dqn_training_curves.png`)

Usage:
    python -m v2.training.train_dqn --episodes 100
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Deque, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_THIS_DIR = Path(__file__).resolve().parent
_V2_ROOT = _THIS_DIR.parent
if str(_V2_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT.parent))

from v2 import config  # noqa: E402
from v2.controllers.dqn import (  # noqa: E402
    QNetwork,
    SafetySignalManager,
    compute_junction_reward,
    get_junction_state,
)
from v2.traffic.generate_traffic import generate_route_file  # noqa: E402

_SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")
_SUMO_TOOLS = Path(_SUMO_HOME) / "tools"
if str(_SUMO_TOOLS) not in sys.path:
    sys.path.insert(0, str(_SUMO_TOOLS))

import traci  # noqa: E402


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


def compute_epsilon(ep_idx: int, total_episodes: int) -> float:
    """
    Linearly decay epsilon from `DQN_EPSILON_START` (1.0) to `DQN_EPSILON_END` (0.05).

    At `ep_idx == total_episodes - 1`, this returns `DQN_EPSILON_END` exactly.
    """
    if total_episodes <= 1:
        return float(config.DQN_EPSILON_END)
    frac = ep_idx / float(total_episodes - 1)
    eps = config.DQN_EPSILON_START - (
        config.DQN_EPSILON_START - config.DQN_EPSILON_END
    ) * frac
    return round(float(eps), 6)


def curriculum_demand(ep_idx: int, total_episodes: int, rng: random.Random) -> str:
    """Simple 4-stage curriculum: Low -> Medium -> High -> Random Mix."""
    frac = ep_idx / float(max(1, total_episodes))
    if frac < 0.25:
        return "low"
    if frac < 0.55:
        return "medium"
    if frac < 0.80:
        return "high"
    return rng.choice(["low", "medium", "high"])


def _train_step(
    policy_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    replay: Deque[Transition],
    rng: random.Random,
) -> float:
    """Sample one mini-batch from replay and perform one gradient step."""
    batch = rng.sample(replay, config.DQN_BATCH_SIZE)
    states = torch.from_numpy(np.stack([b[0] for b in batch]))
    actions = torch.tensor([b[1] for b in batch], dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    next_states = torch.from_numpy(np.stack([b[3] for b in batch]))
    dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

    q_values = policy_net(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1).values
        targets = rewards + config.DQN_GAMMA * next_q * (1.0 - dones)

    loss = loss_fn(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def plot_training_curves(rows: List[Dict[str, object]], out_png: Path) -> None:
    """Plot Reward, Mean Loss, and Epsilon curves into a 3-panel PNG."""
    episodes = [int(r["episode"]) for r in rows]
    rewards = [float(r["total_reward"]) for r in rows]
    losses = [float(r["mean_loss"]) for r in rows]
    epsilons = [float(r["epsilon"]) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(episodes, rewards, color="#2563eb", linewidth=1.6)
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("v2 Shared DQN Training Curves (J1-J2-J3 Corridor)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(episodes, losses, color="#dc2626", linewidth=1.6)
    axes[1].set_ylabel("Mean Huber Loss")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(episodes, epsilons, color="#059669", linewidth=1.8)
    axes[2].set_ylabel("Epsilon")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(True, alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def train_dqn(
    episodes: int = config.DQN_TOTAL_EPISODES,
    duration_s: int = config.DQN_TRAIN_DURATION_S,
    models_dir: Path = config.MODELS_DIR,
    results_dir: Path = config.TRAINING_RESULTS_DIR,
    seed: int = 42,
) -> Dict[str, object]:
    """Train the shared DQN agent and save checkpoints, CSV log, and PNG curves."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    sumo_exe = Path(_SUMO_HOME) / "bin" / "sumo.exe"

    device = torch.device("cpu")
    policy_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config.DQN_LR)
    loss_fn = nn.SmoothL1Loss()
    replay: Deque[Transition] = deque(maxlen=config.DQN_REPLAY_CAPACITY)

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / "dqn_best.pt"
    last_path = models_dir / "dqn_last.pt"
    csv_path = results_dir / "dqn_training_log.csv"
    png_path = results_dir / "dqn_training_curves.png"

    best_reward = -float("inf")
    log_rows: List[Dict[str, object]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for ep_idx in range(episodes):
            ep_t0 = time.perf_counter()
            ep_num = ep_idx + 1
            demand = curriculum_demand(ep_idx, episodes, rng)
            train_seed = config.TRAIN_SEEDS[ep_idx % len(config.TRAIN_SEEDS)]
            epsilon = compute_epsilon(ep_idx, episodes)

            rou_path = tmp_path / f"train_ep{ep_num}_{demand}_s{train_seed}.rou.xml"
            generate_route_file(
                demand_level=demand,
                seed=train_seed,
                duration_s=duration_s,
                output_path=rou_path,
            )

            traci.start(
                [
                    str(sumo_exe),
                    "-n",
                    str(config.NET_FILE),
                    "-r",
                    str(rou_path),
                    "--seed",
                    str(train_seed),
                    "--step-length",
                    str(config.STEP_LENGTH_S),
                    "--no-step-log",
                    "true",
                    "--duration-log.disable",
                    "true",
                ]
            )

            ep_reward = 0.0
            ep_losses: List[float] = []
            veh_wait: Dict[str, float] = {}

            try:
                managers = {
                    jid: SafetySignalManager(jid) for jid in config.JUNCTION_COORDS
                }
                last_obs: Dict[str, np.ndarray] = {}
                last_act: Dict[str, int] = {}
                switched_flag: Dict[str, bool] = {jid: False for jid in managers}

                for step_idx in range(duration_s):
                    make_decision = step_idx % config.DQN_DECISION_INTERVAL_S == 0
                    for jid, mgr in managers.items():
                        req_action = None
                        if make_decision and mgr.state == "GREEN":
                            obs = get_junction_state(
                                jid, mgr.current_action, mgr.time_in_green
                            )
                            rew = compute_junction_reward(jid, switched_flag[jid])
                            if jid in last_obs:
                                replay.append(
                                    (last_obs[jid], last_act[jid], rew, obs, False)
                                )
                                ep_reward += rew

                            if rng.random() < epsilon:
                                act = rng.randrange(config.DQN_ACTION_DIM)
                            else:
                                with torch.no_grad():
                                    q_vals = policy_net(torch.from_numpy(obs).unsqueeze(0))
                                    act = int(torch.argmax(q_vals, dim=1).item())

                            last_obs[jid] = obs
                            last_act[jid] = act
                            switched_flag[jid] = False
                            req_action = act

                        did_switch = mgr.apply_decision_and_step(req_action)
                        if did_switch:
                            switched_flag[jid] = True

                    if make_decision and len(replay) >= config.DQN_MIN_REPLAY_TO_TRAIN:
                        ep_losses.append(
                            _train_step(
                                policy_net, target_net, optimizer, loss_fn, replay, rng
                            )
                        )

                    traci.simulationStep()
                    for vid in traci.vehicle.getIDList():
                        veh_wait[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)

                total_overrides = sum(m.safety_overrides for m in managers.values())
            finally:
                traci.close()

            if ep_num % config.DQN_TARGET_SYNC_EPISODES == 0:
                target_net.load_state_dict(policy_net.state_dict())

            mean_loss = sum(ep_losses) / len(ep_losses) if ep_losses else 0.0
            avg_wait = sum(veh_wait.values()) / len(veh_wait) if veh_wait else 0.0
            wall_s = time.perf_counter() - ep_t0

            if ep_reward > best_reward:
                best_reward = ep_reward
                torch.save(policy_net.state_dict(), best_path)

            row = {
                "episode": ep_num,
                "demand": demand,
                "seed": train_seed,
                "total_reward": round(ep_reward, 2),
                "mean_loss": round(mean_loss, 4),
                "epsilon": epsilon,
                "avg_waiting_time_s": round(avg_wait, 2),
                "safety_overrides": total_overrides,
                "episode_wall_s": round(wall_s, 2),
            }
            log_rows.append(row)
            print(
                f"[Ep {ep_num:3d}/{episodes}] demand={demand:6s} seed={train_seed:2d} | "
                f"eps={epsilon:.3f} | reward={ep_reward:9.1f} | loss={mean_loss:6.3f} | "
                f"wait={avg_wait:5.1f}s | overrides={total_overrides:2d} | {wall_s:.2f}s"
            )

    torch.save(policy_net.state_dict(), last_path)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    plot_training_curves(log_rows, png_path)

    return {
        "episodes": episodes,
        "final_epsilon": log_rows[-1]["epsilon"],
        "best_reward": round(best_reward, 2),
        "mean_wall_s": round(
            sum(float(r["episode_wall_s"]) for r in log_rows) / len(log_rows), 2
        ),
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v2 shared DQN controller.")
    parser.add_argument("--episodes", type=int, default=config.DQN_TOTAL_EPISODES)
    parser.add_argument("--duration", type=int, default=config.DQN_TRAIN_DURATION_S)
    args = parser.parse_args()

    summary = train_dqn(episodes=args.episodes, duration_s=args.duration)
    print("\nTraining complete:", summary)


if __name__ == "__main__":
    main()
