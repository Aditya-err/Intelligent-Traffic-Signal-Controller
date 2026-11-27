# Intelligent Traffic Signal Controller Using Reinforcement Learning

An academic reinforcement learning traffic signal control system built with **Eclipse SUMO 1.28.0**, **TraCI**, **CPU PyTorch**, and a **6-page Streamlit + HTML5 Canvas dashboard** for **Indian Left-Hand Traffic (LHT)** mixed-vehicle corridors.

---

## Project Versions

| Version | Scope | Status | Documentation |
| :--- | :--- | :---: | :--- |
| **[v2](v2/README.md)** (Latest) | **3-junction curved corridor (`J1–J2–J3`)**, Shared DQN (parameter sharing), pedestrians, 60 FPS HTML5 Canvas replay, Holm-corrected statistics | Complete (`19/19` tests) | [v2/README.md](v2/README.md) · [v2/CHANGELOG.md](v2/CHANGELOG.md) |
| **[v1](v1/README.md)** (Archived, tag `v1.0`) | Single 4-way Indian LHT intersection, Fixed-Time, Q-Learning, SARSA, DQN Curriculum | Preserved & runnable | [v1/README.md](v1/README.md) |

---

## v2 Key Features

- **Realistic 3-Junction Curved Corridor (`v2/network/`):** `J1–J2–J3` Indian LHT (`--lefthand true`) arterial with cubic Bezier curved segments (`215.9 m – 298.1 m`), 2 motorized lanes + `2.0 m` sidewalks per direction, and zebra pedestrian crossings.
- **Mixed Indian Traffic & Pedestrians (`v2/traffic/`):** Poisson arrivals across `Low (~720)`, `Medium (~1330)`, and `High (~1660)` veh/hr (`45%` motorcycle, `25%` car, `10%` auto, `10%` bus, `10%` truck) plus zebra-crossing pedestrians (`120` ped/hr).
- **Shared DQN with Hard Safety Layer (`v2/controllers/dqn.py`):** One shared MLP (`10 → 128 → 64 → 2`) across all 3 junctions with an external hard safety state machine enforcing `10 s` min green, `3 s` yellow, and `2 s` all-red.
- **60 FPS HTML5 Canvas Replay (`v2/dashboard/app.py`):** Pre-recorded `0.5 s` TraCI clips replayed smoothly in-browser with zero Streamlit flicker, rotated vehicle sprites, live 3-lamp traffic signals with countdowns, and an **RL Decision Panel**.
- **Rigorous Evaluation (`v2/evaluation/`):** `3` controllers × `3` demand levels × `10` held-out seeds (`1001–1010`, full 1-hour episodes) with 95% CIs, paired t-tests, exact `2^10 = 1024` sign-flip permutation tests, Holm-Bonferroni correction, and `Direction` labels.

---

## v2 Results Summary (`seeds 1001–1010`, 1-hour episodes)

| Controller | Low Wait (s) | Medium Wait (s) | High Wait (s) | Improvement vs Fixed-Time |
| :--- | :---: | :---: | :---: | :---: |
| **Fixed-Time** (`25s NS / 20s EW`) | `15.48 ± 0.44` | `16.39 ± 0.46` | `16.84 ± 0.48` | — |
| **SUMO Actuated** (`type="actuated"`) | `9.94 ± 0.27` | `10.61 ± 0.23` | `11.62 ± 0.54` | `+31.0% to +35.8%` |
| **Shared DQN** (`10→128→64→2`) | `11.60 ± 0.26` | `11.10 ± 0.20` | **`11.43 ± 0.17`** | **`+25.1% to +32.3%`** |

- Shared DQN significantly outperforms Fixed-Time across all 3 demand levels (`p_perm_holm = 0.0176`) with **0 teleports and 0 collisions** across all 30 evaluation episodes, and edges out SUMO Actuated in High demand (`11.43 s` vs `11.62 s`). In Low and Medium demand, DQN trails Actuated because it decides every `5 s` rather than `1 s` induction-loop gap-out.

---

## Quick Start (v2)

```powershell
# 1. Install dependencies (requires Eclipse SUMO 1.28.0 with SUMO_HOME set)
pip install -r requirements.txt

# 2. Launch the 6-page Streamlit + HTML5 Canvas dashboard
streamlit run v2/dashboard/app.py

# 3. Run the full v2 test suite (19 tests)
python -m unittest discover -s v2\tests -t . -v
```

See [v2/README.md](v2/README.md) for full network-building, traffic-generation, training, and evaluation commands.
