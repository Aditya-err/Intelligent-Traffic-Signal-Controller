# Intelligent Traffic Signal Controller Using Reinforcement Learning

An India-specific adaptive traffic signal control system built on **Eclipse SUMO**, **TraCI**, and **CPU-based PyTorch** (with Gymnasium and Streamlit planned in subsequent phases).

> **Current Status:** **Phase 1 (Baseline SUMO Simulation & Repository Foundation)**
> All Phase 1 SUMO network definitions, Indian mixed-traffic routes, signal timing plans, and structural validation tests are ready. **SUMO simulation execution is pending SUMO installation on the `E:` drive.**

---

## Project Objectives

1. Model a realistic **Indian four-way urban intersection** with **left-hand traffic (`--lefthand true`)** and mixed vehicle dynamics.
2. Establish a safe, non-conflicting **fixed-time baseline** signal plan (North–South and East–West groups with protected right turns, yellow, and all-red clearance phases).
3. Build a clean, modular codebase ready to integrate a **TraCI + Gymnasium RL environment** (Phase 2), **tabular / deep RL agents** (Phase 3), and **evaluation & Streamlit visualization** (Phase 4).

---

## Indian Traffic & Road Geometry Assumptions

| Aspect | Configuration | Rationale |
| :--- | :--- | :--- |
| **Driving Side** | Left-hand traffic (`<lefthand value="true"/>`) | Standard Indian road rule (keep left, right-hand-drive vehicles) |
| **Intersection** | 4-way signalized junction (`N`, `S`, `E`, `W`), 250 m arms | Representative urban arterial intersection |
| **Lane Geometry** | 2 lanes/direction, 3.5 m lane width, 50 km/h (`13.89 m/s`) | Aligned with Indian Roads Congress (IRC:86) urban arterial standards |
| **Lane Assignment (LHT)** | Lane `0` (curb/left): Left + Straight; Lane `1` (median/right): Protected Right | Separates opposing right-turning traffic from straight/left movements |
| **Vehicle Mix** | Motorcycles (`38%`), Cars (`30%`), Auto-rickshaws (`20%`), Buses (`6%`), Trucks (`6%`) | Reflects heterogeneous Indian urban traffic |
| **Signal Plan (94 s cycle)** | 4 protected green stages (`25s` NS S+L, `12s` NS R, `25s` EW S+L, `12s` EW R) | Every green stage is followed by `3s` Yellow + `2s` All-Red clearance |

---

## Repository Structure

```text
Intelligent-Traffic-RL/
├── data/                   # Traffic demand profiles & processed datasets (Phase 2+)
├── environment/            # Gymnasium + TraCI environment wrapper (Phase 2)
├── algorithms/             # RL agents: Q-Learning, SARSA, DQN (Phase 3)
├── simulation/
│   ├── network/            # SUMO nodes, edges, connections, tlLogic, netconvert config
│   ├── routes/             # Indian vehicle types & traffic demand flows
│   └── config/             # Lightweight .sumocfg simulation entry point
├── evaluation/             # Baseline vs. RL metrics & comparison scripts (Phase 4)
├── dashboard/              # Streamlit monitoring & analytics UI (Phase 4)
├── models/                 # Saved CPU PyTorch checkpoints (gitignored)
├── reports/                # SUMO tripinfo/summary outputs & plots (gitignored)
├── tests/                  # Configuration & unit tests
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Hardware & Storage Policy

- **Target Hardware:** Intel Core i7-10610U (4C/8T), 16 GB RAM, Intel UHD Graphics (CPU-only PyTorch).
- **Storage Policy:** Keep all project files, SUMO installation, dedicated Conda environment, datasets, and caches on **`E:`** to avoid filling `C:` and `D:`.

---

## Phase 1 Quickstart (Once SUMO Is Installed on `E:`)

1. **Install Eclipse SUMO on `E:`** (e.g., `E:\Sumo`) and set `SUMO_HOME=E:\Sumo` and add `E:\Sumo\bin` to `PATH`.
2. **Validate Phase 1 XML & configuration files** (runs with standard Python, no extra packages required):
   ```powershell
   python -m unittest discover -s tests -v
   ```
3. **Build the compiled SUMO network (`intersection.net.xml`):**
   ```powershell
   cd "E:\MCA projects\Traffic Signal Controller\Intelligent-Traffic-RL\simulation\network"
   netconvert -c intersection.netccfg
   ```
4. **Run the lightweight baseline simulation (1800 s):**
   ```powershell
   cd "E:\MCA projects\Traffic Signal Controller\Intelligent-Traffic-RL\simulation\config"
   sumo -c intersection.sumocfg
   ```
   *(Or use `sumo-gui -c intersection.sumocfg` to inspect left-hand driving visually.)*

---

## Roadmap

- **Phase 1 (Current):** Repository structure, Indian LHT 4-way SUMO network, mixed-traffic routes, safe signal timing, XML validation tests, GitHub readiness.
- **Phase 2:** Dedicated `E:` Conda environment (`traci`, `sumolib`, `gymnasium`, `torch` CPU), TraCI bridge, and Gymnasium MDP environment.
- **Phase 3:** Fixed-time baseline benchmarking + RL controllers (Tabular Q-Learning / SARSA / CPU-based DQN).
- **Phase 4:** Multi-metric evaluation (average waiting time, queue length, throughput, travel time) and Streamlit dashboard.
