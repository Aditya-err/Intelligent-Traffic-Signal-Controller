# Intelligent-Traffic-RL v2 — Multi-Junction Curved Corridor

**v2** upgrades the single 4-way intersection from [v1](../v1/README.md) into a realistic **3-junction (`J1`–`J2`–`J3`) Indian Left-Hand Traffic (LHT) arterial corridor** with curved road geometry, pedestrian sidewalks, zebra crossings, shared-parameter DQN control, 60 FPS HTML5 Canvas replay, and full statistical evaluation.

> **v1 Preservation:** All v1 code, trained models, and Phase 8 evaluation results remain intact and runnable inside `v1/` (tagged `v1.0`).

---

## Corridor Layout

```text
                N1 (-15, 230)               N2 (315, 265)                N3 (605, 230)
                    |                           |                            |
                 (~235m)                     (~235m)                      (~235m)
                    |                           |                            |
W1 (-260,-25) ~~~ [ J1 ] ~~~~~ (~315m) ~~~~~ [ J2 ] ~~~~~ (~335m) ~~~~~ [ J3 ] ~~~ E3 (880, 25)
                 (0, 0)         curved       (300,35)       curved       (620, 0)
                    |                           |                            |
                 (~235m)                     (~235m)                      (~235m)
                    |                           |                            |
                S1 (15, -230)               S2 (285, -195)               S3 (635, -230)
```

---

## Project Structure

```text
v2/
├── config.py                     # All geometry, traffic, safety, and DQN constants
├── network/
│   ├── build_network.py          # Generates .nod.xml, .edg.xml, .netccfg & compiles corridor.net.xml
│   └── corridor.net.xml          # Compiled SUMO 1.28.0 Indian LHT curved corridor
├── traffic/
│   └── generate_traffic.py       # Poisson mixed-vehicle & pedestrian .rou.xml generator
├── controllers/
│   ├── fixed_time.py             # Fixed-Time baseline (25s NS / 20s EW)
│   ├── actuated.py               # SUMO built-in type="actuated" baseline
│   └── dqn.py                    # Shared DQN (10->128->64->2) + external hard safety layer
├── training/
│   └── train_dqn.py              # 100-episode curriculum trainer, CSV logger, curve plotter
├── evaluation/
│   ├── run_episode.py            # Reusable 1-hour TraCI episode runner
│   ├── run_baselines.py          # Baseline runner (seeds 1001-1003)
│   ├── compare_dqn.py            # Quick 3-seed comparison
│   ├── stats.py                  # 95% CI, paired t-test, exact permutation, Holm, Direction
│   └── final_eval.py             # Full 3x10 held-out evaluation (seeds 1001-1010)
├── replays/
│   ├── network_geometry.py       # Extracts lanes, sidewalks, crossings from corridor.net.xml
│   └── record_replay.py          # Records 10-min 0.5s TraCI clips to compact JSON
├── dashboard/
│   └── app.py                    # 6-page Streamlit dashboard + 60 FPS HTML5 Canvas replay
├── models/                       # dqn_best.pt, dqn_last.pt
├── results/                      # baselines/, training/, final/ CSVs & plots
├── tests/                        # Unit & SUMO smoke tests (Phases 1-6)
├── CHANGELOG.md
└── README.md
```

---

## Setup

1. **Install Eclipse SUMO 1.28.0** and set `SUMO_HOME` (e.g., `C:\Program Files (x86)\SUMO`).
2. **Install Python dependencies** (matching v1 environment: `torch`, `numpy`, `pandas`, `plotly`, `streamlit`, `matplotlib`, `traci`, `sumolib`):
   ```powershell
   pip install -r ../v1/requirements.txt
   pip install matplotlib
   ```

---

## End-to-End Commands

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\SUMO"

# 1. Build the 3-junction curved LHT SUMO network
E:\conda_envs\traffic_rl\python.exe -m v2.network.build_network

# 2. Generate sample traffic & pedestrian routes
E:\conda_envs\traffic_rl\python.exe -m v2.traffic.generate_traffic --all-demands --seed 1001

# 3. Train the Shared DQN (100-episode curriculum, TRAIN_SEEDS only)
E:\conda_envs\traffic_rl\python.exe -m v2.training.train_dqn --episodes 100

# 4. Run full 3x10 held-out evaluation (seeds 1001-1010) + statistical tests
E:\conda_envs\traffic_rl\python.exe -m v2.evaluation.final_eval

# 5. Record 10-minute 2D replay JSONs for all controllers & demands
E:\conda_envs\traffic_rl\python.exe -m v2.replays.record_replay --all

# 6. Launch the 6-page Streamlit dashboard
E:\conda_envs\traffic_rl\python.exe -m streamlit run v2/dashboard/app.py

# 7. Run the full v2 test suite
E:\conda_envs\traffic_rl\python.exe -m unittest discover -s v2\tests -t . -v
```

---

## Shared DQN Design Notes
- **Why Parameter Sharing?** All 3 junctions (`J1`, `J2`, `J3`) share identical 4-way LHT geometry and phase structure, so one shared MLP (`10 → 128 → 64 → 2`) collects `3×` more experience per episode and trains cleanly on CPU.
- **Hard Safety Layer:** Enforces `10 s` min green, `3 s` yellow, `2 s` all-red, and `60 s` max green outside the neural network.
- **Reward Formula:** $r = -(1.0 \sum q_i + 0.05 \sum \text{wait}_i + 2.0 \cdot \mathbb{1}_{\text{switch}})$.
- **Summary of Findings:** Shared DQN cuts average waiting time by ~25–32% versus Fixed-Time across all demand levels, and trails SUMO Actuated at Low and Medium demand. A likely reason (not tested) is that DQN acts on a `5 s` decision interval rather than `1 s` induction-loop gap-out.
- **Controllers not in v2:** Q-Learning and SARSA from v1 are preserved in the `/v1` folder and are not part of the v2 comparison.

---

## Limitations
- **Simulation only:** Evaluated in Eclipse SUMO 1.28.0, not on live road hardware.
- **Single corridor:** One 3-junction arterial (`J1–J2–J3`); results may differ on larger city grids.
- **Non-saturated High demand:** High demand (~1660 veh/hr) does not saturate this corridor, so the benefit of RL under heavy congestion is not tested.
- **Evaluation teleports:** A few 1-hour evaluation episodes recorded SUMO teleports: `fixed_time` (`medium`, `seed=1008`, 2 teleports), `fixed_time` (`high`, `seed=1008`, 2 teleports), and `actuated` (`high`, `seed=1007`, 5 teleports).
- **v1 vs v2 comparability:** v1 and v2 results are not directly comparable because the road network is different.
- **Training vs evaluation horizon:** DQN was trained with `900 s` episodes but evaluated on 1-hour (`3600 s`) episodes.
- **Simple pedestrian model:** Small crossing rate (`120` ped/hr) using SUMO's default yielding without a pedestrian-specific RL reward.
