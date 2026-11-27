# Intelligent-Traffic-RL Dashboard

Interactive Streamlit dashboard presenting verified Phase 6–8 SUMO traffic simulation and Reinforcement Learning results, plus a **Live 2D Top-Down SUMO Simulation** across all five controllers:
- **Fixed-Time**
- **Q-Learning**
- **SARSA**
- **DQN Medium-only**
- **DQN Curriculum**

## Launch Command (Windows)

From the repository root (`Intelligent-Traffic-RL/`):

```powershell
E:\conda_envs\traffic_rl\python.exe -m streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit (typically `http://localhost:8501`).

## Live 2D Simulation Controls (`Traffic Simulation` Section)
- **Traffic Demand:** `Low (~720 veh/hr)`, `Medium (~1,330 veh/hr)`, `High (~2,200 veh/hr)`
- **Controller:** Any of the 5 frozen controllers (`epsilon = 0.0`)
- **Simulation Speed:** `1x`, `2x`, `4x`
- **Playback:** `Start`, `Pause`, `Reset`
- Displays live SUMO vehicle `(x, y)` positions by Indian vehicle type, 12-link North/South/East/West signal states, selected action, waiting time, queue length, and active vehicle count.

## Data Sources
Static analytics load directly from existing Phase 6–8 outputs in `reports/`:
- `reports/final_eval_summary.csv`
- `reports/final_eval_raw.csv`
- `reports/final_eval_stats.csv`
- `reports/dqn_training_history.csv`
- `reports/dqn_curriculum_hparam_experiments.csv`
