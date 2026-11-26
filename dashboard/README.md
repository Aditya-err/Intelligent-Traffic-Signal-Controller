# Intelligent-Traffic-RL Dashboard

Interactive Streamlit dashboard presenting verified Phase 6–8 SUMO traffic simulation and Reinforcement Learning results across all five controllers:
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

*(Or `streamlit run dashboard/app.py` if your virtual/Conda environment is already activated.)*

Open the local URL printed by Streamlit (typically `http://localhost:8501`).

## Data Sources
All metrics are loaded directly from existing Phase 6–8 outputs in `reports/`:
- `reports/final_eval_summary.csv`
- `reports/final_eval_raw.csv`
- `reports/final_eval_stats.csv`
- `reports/dqn_training_history.csv`
- `reports/dqn_curriculum_hparam_experiments.csv`
