# Phase 5: Tabular Q-Learning & SARSA

## 1. Compact Discrete State Representation
To avoid combinatorial explosion on CPU/RAM while keeping the 18-dimensional continuous observation intact for future DQN, `discretize_state()` maps `TrafficRLWrapper` observations to a 4-tuple:

$$\tilde{s}_t = (\text{ns\_level},\; \text{ew\_level},\; \text{phase},\; \text{green\_bin})$$

| Feature | Source in 18-D State | Bins | Values |
| :--- | :--- | ---: | :--- |
| `ns_level` | `mean(obs[0:4])` | 4 | `{0, 1, 2, 3}` |
| `ew_level` | `mean(obs[4:8])` | 4 | `{0, 1, 2, 3}` |
| `phase` | `obs[16]` | 2 | `{0 (NS), 1 (EW)}` |
| `green_bin` | `obs[17]` | 3 | `{0, 1, 2}` |

- **Total Possible Discrete States:** $4 \times 4 \times 2 \times 3 = \mathbf{96}$ states.
- **State-Action Pairs:** $96 \times 2 = \mathbf{192}$ entries.

---

## 2. Algorithms & Update Rules

### Q-Learning (`algorithms/q_learning.py`, Off-Policy)
$$Q(s, a) \leftarrow Q(s, a) + \alpha \Big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\Big]$$

### SARSA (`algorithms/sarsa.py`, On-Policy)
$$Q(s, a) \leftarrow Q(s, a) + \alpha \Big[r + \gamma Q(s', a') - Q(s, a)\Big]$$

For both algorithms:
- When `terminated == True`, the future bootstrap term ($\gamma \dots$) is `0.0`.
- When `truncated == True` without termination, bootstrapping is preserved.

---

## 3. Training Configuration & Reproducibility
- **Training Scenario:** `simulation/config/scenario_medium.sumocfg` (`1800 s`, Indian LHT, 5 vehicle types).
- **Episodes:** `15` episodes per agent (`seed = 42 + episode`).
- **Hyperparameters:**
  - Learning rate $\alpha = 0.15$
  - Discount factor $\gamma = 0.95$
  - Initial $\epsilon = 1.0$, $\epsilon_{\min} = 0.05$, decay $= 0.85$ per episode
- **Saved Artifacts (on `E:`):**
  - `models/q_learning_qtable.json`
  - `models/sarsa_qtable.json`
  - `reports/tabular_training_history.csv`
  - `reports/tabular_training_rewards.png`
  - `reports/phase5_comparison.csv`
  - `reports/phase5_controller_comparison.png`

---

## 4. Fair Evaluation Protocol
All three controllers (`fixed_time`, `q_learning` with `explore=False`, `sarsa` with `explore=False`) are evaluated inside the identical `TrafficRLWrapper(SumoTrafficEnv(...))` pipeline across **Low**, **Medium**, and **High** scenarios (`1800 s`, `seed = 42`).
