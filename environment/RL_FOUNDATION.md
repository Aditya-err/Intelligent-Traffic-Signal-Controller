# Phase 4: Reinforcement Learning Foundation (`Intelligent-Traffic-RL`)

## 1. Formal MDP Problem Formulation

| MDP Component | Definition |
| :--- | :--- |
| **Agent** | Signal control policy implementing `BaseRLAgent` (`algorithms/base_agent.py`), supporting Tabular Q-Learning/SARSA and CPU-based PyTorch DQN. |
| **Environment** | `TrafficRLWrapper(SumoTrafficEnv)` wrapping the Indian Left-Hand-Traffic (`lefthand="true"`) 4-way SUMO intersection (`center`). |
| **State Space ($\mathcal{S}$)** | Continuous 18-dimensional normalized vector $s_t \in [0, 1]^{18}$ (`float32`), plus optional compact discrete tuple `(ns_level, ew_level, phase, green_bin)` via `discretize_state()`. |
| **Action Space ($\mathcal{A}$)** | Discrete binary action $a_t \in \{0, 1\}$ where `0` = North–South (NS) and `1` = East–West (EW). |
| **Episode** | Fixed-horizon simulation from $t = 0$ to $t = T_{\max}$ (default `1800 s`) or until no expected vehicles remain in SUMO (`terminated=True`). |

---

## 2. State Representation ($s_t \in [0, 1]^{18}$)

At each decision step $t$:

$$s_t = \big[\tilde{q}_1, \dots, \tilde{q}_8,\; \tilde{w}_1, \dots, \tilde{w}_8,\; p_t,\; \tilde{g}_t\big]^\top \in [0, 1]^{18}$$

- **$\tilde{q}_i = \min\left(\frac{h_i}{Q_{\text{ref}}},\, 1\right)$ ($i=1\dots 8$):** Normalized halting queue lengths on the 8 incoming lanes (`N_to_C_0/1`, `S_to_C_0/1`, `E_to_C_0/1`, `W_to_C_0/1`), with `queue_ref_per_lane = 20.0`.
- **$\tilde{w}_i = \min\left(\frac{W^{\text{lane}}_i}{W_{\text{ref}}},\, 1\right)$ ($i=1\dots 8$):** Normalized accumulated lane waiting times (`traci.lane.getWaitingTime`), with `wait_ref_per_lane = 600.0` s.
- **$p_t \in \{0.0, 1.0\}$:** Current active signal group (`0.0` = NS, `1.0` = EW).
- **$\tilde{g}_t = \min\left(\frac{g_t}{G_{\text{ref}}},\, 1\right)$:** Normalized elapsed green duration on the active phase (`max_green_ref = 60.0` s), enabling the agent to anticipate `min_green` eligibility and avoid starving cross-streets.

---

## 3. Safe Action Execution & Signal Transitions

Safety is enforced inside `SumoTrafficEnv` **independently of the agent's action**:
1. **Minimum Green (`min_green = 10 s`):** If `a_t != p_t` while $g_t < 10\text{ s}$, the switch is blocked (`min_green_blocked=True`) and the active green continues.
2. **Yellow & All-Red Clearance:** Every approved group switch passes through `yellow_time = 3 s` followed by `all_red_time = 2 s` (`rrrrrrrrrrrr`) before energizing the target green.
3. **Protected Right-Turn Service:** Under Indian LHT, median-lane right turns (`_1` lanes) cross oncoming traffic and are served exclusively in non-conflicting protected phases (`NS_GREEN_R`, `EW_GREEN_R`).

---

## 4. Mathematical Review of Reward & Anti-Exploitation

### Phase 2 Base Reward
$$r_t^{\text{base}} = -\left( w_{\text{wait}} \cdot \frac{W_t}{K_t \cdot N_{\text{lanes}} \cdot Q_{\max}} + w_{\text{queue}} \cdot \frac{Q_t}{N_{\text{lanes}} \cdot Q_{\max}} \right)$$

### Identified Scaling & Exploitation Issues
1. **Weak Signal Magnitude (`Q_max = 40.0`):** Under Low/Medium demand (`avg_queue = 2.45–6.29`), $N_{\text{lanes}} \cdot Q_{\max} = 320$, compressing rewards into $[-0.02, 0]$. Tightening the reference to `queue_ref_per_lane = 20.0` ($8 \times 20 = 160$) doubles gradient sensitivity in realistic urban regimes while preserving $[0, 1]$ bounds.
2. **Phase Chattering:** Without an explicit switching cost, an agent could oscillate NS $\leftrightarrow$ EW every `10 s`, wasting `5 s` per switch in yellow + all-red clearance.
3. **Minor-Street Starvation:** Including per-lane accumulated waiting times (`obs[8:16]`) and `green_duration` (`obs[17]`) makes cross-street starvation observable, while the shaped reward penalizes both waiting vehicle-seconds and standing queues.

### Phase 4 Shaped Reward
$$r_t = -\left( w_{\text{wait}} \cdot \tilde{W}_t + w_{\text{queue}} \cdot \tilde{Q}_t + \lambda_{\text{switch}} \cdot \mathbb{I}[\text{switch}_t] \right)$$
with defaults $w_{\text{wait}} = 0.6$, $w_{\text{queue}} = 0.4$, $\lambda_{\text{switch}} = 0.02$.

---

## 5. Episode Termination, Reset & Reproducibility
- **Reset:** `env.reset(seed=S)` closes any prior TraCI session cleanly, launches a fresh headless `sumo` instance with `--seed S`, initializes NS green (`t=0`), and returns `(state_0, info)`.
- **Termination:** `terminated=True` when `sim_time >= max_steps` or `getMinExpectedNumber() <= 0`; `truncated=False`.
- **Reproducibility:** Passing identical `seed` and `sumocfg_path` yields identical initial states and deterministic demand sequences across Fixed-Time and future RL evaluations.

---

## 6. Modular RL Architecture (Ready for Phase 5)
- `SumoTrafficEnv` (`environment/traffic_env.py`): TraCI lifecycle & signal safety.
- `TrafficRLWrapper` (`environment/rl_wrapper.py`): 18-dim state encoding, shaped reward, and `discretize_state()`.
- `BaseRLAgent` (`algorithms/base_agent.py`): Unified interface (`select_action`, `observe_transition`, `on_episode_start`, `on_episode_end`, `save`, `load`) for Phase 5 Tabular Q-Learning and CPU-only PyTorch DQN.
