# Phase 7: Curriculum Training Methodology & Experimental Limitations

## 1. Curriculum Design
To address the generalization gap observed in Phase 6 (Medium-only DQN training), Phase 7 trains `DQNAgent` progressively across all three Indian traffic demand regimes:
1. **Stage 1 — Low Demand (`episodes 1–4`):** ~720 veh/hr (`scenario_low.sumocfg`) to learn clean baseline switching and clearance timing.
2. **Stage 2 — Medium Demand (`episodes 5–8`):** ~1,330 veh/hr (`scenario_medium.sumocfg`) to balance arterial vs. cross-street queues.
3. **Stage 3 — High Demand (`episodes 9–12`):** ~2,200 veh/hr (`scenario_high.sumocfg`) to experience near-saturation queues and spillback recovery.
4. **Stage 4 — Mixed Cycling (`episodes 13–15`):** Cycles `low -> medium -> high` to consolidate multi-regime Q-value generalization.

## 2. Controlled Hyperparameter Experiments
Each experiment varies **one hyperparameter** from `exp0_baseline` (`lr=1e-3`, `epsilon_decay=0.85`, `batch_size=64`, `target_update_freq=100`, `replay_capacity=20,000`):
- `exp1_lr_5e4`: `learning_rate = 5e-4`
- `exp2_eps_decay_090`: `epsilon_decay = 0.90`
- `exp3_batch_128`: `batch_size = 128`
- `exp4_target_freq_50`: `target_update_freq = 50`
- `exp5_replay_50k`: `replay_capacity = 50,000`

## 3. Strict Train / Evaluation Separation
- **Training Seeds:** `42 + episode` (`43..57`).
- **Checkpoint Selection:** Winning experiment and checkpoint (`models/dqn_curriculum_best.pt`) are selected **strictly from training-time `stage4_mixed` episode rewards** — never from evaluation test results.
- **Evaluation Seeds:** Held-out seeds `101`, `202`, `303` across all 3 scenarios (`1800 s` each), using greedy actions (`epsilon = 0.0`) with zero learning updates (`learn_steps` unchanged).

## 4. Experimental Limitations
1. **Training Horizon:** Each curriculum run uses 15 episodes (`2.5` hours of simulated traffic per stage) to remain CPU-practical on a 4-core laptop; longer schedules (50–100+ episodes) may yield stronger convergence.
2. **Single-Intersection Scope:** Evaluates an isolated 4-way Indian LHT junction; multi-intersection corridor coordination is out of scope.
3. **3-Seed Evaluation:** Evaluates 3 held-out seeds per scenario (`101, 202, 303`); larger seed counts would narrow confidence intervals further.
