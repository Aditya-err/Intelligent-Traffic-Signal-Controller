# Phase 8: Final Controller Evaluation & Statistical Comparison

## 1. Evaluation Configuration & Frozen Model Audit
- **Total Experiments:** **150** live SUMO episodes (`5` controllers × `3` scenarios × `10` held-out seeds).
- **Held-Out Evaluation Seeds:** `1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010` (strictly disjoint from training seeds `42..57`).
- **Simulation Settings:** Identical Indian Left-Hand-Traffic (`lefthand="true"`) 4-way intersection (`intersection.net.xml`), `1800 s` duration, `delta_time = 5 s`, `min_green = 10 s`, `yellow_time = 3 s`, `all_red_time = 2 s`, `right_turn_green = 5 s`.
- **Frozen Controllers:**
  1. `fixed_time`: `FixedTimeController(ns_green_steps=5, ew_green_steps=4)`
  2. `q_learning`: `models/q_learning_qtable.json` (`epsilon = 0.0`, greedy)
  3. `sarsa`: `models/sarsa_qtable.json` (`epsilon = 0.0`, greedy)
  4. `dqn_medium_only`: `models/dqn_checkpoint.pt` (`epsilon = 0.0`, `q_net.eval()`, no gradient updates)
  5. `dqn_curriculum`: `models/dqn_curriculum_best.pt` (`exp1_lr_5e4`, `epsilon = 0.0`, `q_net.eval()`, no gradient updates)

---

## 2. Summarized Results (`Mean ± Std` & `95% CI`, `n = 10` Seeds)

### Low Demand (~720 veh/hr)
| Controller | Avg Wait (s) | 95% CI | % vs FT | Avg Queue | Avg Travel (s) | Throughput (veh/hr) | Arrived / Rem | Episode Reward |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_time` | `13.79 ± 0.47` | `±0.34` | `0.00%` | `2.52 ± 0.12` | `57.94 ± 0.61` | `704.4 ± 3.24` | `352.2 / 9.8` | `-4.47 ± 0.14` |
| `q_learning` | `20.26 ± 0.30` | `±0.22` | `-46.91%` | `4.27 ± 0.06` | `63.92 ± 0.46` | `710.0 ± 2.11` | `355.0 / 7.0` | `-7.24 ± 0.09` |
| **`sarsa`** | **`10.50 ± 0.22`** | `±0.16` | **`+23.83%`** | **`1.86 ± 0.05`** | **`54.77 ± 0.43`** | **`712.0 ± 1.89`** | **`356.0 / 6.0`** | `-4.33 ± 0.05` |
| `dqn_medium_only` | `14.51 ± 0.60` | `±0.43` | `-5.18%` | `2.73 ± 0.12` | `58.52 ± 0.66` | `707.2 ± 1.93` | `353.6 / 8.4` | `-5.05 ± 0.15` |
| **`dqn_curriculum`** | `11.19 ± 0.70` | `±0.50` | **`+18.84%`** | `1.92 ± 0.13` | `55.09 ± 0.83` | `710.6 ± 2.32` | `355.3 / 6.7` | **`-4.04 ± 0.22`** |

### Medium Demand (~1,330 veh/hr)
| Controller | Avg Wait (s) | 95% CI | % vs FT | Avg Queue | Avg Travel (s) | Throughput (veh/hr) | Arrived / Rem | Episode Reward |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_time` | `19.67 ± 0.61` | `±0.44` | `0.00%` | `6.11 ± 0.15` | `65.42 ± 0.75` | `1291.2 ± 3.16` | `645.6 / 21.4` | `-7.69 ± 0.12` |
| `q_learning` | `45.59 ± 1.69` | `±1.21` | `-131.77%` | `16.13 ± 0.66` | `95.39 ± 2.05` | `1271.0 ± 4.35` | `635.5 / 31.5` | `-18.35 ± 0.70` |
| `sarsa` | `16.34 ± 1.11` | `±0.80` | `+16.93%` | **`4.66 ± 0.36`** | `62.13 ± 1.47` | **`1296.2 ± 3.71`** | **`648.1 / 18.9`** | `-6.53 ± 0.34` |
| `dqn_medium_only` | `18.00 ± 0.67` | `±0.48` | `+8.49%` | `5.45 ± 0.18` | `63.34 ± 0.78` | `1295.4 ± 4.33` | `647.7 / 19.3` | `-7.09 ± 0.19` |
| **`dqn_curriculum`** | **`16.04 ± 0.48`** | `±0.35` | **`+18.49%`** | `4.75 ± 0.12` | **`61.38 ± 0.66`** | `1295.6 ± 4.70` | `647.8 / 19.2` | **`-6.36 ± 0.15`** |

### High Demand (~2,200 veh/hr)
| Controller | Avg Wait (s) | 95% CI | % vs FT | Avg Queue | Avg Travel (s) | Throughput (veh/hr) | Arrived / Rem | Episode Reward |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_time` | `119.20 ± 7.17` | `±5.13` | `0.00%` | `67.58 ± 3.21` | `215.60 ± 11.50` | `1658.6 ± 28.36` | `829.3 / 271.7` | `-65.08 ± 2.87` |
| `q_learning` | `124.38 ± 7.02` | `±5.02` | `-4.34%` | `69.69 ± 3.59` | `218.96 ± 9.78` | `1664.0 ± 15.43` | `832.0 / 269.0` | `-67.89 ± 2.92` |
| `sarsa` | `122.12 ± 10.67` | `±7.64` | `-2.45%` | `68.74 ± 3.93` | **`201.85 ± 10.29`** | `1607.6 ± 63.39` | `803.8 / 297.2` | `-72.10 ± 5.00` |
| **`dqn_medium_only`** | **`112.72 ± 6.61`** | `±4.73` | **`+5.44%`** | **`62.41 ± 3.93`** | `202.84 ± 9.69` | `1663.8 ± 20.71` | `831.9 / 269.1` | **`-60.09 ± 3.57`** |
| `dqn_curriculum` | `116.80 ± 6.75` | `±4.83` | `+2.02%` | `72.90 ± 4.10` | `212.88 ± 12.36` | **`1668.2 ± 27.99`** | **`834.1 / 266.9`** | `-70.88 ± 3.46` |

---

## 3. Paired Statistical Tests (`Avg Waiting Time`, `n = 10` Matched Seeds)
Both a **two-sided Paired $t$-test** and an **Exact Paired Sign-Flip Permutation Test** ($2^{10} = 1024$ permutations) were run on matched seeds (`α = 0.05`):

| Scenario | Comparison | Mean Diff (s) | $t$-stat | Paired $t$ $p$ | Exact Perm $p$ | Significant ($p < 0.05$) |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **Low** | `sarsa` vs `fixed_time` | `-3.286` | `-18.165` | `< 0.0001` | `0.00195` | **Yes** |
| **Low** | `dqn_curriculum` vs `fixed_time` | `-2.598` | `-8.851` | `0.00001` | `0.00195` | **Yes** |
| **Low** | `dqn_curriculum` vs `dqn_medium_only` | `-3.312` | `-12.092` | `0.000001` | `0.00195` | **Yes** |
| **Medium** | `dqn_curriculum` vs `fixed_time` | `-3.637` | `-16.010` | `< 0.0001` | `0.00195` | **Yes** |
| **Medium** | `sarsa` vs `fixed_time` | `-3.331` | `-8.812` | `0.00001` | `0.00195` | **Yes** |
| **Medium** | `dqn_curriculum` vs `dqn_medium_only` | `-1.966` | `-7.870` | `0.000025` | `0.00195` | **Yes** |
| **High** | `dqn_medium_only` vs `fixed_time` | `-6.479` | `-2.426` | `0.0382` | `0.0332` | **Yes** |
| **High** | `dqn_curriculum` vs `fixed_time` | `-2.404` | `-0.928` | `0.3777` | `0.3750` | No |
| **High** | `dqn_curriculum` vs `dqn_medium_only` | `+4.075` | `+1.090` | `0.3040` | `0.3184` | No |

*(Full 45-row test table including `avg_queue_length` and `episode_reward` is stored in `reports/final_eval_stats.csv`.)*

### Statistical Assumptions & Limitations
- **Paired $t$-test** assumes approximate normality of paired seed differences $d_i$ ($df = 9$).
- **Exact Sign-Flip Permutation Test** avoids normality assumptions and relies only on exchangeability under the null hypothesis, with minimum attainable two-sided $p = 2 / 1024 = 0.001953$.
- No multiple-comparison correction (e.g., Bonferroni) is applied; borderline $p$-values (`~0.03–0.04` on High) should be interpreted cautiously.

---

## 4. Controller Rankings & Trade-Off Analysis
- **Low Demand (by Avg Wait):** 1. `sarsa` (`10.50 s`), 2. `dqn_curriculum` (`11.19 s`, best reward `-4.04`), 3. `fixed_time` (`13.79 s`), 4. `dqn_medium_only` (`14.51 s`), 5. `q_learning` (`20.26 s`).
- **Medium Demand (by Avg Wait):** 1. `dqn_curriculum` (`16.04 s`, best reward `-6.36`, best travel `61.38 s`), 2. `sarsa` (`16.34 s`), 3. `dqn_medium_only` (`18.00 s`), 4. `fixed_time` (`19.67 s`), 5. `q_learning` (`45.59 s`).
- **High Demand (by Avg Wait):** 1. `dqn_medium_only` (`112.72 s`, best queue `62.41`, best reward `-60.09`), 2. `dqn_curriculum` (`116.80 s`, highest throughput `1668.2 veh/hr`, most arrived `834.1`), 3. `fixed_time` (`119.20 s`), 4. `sarsa` (`122.12 s`), 5. `q_learning` (`124.38 s`).

**Key Trade-Offs:**
- Under **High** demand, `dqn_curriculum` maximized vehicle throughput (`1668.2 veh/hr`) and arrived count (`834.1`) by prioritizing arterial discharge, but allowed larger cross-street queues (`72.90` avg queue, `150.3` max queue) than `dqn_medium_only` (`62.41` avg queue).
- Under **Low** and **Medium** demand, `dqn_curriculum` significantly outperformed both `fixed_time` (`+18.84%` Low, `+18.49%` Medium) and `dqn_medium_only` (`p = 0.00195`).
- `q_learning` suffered from limited 15-episode tabular exploration (`14` states visited), confirming the advantage of continuous function approximation (`DQN`) and on-policy `SARSA` under short training budgets.

---

## 5. Generated Reports & Plots (on `E:`)
- `reports/final_eval_raw.csv` (150 episode rows)
- `reports/final_eval_summary.csv` (mean, std, 95% CI, % vs Fixed-Time)
- `reports/final_eval_stats.csv` (paired $t$-test + exact permutation test)
- `reports/final_controller_comparison.png`
- `reports/final_scenario_performance.png`
- `reports/final_waiting_distribution.png`
- `reports/final_seed_variability.png`
