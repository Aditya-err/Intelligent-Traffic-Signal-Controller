# Phase 3 Experimental Methodology & Baseline Limitations

## 1. Experimental Setup
- **Simulator:** Eclipse SUMO `1.28.0` via TraCI + `SumoTrafficEnv` (`lefthand="true"`).
- **Intersection:** 4-way Indian urban arterial (`N`, `S`, `E`, `W`), 250 m arms, 2 lanes/direction (3.5 m width, 50 km/h).
- **Vehicle Mix:** `motorcycle` (38%), `car` (30%), `auto_rickshaw` (20%), `bus` (6%), `truck` (6%).
- **Controller:** `FixedTimeController(ns_green_steps=5, ew_green_steps=4)` operating at `delta_time=5 s` (`min_green=10 s`, `yellow_time=3 s`, `all_red_time=2 s`, `right_turn_green=5 s`).
- **Reproducibility:** Identical network (`intersection.net.xml`), simulation horizon (`1800 s`), and random seed (`42`) across all scenarios.

## 2. Traffic Demand Scenarios
| Scenario | Config | Total Demand | Description |
| :--- | :--- | ---: | :--- |
| **Low** | `scenario_low.sumocfg` | ~720 veh/hr | Off-peak free-flowing urban traffic |
| **Medium** | `scenario_medium.sumocfg` | ~1,330 veh/hr | Moderate arterial demand (Phase 1 reference) |
| **High** | `scenario_high.sumocfg` | ~2,200 veh/hr | Peak-hour congested arterial demand |

## 3. Collected Metrics
1. **Average Waiting Time (s):** Mean vehicle waiting time from SUMO `tripinfo_<scenario>.xml`.
2. **Average Queue Length (veh):** Mean halting vehicles across the 8 incoming lanes per decision step.
3. **Average Travel Time (s):** Mean trip duration from `tripinfo_<scenario>.xml`.
4. **Vehicle Throughput (veh/hr):** `vehicles_arrived / (sim_duration_s / 3600)`.
5. **Maximum Queue Length (veh):** Peak halting vehicle count across the 8 incoming lanes.
6. **Vehicles Arrived:** Completed trips at `t = 1800 s`.
7. **Vehicles Remaining:** Vehicles still in-network (`running + waiting`) at `t = 1800 s`.

## 4. Known Baseline Limitations
1. **No Demand Responsiveness:** Fixed-time switching ignores real-time queue imbalances, wasting green time on empty approaches during asymmetric surges.
2. **Sub-optimal High-Demand Saturation:** Under peak demand (`High`), fixed cycle lengths cannot extend arterial green to clear spillbacks, increasing `max_queue_length` and `vehicles_remaining`.
3. **Single-Intersection Scope:** Evaluates an isolated 4-way junction without upstream/downstream corridor coordination.
4. **Single-Seed Deterministic Benchmark:** Uses `seed=42` for reproducible baseline comparison; multi-seed variance evaluation will be added alongside RL benchmarking in later phases.
