"""Phase 3 unit and live SUMO integration tests for fixed-time baseline."""

from pathlib import Path
import tempfile
import unittest

from algorithms.fixed_time import FixedTimeController
from evaluation.baseline_eval import (
    SCENARIO_CONFIGS,
    evaluate_scenario,
    generate_comparison_plots,
    save_results_csv,
)


class TestFixedTimeController(unittest.TestCase):
    """Unit tests for deterministic fixed-time schedule."""

    def test_invalid_steps_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            FixedTimeController(ns_green_steps=0, ew_green_steps=4)
        with self.assertRaises(ValueError):
            FixedTimeController(ns_green_steps=5, ew_green_steps=-1)

    def test_cycle_sequence_and_reset(self) -> None:
        ctrl = FixedTimeController(ns_green_steps=3, ew_green_steps=2)
        seq = [ctrl.select_action() for _ in range(10)]
        self.assertEqual(seq, [0, 0, 0, 1, 1, 0, 0, 0, 1, 1])

        ctrl.reset()
        self.assertEqual(ctrl.select_action(), 0)


class TestBaselineEvaluationIntegration(unittest.TestCase):
    """Verify scenario configs, live SUMO baseline metrics, CSV, and plots."""

    def test_scenario_configs_exist(self) -> None:
        self.assertEqual(set(SCENARIO_CONFIGS.keys()), {"low", "medium", "high"})
        for path in SCENARIO_CONFIGS.values():
            self.assertTrue(path.is_file(), f"Missing scenario config: {path}")

    def test_live_short_baseline_and_outputs(self) -> None:
        row = evaluate_scenario(
            "medium",
            SCENARIO_CONFIGS["medium"],
            max_steps=120,
            seed=42,
        )
        expected_keys = {
            "scenario",
            "controller",
            "sim_duration_s",
            "seed",
            "avg_waiting_time_s",
            "avg_queue_length",
            "avg_travel_time_s",
            "throughput_veh_per_hr",
            "max_queue_length",
            "vehicles_arrived",
            "vehicles_remaining",
        }
        self.assertEqual(set(row.keys()), expected_keys)
        self.assertGreater(row["vehicles_arrived"], 0)
        self.assertGreater(row["throughput_veh_per_hr"], 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "baseline.csv"
            plot_path = tmp / "baseline.png"
            save_results_csv([row], csv_path)
            generate_comparison_plots([row], plot_path)
            self.assertTrue(csv_path.is_file())
            self.assertTrue(plot_path.is_file())
            self.assertGreater(csv_path.stat().st_size, 0)
            self.assertGreater(plot_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
