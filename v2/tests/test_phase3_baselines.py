"""Phase 3 tests: verify Fixed-Time & Actuated baselines and run_episode metrics."""

import unittest

from v2.controllers.actuated import ActuatedController
from v2.controllers.fixed_time import FixedTimeController
from v2.evaluation.run_episode import run_episode


EXPECTED_KEYS = {
    "controller",
    "demand",
    "seed",
    "duration_s",
    "avg_waiting_time_s",
    "avg_queue_length",
    "avg_travel_time_s",
    "throughput_vph",
    "vehicles_arrived",
    "vehicles_remaining",
    "j1_avg_wait_s",
    "j2_avg_wait_s",
    "j3_avg_wait_s",
    "teleports",
    "collisions",
    "max_vehicle_wait_s",
    "gridlock_warning",
    "log_warnings",
}


class TestPhase3Baselines(unittest.TestCase):
    """Run a 120s smoke check on both baseline controllers and verify metrics."""

    def test_fixed_time_120s_metrics(self) -> None:
        res = run_episode(FixedTimeController(), demand="medium", seed=1001, duration_s=120)
        self.assertTrue(EXPECTED_KEYS.issubset(res.keys()))
        self.assertEqual(res["controller"], "fixed_time")
        self.assertGreater(res["vehicles_arrived"], 0)
        self.assertEqual(res["collisions"], 0)
        self.assertEqual(res["teleports"], 0)

    def test_actuated_120s_metrics(self) -> None:
        res = run_episode(ActuatedController(), demand="medium", seed=1001, duration_s=120)
        self.assertTrue(EXPECTED_KEYS.issubset(res.keys()))
        self.assertEqual(res["controller"], "actuated")
        self.assertGreater(res["vehicles_arrived"], 0)
        self.assertEqual(res["collisions"], 0)
        self.assertEqual(res["teleports"], 0)


if __name__ == "__main__":
    unittest.main()
