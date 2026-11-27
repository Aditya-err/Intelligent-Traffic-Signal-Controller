"""Phase 6 tests: verify summary stats, Holm correction, Direction column, and dashboard."""

import unittest

from v2.evaluation.stats import (
    build_statistical_tests,
    build_summary_table,
    exact_sign_flip_permutation_test,
    holm_bonferroni,
    paired_t_test,
)


class TestPhase6Statistics(unittest.TestCase):
    """Validate statistical utilities, Holm correction, and Direction labels."""

    def test_holm_bonferroni_monotonic(self) -> None:
        raw = [0.001, 0.04, 0.03]
        adj = holm_bonferroni(raw)
        self.assertEqual(adj, [0.003, 0.06, 0.06])

    def test_summary_and_statistical_tests_direction(self) -> None:
        # Synthetic 10-seed rows where actuated < dqn < fixed_time in wait time
        rows = []
        for s in range(1001, 1011):
            offset = (s - 1001) * 0.1
            rows.append(
                {
                    "controller": "fixed_time",
                    "demand": "low",
                    "seed": s,
                    "avg_waiting_time_s": 16.0 + offset,
                    "avg_queue_length": 4.0,
                    "avg_travel_time_s": 70.0,
                    "throughput_vph": 700.0,
                    "j1_avg_wait_s": 15.0,
                    "j2_avg_wait_s": 16.0,
                    "j3_avg_wait_s": 17.0,
                }
            )
            rows.append(
                {
                    "controller": "actuated",
                    "demand": "low",
                    "seed": s,
                    "avg_waiting_time_s": 10.0 + offset,
                    "avg_queue_length": 2.0,
                    "avg_travel_time_s": 60.0,
                    "throughput_vph": 710.0,
                    "j1_avg_wait_s": 10.0,
                    "j2_avg_wait_s": 10.0,
                    "j3_avg_wait_s": 10.0,
                }
            )
            rows.append(
                {
                    "controller": "dqn",
                    "demand": "low",
                    "seed": s,
                    "avg_waiting_time_s": 12.0 + offset,
                    "avg_queue_length": 2.5,
                    "avg_travel_time_s": 63.0,
                    "throughput_vph": 705.0,
                    "j1_avg_wait_s": 12.0,
                    "j2_avg_wait_s": 12.0,
                    "j3_avg_wait_s": 12.0,
                }
            )

        summary = build_summary_table(rows)
        self.assertEqual(len(summary), 3)
        stats = build_statistical_tests(rows)
        self.assertEqual(len(stats), 3)

        by_comp = {r["comparison"]: r for r in stats}
        self.assertEqual(by_comp["dqn vs fixed_time"]["direction"], "Better than fixed_time")
        self.assertEqual(by_comp["dqn vs actuated"]["direction"], "Worse than actuated")
        self.assertEqual(by_comp["dqn vs fixed_time"]["significant"], "Yes")

    def test_paired_and_permutation_pvalues(self) -> None:
        a = [10.0, 11.0, 12.0, 10.5, 11.5]
        b = [15.1, 15.8, 17.2, 15.4, 16.9]
        _, pt = paired_t_test(a, b)
        pp = exact_sign_flip_permutation_test(a, b)
        self.assertLess(pt, 0.01)
        self.assertLess(pp, 0.1)


if __name__ == "__main__":
    unittest.main()
