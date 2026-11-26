"""Phase 8 tests for statistical utilities, frozen evaluation, and seed matching."""

import unittest

import torch

from evaluation.baseline_eval import SCENARIO_CONFIGS
from evaluation.final_eval import (
    FINAL_EVAL_SEEDS,
    load_frozen_controllers,
    summarize_and_test,
)
from evaluation.stats_utils import (
    compute_summary_stats,
    exact_permutation_test,
    paired_t_test,
    pct_improvement,
)
from evaluation.train_tabular_rl import evaluate_controller_on_scenario


class TestStatisticalUtilities(unittest.TestCase):
    """Verify CI, paired t-test, permutation test, and improvement math."""

    def test_summary_stats_and_ci95(self) -> None:
        st = compute_summary_stats([10.0, 12.0, 14.0, 16.0])
        self.assertAlmostEqual(st["mean"], 13.0, places=6)
        self.assertGreater(st["std"], 0.0)
        self.assertGreater(st["ci95_half"], 0.0)
        self.assertAlmostEqual(st["ci95_high"] - st["mean"], st["mean"] - st["ci95_low"], places=6)

    def test_paired_t_and_permutation_tests(self) -> None:
        a = [10.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1, 10.4, 9.9, 10.3]
        b = [14.0, 15.0, 14.5, 13.5, 14.2, 13.8, 14.1, 14.4, 13.9, 14.3]
        res = paired_t_test(a, b)
        perm_p = exact_permutation_test(a, b)
        self.assertAlmostEqual(res["mean_diff"], -4.0, places=5)
        self.assertLess(res["p_value"], 0.001)
        self.assertLess(perm_p, 0.01)

    def test_pct_improvement(self) -> None:
        self.assertAlmostEqual(
            pct_improvement(20.0, 15.0, lower_is_better=True), 25.0, places=6
        )
        self.assertAlmostEqual(
            pct_improvement(1000.0, 1100.0, lower_is_better=False), 10.0, places=6
        )


class TestFinalEvaluationIntegrity(unittest.TestCase):
    """Verify frozen weights, paired seed matching, and aggregation."""

    def test_held_out_seeds_count_and_frozen_models(self) -> None:
        self.assertEqual(len(FINAL_EVAL_SEEDS), 10)
        controllers = dict(load_frozen_controllers())
        for name in ("q_learning", "sarsa", "dqn_medium_only", "dqn_curriculum"):
            self.assertEqual(controllers[name].epsilon, 0.0)

        dqn = controllers["dqn_curriculum"]
        learn_before = dqn.learn_steps
        w_before = [p.clone() for p in dqn.q_net.parameters()]
        r1 = evaluate_controller_on_scenario(
            dqn, "dqn_curriculum", "low", SCENARIO_CONFIGS["low"], max_steps=40, seed=1001
        )
        r2 = evaluate_controller_on_scenario(
            dqn, "dqn_curriculum", "low", SCENARIO_CONFIGS["low"], max_steps=40, seed=1001
        )
        self.assertEqual(dqn.learn_steps, learn_before)
        for wb, wa in zip(w_before, dqn.q_net.parameters()):
            self.assertTrue(torch.equal(wb, wa))
        self.assertEqual(r1["avg_waiting_time_s"], r2["avg_waiting_time_s"])
        self.assertEqual(r1["vehicles_arrived"], r2["vehicles_arrived"])

    def test_summarize_and_test_paired_seed_matching(self) -> None:
        fake_raw = []
        for sc in ("low", "medium", "high"):
            for ctrl_idx, ctrl in enumerate(
                ("fixed_time", "q_learning", "sarsa", "dqn_medium_only", "dqn_curriculum")
            ):
                for s_idx, seed in enumerate((1001, 1002, 1003)):
                    fake_raw.append(
                        {
                            "scenario": sc,
                            "controller": ctrl,
                            "seed": seed,
                            "avg_waiting_time_s": 20.0 - ctrl_idx + s_idx * 0.5,
                            "avg_queue_length": 5.0 + s_idx * 0.2,
                            "avg_travel_time_s": 60.0 - ctrl_idx,
                            "throughput_veh_per_hr": 1000.0 + ctrl_idx * 10,
                            "max_queue_length": 12.0,
                            "vehicles_arrived": 500 + ctrl_idx,
                            "vehicles_remaining": 10,
                            "episode_reward": -8.0 + ctrl_idx * 0.5,
                        }
                    )
        summary, stats = summarize_and_test(fake_raw)
        self.assertEqual(len(summary), 15)
        self.assertEqual(len(stats), 45)


if __name__ == "__main__":
    unittest.main()
