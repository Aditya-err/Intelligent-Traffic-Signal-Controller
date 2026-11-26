"""Phase 7 unit and live SUMO integration tests for curriculum DQN."""

from pathlib import Path
import tempfile
import unittest

import torch

from algorithms.dqn import DQNAgent
from evaluation.baseline_eval import SCENARIO_CONFIGS
from evaluation.train_dqn_curriculum import (
    build_curriculum_schedule,
    train_dqn_curriculum,
)
from evaluation.train_tabular_rl import evaluate_controller_on_scenario


class TestCurriculumScheduleAndReproducibility(unittest.TestCase):
    """Verify curriculum schedule, reproducibility, and checkpoint selection."""

    def test_curriculum_schedule_order(self) -> None:
        sched = build_curriculum_schedule(
            low_episodes=2, medium_episodes=2, high_episodes=2, mixed_episodes=3
        )
        self.assertEqual(len(sched), 9)
        self.assertEqual(sched[0], ("stage1_low", "low"))
        self.assertEqual(sched[2], ("stage2_medium", "medium"))
        self.assertEqual(sched[4], ("stage3_high", "high"))
        self.assertEqual(
            sched[6:],
            [
                ("stage4_mixed", "low"),
                ("stage4_mixed", "medium"),
                ("stage4_mixed", "high"),
            ],
        )

    def test_reproducible_curriculum_training_and_checkpoint_selection(self) -> None:
        sched = [("stage1_low", "low"), ("stage2_medium", "medium")]

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt1 = Path(tmpdir) / "curr1.pt"
            ckpt2 = Path(tmpdir) / "curr2.pt"

            a1 = DQNAgent(batch_size=8, min_replay_size=8, seed=77)
            h1 = train_dqn_curriculum(
                a1, sched, max_steps=40, base_seed=77, best_ckpt_path=ckpt1
            )

            a2 = DQNAgent(batch_size=8, min_replay_size=8, seed=77)
            h2 = train_dqn_curriculum(
                a2, sched, max_steps=40, base_seed=77, best_ckpt_path=ckpt2
            )

            self.assertTrue(ckpt1.is_file())
            self.assertEqual(h1[0]["episode_reward"], h2[0]["episode_reward"])
            self.assertEqual(h1[1]["episode_reward"], h2[1]["episode_reward"])
            for p1, p2 in zip(a1.q_net.parameters(), a2.q_net.parameters()):
                self.assertTrue(torch.allclose(p1, p2, atol=1e-6))

            # Verify evaluation isolation (epsilon=0, no weight/learn_step updates)
            a1.epsilon = 0.0
            learn_before = a1.learn_steps
            weights_before = [p.clone() for p in a1.q_net.parameters()]
            row = evaluate_controller_on_scenario(
                a1, "dqn_curr", "high", SCENARIO_CONFIGS["high"], max_steps=40, seed=101
            )
            self.assertEqual(a1.learn_steps, learn_before)
            for wb, wa in zip(weights_before, a1.q_net.parameters()):
                self.assertTrue(torch.equal(wb, wa))
            self.assertGreater(row["vehicles_arrived"], 0)


if __name__ == "__main__":
    unittest.main()
