"""Phase 4 tests: verify shared DQN training, epsilon decay, checkpoints, and CSV."""

from pathlib import Path
import tempfile
import unittest

from v2 import config
from v2.controllers.dqn import DQNController
from v2.evaluation.run_episode import run_episode
from v2.training.train_dqn import compute_epsilon, train_dqn


class TestPhase4DQN(unittest.TestCase):
    """Validate epsilon schedule and 2-episode end-to-end DQN training."""

    def test_epsilon_reaches_configured_final_value(self) -> None:
        self.assertEqual(compute_epsilon(0, 100), config.DQN_EPSILON_START)
        self.assertEqual(compute_epsilon(99, 100), config.DQN_EPSILON_END)
        self.assertEqual(compute_epsilon(1, 2), config.DQN_EPSILON_END)

    def test_two_episode_training_writes_csv_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            models_dir = tmp / "models"
            results_dir = tmp / "results"
            summary = train_dqn(
                episodes=2,
                duration_s=120,
                models_dir=models_dir,
                results_dir=results_dir,
                seed=7,
            )
            self.assertEqual(summary["final_epsilon"], config.DQN_EPSILON_END)
            self.assertTrue((results_dir / "dqn_training_log.csv").exists())
            self.assertTrue((results_dir / "dqn_training_curves.png").exists())
            self.assertTrue((models_dir / "dqn_best.pt").exists())
            self.assertTrue((models_dir / "dqn_last.pt").exists())

            # Verify saved checkpoint runs cleanly in run_episode
            res = run_episode(
                DQNController(models_dir / "dqn_best.pt"),
                demand="medium",
                seed=1001,
                duration_s=120,
            )
            self.assertEqual(res["controller"], "dqn")
            self.assertGreater(res["vehicles_arrived"], 0)
            self.assertEqual(res["collisions"], 0)


if __name__ == "__main__":
    unittest.main()
