"""Phase 5 unit and live SUMO integration tests for Q-Learning and SARSA."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent
from evaluation.baseline_eval import SCENARIO_CONFIGS
from evaluation.train_tabular_rl import (
    evaluate_controller_on_scenario,
    train_tabular_agent,
)


class TestTabularUpdateMath(unittest.TestCase):
    """Verify exact Q-Learning, SARSA, epsilon decay, and save/load math."""

    @staticmethod
    def _make_state(ns_q: float, ew_q: float, phase: int, green: float) -> np.ndarray:
        s = np.zeros(18, dtype=np.float32)
        s[0:4] = ns_q
        s[4:8] = ew_q
        s[16] = float(phase)
        s[17] = green
        return s

    def test_q_learning_update_and_terminal_handling(self) -> None:
        agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.0, seed=42)
        s = self._make_state(0.5, 0.1, 0, 0.2)
        ns = self._make_state(0.3, 0.4, 1, 0.1)

        # Pre-seed Q(ns, 1) = 4.0 so max_a' Q(ns, a') = 4.0
        ns_key = agent._encode(ns)
        agent.q_table[ns_key] = np.array([1.0, 4.0], dtype=np.float64)

        # Non-terminal: 0 + 0.5 * (-1.0 + 0.9 * 4.0 - 0) = 1.3
        agent.observe_transition(s, 0, -1.0, ns, terminated=False, truncated=False)
        s_key = agent._encode(s)
        self.assertAlmostEqual(agent.q_table[s_key][0], 1.3, places=6)

        # Terminal: no bootstrap -> 1.3 + 0.5 * (-2.0 - 1.3) = -0.35
        agent.observe_transition(s, 0, -2.0, ns, terminated=True, truncated=False)
        self.assertAlmostEqual(agent.q_table[s_key][0], -0.35, places=6)

        # Truncated (non-terminal): bootstraps -> -0.35 + 0.5 * (-1.0 + 0.9*4.0 - (-0.35))
        agent.observe_transition(s, 0, -1.0, ns, terminated=False, truncated=True)
        self.assertAlmostEqual(agent.q_table[s_key][0], 1.125, places=6)

    def test_sarsa_update_and_persistence(self) -> None:
        agent = SARSAAgent(alpha=0.5, gamma=0.9, epsilon=1.0, epsilon_decay=0.5, seed=42)
        s = self._make_state(0.2, 0.6, 1, 0.3)
        ns = self._make_state(0.1, 0.2, 0, 0.1)

        ns_key = agent._encode(ns)
        agent.q_table[ns_key] = np.array([2.0, 8.0], dtype=np.float64)

        # SARSA uses actual next_action=0 (value 2.0), NOT max (8.0)
        agent.observe_transition(
            s, 1, -1.0, ns, terminated=False, truncated=False, info={"next_action": 0}
        )
        s_key = agent._encode(s)
        self.assertAlmostEqual(agent.q_table[s_key][1], 0.4, places=6)

        agent.on_episode_end(1, {})
        self.assertAlmostEqual(agent.epsilon, 0.5, places=6)

        with tempfile.TemporaryDirectory() as tmpdir:
            q_path = Path(tmpdir) / "sarsa.json"
            agent.save(q_path)
            loaded = SARSAAgent(seed=42)
            loaded.load(q_path)
            self.assertAlmostEqual(loaded.epsilon, 0.5, places=6)
            self.assertAlmostEqual(loaded.q_table[s_key][1], 0.4, places=6)


class TestTabularLiveSumoIntegration(unittest.TestCase):
    """Verify Q-Learning & SARSA actually train and evaluate on live SUMO."""

    def test_live_sumo_training_and_eval_q_learning_and_sarsa(self) -> None:
        q_agent = QLearningAgent(seed=42)
        q_hist = train_tabular_agent(
            q_agent,
            "q_learning",
            episodes=1,
            max_steps=60,
            base_seed=42,
            sumocfg_path=SCENARIO_CONFIGS["medium"],
        )
        self.assertEqual(len(q_hist), 1)
        self.assertGreater(len(q_agent.q_table), 0)

        sarsa_agent = SARSAAgent(seed=42)
        s_hist = train_tabular_agent(
            sarsa_agent,
            "sarsa",
            episodes=1,
            max_steps=60,
            base_seed=42,
            sumocfg_path=SCENARIO_CONFIGS["medium"],
        )
        self.assertEqual(len(s_hist), 1)
        self.assertGreater(len(sarsa_agent.q_table), 0)

        eval_q = evaluate_controller_on_scenario(
            q_agent,
            "q_learning",
            "medium",
            SCENARIO_CONFIGS["medium"],
            max_steps=60,
            seed=42,
        )
        self.assertIn("episode_reward", eval_q)
        self.assertGreater(eval_q["vehicles_arrived"], 0)


if __name__ == "__main__":
    unittest.main()
