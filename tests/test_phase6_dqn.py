"""Phase 6 unit and live SUMO integration tests for CPU-based DQN."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from algorithms.dqn import DQNAgent, QNetwork
from algorithms.replay_buffer import ReplayBuffer
from evaluation.baseline_eval import SCENARIO_CONFIGS
from evaluation.train_dqn import train_dqn_agent
from evaluation.train_tabular_rl import evaluate_controller_on_scenario


class TestDQNComponents(unittest.TestCase):
    """Mathematical and architectural unit tests for QNetwork, ReplayBuffer, DQNAgent."""

    def test_network_dimensions_and_cpu_device(self) -> None:
        net = QNetwork(state_dim=18, action_dim=2)
        x = torch.zeros((4, 18), dtype=torch.float32)
        out = net(x)
        self.assertEqual(out.shape, (4, 2))
        self.assertEqual(out.device.type, "cpu")

    def test_replay_buffer_push_wrap_and_sample(self) -> None:
        buf = ReplayBuffer(capacity=5, state_dim=18, seed=42)
        for i in range(8):
            s = np.full(18, float(i), dtype=np.float32)
            ns = np.full(18, float(i + 1), dtype=np.float32)
            buf.push(s, i % 2, -float(i), ns, i == 7)
        self.assertEqual(len(buf), 5)

        states, actions, rewards, next_states, terms = buf.sample(3)
        self.assertEqual(states.shape, (3, 18))
        self.assertEqual(actions.shape, (3,))
        self.assertEqual(rewards.shape, (3,))
        self.assertEqual(next_states.shape, (3, 18))
        self.assertEqual(terms.shape, (3,))

    def test_bellman_targets_terminal_vs_nonterminal(self) -> None:
        agent = DQNAgent(gamma=0.9, seed=42)
        # Set target_net to output constant [1.0, 4.0] so max_a' Q_target = 4.0
        with torch.no_grad():
            for p in agent.target_net.parameters():
                p.zero_()
            agent.target_net.net[-1].bias.copy_(torch.tensor([1.0, 4.0]))

        rewards = torch.tensor([-1.0, -2.0], dtype=torch.float32)
        next_states = torch.zeros((2, 18), dtype=torch.float32)
        terminateds = torch.tensor([0.0, 1.0], dtype=torch.float32)
        targets = agent.compute_bellman_targets(rewards, next_states, terminateds)
        # Row 0 (non-terminal/truncated): -1.0 + 0.9 * 4.0 = 2.6
        # Row 1 (terminal): -2.0 + 0 = -2.0
        self.assertAlmostEqual(float(targets[0].item()), 2.6, places=5)
        self.assertAlmostEqual(float(targets[1].item()), -2.0, places=5)

    def test_gradient_update_target_sync_and_checkpoint(self) -> None:
        agent = DQNAgent(
            batch_size=4, min_replay_size=4, target_update_freq=2, seed=42
        )
        initial_target = [p.clone() for p in agent.target_net.parameters()]

        for i in range(4):
            s = np.random.default_rng(i).random(18, dtype=np.float32)
            ns = np.random.default_rng(i + 10).random(18, dtype=np.float32)
            metrics = agent.observe_transition(
                s, i % 2, -0.5, ns, terminated=False, truncated=(i == 3)
            )
        self.assertIsNotNone(metrics)
        self.assertEqual(agent.learn_steps, 1)

        # Target network must NOT change before target_update_freq (2) steps
        for p0, p1 in zip(initial_target, agent.target_net.parameters()):
            self.assertTrue(torch.equal(p0, p1))

        # Second train_step triggers target sync
        agent.observe_transition(
            np.zeros(18, dtype=np.float32),
            0,
            -0.5,
            np.ones(18, dtype=np.float32),
            terminated=False,
            truncated=False,
        )
        self.assertEqual(agent.learn_steps, 2)
        for q_p, t_p in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            self.assertTrue(torch.equal(q_p, t_p))

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "dqn.pt"
            agent.save(ckpt)
            loaded = DQNAgent(seed=42)
            loaded.load(ckpt)
            for p_orig, p_load in zip(
                agent.q_net.parameters(), loaded.q_net.parameters()
            ):
                self.assertTrue(torch.equal(p_orig, p_load))


class TestDQNLiveSumoIntegration(unittest.TestCase):
    """Verify live SUMO training and greedy evaluation for DQNAgent."""

    def test_live_sumo_dqn_training_and_greedy_eval(self) -> None:
        agent = DQNAgent(
            batch_size=8, min_replay_size=8, target_update_freq=5, seed=42
        )
        hist = train_dqn_agent(
            agent,
            episodes=1,
            max_steps=80,
            base_seed=42,
            sumocfg_path=SCENARIO_CONFIGS["medium"],
        )
        self.assertEqual(len(hist), 1)
        self.assertGreater(len(agent.replay_buffer), 8)
        self.assertGreater(agent.learn_steps, 0)

        learn_before = agent.learn_steps
        row = evaluate_controller_on_scenario(
            agent,
            "dqn",
            "medium",
            SCENARIO_CONFIGS["medium"],
            max_steps=60,
            seed=42,
        )
        self.assertEqual(agent.learn_steps, learn_before)
        self.assertGreater(row["vehicles_arrived"], 0)
        self.assertIn("episode_reward", row)


if __name__ == "__main__":
    unittest.main()
