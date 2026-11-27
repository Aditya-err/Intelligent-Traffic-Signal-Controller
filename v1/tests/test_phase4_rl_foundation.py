"""Phase 4 tests for RL state representation, reward shaping, safety, and BaseRLAgent."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from algorithms.base_agent import BaseRLAgent
from environment.rl_wrapper import TrafficRLWrapper, discretize_state
from environment.traffic_env import SumoTrafficEnv


class _DummyRLAgent(BaseRLAgent):
    """Minimal concrete subclass to verify BaseRLAgent contract."""

    def __init__(self, state_dim: int = 18, action_dim: int = 2, seed: int = 42) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, seed=seed)
        self.transitions: list[tuple[int, float]] = []

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        return int(round(float(state[16]))) % self.action_dim

    def observe_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict | None = None,
    ) -> dict[str, float]:
        self.transitions.append((action, reward))
        return {"last_reward": float(reward)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text("dummy", encoding="utf-8")

    def load(self, path: str | Path) -> None:
        Path(path).read_text(encoding="utf-8")


class TestBaseRLAgentContract(unittest.TestCase):
    """Verify BaseRLAgent abstract interface and discretization helper."""

    def test_cannot_instantiate_abstract_base_agent(self) -> None:
        with self.assertRaises(TypeError):
            BaseRLAgent(state_dim=18, action_dim=2)  # type: ignore[abstract]

    def test_dummy_agent_and_discretization(self) -> None:
        agent = _DummyRLAgent()
        cfg = agent.get_config()
        self.assertEqual(cfg["state_dim"], 18)
        self.assertEqual(cfg["action_dim"], 2)

        sample_state = np.zeros(18, dtype=np.float32)
        sample_state[0:4] = 0.5
        sample_state[16] = 1.0
        sample_state[17] = 0.5
        key = discretize_state(sample_state)
        self.assertEqual(len(key), 4)
        self.assertEqual(key[2], 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "dummy.ckpt"
            agent.save(ckpt)
            agent.load(ckpt)


class TestTrafficRLWrapperIntegration(unittest.TestCase):
    """Live SUMO integration tests for Phase 4 RL state, reward, safety, episodes."""

    def setUp(self) -> None:
        self.base_env = SumoTrafficEnv(
            use_gui=False,
            max_steps=60,
            delta_time=5,
            min_green=10,
            yellow_time=3,
            all_red_time=2,
            right_turn_green=5,
            sumo_seed=42,
        )
        self.env = TrafficRLWrapper(self.base_env)

    def tearDown(self) -> None:
        self.env.close()

    def test_state_shape_bounds_and_features(self) -> None:
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, (18,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertEqual(obs[16], 0.0)
        self.assertEqual(obs[17], 0.0)
        self.assertEqual(len(info["lane_waiting_times"]), 8)

        # Advance 2 steps holding NS (0): green duration must increase
        obs, reward, _, _, info = self.env.step(0)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertGreater(obs[17], 0.0)
        self.assertEqual(info["green_duration_s"], 5)
        self.assertTrue(np.isfinite(reward))
        self.assertLessEqual(reward, 0.0)

    def test_reward_switch_penalty_and_action_safety(self) -> None:
        self.env.reset(seed=42)

        # Premature switch at t=0 must be blocked by min_green -> switch_penalty == 0
        _, _, _, _, info1 = self.env.step(1)
        self.assertTrue(info1["min_green_blocked"])
        self.assertEqual(info1["switch_penalty"], 0.0)

        # Step once more on NS so green_elapsed reaches 10s
        _, _, _, _, info2 = self.env.step(0)
        self.assertEqual(info2["green_duration_s"], 10)

        # Now switch to EW (1) -> switch executes and switch_penalty applies
        obs3, r3, _, _, info3 = self.env.step(1)
        self.assertTrue(info3["switch_executed"])
        self.assertEqual(obs3[16], 1.0)
        self.assertAlmostEqual(info3["switch_penalty"], self.env.switch_penalty)
        expected_r3 = -(
            self.env.wait_weight * info3["shaped_norm_wait"]
            + self.env.queue_weight * info3["shaped_norm_queue"]
            + info3["switch_penalty"]
        )
        self.assertAlmostEqual(r3, expected_r3, places=6)

    def test_episode_reproducibility_and_agent_loop(self) -> None:
        agent = _DummyRLAgent()

        def run_episode() -> tuple[np.ndarray, list[float]]:
            state, _ = self.env.reset(seed=42)
            agent.on_episode_start(0)
            rewards: list[float] = []
            terminated = False
            while not terminated:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.assertFalse(truncated)
                agent.observe_transition(
                    state, action, reward, next_state, terminated, truncated, info
                )
                rewards.append(reward)
                state = next_state
            agent.on_episode_end(0, {"total_reward": sum(rewards)})
            return state, rewards

        s1, r1 = run_episode()
        s2, r2 = run_episode()
        np.testing.assert_allclose(s1, s2, atol=1e-6)
        self.assertEqual(len(r1), len(r2))
        for a, b in zip(r1, r2):
            self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
