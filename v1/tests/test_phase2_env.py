"""Phase 2 integration tests for SumoTrafficEnv (live SUMO + TraCI, no mocks)."""

import unittest

from gymnasium.utils.env_checker import check_env
import numpy as np

from environment.traffic_env import SumoTrafficEnv


# Valid, non-conflicting 12-link signal states allowed in Indian LHT intersection
ALLOWED_SIGNAL_STATES = {
    SumoTrafficEnv.NS_GREEN_SL,
    SumoTrafficEnv.NS_YELLOW_SL,
    SumoTrafficEnv.NS_GREEN_R,
    SumoTrafficEnv.NS_YELLOW_R,
    SumoTrafficEnv.EW_GREEN_SL,
    SumoTrafficEnv.EW_YELLOW_SL,
    SumoTrafficEnv.EW_GREEN_R,
    SumoTrafficEnv.EW_YELLOW_R,
    SumoTrafficEnv.ALL_RED,
}


class TestSumoTrafficEnvIntegration(unittest.TestCase):
    """Real SUMO + TraCI integration tests for Phase 2 Gymnasium environment."""

    def setUp(self) -> None:
        self.env = SumoTrafficEnv(
            use_gui=False,
            max_steps=120,
            delta_time=5,
            min_green=10,
            yellow_time=3,
            all_red_time=2,
            right_turn_green=5,
        )

    def tearDown(self) -> None:
        self.env.close()

    def test_gymnasium_api_compliance(self) -> None:
        check_env(self.env, skip_render_check=True)

    def test_observation_shape_and_validity(self) -> None:
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, (9,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertEqual(obs[8], 0.0)
        self.assertEqual(info["sim_time"], 0)

        for action in (0, 0, 1, 1):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.assertEqual(obs.shape, (9,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertIsInstance(reward, float)
            self.assertLessEqual(reward, 0.0)
            self.assertFalse(truncated)
            if terminated:
                break

    def test_action_validation(self) -> None:
        self.env.reset(seed=42)
        self.assertEqual(self.env.action_space.n, 2)
        for invalid_action in (-1, 2, "invalid"):
            with self.assertRaises(ValueError):
                self.env.step(invalid_action)  # type: ignore[arg-type]

    def test_signal_safety_min_green_and_clearance(self) -> None:
        self.env.reset(seed=42)

        # t=0, green_elapsed=0 < min_green(10): switching to EW (1) must be blocked
        obs, _, _, _, info1 = self.env.step(1)
        self.assertTrue(info1["min_green_blocked"])
        self.assertFalse(info1["switch_executed"])
        self.assertEqual(obs[8], 0.0)

        # t=5, green_elapsed=5 < min_green(10): switching to EW (1) still blocked
        obs, _, _, _, info2 = self.env.step(1)
        self.assertTrue(info2["min_green_blocked"])
        self.assertFalse(info2["switch_executed"])
        self.assertEqual(obs[8], 0.0)

        # t=10, green_elapsed=10 >= min_green(10): switching to EW (1) must execute safely
        obs, _, _, _, info3 = self.env.step(1)
        self.assertFalse(info3["min_green_blocked"])
        self.assertTrue(info3["switch_executed"])
        self.assertEqual(obs[8], 1.0)

        states = info3["signal_states"]
        self.assertTrue(any("y" in s for s in states), "Missing yellow clearance")
        self.assertIn(SumoTrafficEnv.ALL_RED, states, "Missing all-red clearance")
        self.assertIn(SumoTrafficEnv.EW_GREEN_SL, states, "Missing target EW green")

        # Verify no conflicting green states ever occurred
        for state in states:
            self.assertIn(state, ALLOWED_SIGNAL_STATES)

    def test_reset_and_episode_termination(self) -> None:
        short_env = SumoTrafficEnv(
            use_gui=False,
            max_steps=25,
            delta_time=5,
            min_green=10,
        )
        try:
            obs, info = short_env.reset(seed=7)
            self.assertEqual(info["sim_time"], 0)
            self.assertEqual(obs.shape, (9,))

            terminated = False
            steps = 0
            while not terminated and steps < 20:
                _, _, terminated, truncated, info = short_env.step(steps % 2)
                self.assertFalse(truncated)
                steps += 1

            self.assertTrue(terminated)
            self.assertGreaterEqual(info["sim_time"], 25)

            # Re-reset must start a clean episode
            obs2, info2 = short_env.reset(seed=8)
            self.assertEqual(info2["sim_time"], 0)
            self.assertEqual(obs2[8], 0.0)
        finally:
            short_env.close()

    def test_traci_connection_and_cleanup(self) -> None:
        self.env.reset(seed=42)
        self.assertIsNotNone(self.env._conn)
        self.env.close()
        self.assertIsNone(self.env._conn)
        with self.assertRaises(RuntimeError):
            self.env.step(0)


if __name__ == "__main__":
    unittest.main()
