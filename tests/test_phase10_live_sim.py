"""Phase 10 tests for 1-second live 2D HTML5 Canvas SUMO simulation."""

import hashlib
from pathlib import Path
import unittest

from dashboard.canvas_viz import build_live_canvas_html, get_diagnostic_rows
from dashboard.live_sim import LiveSimSession


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_SIGNAL_CHARS = set("Ggyr")
INDIAN_VTYPES = {"car", "motorcycle", "auto_rickshaw", "bus", "truck"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestLive2DSimulation(unittest.TestCase):
    """Verify 1-sec continuous movement, HTML5 Canvas rendering, and cleanup."""

    def test_all_five_controllers_step_with_live_sumo(self) -> None:
        summary_csv = PROJECT_ROOT / "reports" / "final_eval_summary.csv"
        before_hash = _sha256(summary_csv)

        for ctrl_key in (
            "fixed_time",
            "q_learning",
            "sarsa",
            "dqn_medium_only",
            "dqn_curriculum",
        ):
            session = LiveSimSession(
                scenario="medium", controller_key=ctrl_key, seed=1001, max_steps=120
            )
            try:
                snap = session.reset(warmup_steps=20)
                self.assertGreaterEqual(snap.sim_time, 20)
                self.assertGreater(snap.vehicle_count, 0)
                self.assertGreaterEqual(snap.total_spawned, snap.vehicle_count)
                self.assertEqual(len(snap.tls_state), 12)
                self.assertTrue(set(snap.tls_state).issubset(ALLOWED_SIGNAL_CHARS))

                # Confirm vehicles from all 4 approaches (N, S, E, W) spawn
                prefixes = {v["id"].split("_")[1] for v in snap.vehicles if "_" in v["id"]}
                self.assertEqual(prefixes, {"N", "S", "E", "W"})

                # Confirm diagnostics and Canvas payload match
                diag_rows = get_diagnostic_rows(snap)
                self.assertEqual(len(diag_rows), snap.vehicle_count)
                rendered_ids = [
                    r["vehicle_id"]
                    for r in diag_rows
                    if r["rendered_status"] == "RENDERED"
                ]
                self.assertGreater(len(rendered_ids), 0)

                for v in snap.vehicles:
                    self.assertIn(v["vtype"], INDIAN_VTYPES)
                    self.assertIn("angle", v)
                    self.assertTrue(v.get("lane_id"))

                # Step 1 second forward and verify continuous coordinate updates
                prev_pos = {v["id"]: (v["x"], v["y"]) for v in snap.vehicles}
                snap2 = session.step_once()
                self.assertGreater(snap2.sim_time, snap.sim_time)
                moved = any(
                    (v["x"], v["y"]) != prev_pos[v["id"]]
                    for v in snap2.vehicles
                    if v["id"] in prev_pos and v["speed"] > 0.5
                )
                self.assertTrue(moved, "Vehicles did not move between consecutive 1s steps")

                html = build_live_canvas_html(snap2)
                self.assertIn("<canvas id=\"simCanvas\"", html)
                for r in get_diagnostic_rows(snap2):
                    if r["rendered_status"] == "RENDERED":
                        self.assertIn(r["vehicle_id"], html)
            finally:
                session.close()

        self.assertEqual(_sha256(summary_csv), before_hash)

    def test_start_pause_resume_reset_and_cleanup(self) -> None:
        session = LiveSimSession(
            scenario="low", controller_key="fixed_time", seed=1001, max_steps=60
        )
        try:
            snap0 = session.reset()
            self.assertFalse(session.is_running)
            self.assertEqual(snap0.sim_time, 0)

            session.is_running = True
            snap1 = session.step_once()
            self.assertEqual(snap1.sim_time, 1)

            session.is_running = False
            self.assertEqual(session.snapshot.sim_time, 1)  # type: ignore[union-attr]

            session.is_running = True
            snap2 = session.step_once()
            self.assertEqual(snap2.sim_time, 2)

            session.is_running = False
            snap_reset = session.reset()
            self.assertEqual(snap_reset.sim_time, 0)
            self.assertEqual(snap_reset.total_spawned, 0)
        finally:
            session.close()
            self.assertFalse(session.is_running)


if __name__ == "__main__":
    unittest.main()
