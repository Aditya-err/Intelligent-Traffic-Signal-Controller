"""Phase 5 tests: verify replay JSON recorder produces valid dashboard-loadable JSON."""

from pathlib import Path
import tempfile
import unittest

from v2.dashboard.app import build_canvas_html, load_replay_json
from v2.replays.record_replay import record_replay


class TestPhase5Replay(unittest.TestCase):
    """Verify the recorder creates valid JSON that the dashboard loader/HTML builder accepts."""

    def test_record_and_load_replay_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_replay.json"
            record_replay(
                controller_name="fixed_time",
                demand="low",
                seed=1001,
                duration_s=20,
                step_s=0.5,
                output_path=out,
            )
            self.assertTrue(out.exists())
            data = load_replay_json(out)
            self.assertEqual(data["meta"]["controller"], "fixed_time")
            self.assertGreater(len(data["frames"]), 0)
            first = data["frames"][0]
            self.assertIn("tls", first)
            self.assertIn("st", first)
            html = build_canvas_html(data)
            self.assertIn("corridorCanvas", html)


if __name__ == "__main__":
    unittest.main()
