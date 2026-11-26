"""Phase 9 unit tests for dashboard data loading, validation, and rendering."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from dashboard.data_loader import load_dashboard_datasets, load_validated_csv
from dashboard.intersection_viz import create_indian_intersection_figure


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDashboardDataLoader(unittest.TestCase):
    """Verify CSV loading, missing/invalid handling, and metric parity."""

    def test_load_existing_phase6_to_8_datasets(self) -> None:
        datasets = load_dashboard_datasets(PROJECT_ROOT / "reports")
        for name, res in datasets.items():
            self.assertTrue(res.ok, f"Dataset {name} failed: {res.error}")
            self.assertIsNotNone(res.df)

        summary_df = datasets["final_summary"].df
        assert summary_df is not None
        self.assertEqual(len(summary_df), 15)

        # Verify exact parity with Phase 8 summary CSV
        raw_csv = pd.read_csv(PROJECT_ROOT / "reports" / "final_eval_summary.csv")
        pd.testing.assert_frame_equal(summary_df, raw_csv)

    def test_missing_and_invalid_csv_handled_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            missing_res = load_validated_csv(tmp / "does_not_exist.csv", ["a", "b"])
            self.assertFalse(missing_res.ok)
            self.assertIsNotNone(missing_res.error)

            bad_csv = tmp / "bad.csv"
            bad_csv.write_text("col_x,col_y\n1,2\n", encoding="utf-8")
            bad_res = load_validated_csv(bad_csv, ["scenario", "controller"])
            self.assertFalse(bad_res.ok)
            self.assertIn("missing required columns", bad_res.error or "")

            empty_csv = tmp / "empty.csv"
            empty_csv.write_text("scenario,controller\n", encoding="utf-8")
            empty_res = load_validated_csv(empty_csv, ["scenario", "controller"])
            self.assertFalse(empty_res.ok)
            self.assertIn("empty", empty_res.error or "")


class TestDashboardComponents(unittest.TestCase):
    """Verify intersection Plotly diagram and Streamlit AppTest smoke run."""

    def test_indian_intersection_figure_builds(self) -> None:
        fig = create_indian_intersection_figure(
            "Medium (~1,330 veh/hr)", "North-South (Straight + Left)", avg_queue_ns_ew=4.75
        )
        self.assertEqual(len(fig.data), 1)
        self.assertIn("Indian LHT 4-Way Intersection", fig.layout.title.text)

    def test_streamlit_app_renders_without_exception(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"))
        app.run(timeout=30)
        self.assertFalse(app.exception, f"Streamlit raised exception: {app.exception}")
        self.assertGreater(len(app.metric), 0)


if __name__ == "__main__":
    unittest.main()
