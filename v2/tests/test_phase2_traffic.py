"""Phase 2 tests: verify v2 random traffic, pedestrians, and 60s SUMO smoke run."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

from v2 import config
from v2.traffic.generate_traffic import generate_route_file

_SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")
_SUMO_TOOLS = Path(_SUMO_HOME) / "tools"
if str(_SUMO_TOOLS) not in sys.path:
    sys.path.insert(0, str(_SUMO_TOOLS))

import traci  # noqa: E402


class TestPhase2Traffic(unittest.TestCase):
    """Validate route generation, demand levels, vehicle mix, and 60s SUMO behavior."""

    def test_train_and_eval_seeds_are_disjoint(self) -> None:
        self.assertEqual(config.EVAL_SEEDS, list(range(1001, 1011)))
        self.assertTrue(set(config.TRAIN_SEEDS).isdisjoint(set(config.EVAL_SEEDS)))

    def test_vehicle_mix_sums_to_one(self) -> None:
        self.assertAlmostEqual(sum(config.VEHICLE_MIX.values()), 1.0, places=6)

    def test_demand_levels_and_vehicle_mix_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for lvl, target_vph in config.DEMAND_LEVELS_VPH.items():
                out = Path(tmpdir) / f"{lvl}.rou.xml"
                summary = generate_route_file(
                    demand_level=lvl, seed=1001, duration_s=3600, output_path=out
                )
                self.assertTrue(out.exists())
                # Poisson count within +/- 15% of target rate over 1 hour
                self.assertGreater(summary["total_vehicles"], target_vph * 0.85)
                self.assertLess(summary["total_vehicles"], target_vph * 1.15)
                self.assertGreater(summary["total_pedestrians"], 0)

                # Check all 5 vehicle types present and motorcycle is largest share
                counts = summary["vehicle_counts"]
                self.assertEqual(set(counts.keys()), set(config.VEHICLE_TYPES.keys()))
                for k, cnt in counts.items():
                    self.assertGreater(cnt, 0, f"{k} missing in {lvl}")
                self.assertGreater(counts["motorcycle"], counts["car"])

    def test_poisson_arrivals_and_non_fixed_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "med.rou.xml"
            generate_route_file(
                demand_level="medium", seed=42, duration_s=600, output_path=out
            )
            root = ET.parse(out).getroot()
            vehicles = root.findall("vehicle")
            departs = [float(v.attrib["depart"]) for v in vehicles]
            self.assertEqual(departs, sorted(departs))
            # Inter-arrival times must vary (not constant spacing)
            gaps = {round(departs[i + 1] - departs[i], 2) for i in range(len(departs) - 1)}
            self.assertGreater(len(gaps), 10)
            # Multiple distinct entry edges used
            first_edges = {v.find("route").attrib["edges"].split()[0] for v in vehicles}
            self.assertEqual(first_edges, set(config.ENTRY_EDGES))

    def test_60s_sumo_smoke_no_teleports_no_collisions_no_crossing_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "smoke.rou.xml"
            generate_route_file(
                demand_level="medium", seed=1001, duration_s=120, output_path=out
            )
            sumo_exe = Path(_SUMO_HOME) / "bin" / "sumo.exe"
            res = subprocess.run(
                [
                    str(sumo_exe),
                    "-n",
                    str(config.NET_FILE),
                    "-r",
                    str(out),
                    "--end",
                    "60",
                    "--no-step-log",
                    "true",
                    "--duration-log.disable",
                    "true",
                    "--collision.action",
                    "warn",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            combined_log = (res.stdout + "\n" + res.stderr).lower()
            self.assertNotIn("teleport", combined_log)
            self.assertNotIn("collision", combined_log)

            # TraCI check: step 60s and verify 0 collisions, 0 teleports, vehicles moving
            traci.start(
                [
                    str(sumo_exe),
                    "-n",
                    str(config.NET_FILE),
                    "-r",
                    str(out),
                    "--no-step-log",
                    "true",
                    "--duration-log.disable",
                    "true",
                ]
            )
            try:
                stopped_on_crossing = 0
                max_active = 0
                for _ in range(60):
                    traci.simulationStep()
                    self.assertEqual(traci.simulation.getCollidingVehiclesNumber(), 0)
                    self.assertEqual(traci.simulation.getStartingTeleportNumber(), 0)
                    vids = traci.vehicle.getIDList()
                    max_active = max(max_active, len(vids))
                    for vid in vids:
                        lane_id = traci.vehicle.getLaneID(vid)
                        if "_c" in lane_id and traci.vehicle.getSpeed(vid) < 0.1:
                            stopped_on_crossing += 1
                self.assertGreater(max_active, 0)
                self.assertEqual(stopped_on_crossing, 0)
            finally:
                traci.close()


if __name__ == "__main__":
    unittest.main()
