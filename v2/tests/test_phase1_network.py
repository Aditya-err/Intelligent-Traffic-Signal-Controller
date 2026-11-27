"""Phase 1 tests: verify v2 3-junction curved LHT SUMO road network."""

import os
import subprocess
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from v2 import config


class TestPhase1Network(unittest.TestCase):
    """Validate generated corridor.net.xml geometry, LHT rules, crossings, and SUMO load."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.net_path = config.NET_FILE
        assert cls.net_path.exists(), f"Missing compiled network: {cls.net_path}"
        cls.tree = ET.parse(cls.net_path)
        cls.root = cls.tree.getroot()

    def test_required_network_files_exist(self) -> None:
        for path in (
            config.NOD_FILE,
            config.EDG_FILE,
            config.NETCCFG_FILE,
            config.NET_FILE,
        ):
            self.assertTrue(path.exists(), f"Expected file missing: {path}")

    def test_lefthand_enabled(self) -> None:
        self.assertEqual(self.root.attrib.get("lefthand"), "true")

    def test_three_signalized_junctions(self) -> None:
        tl_ids = sorted(el.attrib["id"] for el in self.root.findall("tlLogic"))
        self.assertEqual(tl_ids, ["J1", "J2", "J3"])
        for tl in self.root.findall("tlLogic"):
            phases = tl.findall("phase")
            self.assertGreaterEqual(len(phases), 4)

    def test_road_lengths_lanes_sidewalks_and_curves(self) -> None:
        normal_edges = [
            el
            for el in self.root.findall("edge")
            if el.attrib.get("function") not in ("internal", "walkingarea", "crossing")
        ]
        # 4 corridor segments * 2 directions + 6 cross arms * 2 directions = 20 edges
        self.assertEqual(len(normal_edges), 20)

        for edge in normal_edges:
            eid = edge.attrib["id"]
            lanes = edge.findall("lane")
            ped_lanes = [l for l in lanes if "pedestrian" in l.attrib.get("allow", "")]
            veh_lanes = [l for l in lanes if "pedestrian" not in l.attrib.get("allow", "")]

            self.assertEqual(len(veh_lanes), 2, f"{eid} should have 2 vehicle lanes")
            self.assertEqual(len(ped_lanes), 1, f"{eid} should have 1 sidewalk lane")

            max_len = max(float(l.attrib["length"]) for l in lanes)
            self.assertGreaterEqual(max_len, 200.0, f"{eid} too short: {max_len}")
            self.assertLessEqual(max_len, 400.0, f"{eid} too long: {max_len}")

            shape_str = edge.attrib.get("shape") or veh_lanes[0].attrib.get("shape", "")
            pts = shape_str.split()
            self.assertGreater(len(pts), 2, f"{eid} is not curved (only {len(pts)} pts)")

    def test_zebra_crossings_and_turning_directions(self) -> None:
        crossings = [
            el for el in self.root.findall("edge") if el.attrib.get("function") == "crossing"
        ]
        # At least 12 zebra crossings (4 arms at each of J1, J2, J3)
        self.assertGreaterEqual(len(crossings), 12)
        for jid in ("J1", "J2", "J3"):
            j_crossings = [c for c in crossings if c.attrib["id"].startswith(f":{jid}_")]
            self.assertGreaterEqual(len(j_crossings), 4, f"{jid} should have >=4 zebra crossings")

        # Verify straight, left, and right connections exist at every junction
        for jid in ("J1", "J2", "J3"):
            dirs = {
                c.attrib.get("dir")
                for c in self.root.findall("connection")
                if c.attrib.get("tl") == jid
            }
            self.assertTrue({"s", "l", "r"}.issubset(dirs), f"{jid} missing turns: {dirs}")

    def test_sumo_smoke_load_60s(self) -> None:
        sumo_home = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")
        sumo_exe = Path(sumo_home) / "bin" / "sumo.exe"
        self.assertTrue(sumo_exe.exists(), f"SUMO binary not found: {sumo_exe}")

        res = subprocess.run(
            [
                str(sumo_exe),
                "-n",
                str(self.net_path),
                "--end",
                "60",
                "--no-step-log",
                "true",
                "--duration-log.disable",
                "true",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            res.returncode,
            0,
            f"SUMO failed on corridor.net.xml:\n{res.stdout}\n{res.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
