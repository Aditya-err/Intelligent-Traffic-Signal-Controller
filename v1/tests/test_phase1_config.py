"""Phase 1 configuration and structural validation tests (stdlib only)."""

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase1ProjectStructure(unittest.TestCase):
    """Verify required directories, GitHub files, and SUMO XML configs."""

    REQUIRED_DIRS = [
        "data",
        "environment",
        "algorithms",
        "simulation/network",
        "simulation/routes",
        "simulation/config",
        "evaluation",
        "dashboard",
        "models",
        "reports",
        "tests",
    ]

    REQUIRED_FILES = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "simulation/network/intersection.nod.xml",
        "simulation/network/intersection.edg.xml",
        "simulation/network/intersection.con.xml",
        "simulation/network/intersection.tll.xml",
        "simulation/network/intersection.netccfg",
        "simulation/routes/intersection.rou.xml",
        "simulation/config/intersection.sumocfg",
    ]

    def test_required_directories_exist(self) -> None:
        for rel_dir in self.REQUIRED_DIRS:
            path = PROJECT_ROOT / rel_dir
            self.assertTrue(path.is_dir(), f"Missing directory: {rel_dir}")

    def test_required_files_exist(self) -> None:
        for rel_file in self.REQUIRED_FILES:
            path = PROJECT_ROOT / rel_file
            self.assertTrue(path.is_file(), f"Missing file: {rel_file}")


class TestSumoXmlConfigs(unittest.TestCase):
    """Validate XML well-formedness and Indian LHT / signal safety rules."""

    def _parse(self, rel_path: str) -> ET.Element:
        tree = ET.parse(PROJECT_ROOT / rel_path)
        return tree.getroot()

    def test_all_xml_files_are_well_formed(self) -> None:
        xml_files = [
            "simulation/network/intersection.nod.xml",
            "simulation/network/intersection.edg.xml",
            "simulation/network/intersection.con.xml",
            "simulation/network/intersection.tll.xml",
            "simulation/network/intersection.netccfg",
            "simulation/routes/intersection.rou.xml",
            "simulation/config/intersection.sumocfg",
        ]
        for rel_path in xml_files:
            root = self._parse(rel_path)
            self.assertIsNotNone(root.tag, f"Empty root tag in {rel_path}")

    def test_netconvert_enforces_lefthand_traffic(self) -> None:
        root = self._parse("simulation/network/intersection.netccfg")
        lefthand = root.find("./processing/lefthand")
        self.assertIsNotNone(lefthand, "Missing <lefthand> setting in netccfg")
        self.assertEqual(lefthand.attrib.get("value"), "true")

    def test_edges_and_connections_consistency(self) -> None:
        edg_root = self._parse("simulation/network/intersection.edg.xml")
        con_root = self._parse("simulation/network/intersection.con.xml")
        rou_root = self._parse("simulation/routes/intersection.rou.xml")

        edge_ids = {e.attrib["id"] for e in edg_root.findall("edge")}
        self.assertEqual(len(edge_ids), 8)

        connections = con_root.findall("connection")
        self.assertEqual(len(connections), 12)
        for conn in connections:
            self.assertIn(conn.attrib["from"], edge_ids)
            self.assertIn(conn.attrib["to"], edge_ids)

        for route in rou_root.findall("route"):
            for edge_id in route.attrib["edges"].split():
                self.assertIn(edge_id, edge_ids)

    def test_indian_vehicle_types_and_distribution(self) -> None:
        rou_root = self._parse("simulation/routes/intersection.rou.xml")
        vtype_ids = {vt.attrib["id"] for vt in rou_root.findall("vType")}
        expected = {"car", "motorcycle", "auto_rickshaw", "bus", "truck"}
        self.assertTrue(expected.issubset(vtype_ids))

        dist = rou_root.find("./vTypeDistribution[@id='indian_mix']")
        self.assertIsNotNone(dist)
        dist_refs = dist.findall("vType")
        self.assertEqual({d.attrib["refId"] for d in dist_refs}, expected)
        total_prob = sum(float(d.attrib["probability"]) for d in dist_refs)
        self.assertAlmostEqual(total_prob, 1.0, places=6)

    def test_signal_plan_has_yellow_and_all_red_clearance(self) -> None:
        tll_root = self._parse("simulation/network/intersection.tll.xml")
        con_root = self._parse("simulation/network/intersection.con.xml")
        num_links = len(con_root.findall("connection"))

        phases = tll_root.findall("./tlLogic/phase")
        self.assertEqual(len(phases), 12)

        states = [p.attrib["state"] for p in phases]
        for state in states:
            self.assertEqual(len(state), num_links)

        has_yellow = any("y" in s for s in states)
        has_all_red = any(set(s) == {"r"} for s in states)
        self.assertTrue(has_yellow, "Signal plan must include yellow clearance phases")
        self.assertTrue(has_all_red, "Signal plan must include all-red clearance phases")


if __name__ == "__main__":
    unittest.main()
