"""
Reproducible SUMO Road Network Builder for Intelligent-Traffic-RL v2.

Generates a 3-junction (J1, J2, J3) Indian Left-Hand Traffic (LHT) arterial
corridor with:
- Realistic 220m - 340m road segments
- Genuinely curved road geometry (cubic Bezier spline shapes + spreadType="center")
- 2 motorized lanes per direction + pedestrian sidewalks on every road
- Straight, left-turn, and right-turn connections under LHT rules
- Zebra pedestrian crossings on all 4 arms of every signalized junction

Usage:
    python -m v2.network.build_network
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import xml.etree.ElementTree as ET

# Allow running both as `python -m v2.network.build_network` and `python build_network.py`
_THIS_DIR = Path(__file__).resolve().parent
_V2_ROOT = _THIS_DIR.parent
if str(_V2_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT.parent))

from v2 import config  # noqa: E402

Point = Tuple[float, float]


def _find_netconvert() -> str:
    """Locate the SUMO `netconvert` executable."""
    sumo_home = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")
    candidate = Path(sumo_home) / "bin" / "netconvert.exe"
    if candidate.exists():
        return str(candidate)
    which_path = shutil.which("netconvert")
    if which_path:
        return which_path
    raise FileNotFoundError(
        "Could not find SUMO netconvert. Set SUMO_HOME or add SUMO/bin to PATH."
    )


def _cubic_bezier(
    p0: Point, p1: Point, p2: Point, p3: Point, n_points: int
) -> List[Point]:
    """Sample `n_points` along a cubic Bezier curve from p0 to p3."""
    pts: List[Point] = []
    for i in range(n_points):
        t = i / (n_points - 1)
        u = 1.0 - t
        x = (
            (u**3) * p0[0]
            + 3 * (u**2) * t * p1[0]
            + 3 * u * (t**2) * p2[0]
            + (t**3) * p3[0]
        )
        y = (
            (u**3) * p0[1]
            + 3 * (u**2) * t * p1[1]
            + 3 * u * (t**2) * p2[1]
            + (t**3) * p3[1]
        )
        pts.append((round(x, 2), round(y, 2)))
    return pts


def _offset_polyline_left(pts: Sequence[Point], offset_m: float) -> List[Point]:
    """
    Offset a polyline laterally to the LEFT of its travel direction.

    In Indian LHT (`--lefthand true`), opposing carriageways with `spreadType="center"`
    are shifted to the left of their travel direction by `CARRIAGEWAY_OFFSET_M` so
    they sit cleanly on opposite sides of the road median.
    """
    out: List[Point] = []
    n = len(pts)
    for i, (x, y) in enumerate(pts):
        if i == 0:
            dx, dy = pts[1][0] - x, pts[1][1] - y
        elif i == n - 1:
            dx, dy = x - pts[n - 2][0], y - pts[n - 2][1]
        else:
            dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        length = math.hypot(dx, dy) or 1.0
        # Left normal vector is (-dy, dx) / length
        nx, ny = -dy / length, dx / length
        out.append((round(x + offset_m * nx, 2), round(y + offset_m * ny, 2)))
    return out


def _format_shape(pts: Iterable[Point]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def _curved_corridor_centerline(
    start: Point, end: Point, bow_y: float, n_points: int
) -> List[Point]:
    """Build a smooth S/arc Bezier centerline between two corridor nodes."""
    dx = end[0] - start[0]
    p1 = (start[0] + 0.33 * dx, start[1] + bow_y)
    p2 = (start[0] + 0.67 * dx, end[1] - bow_y * 0.5)
    return _cubic_bezier(start, p1, p2, end, n_points)


def _curved_cross_centerline(
    start: Point, end: Point, bow_x: float, n_points: int
) -> List[Point]:
    """Build a gentle curved centerline for a North-South approach arm."""
    dy = end[1] - start[1]
    p1 = (start[0] + bow_x, start[1] + 0.35 * dy)
    p2 = (end[0] - bow_x * 0.4, start[1] + 0.70 * dy)
    return _cubic_bezier(start, p1, p2, end, n_points)


def write_nodes_xml(path: Path) -> None:
    """Write `corridor.nod.xml` defining 3 traffic_light junctions and 8 boundary nodes."""
    root = ET.Element("nodes")
    for jid, (x, y) in config.JUNCTION_COORDS.items():
        ET.SubElement(
            root,
            "node",
            id=jid,
            x=f"{x:.2f}",
            y=f"{y:.2f}",
            type="traffic_light",
            tlType="static",
        )
    for bid, (x, y) in config.BOUNDARY_COORDS.items():
        ET.SubElement(
            root,
            "node",
            id=bid,
            x=f"{x:.2f}",
            y=f"{y:.2f}",
            type="priority",
        )
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _add_two_way_road(
    root: ET.Element,
    edge_ab: str,
    edge_ba: str,
    node_a: str,
    node_b: str,
    centerline_ab: List[Point],
    num_lanes: int,
    speed_mps: float,
    priority: int,
) -> None:
    """Add opposing LHT carriageways with `spreadType='center'` and sidewalks."""
    shape_ab = _offset_polyline_left(centerline_ab, config.CARRIAGEWAY_OFFSET_M)
    centerline_ba = list(reversed(centerline_ab))
    shape_ba = _offset_polyline_left(centerline_ba, config.CARRIAGEWAY_OFFSET_M)

    for eid, frm, to, shp in (
        (edge_ab, node_a, node_b, shape_ab),
        (edge_ba, node_b, node_a, shape_ba),
    ):
        ET.SubElement(
            root,
            "edge",
            id=eid,
            **{"from": frm},
            to=to,
            priority=str(priority),
            numLanes=str(num_lanes),
            speed=f"{speed_mps:.2f}",
            width=f"{config.VEHICLE_LANE_WIDTH_M:.2f}",
            sidewalkWidth=f"{config.SIDEWALK_WIDTH_M:.2f}",
            spreadType="center",
            shape=_format_shape(shp),
            disallow="pedestrian",
        )


def write_edges_xml(path: Path) -> None:
    """Write `corridor.edg.xml` with curved corridor links and cross-street arms."""
    root = ET.Element("edges")
    all_nodes = {**config.JUNCTION_COORDS, **config.BOUNDARY_COORDS}
    n_pts = config.CURVE_SAMPLE_POINTS

    # Main East-West arterial corridor segments (220m - 340m, gently curved)
    corridor_segments = [
        ("W1_J1", "J1_W1", "W1", "J1", 28.0),
        ("J1_J2", "J2_J1", "J1", "J2", 42.0),
        ("J2_J3", "J3_J2", "J2", "J3", -42.0),
        ("J3_E3", "E3_J3", "J3", "E3", -28.0),
    ]
    for e_ab, e_ba, n_a, n_b, bow_y in corridor_segments:
        cl = _curved_corridor_centerline(all_nodes[n_a], all_nodes[n_b], bow_y, n_pts)
        _add_two_way_road(
            root,
            e_ab,
            e_ba,
            n_a,
            n_b,
            cl,
            num_lanes=config.CORRIDOR_VEHICLE_LANES,
            speed_mps=config.CORRIDOR_SPEED_MPS,
            priority=3,
        )

    # North-South cross streets at J1, J2, J3 (225m - 235m each, gentle curves)
    cross_segments = [
        ("N1_J1", "J1_N1", "N1", "J1", 16.0),
        ("S1_J1", "J1_S1", "S1", "J1", -16.0),
        ("N2_J2", "J2_N2", "N2", "J2", -18.0),
        ("S2_J2", "J2_S2", "S2", "J2", 18.0),
        ("N3_J3", "J3_N3", "N3", "J3", 16.0),
        ("S3_J3", "J3_S3", "S3", "J3", -16.0),
    ]
    for e_ab, e_ba, n_a, n_b, bow_x in cross_segments:
        cl = _curved_cross_centerline(all_nodes[n_a], all_nodes[n_b], bow_x, n_pts)
        _add_two_way_road(
            root,
            e_ab,
            e_ba,
            n_a,
            n_b,
            cl,
            num_lanes=config.CROSS_STREET_VEHICLE_LANES,
            speed_mps=config.CROSS_STREET_SPEED_MPS,
            priority=2,
        )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_netccfg(path: Path) -> None:
    """Write `corridor.netccfg` so the network can be rebuilt directly with netconvert."""
    root = ET.Element("configuration")

    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "node-files", value=config.NOD_FILE.name)
    ET.SubElement(inp, "edge-files", value=config.EDG_FILE.name)

    out = ET.SubElement(root, "output")
    ET.SubElement(out, "output-file", value=config.NET_FILE.name)

    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "lefthand", value="true")
    ET.SubElement(proc, "no-turnarounds", value="true")
    ET.SubElement(proc, "crossings.guess", value="true")
    ET.SubElement(proc, "walkingareas", value="true")
    ET.SubElement(proc, "junctions.corner-detail", value="5")

    tls = ET.SubElement(root, "tls_building")
    ET.SubElement(tls, "tls.default-type", value="static")
    ET.SubElement(tls, "tls.green.time", value="25")
    ET.SubElement(tls, "tls.yellow.time", value=str(config.YELLOW_S))
    ET.SubElement(tls, "tls.allred.time", value=str(config.ALL_RED_S))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def build_network(verbose: bool = True) -> Path:
    """Generate XML inputs and compile `corridor.net.xml` using SUMO netconvert."""
    config.NETWORK_DIR.mkdir(parents=True, exist_ok=True)
    write_nodes_xml(config.NOD_FILE)
    write_edges_xml(config.EDG_FILE)
    write_netccfg(config.NETCCFG_FILE)

    netconvert = _find_netconvert()
    cmd = [netconvert, "-c", str(config.NETCCFG_FILE)]
    res = subprocess.run(
        cmd,
        cwd=str(config.NETWORK_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"netconvert failed (exit {res.returncode}):\n{res.stdout}\n{res.stderr}"
        )
    if verbose:
        print(f"[build_network] Compiled {config.NET_FILE}")
        if res.stdout.strip():
            print(res.stdout.strip())
        if res.stderr.strip():
            print(f"[build_network warnings]\n{res.stderr.strip()}")
    return config.NET_FILE


if __name__ == "__main__":
    build_network(verbose=True)
