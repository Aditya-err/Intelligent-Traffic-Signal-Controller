"""
Extract road, sidewalk, crossing, and junction geometry from `corridor.net.xml`
for the HTML5 Canvas 2D replay viewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import xml.etree.ElementTree as ET

from v2 import config


def _parse_shape(shape_str: str) -> List[List[float]]:
    pts: List[List[float]] = []
    for token in shape_str.strip().split():
        if "," not in token:
            continue
        x_str, y_str = token.split(",")[:2]
        pts.append([round(float(x_str), 2), round(float(y_str), 2)])
    return pts


def extract_network_geometry(net_path: Path = config.NET_FILE) -> Dict[str, Any]:
    """Parse `corridor.net.xml` into a compact JSON-serializable geometry dict."""
    root = ET.parse(net_path).getroot()

    vehicle_lanes: List[Dict[str, Any]] = []
    sidewalk_lanes: List[Dict[str, Any]] = []
    crossings: List[Dict[str, Any]] = []
    junction_polys: List[Dict[str, Any]] = []

    all_x: List[float] = []
    all_y: List[float] = []

    for edge in root.findall("edge"):
        fn = edge.attrib.get("function", "")
        if fn == "internal":
            continue
        if fn == "crossing":
            for lane in edge.findall("lane"):
                pts = _parse_shape(lane.attrib.get("shape", ""))
                if pts:
                    crossings.append(
                        {
                            "id": lane.attrib["id"],
                            "shape": pts,
                            "width": round(float(lane.attrib.get("width", "3.0")), 2),
                        }
                    )
            continue
        if fn == "walkingarea":
            continue

        for lane in edge.findall("lane"):
            pts = _parse_shape(lane.attrib.get("shape", ""))
            if not pts:
                continue
            for x, y in pts:
                all_x.append(x)
                all_y.append(y)
            allow = lane.attrib.get("allow", "")
            width = round(float(lane.attrib.get("width", "3.5")), 2)
            item = {"id": lane.attrib["id"], "shape": pts, "width": width}
            if "pedestrian" in allow:
                sidewalk_lanes.append(item)
            else:
                vehicle_lanes.append(item)

    for junc in root.findall("junction"):
        jtype = junc.attrib.get("type", "")
        if jtype in ("internal", ""):
            continue
        shape_str = junc.attrib.get("shape", "")
        pts = _parse_shape(shape_str)
        if pts:
            junction_polys.append(
                {
                    "id": junc.attrib["id"],
                    "type": jtype,
                    "x": round(float(junc.attrib.get("x", 0.0)), 2),
                    "y": round(float(junc.attrib.get("y", 0.0)), 2),
                    "shape": pts,
                }
            )

    bounds = {
        "min_x": round(min(all_x), 1) if all_x else -300.0,
        "max_x": round(max(all_x), 1) if all_x else 900.0,
        "min_y": round(min(all_y), 1) if all_y else -260.0,
        "max_y": round(max(all_y), 1) if all_y else 280.0,
    }

    return {
        "bounds": bounds,
        "vehicle_lanes": vehicle_lanes,
        "sidewalk_lanes": sidewalk_lanes,
        "crossings": crossings,
        "junctions": junction_polys,
        "tl_coords": {
            jid: [round(x, 2), round(y, 2)]
            for jid, (x, y) in config.JUNCTION_COORDS.items()
        },
    }
