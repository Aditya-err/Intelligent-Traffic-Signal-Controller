"""
SUMO Actuated Baseline Controller for Intelligent-Traffic-RL v2.

Uses SUMO's built-in `type="actuated"` traffic light logic (gap-out vehicle
actuation) with the same safety timings from `v2/config.py`:
- Min green: 10 s (`MIN_GREEN_S`)
- Max green: 60 s (`MAX_GREEN_S`)
- Yellow: 3 s (`YELLOW_S`)
- All-Red: 2 s (`ALL_RED_S`)
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from v2 import config


def build_actuated_tls_additional(output_path: Path) -> Path:
    """
    Generate a SUMO `.add.xml` file switching J1, J2, J3 to `type="actuated"`
    while preserving the phase state strings from `corridor.net.xml`.
    """
    net_root = ET.parse(config.NET_FILE).getroot()
    add_root = ET.Element("additional")

    main_min_dur = max(1, config.MIN_GREEN_S - config.PED_CLEARANCE_S)
    main_max_dur = max(main_min_dur, config.MAX_GREEN_S - config.PED_CLEARANCE_S)

    for tl in net_root.findall("tlLogic"):
        jid = tl.attrib["id"]
        if jid not in config.JUNCTION_COORDS:
            continue
        phases = tl.findall("phase")
        if len(phases) != 8:
            continue

        new_tl = ET.SubElement(
            add_root,
            "tlLogic",
            id=jid,
            type="actuated",
            programID="actuated",
            offset="0",
        )
        ET.SubElement(new_tl, "param", key="max-gap", value=str(config.ACTUATED_MAX_GAP_S))
        ET.SubElement(
            new_tl, "param", key="detector-gap", value=str(config.ACTUATED_DETECTOR_GAP_S)
        )

        for idx, ph in enumerate(phases):
            state = ph.attrib["state"]
            if idx in (0, 4):
                # Main green phase (actuated between minDur and maxDur)
                ET.SubElement(
                    new_tl,
                    "phase",
                    duration=str(config.FIXED_TIME_NS_GREEN_S - config.PED_CLEARANCE_S),
                    minDur=str(main_min_dur),
                    maxDur=str(main_max_dur),
                    state=state,
                )
            elif idx in (1, 5):
                # Fixed pedestrian clearance (completes MIN_GREEN_S)
                ET.SubElement(
                    new_tl,
                    "phase",
                    duration=str(config.PED_CLEARANCE_S),
                    state=state,
                )
            elif idx in (2, 6):
                # Fixed yellow
                ET.SubElement(
                    new_tl,
                    "phase",
                    duration=str(config.YELLOW_S),
                    state=state,
                )
            else:
                # Fixed all-red
                ET.SubElement(
                    new_tl,
                    "phase",
                    duration=str(config.ALL_RED_S),
                    state=state,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(add_root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


class ActuatedController:
    """SUMO built-in actuated traffic signal controller for J1, J2, J3."""

    name = "actuated"

    def __init__(self) -> None:
        self._add_file = config.NETWORK_DIR / "actuated_tls.add.xml"
        build_actuated_tls_additional(self._add_file)

    def extra_sumo_args(self) -> list[str]:
        """Load the actuated TLS additional XML file into SUMO."""
        return ["--additional-files", str(self._add_file)]

    def on_simulation_start(self) -> None:
        """SUMO activates `programID='actuated'` from the additional file."""
        return None

    def step(self, sim_time_s: float) -> None:
        """SUMO handles actuated phase switching internally."""
        return None
