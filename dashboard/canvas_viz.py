"""HTML5 Canvas 2D top-down renderer for the live Indian LHT SUMO simulation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dashboard.live_sim import SimSnapshot


CANVAS_WIDTH = 920
CANVAS_HEIGHT = 620
PX_PER_METER = 4.2
VIEW_HALF_X_M = (CANVAS_WIDTH / 2.0) / PX_PER_METER   # ~109.5 m
VIEW_HALF_Y_M = (CANVAS_HEIGHT / 2.0) / PX_PER_METER  # ~73.8 m


def get_diagnostic_rows(snapshot: "SimSnapshot") -> list[dict[str, Any]]:
    """Return per-vehicle TraCI diagnostics and whether each is inside viewport."""
    rows: list[dict[str, Any]] = []
    for v in snapshot.vehicles:
        in_view = (
            abs(v["x"]) <= VIEW_HALF_X_M + 5.0
            and abs(v["y"]) <= VIEW_HALF_Y_M + 5.0
        )
        rows.append(
            {
                "vehicle_id": v["id"],
                "type": v["vtype"],
                "lane_id": v.get("lane_id", ""),
                "x": round(v["x"], 2),
                "y": round(v["y"], 2),
                "angle": round(v.get("angle", 0.0), 1),
                "rendered_status": "RENDERED" if in_view else "OUT_OF_VIEWPORT",
            }
        )
    return rows


def build_live_canvas_html(snapshot: "SimSnapshot") -> str:
    """Build a self-contained HTML5 Canvas rendering SUMO vehicles & 3-lamp signals."""
    payload = {
        "sim_time": snapshot.sim_time,
        "scenario": snapshot.scenario,
        "controller": snapshot.controller_name,
        "tls_state": snapshot.tls_state.ljust(12, "r"),
        "vehicles": [
            {
                "id": v["id"],
                "x": v["x"],
                "y": v["y"],
                "angle": v.get("angle", 0.0),
                "vtype": v["vtype"],
                "speed": v["speed"],
            }
            for v in snapshot.vehicles
            if abs(v["x"]) <= VIEW_HALF_X_M + 8.0 and abs(v["y"]) <= VIEW_HALF_Y_M + 8.0
        ],
    }
    data_json = json.dumps(payload)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  html, body {{
    margin: 0;
    padding: 0;
    background: #0b1120;
    overflow: hidden;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }}
  #wrap {{
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
  }}
  canvas {{
    width: 100%;
    max-width: {CANVAS_WIDTH}px;
    height: auto;
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.25);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.45);
    display: block;
  }}
</style>
</head>
<body>
<div id="wrap">
  <canvas id="simCanvas" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}"></canvas>
</div>
<script>
(function() {{
  const DATA = {data_json};
  const canvas = document.getElementById("simCanvas");
  const ctx = canvas.getContext("2d");
  const W = {CANVAS_WIDTH};
  const H = {CANVAS_HEIGHT};
  const CX = W / 2;
  const CY = H / 2;
  const SCALE = {PX_PER_METER};

  const toPxX = (x) => CX + x * SCALE;
  const toPxY = (y) => CY - y * SCALE;

  function drawRoadBackground() {{
    // 1. Lush green roadside grass with subtle turf pattern
    ctx.fillStyle = "#2d5a27";
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(255,255,255,0.035)";
    ctx.lineWidth = 1;
    for (let gx = 0; gx < W; gx += 28) {{
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }}
    for (let gy = 0; gy < H; gy += 28) {{
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }}

    const roadHalf = 7.4 * SCALE;   // ~31px
    const walkPad = 2.4 * SCALE;    // sidewalk width ~10px
    const stopPx = 11.2 * SCALE;    // ~47px

    // 2. Concrete sidewalks
    ctx.fillStyle = "#94a3b8";
    ctx.fillRect(CX - roadHalf - walkPad, 0, (roadHalf + walkPad) * 2, H);
    ctx.fillRect(0, CY - roadHalf - walkPad, W, (roadHalf + walkPad) * 2);

    // Curb edge lines
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 2;
    ctx.strokeRect(CX - roadHalf - walkPad, -2, (roadHalf + walkPad) * 2, H + 4);
    ctx.strokeRect(-2, CY - roadHalf - walkPad, W + 4, (roadHalf + walkPad) * 2);

    // 3. Dark asphalt roads
    ctx.fillStyle = "#2f353f";
    ctx.fillRect(CX - roadHalf, 0, roadHalf * 2, H);
    ctx.fillRect(0, CY - roadHalf, W, roadHalf * 2);

    // Central junction box
    ctx.fillStyle = "#353c47";
    ctx.fillRect(CX - roadHalf, CY - roadHalf, roadHalf * 2, roadHalf * 2);

    // 4. White road edge markings
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1.6;
    [CX - roadHalf + 2, CX + roadHalf - 2].forEach(x => {{
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, CY - stopPx); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x, CY + stopPx); ctx.lineTo(x, H); ctx.stroke();
    }});
    [CY - roadHalf + 2, CY + roadHalf - 2].forEach(y => {{
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(CX - stopPx, y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(CX + stopPx, y); ctx.lineTo(W, y); ctx.stroke();
    }});

    // 5. Dashed white lane dividers (+-3.5m)
    ctx.save();
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([10, 10]);
    [-3.5 * SCALE, 3.5 * SCALE].forEach(off => {{
      ctx.beginPath(); ctx.moveTo(CX + off, 0); ctx.lineTo(CX + off, CY - stopPx); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(CX + off, CY + stopPx); ctx.lineTo(CX + off, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, CY + off); ctx.lineTo(CX - stopPx, CY + off); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(CX + stopPx, CY + off); ctx.lineTo(W, CY + off); ctx.stroke();
    }});
    ctx.restore();

    // 6. Double yellow center lines
    ctx.strokeStyle = "#facc15";
    ctx.lineWidth = 2;
    [-2.2, 2.2].forEach(d => {{
      ctx.beginPath(); ctx.moveTo(CX + d, 0); ctx.lineTo(CX + d, CY - stopPx); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(CX + d, CY + stopPx); ctx.lineTo(CX + d, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, CY + d); ctx.lineTo(CX - stopPx, CY + d); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(CX + stopPx, CY + d); ctx.lineTo(W, CY + d); ctx.stroke();
    }});

    // 7. Zebra crossings on all 4 approaches
    ctx.fillStyle = "#f8fafc";
    const stripeW = 4;
    const stripeLen = 14;
    for (let x = CX - roadHalf + 4; x < CX + roadHalf - 4; x += 9) {{
      ctx.fillRect(x, CY - stopPx + 2, stripeW, stripeLen);
      ctx.fillRect(x, CY + stopPx - stripeLen - 2, stripeW, stripeLen);
    }}
    for (let y = CY - roadHalf + 4; y < CY + roadHalf - 4; y += 9) {{
      ctx.fillRect(CX - stopPx + 2, y, stripeLen, stripeW);
      ctx.fillRect(CX + stopPx - stripeLen - 2, y, stripeLen, stripeW);
    }}

    // 8. Indian LHT directional labels
    ctx.fillStyle = "rgba(241, 245, 249, 0.78)";
    ctx.font = "bold 11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("N → S (Keep Left)", CX + roadHalf + 66, 20);
    ctx.fillText("S → N (Keep Left)", CX - roadHalf - 66, H - 12);
    ctx.fillText("W → E (Keep Left)", 72, CY - roadHalf - 16);
    ctx.fillText("E → W (Keep Left)", W - 72, CY + roadHalf + 24);
  }}

  function activeLamp(sliceStr) {{
    if (sliceStr.includes("G") || sliceStr.includes("g")) return "green";
    if (sliceStr.includes("y")) return "yellow";
    return "red";
  }}

  function drawSignalHead(label, px, py, sliceStr) {{
    const active = activeLamp(sliceStr);
    const hw = 26, hh = 68;

    // Housing
    ctx.save();
    ctx.fillStyle = "#0f172a";
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(px - hw / 2, py - hh / 2, hw, hh, 6);
    ctx.fill();
    ctx.stroke();

    // Approach tag
    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(label, px, py - hh / 2 - 5);

    const lamps = [
      ["red", py - 20, "#ff2e2e", "#450a0a"],
      ["yellow", py, "#ffd60a", "#422006"],
      ["green", py + 20, "#00e676", "#052e16"],
    ];
    lamps.forEach(([name, ly, bright, dim]) => {{
      const isOn = active === name;
      ctx.beginPath();
      ctx.arc(px, ly, 8.2, 0, Math.PI * 2);
      ctx.fillStyle = isOn ? bright : dim;
      if (isOn) {{
        ctx.shadowColor = bright;
        ctx.shadowBlur = 14;
      }} else {{
        ctx.shadowBlur = 0;
      }}
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = isOn ? "#ffffff" : "#334155";
      ctx.lineWidth = isOn ? 1.8 : 1;
      ctx.stroke();
    }});
    ctx.restore();
  }}

  function drawAllSignals() {{
    const tls = DATA.tls_state;
    const off = 66;
    drawSignalHead("N", CX + off, CY - off - 12, tls.slice(6, 9));
    drawSignalHead("S", CX - off, CY + off + 12, tls.slice(0, 3));
    drawSignalHead("E", CX + off + 16, CY + off - 4, tls.slice(3, 6));
    drawSignalHead("W", CX - off - 16, CY - off + 4, tls.slice(9, 12));
  }}

  function drawVehicle(v) {{
    const px = toPxX(v.x);
    const py = toPxY(v.y);
    const rad = (v.angle * Math.PI) / 180.0;

    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(rad);

    if (v.vtype === "motorcycle") {{
      // Top-down motorcycle + rider
      ctx.fillStyle = "#111827";
      ctx.fillRect(-2, -6, 4, 12);
      ctx.fillStyle = "#f97316";
      ctx.beginPath();
      ctx.roundRect(-2.5, -4, 5, 8, 2);
      ctx.fill();
      ctx.fillStyle = "#f8fafc";
      ctx.beginPath();
      ctx.arc(0, 0, 2.3, 0, Math.PI * 2);
      ctx.fill();
    }} else if (v.vtype === "auto_rickshaw") {{
      // Indian green/yellow auto-rickshaw
      ctx.fillStyle = "#065f46";
      ctx.beginPath();
      ctx.roundRect(-4.5, -7.5, 9, 15, 3);
      ctx.fill();
      ctx.fillStyle = "#facc15";
      ctx.beginPath();
      ctx.roundRect(-3.8, -3.5, 7.6, 9, 2);
      ctx.fill();
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(-3.2, -6.5, 6.4, 2.2);
    }} else if (v.vtype === "bus") {{
      // City bus
      ctx.fillStyle = "#dc2626";
      ctx.beginPath();
      ctx.roundRect(-6.5, -20, 13, 40, 3);
      ctx.fill();
      ctx.strokeStyle = "#fecaca";
      ctx.lineWidth = 1;
      ctx.strokeRect(-6.5, -20, 13, 40);
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(-5, -18, 10, 4);
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(-4, -8, 8, 6);
      ctx.fillRect(-4, 4, 8, 6);
    }} else if (v.vtype === "truck") {{
      // Goods truck (cab + cargo box)
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.roundRect(-6, -15, 12, 8, 2);
      ctx.fill();
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(-4.5, -14, 9, 2.5);
      ctx.fillStyle = "#cbd5e1";
      ctx.fillRect(-6, -6, 12, 21);
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 1;
      ctx.strokeRect(-6, -6, 12, 21);
    }} else {{
      // Car (sedan/hatchback)
      const carCols = ["#ef4444", "#3b82f6", "#e2e8f0", "#f59e0b"];
      let hash = 0;
      for (let i = 0; i < v.id.length; i++) hash += v.id.charCodeAt(i);
      ctx.fillStyle = carCols[hash % carCols.length];
      ctx.beginPath();
      ctx.roundRect(-4.8, -9.5, 9.6, 19, 3);
      ctx.fill();
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 1;
      ctx.stroke();
      // Windshield & rear window
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(-3.6, -5.5, 7.2, 3.2);
      ctx.fillRect(-3.4, 3.2, 6.8, 2.5);
      // Headlights
      ctx.fillStyle = "#fef08a";
      ctx.fillRect(-4, -9.5, 2, 1.5);
      ctx.fillRect(2, -9.5, 2, 1.5);
    }}

    ctx.restore();
  }}

  function drawLegendAndHud() {{
    // Compact legend bar at top-left
    ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
    ctx.beginPath();
    ctx.roundRect(10, 10, 200, 96, 8);
    ctx.fill();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 11px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`SUMO Time: ${{DATA.sim_time}} s`, 20, 28);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText(`Rendered Vehicles: ${{DATA.vehicles.length}}`, 20, 45);
    ctx.fillText(`TLS: ${{DATA.tls_state}}`, 20, 62);

    const items = [
      ["Car", "#3b82f6"],
      ["Moto", "#f97316"],
      ["Auto", "#facc15"],
      ["Bus", "#dc2626"],
      ["Truck", "#cbd5e1"],
    ];
    let ix = 20;
    items.forEach(([lbl, col]) => {{
      ctx.fillStyle = col;
      ctx.fillRect(ix, 75, 8, 8);
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "9px system-ui, sans-serif";
      ctx.fillText(lbl, ix + 11, 83);
      ix += 37;
    }});
  }}

  drawRoadBackground();
  DATA.vehicles.forEach(drawVehicle);
  drawAllSignals();
  drawLegendAndHud();
}})();
</script>
</body>
</html>
"""
