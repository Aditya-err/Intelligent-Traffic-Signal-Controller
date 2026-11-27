"""HTML5 Canvas 2D top-down renderer for the live Indian LHT SUMO simulation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dashboard.live_sim import SimSnapshot


CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 680
PX_PER_METER = 8.4
VIEW_HALF_X_M = (CANVAS_WIDTH / 2.0) / PX_PER_METER   # ~59.5 m
VIEW_HALF_Y_M = (CANVAS_HEIGHT / 2.0) / PX_PER_METER  # ~40.5 m


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


def build_live_canvas_html(snapshot: "SimSnapshot", duration_ms: int = 420) -> str:
    """Build a self-contained HTML5 Canvas with smooth TraCI interpolation & countdowns."""
    prev_map = {v["id"]: v for v in snapshot.prev_vehicles}
    vehs_payload = []
    for v in snapshot.vehicles:
        if abs(v["x"]) > VIEW_HALF_X_M + 8.0 or abs(v["y"]) > VIEW_HALF_Y_M + 8.0:
            continue
        pv = prev_map.get(v["id"])
        vehs_payload.append(
            {
                "id": v["id"],
                "x0": pv["x"] if pv is not None else v["x"],
                "y0": pv["y"] if pv is not None else v["y"],
                "a0": pv.get("angle", v.get("angle", 0.0)) if pv is not None else v.get("angle", 0.0),
                "x1": v["x"],
                "y1": v["y"],
                "a1": v.get("angle", 0.0),
                "vtype": v["vtype"],
                "speed": v["speed"],
            }
        )

    payload = {
        "sim_time": snapshot.sim_time,
        "scenario": snapshot.scenario,
        "controller": snapshot.controller_name,
        "tls_state": snapshot.tls_state.ljust(12, "r"),
        "ns_countdown": snapshot.ns_countdown,
        "ew_countdown": snapshot.ew_countdown,
        "duration_ms": duration_ms,
        "vehicles": vehs_payload,
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
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.28);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.5);
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

  function lerpAngle(a0, a1, t) {{
    let diff = ((a1 - a0 + 540) % 360) - 180;
    return a0 + diff * t;
  }}

  function drawRoadBackground() {{
    // 1. Lush green roadside grass with subtle turf pattern
    ctx.fillStyle = "#2d5a27";
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(255,255,255,0.035)";
    ctx.lineWidth = 1;
    for (let gx = 0; gx < W; gx += 32) {{
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }}
    for (let gy = 0; gy < H; gy += 32) {{
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }}

    const roadHalf = 7.4 * SCALE;   // ~45.9px (92px total road width)
    const walkPad = 2.6 * SCALE;    // ~16px sidewalk
    const stopPx = 11.4 * SCALE;    // ~70.7px

    // 2. Concrete sidewalks
    ctx.fillStyle = "#94a3b8";
    ctx.fillRect(CX - roadHalf - walkPad, 0, (roadHalf + walkPad) * 2, H);
    ctx.fillRect(0, CY - roadHalf - walkPad, W, (roadHalf + walkPad) * 2);

    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 2;
    ctx.strokeRect(CX - roadHalf - walkPad, -2, (roadHalf + walkPad) * 2, H + 4);
    ctx.strokeRect(-2, CY - roadHalf - walkPad, W + 4, (roadHalf + walkPad) * 2);

    // 3. Dark asphalt roads
    ctx.fillStyle = "#2f353f";
    ctx.fillRect(CX - roadHalf, 0, roadHalf * 2, H);
    ctx.fillRect(0, CY - roadHalf, W, roadHalf * 2);

    // Central junction box + subtle curved turning guide arcs
    ctx.fillStyle = "#363d49";
    ctx.fillRect(CX - roadHalf, CY - roadHalf, roadHalf * 2, roadHalf * 2);
    ctx.save();
    ctx.strokeStyle = "rgba(226, 232, 240, 0.14)";
    ctx.lineWidth = 1.4;
    ctx.setLineDash([5, 6]);
    const turnR = 6.0 * SCALE;
    [
      [CX + roadHalf, CY - roadHalf, Math.PI * 0.5, Math.PI],
      [CX - roadHalf, CY - roadHalf, 0, Math.PI * 0.5],
      [CX - roadHalf, CY + roadHalf, Math.PI * 1.5, Math.PI * 2],
      [CX + roadHalf, CY + roadHalf, Math.PI, Math.PI * 1.5],
    ].forEach(([ax, ay, sA, eA]) => {{
      ctx.beginPath();
      ctx.arc(ax, ay, turnR, sA, eA);
      ctx.stroke();
    }});
    ctx.restore();

    // 4. White road edge markings
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 2;
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
    ctx.lineWidth = 1.8;
    ctx.setLineDash([12, 12]);
    [-3.5 * SCALE, 3.5 * SCALE].forEach(off => {{
      ctx.beginPath(); ctx.moveTo(CX + off, 0); ctx.lineTo(CX + off, CY - stopPx); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(CX + off, CY + stopPx); ctx.lineTo(CX + off, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, CY + off); ctx.lineTo(CX - stopPx, CY + off); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(CX + stopPx, CY + off); ctx.lineTo(W, CY + off); ctx.stroke();
    }});
    ctx.restore();

    // 6. Double yellow center lines
    ctx.strokeStyle = "#facc15";
    ctx.lineWidth = 2.4;
    [-2.5, 2.5].forEach(d => {{
      ctx.beginPath(); ctx.moveTo(CX + d, 0); ctx.lineTo(CX + d, CY - stopPx); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(CX + d, CY + stopPx); ctx.lineTo(CX + d, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, CY + d); ctx.lineTo(CX - stopPx, CY + d); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(CX + stopPx, CY + d); ctx.lineTo(W, CY + d); ctx.stroke();
    }});

    // 7. Zebra crossings on all 4 approaches
    ctx.fillStyle = "#f8fafc";
    const stripeW = 5;
    const stripeLen = 18;
    for (let x = CX - roadHalf + 5; x < CX + roadHalf - 5; x += 11) {{
      ctx.fillRect(x, CY - stopPx + 3, stripeW, stripeLen);
      ctx.fillRect(x, CY + stopPx - stripeLen - 3, stripeW, stripeLen);
    }}
    for (let y = CY - roadHalf + 5; y < CY + roadHalf - 5; y += 11) {{
      ctx.fillRect(CX - stopPx + 3, y, stripeLen, stripeW);
      ctx.fillRect(CX + stopPx - stripeLen - 3, y, stripeLen, stripeW);
    }}

    // 8. Indian LHT directional labels
    ctx.fillStyle = "rgba(248, 250, 252, 0.85)";
    ctx.font = "bold 12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("N → S (Keep Left)", CX + roadHalf + 82, 24);
    ctx.fillText("S → N (Keep Left)", CX - roadHalf - 82, H - 14);
    ctx.fillText("W → E (Keep Left)", 86, CY - roadHalf - 22);
    ctx.fillText("E → W (Keep Left)", W - 86, CY + roadHalf + 30);
  }}

  function activeLamp(sliceStr) {{
    if (sliceStr.includes("G") || sliceStr.includes("g")) return "green";
    if (sliceStr.includes("y")) return "yellow";
    return "red";
  }}

  function drawSignalHead(label, px, py, sliceStr, countdown) {{
    const active = activeLamp(sliceStr);
    const hw = 30, hh = 78;

    ctx.save();
    // Housing
    ctx.fillStyle = "#090d16";
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.roundRect(px - hw / 2, py - hh / 2, hw, hh, 7);
    ctx.fill();
    ctx.stroke();

    // Digital countdown badge above housing
    ctx.fillStyle = "#0f172a";
    ctx.strokeStyle = active === "green" ? "#00e676" : (active === "yellow" ? "#ffd60a" : "#ff2e2e");
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(px - 22, py - hh / 2 - 24, 44, 20, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`${{label}}: ${{countdown}}s`, px, py - hh / 2 - 10);

    const lamps = [
      ["red", py - 23, "#ff2e2e", "#450a0a"],
      ["yellow", py, "#ffd60a", "#422006"],
      ["green", py + 23, "#00e676", "#052e16"],
    ];
    lamps.forEach(([name, ly, bright, dim]) => {{
      const isOn = active === name;
      ctx.beginPath();
      ctx.arc(px, ly, 9.5, 0, Math.PI * 2);
      ctx.fillStyle = isOn ? bright : dim;
      if (isOn) {{
        ctx.shadowColor = bright;
        ctx.shadowBlur = 16;
      }} else {{
        ctx.shadowBlur = 0;
      }}
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = isOn ? "#ffffff" : "#334155";
      ctx.lineWidth = isOn ? 2 : 1;
      ctx.stroke();
    }});
    ctx.restore();
  }}

  function drawAllSignals() {{
    const tls = DATA.tls_state;
    const off = 86;
    drawSignalHead("N", CX + off, CY - off - 12, tls.slice(6, 9), DATA.ns_countdown);
    drawSignalHead("S", CX - off, CY + off + 22, tls.slice(0, 3), DATA.ns_countdown);
    drawSignalHead("E", CX + off + 18, CY + off + 8, tls.slice(3, 6), DATA.ew_countdown);
    drawSignalHead("W", CX - off - 18, CY - off - 2, tls.slice(9, 12), DATA.ew_countdown);
  }}

  function drawVehicle(v, t) {{
    const x = v.x0 + (v.x1 - v.x0) * t;
    const y = v.y0 + (v.y1 - v.y0) * t;
    const ang = lerpAngle(v.a0, v.a1, t);
    const px = toPxX(x);
    const py = toPxY(y);
    const rad = (ang * Math.PI) / 180.0;

    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(rad);

    if (v.vtype === "motorcycle") {{
      ctx.fillStyle = "#111827";
      ctx.fillRect(-2.5, -7, 5, 14);
      ctx.fillStyle = "#f97316";
      ctx.beginPath();
      ctx.roundRect(-3, -5, 6, 10, 2);
      ctx.fill();
      ctx.fillStyle = "#f8fafc";
      ctx.beginPath();
      ctx.arc(0, 0, 2.8, 0, Math.PI * 2);
      ctx.fill();
    }} else if (v.vtype === "auto_rickshaw") {{
      ctx.fillStyle = "#065f46";
      ctx.beginPath();
      ctx.roundRect(-5.5, -9, 11, 18, 3.5);
      ctx.fill();
      ctx.fillStyle = "#facc15";
      ctx.beginPath();
      ctx.roundRect(-4.5, -4, 9, 11, 2.5);
      ctx.fill();
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(-4, -8, 8, 2.5);
    }} else if (v.vtype === "bus") {{
      ctx.fillStyle = "#dc2626";
      ctx.beginPath();
      ctx.roundRect(-8, -28, 16, 56, 3.5);
      ctx.fill();
      ctx.strokeStyle = "#fecaca";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(-8, -28, 16, 56);
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(-6.5, -25, 13, 5);
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(-5, -12, 10, 8);
      ctx.fillRect(-5, 5, 10, 8);
    }} else if (v.vtype === "truck") {{
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.roundRect(-7.5, -20, 15, 11, 2.5);
      ctx.fill();
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(-5.5, -18.5, 11, 3.5);
      ctx.fillStyle = "#cbd5e1";
      ctx.fillRect(-7.5, -8, 15, 28);
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(-7.5, -8, 15, 28);
    }} else {{
      const carCols = ["#ef4444", "#3b82f6", "#e2e8f0", "#f59e0b"];
      let hash = 0;
      for (let i = 0; i < v.id.length; i++) hash += v.id.charCodeAt(i);
      ctx.fillStyle = carCols[hash % carCols.length];
      ctx.beginPath();
      ctx.roundRect(-6, -12.5, 12, 25, 3.5);
      ctx.fill();
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(-4.5, -7.5, 9, 4.2);
      ctx.fillRect(-4.2, 4.2, 8.4, 3.2);
      ctx.fillStyle = "#fef08a";
      ctx.fillRect(-5, -12.5, 2.5, 2);
      ctx.fillRect(2.5, -12.5, 2.5, 2);
    }}

    ctx.restore();
  }}

  function drawHud() {{
    ctx.fillStyle = "rgba(15, 23, 42, 0.90)";
    ctx.beginPath();
    ctx.roundRect(12, 12, 232, 92, 8);
    ctx.fill();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 12px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`SUMO Time: ${{DATA.sim_time}} s (${{DATA.scenario.toUpperCase()}})`, 22, 31);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText(`Vehicles in View: ${{DATA.vehicles.length}}`, 22, 49);
    ctx.fillText(`TLS: ${{DATA.tls_state}}`, 22, 66);

    const items = [
      ["Car", "#3b82f6"],
      ["Moto", "#f97316"],
      ["Auto", "#facc15"],
      ["Bus", "#dc2626"],
      ["Truck", "#cbd5e1"],
    ];
    let ix = 22;
    items.forEach(([lbl, col]) => {{
      ctx.fillStyle = col;
      ctx.fillRect(ix, 78, 9, 9);
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "10px system-ui, sans-serif";
      ctx.fillText(lbl, ix + 12, 86);
      ix += 42;
    }});
  }}

  const startTs = performance.now();
  function renderFrame(now) {{
    const t = DATA.duration_ms > 0 ? Math.min(1.0, (now - startTs) / DATA.duration_ms) : 1.0;
    drawRoadBackground();
    DATA.vehicles.forEach(v => drawVehicle(v, t));
    drawAllSignals();
    drawHud();
    if (t < 1.0) {{
      requestAnimationFrame(renderFrame);
    }}
  }}
  requestAnimationFrame(renderFrame);
}})();
</script>
</body>
</html>
"""


def build_realtime_canvas_html(state_url: str) -> str:
    """Build a persistent, zero-blink HTML5 Canvas + HUD polling ``state_url``."""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  html, body {{
    margin: 0;
    padding: 0;
    background: #0b1120;
    color: #f1f5f9;
    overflow: hidden;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }}
  .hud-grid {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px;
    margin-bottom: 10px;
  }}
  .hud-card {{
    background: #111827;
    border: 1px solid rgba(59, 130, 246, 0.28);
    border-radius: 8px;
    padding: 8px 10px;
  }}
  .hud-lbl {{
    font-size: 11px;
    color: #94a3b8;
    margin-bottom: 2px;
  }}
  .hud-val {{
    font-size: 17px;
    font-weight: 700;
    color: #f8fafc;
  }}
  #wrap {{
    width: 100%;
    display: flex;
    justify-content: center;
  }}
  canvas {{
    width: 100%;
    max-width: {CANVAS_WIDTH}px;
    height: auto;
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.28);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.5);
    display: block;
  }}
  .approach-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-top: 8px;
  }}
  .approach-card {{
    background: #111827;
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 8px;
    padding: 7px 10px;
    font-size: 12px;
  }}
  .decision-bar {{
    margin-top: 8px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 13px;
    color: #d1fae5;
  }}
</style>
</head>
<body>
<div class="hud-grid">
  <div class="hud-card"><div class="hud-lbl">Sim Clock</div><div class="hud-val" id="m_time">0 s</div></div>
  <div class="hud-card"><div class="hud-lbl">Controller</div><div class="hud-val" id="m_ctrl">-</div></div>
  <div class="hud-card"><div class="hud-lbl">Total Spawned</div><div class="hud-val" id="m_spawn">0</div></div>
  <div class="hud-card"><div class="hud-lbl">Active Vehicles</div><div class="hud-val" id="m_active">0</div></div>
  <div class="hud-card"><div class="hud-lbl">Completed</div><div class="hud-val" id="m_done">0</div></div>
  <div class="hud-card"><div class="hud-lbl">Waiting / Queue</div><div class="hud-val" id="m_wait">0 / 0</div></div>
</div>
<div id="wrap">
  <canvas id="simCanvas" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}"></canvas>
</div>
<div class="approach-grid">
  <div class="approach-card" id="app_N">North: 0 veh</div>
  <div class="approach-card" id="app_S">South: 0 veh</div>
  <div class="approach-card" id="app_E">East: 0 veh</div>
  <div class="approach-card" id="app_W">West: 0 veh</div>
</div>
<div class="decision-bar" id="decisionBar">Waiting for live SUMO state...</div>
<script>
(function() {{
  const STATE_URL = {json.dumps(state_url)};
  const canvas = document.getElementById("simCanvas");
  const ctx = canvas.getContext("2d");
  const W = {CANVAS_WIDTH};
  const H = {CANVAS_HEIGHT};
  const CX = W / 2;
  const CY = H / 2;
  const SCALE = {PX_PER_METER};
  const CTRL_NAMES = {{
    fixed_time: "Fixed-Time",
    q_learning: "Q-Learning",
    sarsa: "SARSA",
    dqn_medium_only: "DQN Medium",
    dqn_curriculum: "DQN Curriculum",
  }};

  let DATA = null;
  let lastFetchTs = performance.now();

  const toPxX = (x) => CX + x * SCALE;
  const toPxY = (y) => CY - y * SCALE;

  function lerpAngle(a0, a1, t) {{
    let diff = ((a1 - a0 + 540) % 360) - 180;
    return a0 + diff * t;
  }}

  function drawRoadBackground() {{
    ctx.fillStyle = "#2d5a27";
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(255,255,255,0.035)";
    ctx.lineWidth = 1;
    for (let gx = 0; gx < W; gx += 32) {{
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }}
    for (let gy = 0; gy < H; gy += 32) {{
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }}

    const roadHalf = 7.5 * SCALE;   // ~63px (126px wide 4-lane road)
    const walkPad = 2.6 * SCALE;    // ~22px sidewalk
    const stopPx = 11.8 * SCALE;    // ~99px

    // Rounded concrete sidewalks with corner fillets
    ctx.fillStyle = "#94a3b8";
    ctx.fillRect(CX - roadHalf - walkPad, 0, (roadHalf + walkPad) * 2, H);
    ctx.fillRect(0, CY - roadHalf - walkPad, W, (roadHalf + walkPad) * 2);

    // Dark asphalt roads
    ctx.fillStyle = "#2f353f";
    ctx.fillRect(CX - roadHalf, 0, roadHalf * 2, H);
    ctx.fillRect(0, CY - roadHalf, W, roadHalf * 2);

    // Rounded asphalt corner aprons for smooth turning geometry
    const cornerR = 3.8 * SCALE;
    [
      [CX - roadHalf - cornerR, CY - roadHalf - cornerR, CX - roadHalf, CY - roadHalf, 0, Math.PI * 0.5],
      [CX + roadHalf + cornerR, CY - roadHalf - cornerR, CX + roadHalf, CY - roadHalf, Math.PI * 0.5, Math.PI],
      [CX + roadHalf + cornerR, CY + roadHalf + cornerR, CX + roadHalf, CY + roadHalf, Math.PI, Math.PI * 1.5],
      [CX - roadHalf - cornerR, CY + roadHalf + cornerR, CX - roadHalf, CY + roadHalf, Math.PI * 1.5, Math.PI * 2],
    ].forEach(([cx, cy, sx, sy, a0, a1]) => {{
      ctx.fillStyle = "#2f353f";
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.arc(cx, cy, cornerR, a0, a1);
      ctx.closePath();
      ctx.fill();
    }});

    // Central junction box + subtle curved turning lane guides
    ctx.fillStyle = "#363d49";
    ctx.fillRect(CX - roadHalf, CY - roadHalf, roadHalf * 2, roadHalf * 2);
    ctx.save();
    ctx.strokeStyle = "rgba(226, 232, 240, 0.16)";
    ctx.lineWidth = 1.6;
    ctx.setLineDash([6, 6]);
    const turnR = 6.2 * SCALE;
    [
      [CX + roadHalf, CY - roadHalf, Math.PI * 0.5, Math.PI],
      [CX - roadHalf, CY - roadHalf, 0, Math.PI * 0.5],
      [CX - roadHalf, CY + roadHalf, Math.PI * 1.5, Math.PI * 2],
      [CX + roadHalf, CY + roadHalf, Math.PI, Math.PI * 1.5],
    ].forEach(([ax, ay, sA, eA]) => {{
      ctx.beginPath();
      ctx.arc(ax, ay, turnR, sA, eA);
      ctx.stroke();
    }});
    ctx.restore();

    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 2;
    [CX - roadHalf + 2, CX + roadHalf - 2].forEach(x => {{
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, CY - stopPx); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x, CY + stopPx); ctx.lineTo(x, H); ctx.stroke();
    }});
    [CY - roadHalf + 2, CY + roadHalf - 2].forEach(y => {{
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(CX - stopPx, y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(CX + stopPx, y); ctx.lineTo(W, y); ctx.stroke();
    }});

    ctx.save();
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1.8;
    ctx.setLineDash([12, 12]);
    [-3.5 * SCALE, 3.5 * SCALE].forEach(off => {{
      ctx.beginPath(); ctx.moveTo(CX + off, 0); ctx.lineTo(CX + off, CY - stopPx); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(CX + off, CY + stopPx); ctx.lineTo(CX + off, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, CY + off); ctx.lineTo(CX - stopPx, CY + off); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(CX + stopPx, CY + off); ctx.lineTo(W, CY + off); ctx.stroke();
    }});
    ctx.restore();

    ctx.strokeStyle = "#facc15";
    ctx.lineWidth = 2.4;
    [-2.5, 2.5].forEach(d => {{
      ctx.beginPath(); ctx.moveTo(CX + d, 0); ctx.lineTo(CX + d, CY - stopPx); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(CX + d, CY + stopPx); ctx.lineTo(CX + d, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, CY + d); ctx.lineTo(CX - stopPx, CY + d); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(CX + stopPx, CY + d); ctx.lineTo(W, CY + d); ctx.stroke();
    }});

    ctx.fillStyle = "#f8fafc";
    for (let x = CX - roadHalf + 5; x < CX + roadHalf - 5; x += 11) {{
      ctx.fillRect(x, CY - stopPx + 3, 5, 18);
      ctx.fillRect(x, CY + stopPx - 21, 5, 18);
    }}
    for (let y = CY - roadHalf + 5; y < CY + roadHalf - 5; y += 11) {{
      ctx.fillRect(CX - stopPx + 3, y, 18, 5);
      ctx.fillRect(CX + stopPx - 21, y, 18, 5);
    }}

    ctx.fillStyle = "rgba(248, 250, 252, 0.85)";
    ctx.font = "bold 12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("N → S (Keep Left)", CX + roadHalf + 82, 24);
    ctx.fillText("S → N (Keep Left)", CX - roadHalf - 82, H - 14);
    ctx.fillText("W → E (Keep Left)", 86, CY - roadHalf - 22);
    ctx.fillText("E → W (Keep Left)", W - 86, CY + roadHalf + 30);
  }}

  function activeLamp(sliceStr) {{
    if (sliceStr.includes("G") || sliceStr.includes("g")) return "green";
    if (sliceStr.includes("y")) return "yellow";
    return "red";
  }}

  function drawSignalHead(label, px, py, sliceStr, countdown) {{
    const active = activeLamp(sliceStr);
    const hw = 30, hh = 78;
    ctx.save();
    ctx.fillStyle = "#090d16";
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.roundRect(px - hw / 2, py - hh / 2, hw, hh, 7);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#0f172a";
    ctx.strokeStyle = active === "green" ? "#00e676" : (active === "yellow" ? "#ffd60a" : "#ff2e2e");
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(px - 22, py - hh / 2 - 24, 44, 20, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`${{label}}: ${{countdown}}s`, px, py - hh / 2 - 10);

    const lamps = [
      ["red", py - 23, "#ff2e2e", "#450a0a"],
      ["yellow", py, "#ffd60a", "#422006"],
      ["green", py + 23, "#00e676", "#052e16"],
    ];
    lamps.forEach(([name, ly, bright, dim]) => {{
      const isOn = active === name;
      ctx.beginPath();
      ctx.arc(px, ly, 9.5, 0, Math.PI * 2);
      ctx.fillStyle = isOn ? bright : dim;
      ctx.shadowColor = isOn ? bright : "transparent";
      ctx.shadowBlur = isOn ? 16 : 0;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = isOn ? "#ffffff" : "#334155";
      ctx.lineWidth = isOn ? 2 : 1;
      ctx.stroke();
    }});
    ctx.restore();
  }}

  function drawVehicle(v, t) {{
    const x = v.x0 + (v.x1 - v.x0) * t;
    const y = v.y0 + (v.y1 - v.y0) * t;
    const ang = lerpAngle(v.a0, v.a1, t);
    const px = toPxX(x);
    const py = toPxY(y);
    const rad = (ang * Math.PI) / 180.0;

    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(rad);

    if (v.vtype === "motorcycle") {{
      ctx.fillStyle = "#111827";
      ctx.fillRect(-3, -9, 6, 18);
      ctx.fillStyle = "#f97316";
      ctx.beginPath(); ctx.roundRect(-3.8, -6.5, 7.6, 13, 2.5); ctx.fill();
      ctx.fillStyle = "#f8fafc";
      ctx.beginPath(); ctx.arc(0, 0, 3.6, 0, Math.PI * 2); ctx.fill();
    }} else if (v.vtype === "auto_rickshaw") {{
      ctx.fillStyle = "#065f46";
      ctx.beginPath(); ctx.roundRect(-6.8, -11, 13.6, 22, 4); ctx.fill();
      ctx.fillStyle = "#facc15";
      ctx.beginPath(); ctx.roundRect(-5.6, -5, 11.2, 13.5, 3); ctx.fill();
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(-5, -10, 10, 3);
    }} else if (v.vtype === "bus") {{
      ctx.fillStyle = "#dc2626";
      ctx.beginPath(); ctx.roundRect(-10, -35, 20, 70, 4); ctx.fill();
      ctx.strokeStyle = "#fecaca"; ctx.lineWidth = 1.4; ctx.strokeRect(-10, -35, 20, 70);
      ctx.fillStyle = "#0f172a"; ctx.fillRect(-8, -32, 16, 6);
      ctx.fillStyle = "#f8fafc"; ctx.fillRect(-6, -15, 12, 10); ctx.fillRect(-6, 6, 12, 10);
    }} else if (v.vtype === "truck") {{
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath(); ctx.roundRect(-9.2, -25, 18.4, 13, 3); ctx.fill();
      ctx.fillStyle = "#0f172a"; ctx.fillRect(-7, -23, 14, 4);
      ctx.fillStyle = "#cbd5e1"; ctx.fillRect(-9.2, -10, 18.4, 35);
      ctx.strokeStyle = "#475569"; ctx.lineWidth = 1.4; ctx.strokeRect(-9.2, -10, 18.4, 35);
    }} else {{
      const carCols = ["#ef4444", "#3b82f6", "#e2e8f0", "#f59e0b"];
      let hash = 0;
      for (let i = 0; i < v.id.length; i++) hash += v.id.charCodeAt(i);
      ctx.fillStyle = carCols[hash % carCols.length];
      ctx.beginPath(); ctx.roundRect(-7.5, -15.5, 15, 31, 4); ctx.fill();
      ctx.strokeStyle = "#0f172a"; ctx.lineWidth = 1.4; ctx.stroke();
      ctx.fillStyle = "#0f172a"; ctx.fillRect(-5.8, -9.5, 11.6, 5.2); ctx.fillRect(-5.2, 5.2, 10.4, 4);
      ctx.fillStyle = "#fef08a"; ctx.fillRect(-6.2, -15.5, 3, 2.4); ctx.fillRect(3.2, -15.5, 3, 2.4);
    }}
    ctx.restore();
  }}

  function updateHudDom() {{
    if (!DATA || DATA.sim_time === undefined) return;
    document.getElementById("m_time").textContent = `${{DATA.sim_time}} s`;
    document.getElementById("m_ctrl").textContent = CTRL_NAMES[DATA.controller_name] || DATA.controller_name;
    document.getElementById("m_spawn").textContent = `${{DATA.total_spawned}}`;
    document.getElementById("m_active").textContent = `${{DATA.vehicle_count}}`;
    document.getElementById("m_done").textContent = `${{DATA.completed_vehicles}}`;
    document.getElementById("m_wait").textContent = `${{DATA.waiting_vehicles}} / ${{DATA.queue_length}}`;

    const q = DATA.approach_queues || {{}};
    const w = DATA.approach_waits || {{}};
    document.getElementById("app_N").textContent = `North: ${{q.N || 0}} veh (${{Math.round(w.N || 0)}}s wait)`;
    document.getElementById("app_S").textContent = `South: ${{q.S || 0}} veh (${{Math.round(w.S || 0)}}s wait)`;
    document.getElementById("app_E").textContent = `East: ${{q.E || 0}} veh (${{Math.round(w.E || 0)}}s wait)`;
    document.getElementById("app_W").textContent = `West: ${{q.W || 0}} veh (${{Math.round(w.W || 0)}}s wait)`;

    const act = DATA.selected_action === 0 ? "0: North-South" : "1: East-West";
    const ph = DATA.current_group === 0 ? "North-South" : "East-West";
    document.getElementById("decisionBar").textContent =
      `Action: ${{act}} | Phase: ${{ph}} (Next decision: ${{DATA.next_decision_in}}s) — ${{DATA.decision_reason}}`;
  }}

  async function pollState() {{
    try {{
      const res = await fetch(STATE_URL, {{ cache: "no-store" }});
      if (res.ok) {{
        const nextData = await res.json();
        if (!DATA || nextData.sim_time !== DATA.sim_time || nextData.tls_state !== DATA.tls_state || nextData.controller_name !== DATA.controller_name || nextData.scenario !== DATA.scenario) {{
          DATA = nextData;
          lastFetchTs = performance.now();
          updateHudDom();
        }} else {{
          DATA.is_running = nextData.is_running;
        }}
      }}
    }} catch (e) {{}}
    setTimeout(pollState, 120);
  }}

  function animate(now) {{
    drawRoadBackground();
    if (DATA && DATA.vehicles) {{
      const dur = DATA.is_running ? (DATA.poll_ms || 420) : 0;
      const t = dur > 0 ? Math.min(1.0, (now - lastFetchTs) / dur) : 1.0;
      DATA.vehicles.forEach(v => drawVehicle(v, t));
      const tls = (DATA.tls_state || "rrrrrrrrrrrr").padEnd(12, "r");
      const off = 114;
      drawSignalHead("N", CX + off, CY - off - 10, tls.slice(6, 9), DATA.ns_countdown || 0);
      drawSignalHead("S", CX - off, CY + off + 22, tls.slice(0, 3), DATA.ns_countdown || 0);
      drawSignalHead("E", CX + off + 22, CY + off + 10, tls.slice(3, 6), DATA.ew_countdown || 0);
      drawSignalHead("W", CX - off - 22, CY - off - 2, tls.slice(9, 12), DATA.ew_countdown || 0);
    }}
    requestAnimationFrame(animate);
  }}

  pollState();
  requestAnimationFrame(animate);
}})();
</script>
</body>
</html>
"""
