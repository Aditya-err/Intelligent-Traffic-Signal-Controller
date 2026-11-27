"""
Intelligent-Traffic-RL v2 — Finished 6-Page Streamlit Dashboard.

Pages:
1. Overview (best controller per demand, honest findings, Limitations)
2. Traffic Simulation (Phase 5 60 FPS HTML5 Canvas 2D corridor replay)
3. Controller Comparison (Waiting time, Queue, Throughput + 95% CI)
4. Performance Analytics (Per-junction J1/J2/J3 waiting time & travel time)
5. Training Analysis (DQN Reward, Huber Loss, Epsilon, Safety Overrides)
6. Statistical Results (Paired t-test, Exact Permutation, Holm correction, Direction)

Usage:
    streamlit run v2/dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

_THIS_DIR = Path(__file__).resolve().parent
_V2_ROOT = _THIS_DIR.parent
if str(_V2_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT.parent))

from v2 import config  # noqa: E402
from v2.replays.network_geometry import extract_network_geometry  # noqa: E402


CONTROLLER_LABELS = {
    "fixed_time": "Fixed-Time (25s NS / 20s EW)",
    "actuated": "SUMO Actuated",
    "dqn": "Shared DQN (RL)",
}
DEMAND_LABELS = {
    "low": "Low (~720 veh/hr)",
    "medium": "Medium (~1330 veh/hr)",
    "high": "High (~1660 veh/hr)",
}
PAGES = [
    "Overview",
    "Traffic Simulation",
    "Controller Comparison",
    "Performance Analytics",
    "Training Analysis",
    "Statistical Results",
]


def load_replay_json(path: Path) -> dict:
    """Load and validate a recorded replay JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "geometry" not in data or "frames" not in data or not data["frames"]:
        raise ValueError(f"Invalid replay JSON structure in {path}")
    if config.NET_FILE.exists():
        data["geometry"] = extract_network_geometry(config.NET_FILE)
    return data


def build_canvas_html(replay_data: dict) -> str:
    """Build self-contained HTML5 Canvas + JS replay player."""
    payload_json = json.dumps(replay_data, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  body {{ margin: 0; padding: 0; background: #0f172a; color: #e2e8f0; }}
  .wrap {{ width: 100%; padding: 8px; }}
  .stats-grid {{
    display: grid; grid-template-columns: repeat(8, 1fr); gap: 8px; margin-bottom: 8px;
  }}
  .stat-card {{
    background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 8px 10px; text-align: center;
  }}
  .stat-label {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.4px; }}
  .stat-val {{ font-size: 18px; font-weight: 700; color: #f8fafc; margin-top: 2px; }}
  .controls {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
    background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 8px 12px; margin-bottom: 8px;
  }}
  button {{
    background: #2563eb; color: #fff; border: none; border-radius: 6px;
    padding: 6px 12px; font-size: 13px; font-weight: 600; cursor: pointer;
  }}
  button.secondary {{ background: #334155; }}
  button.active {{ background: #059669; }}
  button:hover {{ filter: brightness(1.1); }}
  input[type=range] {{ flex: 1; min-width: 140px; accent-color: #38bdf8; }}
  canvas {{
    width: 100%; height: 640px; display: block;
    background: #1f4d2b; border-radius: 10px; border: 1px solid #334155; cursor: grab;
  }}
  .rl-panel {{
    margin-top: 8px; background: #1e293b; border: 1px solid #334155;
    border-radius: 8px; padding: 12px 14px; transition: box-shadow 0.25s, border-color 0.25s;
  }}
  .rl-panel.pulse {{
    border-color: #38bdf8; box-shadow: 0 0 16px rgba(56, 189, 248, 0.55);
  }}
  .rl-header {{
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;
  }}
  .rl-grid {{
    display: grid; grid-template-columns: 1.2fr 1.2fr 1.6fr; gap: 12px; align-items: center;
  }}
  .qbar-row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 13px; }}
  .qbar-bg {{ flex: 1; height: 14px; background: #0f172a; border-radius: 4px; overflow: hidden; }}
  .qbar-fill {{ height: 100%; background: #38bdf8; transition: width 0.1s; }}
  .explain {{
    background: #0f172a; border-left: 4px solid #38bdf8; padding: 10px 12px;
    border-radius: 4px; font-size: 13px; line-height: 1.4;
  }}
  .explain.override {{ border-left-color: #f59e0b; color: #fde68a; }}
  select.junc-sel {{
    background: #0f172a; color: #f8fafc; border: 1px solid #475569;
    border-radius: 6px; padding: 4px 8px; font-size: 13px;
  }}
  .legend {{
    display: flex; gap: 14px; font-size: 12px; color: #cbd5e1; margin-left: auto;
  }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="stats-grid">
    <div class="stat-card"><div class="stat-label">Sim Clock</div><div class="stat-val" id="st-clock">0.0 s</div></div>
    <div class="stat-card"><div class="stat-label">Spawned</div><div class="stat-val" id="st-sp">0</div></div>
    <div class="stat-card"><div class="stat-label">Active</div><div class="stat-val" id="st-ac">0</div></div>
    <div class="stat-card"><div class="stat-label">Completed</div><div class="stat-val" id="st-co">0</div></div>
    <div class="stat-card"><div class="stat-label">Waiting</div><div class="stat-val" id="st-wa">0</div></div>
    <div class="stat-card"><div class="stat-label">J1 Queue</div><div class="stat-val" id="st-j1">0</div></div>
    <div class="stat-card"><div class="stat-label">J2 Queue</div><div class="stat-val" id="st-j2">0</div></div>
    <div class="stat-card"><div class="stat-label">J3 Queue</div><div class="stat-val" id="st-j3">0</div></div>
  </div>

  <div class="controls">
    <button id="btn-start">Start</button>
    <button id="btn-pause" class="secondary">Pause</button>
    <button id="btn-reset" class="secondary">Reset</button>
    <span style="font-size:12px;color:#94a3b8;margin-left:4px;">Speed:</span>
    <button class="secondary spd-btn" data-spd="0.5">0.5x</button>
    <button class="active spd-btn" data-spd="1">1x</button>
    <button class="secondary spd-btn" data-spd="2">2x</button>
    <button class="secondary spd-btn" data-spd="4">4x</button>
    <span style="font-size:12px;color:#94a3b8;margin-left:6px;">Camera:</span>
    <button class="secondary cam-btn" data-cam="all">Full Corridor</button>
    <button class="secondary cam-btn" data-cam="J1">Zoom J1</button>
    <button class="active cam-btn" data-cam="J2">Zoom J2</button>
    <button class="secondary cam-btn" data-cam="J3">Zoom J3</button>
    <input type="range" id="scrub" min="0" max="1" step="1" value="0"/>
    <div class="legend">
      <span><i class="dot" style="background:#3b82f6"></i>Bike</span>
      <span><i class="dot" style="background:#ef4444"></i>Car</span>
      <span><i class="dot" style="background:#f59e0b"></i>Auto</span>
      <span><i class="dot" style="background:#eab308"></i>Bus</span>
      <span><i class="dot" style="background:#38bdf8"></i>Truck</span>
      <span><i class="dot" style="background:#f8fafc;border-radius:50%"></i>Ped</span>
    </div>
  </div>

  <canvas id="corridorCanvas" width="1440" height="660"></canvas>

  <div class="rl-panel">
    <div class="rl-header">
      <strong style="font-size:14px;">Signal &amp; RL Decision Panel</strong>
      <div>
        <label for="juncSelect" style="font-size:12px;color:#94a3b8;margin-right:6px;">Inspect Junction:</label>
        <select id="juncSelect" class="junc-sel">
          <option value="J1">J1 (West)</option>
          <option value="J2">J2 (Center)</option>
          <option value="J3">J3 (East)</option>
        </select>
      </div>
    </div>
    <div class="rl-grid" id="rl-content">
      <div id="rl-queues" style="font-size:13px;"></div>
      <div id="rl-qvals"></div>
      <div id="rl-explain" class="explain"></div>
    </div>
  </div>
</div>

<script>
const DATA = {payload_json};
const GEO = DATA.geometry;
const FRAMES = DATA.frames;
const META = DATA.meta;

const canvas = document.getElementById("corridorCanvas");
const ctx = canvas.getContext("2d");
const scrub = document.getElementById("scrub");
scrub.max = Math.max(0, FRAMES.length - 1);

const VEH_STYLE = {{
  m: {{ len: 2.2, wid: 0.9,  color: "#3b82f6" }},
  c: {{ len: 4.3, wid: 1.85, color: "#ef4444" }},
  a: {{ len: 2.9, wid: 1.4,  color: "#f59e0b" }},
  b: {{ len: 10.5, wid: 2.55, color: "#eab308" }},
  t: {{ len: 8.0, wid: 2.45, color: "#38bdf8" }},
}};

const b = GEO.bounds;
const worldW = b.max_x - b.min_x;
const worldH = b.max_y - b.min_y;
const fullScale = Math.min((canvas.width - 56) / worldW, (canvas.height - 40) / worldH);

// Camera state with smooth 0.5s ease-out transitions
let camX = GEO.tl_coords.J2 ? GEO.tl_coords.J2[0] : (b.min_x + worldW / 2);
let camY = GEO.tl_coords.J2 ? GEO.tl_coords.J2[1] : (b.min_y + worldH / 2);
let camScale = 7.2;
let animFrom = null, animTo = null, animStart = 0;
const CAM_ANIM_MS = 500;

function setCamera(mode, animate = true) {{
  let tx = camX, ty = camY, ts = camScale;
  if (mode === "all") {{
    tx = b.min_x + worldW / 2;
    ty = b.min_y + worldH / 2;
    ts = fullScale;
  }} else if (GEO.tl_coords[mode]) {{
    tx = GEO.tl_coords[mode][0];
    ty = GEO.tl_coords[mode][1];
    ts = 7.2;
    const sel = document.getElementById("juncSelect");
    if (sel) sel.value = mode;
  }}
  if (!animate) {{
    camX = tx; camY = ty; camScale = ts; animFrom = null;
  }} else {{
    animFrom = {{ x: camX, y: camY, s: camScale }};
    animTo = {{ x: tx, y: ty, s: ts }};
    animStart = performance.now();
  }}
}}
setCamera("J2", false);

function wx(x) {{ return canvas.width / 2 + (x - camX) * camScale; }}
function wy(y) {{ return canvas.height / 2 - (y - camY) * camScale; }}

function drawPolyline(pts, widthPx, color, dash = []) {{
  if (pts.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = widthPx;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(wx(pts[0][0]), wy(pts[0][1]));
  for (let i = 1; i < pts.length; i++) ctx.lineTo(wx(pts[i][0]), wy(pts[i][1]));
  ctx.stroke();
  ctx.restore();
}}

function drawZebraCrossing(pts, widthM) {{
  if (pts.length < 2) return;
  const x0 = wx(pts[0][0]), y0 = wy(pts[0][1]);
  const x1 = wx(pts[pts.length - 1][0]), y1 = wy(pts[pts.length - 1][1]);
  const dx = x1 - x0, dy = y1 - y0;
  const dist = Math.hypot(dx, dy);
  if (dist < 4) return;
  const ang = Math.atan2(dy, dx);
  const stripeLen = Math.max(4, widthM * camScale * 0.85);
  const step = Math.max(3, 1.3 * camScale);
  const barW = Math.max(1.5, 0.65 * camScale);

  ctx.save();
  ctx.translate(x0, y0);
  ctx.rotate(ang);
  ctx.fillStyle = "rgba(248, 250, 252, 0.92)";
  for (let d = step * 0.5; d < dist - step * 0.3; d += step) {{
    ctx.fillRect(d - barW / 2, -stripeLen / 2, barW, stripeLen);
  }}
  ctx.restore();
}}

function offsetPolylineRight(pts, offsetM) {{
  // Offset a polyline laterally to the RIGHT of its travel direction (in world coords)
  const out = [];
  const n = pts.length;
  for (let i = 0; i < n; i++) {{
    const pPrev = pts[Math.max(0, i - 1)];
    const pNext = pts[Math.min(n - 1, i + 1)];
    const dx = pNext[0] - pPrev[0];
    const dy = pNext[1] - pPrev[1];
    const len = Math.hypot(dx, dy) || 1.0;
    // Right normal in world coords is (dy, -dx) / len
    out.push([pts[i][0] + offsetM * (dy / len), pts[i][1] + offsetM * (-dx / len)]);
  }}
  return out;
}}

function drawRoadNetwork() {{
  // Mowed grass background (like v1)
  ctx.fillStyle = "#1f4d2b";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(34, 95, 52, 0.35)";
  const stripeH = 26;
  for (let y = 0; y < canvas.height; y += stripeH * 2) {{
    ctx.fillRect(0, y, canvas.width, stripeH);
  }}

  // Sidewalk curb + concrete footpath
  for (const sw of GEO.sidewalk_lanes) {{
    drawPolyline(sw.shape, Math.max(3, (sw.width + 0.6) * camScale), "#475569");
    drawPolyline(sw.shape, Math.max(2, sw.width * camScale), "#94a3b8");
  }}

  // Asphalt road surface
  for (const vl of GEO.vehicle_lanes) {{
    drawPolyline(vl.shape, Math.max(4, (vl.width + 0.45) * camScale), "#2b313a");
  }}

  // Junction asphalt polygons (J2 is rounded with radius=14m)
  ctx.fillStyle = "#2b313a";
  for (const j of GEO.junctions) {{
    if (j.shape.length < 3) continue;
    ctx.beginPath();
    ctx.moveTo(wx(j.shape[0][0]), wy(j.shape[0][1]));
    for (let i = 1; i < j.shape.length; i++) ctx.lineTo(wx(j.shape[i][0]), wy(j.shape[i][1]));
    ctx.closePath();
    ctx.fill();
  }}

  // Proper LHT lane markings placed BETWEEN lanes (never under vehicles)
  for (const vl of GEO.vehicle_lanes) {{
    if (vl.id.endsWith("_1")) {{
      // Outer white road edge line (left of _1, next to sidewalk)
      drawPolyline(offsetPolylineRight(vl.shape, -1.75), Math.max(1.0, 0.16 * camScale), "rgba(241,245,249,0.75)");
      // Dashed white lane divider (between _1 and _2)
      drawPolyline(offsetPolylineRight(vl.shape, 1.75), Math.max(1.0, 0.16 * camScale), "rgba(255,255,255,0.75)", [Math.max(4, 3 * camScale), Math.max(4, 3 * camScale)]);
    }} else if (vl.id.endsWith("_2")) {{
      // Double yellow median line (inner edge of _2, inside the 1.4m median gap)
      drawPolyline(offsetPolylineRight(vl.shape, 1.85), Math.max(1.0, 0.18 * camScale), "rgba(250,204,21,0.9)");
      drawPolyline(offsetPolylineRight(vl.shape, 2.25), Math.max(1.0, 0.18 * camScale), "rgba(250,204,21,0.9)");
    }}
  }}

  // Zebra crossings
  for (const cr of GEO.crossings) {{
    drawZebraCrossing(cr.shape, cr.width);
  }}

  // Junction ID badges
  ctx.font = "bold 12px system-ui";
  for (const [jid, pt] of Object.entries(GEO.tl_coords)) {{
    const jx = wx(pt[0]), jy = wy(pt[1]);
    ctx.fillStyle = "rgba(15, 23, 42, 0.78)";
    ctx.fillRect(jx - 14, jy - 10, 28, 17);
    ctx.fillStyle = "#f8fafc";
    ctx.textAlign = "center";
    ctx.fillText(jid, jx, jy + 3);
  }}
  ctx.textAlign = "left";
}}

function drawVehicleSprite(vtype, x, y, ang, isBraking) {{
  const st = VEH_STYLE[vtype] || VEH_STYLE.c;
  const L = Math.max(10, st.len * camScale * 1.3);
  const W = Math.max(5.5, st.wid * camScale * 1.3);

  ctx.save();
  ctx.translate(wx(x), wy(y));
  ctx.rotate((ang - 90) * Math.PI / 180);
  // SUMO getPosition() returns the front bumper; shift back by ~45% of length
  ctx.translate(-L * 0.45, 0);

  if (camScale < 2.2) {{
    // Clean compact marker when zoomed out to full corridor
    ctx.fillStyle = st.color;
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 0.8;
    ctx.fillRect(-L / 2, -W / 2, L, W);
    ctx.strokeRect(-L / 2, -W / 2, L, W);
    if (isBraking) {{
      ctx.fillStyle = "#ff2e2e";
      ctx.fillRect(-L / 2 - 1, -W * 0.4, 2, W * 0.8);
    }}
    ctx.restore();
    return;
  }}

  // Detailed v1-grade top-down sprites when zoomed in
  // Drop shadow
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  ctx.fillRect(-L / 2 + 1.5, -W / 2 + 2, L, W);

  if (vtype === "m") {{
    // Motorcycle: tires + slim frame + rider helmet
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(-L * 0.45, -W * 0.25, L * 0.28, W * 0.5);
    ctx.fillRect(L * 0.18, -W * 0.25, L * 0.28, W * 0.5);
    ctx.fillStyle = st.color;
    ctx.fillRect(-L * 0.3, -W * 0.35, L * 0.65, W * 0.7);
    ctx.beginPath();
    ctx.arc(0, 0, W * 0.42, 0, Math.PI * 2);
    ctx.fillStyle = "#f8fafc";
    ctx.fill();
  }} else if (vtype === "b") {{
    // Bus: yellow body + dark windshield + roof AC units + windows
    ctx.fillStyle = st.color;
    ctx.strokeStyle = "#713f12";
    ctx.lineWidth = 1.2;
    ctx.fillRect(-L / 2, -W / 2, L, W);
    ctx.strokeRect(-L / 2, -W / 2, L, W);
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(L * 0.34, -W * 0.4, L * 0.12, W * 0.8);
    ctx.fillStyle = "#fef08a";
    ctx.fillRect(-L * 0.32, -W * 0.3, L * 0.22, W * 0.6);
    ctx.fillRect(0, -W * 0.3, L * 0.22, W * 0.6);
  }} else if (vtype === "t") {{
    // Truck: cargo box + distinct front cab + windshield
    ctx.fillStyle = "#e2e8f0";
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1.2;
    ctx.fillRect(-L / 2, -W / 2, L * 0.68, W);
    ctx.strokeRect(-L / 2, -W / 2, L * 0.68, W);
    ctx.fillStyle = st.color;
    ctx.fillRect(L * 0.18, -W * 0.46, L * 0.32, W * 0.92);
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(L * 0.35, -W * 0.38, L * 0.1, W * 0.76);
  }} else if (vtype === "a") {{
    // Auto-rickshaw: Indian 3-wheeler green/yellow canopy
    ctx.fillStyle = "#16a34a";
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 1;
    ctx.fillRect(-L / 2, -W / 2, L, W);
    ctx.strokeRect(-L / 2, -W / 2, L, W);
    ctx.fillStyle = "#facc15";
    ctx.fillRect(-L * 0.38, -W * 0.4, L * 0.55, W * 0.8);
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(L * 0.22, -W * 0.34, L * 0.14, W * 0.68);
  }} else {{
    // Car: 4 tires + painted sedan body + front/rear windshields
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(-L * 0.38, -W * 0.56, L * 0.2, W * 1.12);
    ctx.fillRect(L * 0.18, -W * 0.56, L * 0.2, W * 1.12);
    ctx.fillStyle = st.color;
    ctx.strokeStyle = "#7f1d1d";
    ctx.lineWidth = 1;
    ctx.fillRect(-L / 2, -W / 2, L, W);
    ctx.strokeRect(-L / 2, -W / 2, L, W);
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(L * 0.12, -W * 0.38, L * 0.18, W * 0.76);
    ctx.fillRect(-L * 0.32, -W * 0.36, L * 0.13, W * 0.72);
  }}

  // Headlights (all vehicles)
  ctx.fillStyle = "#fef9c3";
  ctx.fillRect(L * 0.45, -W * 0.36, L * 0.05, W * 0.2);
  ctx.fillRect(L * 0.45, W * 0.16, L * 0.05, W * 0.2);

  // Red brake lights when speed is near zero
  if (isBraking) {{
    ctx.fillStyle = "#ff2e2e";
    ctx.shadowColor = "#ff2e2e";
    ctx.shadowBlur = 6;
    ctx.fillRect(-L * 0.51, -W * 0.38, L * 0.06, W * 0.24);
    ctx.fillRect(-L * 0.51, W * 0.14, L * 0.06, W * 0.24);
    ctx.shadowBlur = 0;
  }}
  ctx.restore();
}}

function drawTrafficLightHead(cx, cy, lamp, countdown, label, orientation) {{
  // Bug 4: In Full Corridor view (zoomed out), hide text/timers and show only coloured lights
  if (camScale < 2.2) {{
    const col = lamp === "G" ? "#22c55e" : (lamp === "Y" ? "#facc15" : "#ef4444");
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, 4.0, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
    return;
  }}

  // orientation: "N", "S" (vertical) or "E", "W" (horizontal)
  const isHoriz = (orientation === "E" || orientation === "W");
  const w = isHoriz ? 38 : 16;
  const h = isHoriz ? 16 : 38;

  ctx.save();
  ctx.fillStyle = "#0f172a";
  ctx.strokeStyle = "#64748b";
  ctx.lineWidth = 1.4;
  ctx.fillRect(cx - w / 2, cy - h / 2, w, h);
  ctx.strokeRect(cx - w / 2, cy - h / 2, w, h);

  // Order lamps so Red is closest to the intersection / top
  const lamps = isHoriz
    ? [
        ["R", cx + (orientation === "E" ? -11 : 11), cy, "#ef4444", "#450a0a"],
        ["Y", cx, cy, "#facc15", "#422006"],
        ["G", cx + (orientation === "E" ? 11 : -11), cy, "#22c55e", "#052e16"],
      ]
    : [
        ["R", cx, cy + (orientation === "S" ? -11 : 11), "#ef4444", "#450a0a"],
        ["Y", cx, cy, "#facc15", "#422006"],
        ["G", cx, cy + (orientation === "S" ? 11 : -11), "#22c55e", "#052e16"],
      ];

  for (const [code, lx, ly, onCol, offCol] of lamps) {{
    ctx.beginPath();
    ctx.arc(lx, ly, 4.2, 0, Math.PI * 2);
    if (lamp === code) {{
      ctx.shadowColor = onCol;
      ctx.shadowBlur = 10;
      ctx.fillStyle = onCol;
    }} else {{
      ctx.shadowBlur = 0;
      ctx.fillStyle = offCol;
    }}
    ctx.fill();
  }}

  ctx.shadowBlur = 0;
  ctx.fillStyle = "rgba(15, 23, 42, 0.9)";
  ctx.fillRect(cx - 22, cy - h / 2 - 15, 44, 13);
  ctx.fillStyle = "#f8fafc";
  ctx.font = "bold 10px system-ui";
  ctx.textAlign = "center";
  ctx.fillText(`${{label}}:${{countdown}}s`, cx, cy - h / 2 - 5);
  ctx.restore();
}}

let playing = false;
let speed = 1.0;
let floatIdx = 0.0;
let lastTs = null;
const stepS = META.step_s || 0.5;
const lastActionByJunc = {{}};

function lerp(a, b, t) {{ return a + (b - a) * t; }}
function lerpAngle(a, b, t) {{
  let diff = ((b - a + 540) % 360) - 180;
  return a + diff * t;
}}

function drawBoundaryFade() {{
  // Soft border vignette so boundary roads blend smoothly at canvas edges
  const gradL = ctx.createLinearGradient(0, 0, 36, 0);
  gradL.addColorStop(0, "rgba(15, 23, 42, 0.55)");
  gradL.addColorStop(1, "rgba(15, 23, 42, 0)");
  ctx.fillStyle = gradL; ctx.fillRect(0, 0, 36, canvas.height);

  const gradR = ctx.createLinearGradient(canvas.width - 36, 0, canvas.width, 0);
  gradR.addColorStop(0, "rgba(15, 23, 42, 0)");
  gradR.addColorStop(1, "rgba(15, 23, 42, 0.55)");
  ctx.fillStyle = gradR; ctx.fillRect(canvas.width - 36, 0, 36, canvas.height);

  const gradT = ctx.createLinearGradient(0, 0, 0, 28);
  gradT.addColorStop(0, "rgba(15, 23, 42, 0.55)");
  gradT.addColorStop(1, "rgba(15, 23, 42, 0)");
  ctx.fillStyle = gradT; ctx.fillRect(0, 0, canvas.width, 28);

  const gradB = ctx.createLinearGradient(0, canvas.height - 28, 0, canvas.height);
  gradB.addColorStop(0, "rgba(15, 23, 42, 0)");
  gradB.addColorStop(1, "rgba(15, 23, 42, 0.55)");
  ctx.fillStyle = gradB; ctx.fillRect(0, canvas.height - 28, canvas.width, 28);
}}

function renderFrame(fIdx) {{
  const i0 = Math.min(Math.floor(fIdx), FRAMES.length - 1);
  const i1 = Math.min(i0 + 1, FRAMES.length - 1);
  const alpha = fIdx - i0;
  const f0 = FRAMES[i0];
  const f1 = FRAMES[i1];

  drawRoadNetwork();

  const nextMap = new Map();
  for (const v of f1.v) nextMap.set(v[0], v);

  for (const v of f0.v) {{
    const [vid, vtype, x0, y0, a0] = v;
    const nxt = nextMap.get(vid);
    const x = nxt ? lerp(x0, nxt[2], alpha) : x0;
    const y = nxt ? lerp(y0, nxt[3], alpha) : y0;
    const ang = nxt ? lerpAngle(a0, nxt[4], alpha) : a0;
    const isBraking = nxt ? (Math.hypot(nxt[2] - x0, nxt[3] - y0) / stepS < 0.35) : false;
    drawVehicleSprite(vtype, x, y, ang, isBraking);
  }}

  // Pedestrians (slightly larger) + Walk/Stop signal icons at zebra crossings
  const pedR = Math.max(3.6, 0.65 * camScale);
  ctx.fillStyle = "#f8fafc";
  ctx.strokeStyle = "#0284c7";
  ctx.lineWidth = 1.4;
  for (const p of f0.p) {{
    ctx.beginPath();
    ctx.arc(wx(p[0]), wy(p[1]), pedR, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }}

  if (camScale >= 2.2) {{
    for (const cr of GEO.crossings) {{
      if (!cr.shape || cr.shape.length < 2) continue;
      const m = cr.id.match(/^:(J[123])_/);
      if (!m || !f0.tls[m[1]]) continue;
      const jt = f0.tls[m[1]];
      const p0 = cr.shape[0], p1 = cr.shape[cr.shape.length - 1];
      const horizCross = Math.abs(p1[0] - p0[0]) > Math.abs(p1[1] - p0[1]);
      const walkOn = horizCross ? (jt.ew === "G") : (jt.ns === "G");
      const sx = wx(p0[0]), sy = wy(p0[1]);
      ctx.save();
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(sx - 6, sy - 6, 12, 12);
      ctx.beginPath();
      ctx.arc(sx, sy, 3.8, 0, Math.PI * 2);
      ctx.fillStyle = walkOn ? "#22c55e" : "#ef4444";
      ctx.fill();
      ctx.restore();
    }}
  }}

  // 4 Traffic light heads positioned on LHT incoming approach sides (with per-arm countdowns)
  const headOff = camScale < 2.2 ? 12 : Math.max(24, 15 * camScale);
  for (const [jid, pt] of Object.entries(GEO.tl_coords)) {{
    const t = f0.tls[jid];
    if (!t) continue;
    const cdNS = t.cd_ns ?? t.cd ?? 0;
    const cdEW = t.cd_ew ?? t.cd ?? 0;
    const cx = wx(pt[0]), cy = wy(pt[1]);
    drawTrafficLightHead(cx + headOff, cy - headOff, t.ns, cdNS, "N", "N");
    drawTrafficLightHead(cx - headOff, cy + headOff, t.ns, cdNS, "S", "S");
    drawTrafficLightHead(cx + headOff, cy + headOff, t.ew, cdEW, "E", "E");
    drawTrafficLightHead(cx - headOff, cy - headOff, t.ew, cdEW, "W", "W");
  }}

  drawBoundaryFade();

  document.getElementById("st-clock").textContent = f0.t.toFixed(1) + " s";
  document.getElementById("st-sp").textContent = f0.st.sp;
  document.getElementById("st-ac").textContent = f0.st.ac;
  document.getElementById("st-co").textContent = f0.st.co;
  document.getElementById("st-wa").textContent = f0.st.wa;
  document.getElementById("st-j1").textContent = f0.st.jq[0];
  document.getElementById("st-j2").textContent = f0.st.jq[1];
  document.getElementById("st-j3").textContent = f0.st.jq[2];

  const jid = document.getElementById("juncSelect").value;
  const qEl = document.getElementById("rl-queues");
  const qvEl = document.getElementById("rl-qvals");
  const exEl = document.getElementById("rl-explain");
  const panelEl = document.querySelector(".rl-panel");

  if (META.controller !== "dqn" || !f0.rl || !f0.rl[jid]) {{
    qEl.innerHTML = `<strong>${{jid}}</strong> — Controller: <code>${{META.controller}}</code>`;
    qvEl.innerHTML = `<span style="color:#94a3b8;font-size:12px;">No Q-values (rule-based signal)</span>`;
    exEl.className = "explain";
    exEl.textContent = "Rule-based controller, no RL decision";
  }} else {{
    const d = f0.rl[jid];
    const nsSum = d.q[1] + d.q[2];
    const ewSum = d.q[0] + d.q[3];
    const actName = d.act === 0 ? "NS Green" : "EW Green";

    // Briefly highlight RL panel when chosen action changes
    if (lastActionByJunc[jid] !== undefined && lastActionByJunc[jid] !== d.act && panelEl) {{
      panelEl.classList.add("pulse");
      setTimeout(() => panelEl.classList.remove("pulse"), 650);
    }}
    lastActionByJunc[jid] = d.act;

    qEl.innerHTML = `
      <div><strong>${{jid}} Approach Queues:</strong></div>
      <div style="margin-top:4px;">Approach 1 (E/W): <b>${{d.q[0]}}</b> | North: <b>${{d.q[1]}}</b> | South: <b>${{d.q[2]}}</b> | Approach 4 (W/E): <b>${{d.q[3]}}</b></div>
      <div style="margin-top:4px;color:#94a3b8;">Reward: <b style="color:#f8fafc;">${{d.rew}}</b> | Epsilon: <b style="color:#f8fafc;">${{d.eps.toFixed(2)}}</b> | Action: <b style="color:#38bdf8;">${{actName}}</b></div>
    `;

    // Scale Q-value bars from shared baseline [-maxMag, 0] so close values look close
    const maxMag = Math.max(25, Math.abs(d.qv[0]), Math.abs(d.qv[1]));
    const w0 = Math.max(8, Math.min(100, ((d.qv[0] + maxMag) / maxMag) * 100));
    const w1 = Math.max(8, Math.min(100, ((d.qv[1] + maxMag) / maxMag) * 100));
    const qDiff = Math.abs(d.qv[0] - d.qv[1]).toFixed(2);

    qvEl.innerHTML = `
      <div class="qbar-row"><span style="width:72px;">NS Green</span><div class="qbar-bg"><div class="qbar-fill" style="width:${{w0}}%;background:${{d.act===0?'#22c55e':'#475569'}}"></div></div><span>${{d.qv[0].toFixed(2)}}</span></div>
      <div class="qbar-row"><span style="width:72px;">EW Green</span><div class="qbar-bg"><div class="qbar-fill" style="width:${{w1}}%;background:${{d.act===1?'#22c55e':'#475569'}}"></div></div><span>${{d.qv[1].toFixed(2)}}</span></div>
      <div style="font-size:12px;color:#94a3b8;margin-top:4px;">difference: <b style="color:#e2e8f0;">${{qDiff}}</b></div>
    `;

    if (d.ov && d.hold > 0) {{
      exEl.className = "explain override";
      exEl.textContent = `Safety Hold: Min green (10s) not complete — holding ${{d.hold}} s more before RL can switch.`;
    }} else {{
      const totalQ = nsSum + ewSum;
      const loadTag = totalQ >= 8 ? "Heavy Queue Mode" : (totalQ >= 3 ? "Moderate Flow" : "Light Traffic (Fast Switch)");
      exEl.className = "explain";
      exEl.textContent = `[${{loadTag}}] N-S has ${{nsSum}} waiting vs ${{ewSum}} E-W -> RL chose ${{actName}}.`;
    }}
  }}
}}

function loop(ts) {{
  let needDraw = false;
  if (animFrom !== null) {{
    const p = Math.min(1.0, (ts - animStart) / CAM_ANIM_MS);
    const ease = 1.0 - Math.pow(1.0 - p, 3);
    camX = lerp(animFrom.x, animTo.x, ease);
    camY = lerp(animFrom.y, animTo.y, ease);
    camScale = lerp(animFrom.s, animTo.s, ease);
    if (p >= 1.0) animFrom = null;
    needDraw = true;
  }}
  if (playing) {{
    if (lastTs !== null) {{
      const dt = (ts - lastTs) / 1000.0;
      floatIdx += (dt * speed) / stepS;
      if (floatIdx >= FRAMES.length - 1) {{
        floatIdx = FRAMES.length - 1;
        playing = false;
      }}
      scrub.value = Math.floor(floatIdx);
    }}
    lastTs = ts;
    needDraw = true;
  }}
  if (needDraw) renderFrame(floatIdx);
  requestAnimationFrame(loop);
}}

// Mouse-wheel zoom & click-drag pan
let dragging = false, dragX = 0, dragY = 0;
canvas.addEventListener("mousedown", (e) => {{ dragging = true; dragX = e.clientX; dragY = e.clientY; canvas.style.cursor = "grabbing"; }});
window.addEventListener("mouseup", () => {{ dragging = false; canvas.style.cursor = "grab"; }});
window.addEventListener("mousemove", (e) => {{
  if (!dragging) return;
  camX -= (e.clientX - dragX) / camScale;
  camY += (e.clientY - dragY) / camScale;
  dragX = e.clientX; dragY = e.clientY;
  renderFrame(floatIdx);
}});
canvas.addEventListener("wheel", (e) => {{
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.15 : 0.87;
  camScale = Math.max(0.6, Math.min(14.0, camScale * factor));
  renderFrame(floatIdx);
}}, {{ passive: false }});

document.getElementById("btn-start").onclick = () => {{ playing = true; lastTs = null; }};
document.getElementById("btn-pause").onclick = () => {{ playing = false; }};
document.getElementById("btn-reset").onclick = () => {{ playing = false; floatIdx = 0; scrub.value = 0; renderFrame(0); }};
document.getElementById("juncSelect").onchange = () => renderFrame(floatIdx);
scrub.oninput = () => {{ floatIdx = parseFloat(scrub.value); renderFrame(floatIdx); }};
document.querySelectorAll(".spd-btn").forEach(btn => {{
  btn.onclick = () => {{
    speed = parseFloat(btn.dataset.spd);
    document.querySelectorAll(".spd-btn").forEach(b => b.className = "secondary spd-btn");
    btn.className = "active spd-btn";
  }};
}});
document.querySelectorAll(".cam-btn").forEach(btn => {{
  btn.onclick = () => {{
    setCamera(btn.dataset.cam);
    document.querySelectorAll(".cam-btn").forEach(b => b.className = "secondary cam-btn");
    btn.className = "active cam-btn";
    renderFrame(floatIdx);
  }};
}});

renderFrame(0);
requestAnimationFrame(loop);
</script>
</body>
</html>"""


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _page_overview(summary_df: pd.DataFrame | None) -> None:
    st.header("Overview — 3-Junction Curved Corridor (J1–J2–J3)")
    st.write(
        "This dashboard evaluates **Fixed-Time**, **SUMO Actuated**, and **Shared DQN (RL)** "
        "on an Indian Left-Hand Traffic (LHT) 3-junction arterial corridor across 3 demand levels "
        "and 10 held-out seeds (`1001–1010`)."
    )

    if summary_df is None or summary_df.empty:
        st.warning("Final evaluation results not found yet. Run `python -m v2.evaluation.final_eval`.")
    else:
        st.subheader("Best Controller per Demand Level (by Lowest Avg Waiting Time)")
        cols = st.columns(3)
        for idx, d in enumerate(["low", "medium", "high"]):
            sub = summary_df[summary_df["demand"] == d].sort_values("avg_waiting_time_s_mean")
            best = sub.iloc[0]
            with cols[idx]:
                st.metric(
                    label=f"{d.title()} Demand Winner",
                    value=CONTROLLER_LABELS.get(best["controller"], best["controller"]),
                    delta=f"{best['avg_waiting_time_s_mean']:.2f} s ({best['wait_improvement_vs_fixed_pct']:+.1f}% vs Fixed)",
                )

        # Honest comparison note if DQN trails Actuated
        dqn_rows = summary_df[summary_df["controller"] == "dqn"]
        act_rows = summary_df[summary_df["controller"] == "actuated"]
        if not dqn_rows.empty and not act_rows.empty:
            merged = dqn_rows.merge(act_rows, on="demand", suffixes=("_dqn", "_act"))
            if (merged["avg_waiting_time_s_mean_dqn"] > merged["avg_waiting_time_s_mean_act"]).any():
                st.info(
                    "**Honest Finding:** Shared DQN reduces waiting time by ~25–32% compared to "
                    "**Fixed-Time**, and trails **SUMO Actuated** at Low and Medium demand. "
                    "**A likely reason (not tested):** DQN makes decisions on a `5 s` interval with a `100`-episode "
                    "CPU training budget, whereas SUMO Actuated reacts every `1 s` via virtual induction-loop gap-out."
                )

    st.subheader("Controllers Not in v2")
    st.info(
        "Q-Learning and SARSA from v1 are preserved in the `/v1` folder and are not part of the v2 comparison."
    )

    st.subheader("Limitations")
    st.markdown(
        "- **Simulation only:** Evaluated in Eclipse SUMO 1.28.0, not on live road hardware.\n"
        "- **Single corridor:** One 3-junction arterial (`J1–J2–J3`); results may differ on larger city grids.\n"
        "- **Non-saturated High demand:** High demand (~1660 veh/hr) does not saturate this corridor, so the benefit of RL under heavy congestion is not tested.\n"
        "- **Evaluation teleports:** A few 1-hour evaluation episodes had SUMO teleports: `fixed_time` (`medium`, `seed=1008`, 2 teleports), `fixed_time` (`high`, `seed=1008`, 2 teleports), and `actuated` (`high`, `seed=1007`, 5 teleports).\n"
        "- **v1 vs v2 comparability:** v1 and v2 results are not directly comparable because the road network is different.\n"
        "- **Training vs evaluation horizon:** DQN was trained with `900 s` episodes but evaluated on 1-hour (`3600 s`) episodes.\n"
        "- **Fixed demand patterns:** Uses stationary Poisson arrivals at 3 preset levels (`720`, `1330`, `1660` veh/hr).\n"
        "- **Simple pedestrian model:** Small crossing rate (`120` ped/hr) using SUMO's default yielding without a pedestrian-specific RL reward."
    )


def build_compare_html(left_data: dict, right_data: dict) -> str:
    """Side-by-side HTML5 Canvas comparing Fixed-Time and Shared DQN on shared controls."""
    left_json = json.dumps(left_data, separators=(",", ":"))
    right_json = json.dumps(right_data, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  body {{ margin: 0; padding: 0; background: #0f172a; color: #e2e8f0; }}
  .wrap {{ padding: 8px; }}
  .controls {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
    background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 8px 12px; margin-bottom: 8px;
  }}
  button {{
    background: #2563eb; color: #fff; border: none; border-radius: 6px;
    padding: 6px 12px; font-size: 13px; font-weight: 600; cursor: pointer;
  }}
  button.secondary {{ background: #334155; }}
  button.active {{ background: #059669; }}
  input[type=range] {{ flex: 1; min-width: 140px; accent-color: #38bdf8; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  .pane {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 8px; }}
  .pane-hdr {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; }}
  canvas {{ width: 100%; height: 540px; display: block; background: #1f4d2b; border-radius: 6px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="controls">
    <button id="c-start">Start</button>
    <button id="c-pause" class="secondary">Pause</button>
    <button id="c-reset" class="secondary">Reset</button>
    <span style="font-size:12px;color:#94a3b8;">Clock: <b id="c-clock" style="color:#f8fafc;">0.0 s</b></span>
    <span style="font-size:12px;color:#94a3b8;margin-left:4px;">Junction:</span>
    <button class="secondary ccam-btn" data-cam="J1">J1</button>
    <button class="active ccam-btn" data-cam="J2">J2</button>
    <button class="secondary ccam-btn" data-cam="J3">J3</button>
    <input type="range" id="c-scrub" min="0" max="1" step="1" value="0"/>
  </div>
  <div class="grid">
    <div class="pane">
      <div class="pane-hdr"><strong>Fixed-Time</strong><span id="l-stat">Wait: 0 | Done: 0</span></div>
      <canvas id="cLeft" width="700" height="540"></canvas>
    </div>
    <div class="pane">
      <div class="pane-hdr"><strong>Shared DQN (RL)</strong><span id="r-stat">Wait: 0 | Done: 0</span></div>
      <canvas id="cRight" width="700" height="540"></canvas>
    </div>
  </div>
</div>
<script>
const LDATA = {left_json};
const RDATA = {right_json};
const cL = document.getElementById("cLeft"), ctxL = cL.getContext("2d");
const cR = document.getElementById("cRight"), ctxR = cR.getContext("2d");
const scrub = document.getElementById("c-scrub");
const maxF = Math.min(LDATA.frames.length, RDATA.frames.length) - 1;
scrub.max = Math.max(0, maxF);

let focusJ = "J2", playing = false, fIdx = 0, lastTs = null;
const stepS = LDATA.meta.step_s || 0.5;
const VEH_COL = {{ m: "#3b82f6", c: "#ef4444", a: "#f59e0b", b: "#eab308", t: "#38bdf8" }};
const VEH_DIM = {{ m: [2.2, 0.9], c: [4.3, 1.85], a: [2.9, 1.4], b: [10.5, 2.55], t: [8.0, 2.45] }};

function drawPane(ctx, canvas, data, idx) {{
  const geo = data.geometry;
  const f = data.frames[Math.min(Math.floor(idx), data.frames.length - 1)];
  const pt = geo.tl_coords[focusJ] || [550, 260];
  const sc = 5.4;
  const wx = (x) => canvas.width / 2 + (x - pt[0]) * sc;
  const wy = (y) => canvas.height / 2 - (y - pt[1]) * sc;

  ctx.fillStyle = "#1f4d2b";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (const vl of geo.vehicle_lanes) {{
    if (vl.shape.length < 2) continue;
    ctx.strokeStyle = "#2b313a";
    ctx.lineWidth = Math.max(4, (vl.width + 0.4) * sc);
    ctx.beginPath();
    ctx.moveTo(wx(vl.shape[0][0]), wy(vl.shape[0][1]));
    for (let i = 1; i < vl.shape.length; i++) ctx.lineTo(wx(vl.shape[i][0]), wy(vl.shape[i][1]));
    ctx.stroke();
  }}
  ctx.fillStyle = "#2b313a";
  for (const j of geo.junctions) {{
    if (j.shape.length < 3) continue;
    ctx.beginPath();
    ctx.moveTo(wx(j.shape[0][0]), wy(j.shape[0][1]));
    for (let i = 1; i < j.shape.length; i++) ctx.lineTo(wx(j.shape[i][0]), wy(j.shape[i][1]));
    ctx.closePath(); ctx.fill();
  }}
  for (const v of f.v) {{
    const [_, vt, x, y, ang] = v;
    const d = VEH_DIM[vt] || VEH_DIM.c;
    const L = d[0] * sc * 1.3, W = d[1] * sc * 1.3;
    ctx.save();
    ctx.translate(wx(x), wy(y));
    ctx.rotate((ang - 90) * Math.PI / 180);
    ctx.translate(-L * 0.45, 0);
    ctx.fillStyle = VEH_COL[vt] || "#ef4444";
    ctx.fillRect(-L / 2, -W / 2, L, W);
    ctx.restore();
  }}
  const t = f.tls[focusJ];
  if (t) {{
    const cx = wx(pt[0]), cy = wy(pt[1]), off = 65;
    const drawDot = (dx, dy, lamp, cd, lbl) => {{
      ctx.fillStyle = lamp === "G" ? "#22c55e" : (lamp === "Y" ? "#facc15" : "#ef4444");
      ctx.beginPath(); ctx.arc(cx + dx, cy + dy, 7, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#f8fafc"; ctx.font = "bold 10px system-ui";
      ctx.fillText(`${{lbl}}:${{cd}}s`, cx + dx - 12, cy + dy - 10);
    }};
    drawDot(off, -off, t.ns, t.cd_ns ?? t.cd ?? 0, "N");
    drawDot(-off, off, t.ns, t.cd_ns ?? t.cd ?? 0, "S");
    drawDot(off, off, t.ew, t.cd_ew ?? t.cd ?? 0, "E");
    drawDot(-off, -off, t.ew, t.cd_ew ?? t.cd ?? 0, "W");
  }}
  return f;
}}

function renderAll() {{
  const fl = drawPane(ctxL, cL, LDATA, fIdx);
  const fr = drawPane(ctxR, cR, RDATA, fIdx);
  document.getElementById("c-clock").textContent = fl.t.toFixed(1) + " s";
  document.getElementById("l-stat").textContent = `Waiting: ${{fl.st.wa}} | Done: ${{fl.st.co}}`;
  document.getElementById("r-stat").textContent = `Waiting: ${{fr.st.wa}} | Done: ${{fr.st.co}}`;
}}

function loop(ts) {{
  if (playing) {{
    if (lastTs !== null) {{
      fIdx += ((ts - lastTs) / 1000.0) / stepS;
      if (fIdx >= maxF) {{ fIdx = maxF; playing = false; }}
      scrub.value = Math.floor(fIdx);
    }}
    lastTs = ts;
    renderAll();
  }}
  requestAnimationFrame(loop);
}}

document.getElementById("c-start").onclick = () => {{ playing = true; lastTs = null; }};
document.getElementById("c-pause").onclick = () => {{ playing = false; }};
document.getElementById("c-reset").onclick = () => {{ playing = false; fIdx = 0; scrub.value = 0; renderAll(); }};
scrub.oninput = () => {{ fIdx = parseFloat(scrub.value); renderAll(); }};
document.querySelectorAll(".ccam-btn").forEach(btn => {{
  btn.onclick = () => {{
    focusJ = btn.dataset.cam;
    document.querySelectorAll(".ccam-btn").forEach(b => b.className = "secondary ccam-btn");
    btn.className = "active ccam-btn";
    renderAll();
  }};
}});
renderAll();
requestAnimationFrame(loop);
</script>
</body>
</html>"""


def _page_traffic_simulation() -> None:
    st.header("Traffic Simulation — 2D Corridor Replay")
    compare_mode = st.checkbox("Side-by-Side Compare Mode (Fixed-Time vs Shared DQN)", value=False)
    col1, col2, col3 = st.columns([1.4, 1.4, 2.2])
    with col1:
        controller = st.selectbox(
            "Controller",
            options=list(CONTROLLER_LABELS.keys()),
            format_func=lambda k: CONTROLLER_LABELS[k],
            index=2,
            disabled=compare_mode,
        )
    with col2:
        demand = st.selectbox(
            "Demand Level",
            options=list(DEMAND_LABELS.keys()),
            format_func=lambda k: DEMAND_LABELS[k],
            index=1,
        )
    with col3:
        st.info(
            "Pre-recorded 10-min SUMO TraCI clips (seed 1001, 0.5 s step) "
            "rendered at 60 FPS on HTML5 Canvas with zero Streamlit reruns."
        )

    if compare_mode:
        f_path = config.REPLAYS_DIR / f"replay_fixed_time_{demand}_seed1001.json"
        d_path = config.REPLAYS_DIR / f"replay_dqn_{demand}_seed1001.json"
        if not f_path.exists() or not d_path.exists():
            st.error("Both Fixed-Time and DQN replay files are required for compare mode.")
            return
        components.html(
            build_compare_html(load_replay_json(f_path), load_replay_json(d_path)),
            height=680,
            scrolling=False,
        )
        return

    replay_path = config.REPLAYS_DIR / f"replay_{controller}_{demand}_seed1001.json"
    if not replay_path.exists():
        st.error(f"Replay file not found: `{replay_path}`.")
        return
    data = load_replay_json(replay_path)
    components.html(build_canvas_html(data), height=990, scrolling=False)


def _page_controller_comparison(summary_df: pd.DataFrame | None) -> None:
    st.header("Controller Comparison (Seeds 1001–1010)")
    if summary_df is None or summary_df.empty:
        st.warning("Summary table not available yet.")
        return

    fig_w = px.bar(
        summary_df,
        x="demand",
        y="avg_waiting_time_s_mean",
        error_y="avg_waiting_time_s_ci95",
        color="controller",
        barmode="group",
        title="Average Waiting Time (s) with 95% Confidence Intervals",
        labels={"avg_waiting_time_s_mean": "Avg Waiting Time (s)", "demand": "Demand Level"},
    )
    st.plotly_chart(fig_w, use_container_width=True)
    st.caption(
        "**How to understand this chart:** Shorter bars mean vehicles wait less at red lights. "
        "The thin vertical whiskers show the 95% confidence interval across the 10 held-out seeds."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_q = px.bar(
            summary_df,
            x="demand",
            y="avg_queue_length_mean",
            color="controller",
            barmode="group",
            title="Average Queue Length (halting vehicles)",
            labels={"avg_queue_length_mean": "Avg Queue Length", "demand": "Demand Level"},
        )
        st.plotly_chart(fig_q, use_container_width=True)
        st.caption(
            "**How to understand this chart:** Shows the average number of stopped vehicles queued "
            "across all 3 junctions at any moment. Lower is better."
        )
    with c2:
        fig_t = px.bar(
            summary_df,
            x="demand",
            y="throughput_vph_mean",
            color="controller",
            barmode="group",
            title="Corridor Throughput (vehicles/hour)",
            labels={"throughput_vph_mean": "Throughput (veh/hr)", "demand": "Demand Level"},
        )
        st.plotly_chart(fig_t, use_container_width=True)
        st.caption(
            "**How to understand this chart:** Higher bars mean more vehicles completed their trip "
            "through the corridor within the 1-hour simulation."
        )

    st.subheader("Summary Table")
    st.dataframe(summary_df, use_container_width=True)


def _page_performance_analytics(summary_df: pd.DataFrame | None) -> None:
    st.header("Performance Analytics — Per-Junction & Travel Time")
    if summary_df is None or summary_df.empty:
        st.warning("Summary table not available yet.")
        return

    junc_rows = []
    for _, r in summary_df.iterrows():
        for jid, col in (("J1", "j1_avg_wait_s_mean"), ("J2", "j2_avg_wait_s_mean"), ("J3", "j3_avg_wait_s_mean")):
            junc_rows.append(
                {
                    "controller": r["controller"],
                    "demand": r["demand"],
                    "junction": jid,
                    "avg_wait_s": r[col],
                }
            )
    junc_df = pd.DataFrame(junc_rows)

    fig_j = px.bar(
        junc_df,
        x="junction",
        y="avg_wait_s",
        color="controller",
        facet_col="demand",
        barmode="group",
        title="Per-Junction Average Waiting Time (J1, J2, J3)",
        labels={"avg_wait_s": "Avg Waiting Time (s)", "junction": "Junction"},
    )
    st.plotly_chart(fig_j, use_container_width=True)
    st.caption(
        "**How to understand this chart:** Breaks down waiting time at each individual junction (`J1`, `J2`, `J3`). "
        "Useful for spotting whether the center junction (`J2`) bottlenecks under high demand."
    )

    fig_tt = px.bar(
        summary_df,
        x="demand",
        y="avg_travel_time_s_mean",
        error_y="avg_travel_time_s_ci95",
        color="controller",
        barmode="group",
        title="Average Vehicle Travel Time (s)",
        labels={"avg_travel_time_s_mean": "Avg Travel Time (s)", "demand": "Demand Level"},
    )
    st.plotly_chart(fig_tt, use_container_width=True)
    st.caption(
        "**How to understand this chart:** Total time from entering the corridor to exiting it, "
        "including driving time and signal delays. Lower is faster."
    )


def _page_training_analysis(train_df: pd.DataFrame | None) -> None:
    st.header("Training Analysis — Shared DQN Curriculum (100 Episodes)")
    if train_df is None or train_df.empty:
        st.warning("Training log (`dqn_training_log.csv`) not found.")
        return

    c1, c2 = st.columns(2)
    with c1:
        fig_r = px.line(
            train_df,
            x="episode",
            y="total_reward",
            color="demand",
            title="Episode Total Reward Across Curriculum",
        )
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption(
            "**How to understand this chart:** Higher (less negative) reward means smaller queues and waiting times. "
            "Reward steps down when the curriculum switches to heavier demand (`low → medium → high`)."
        )
    with c2:
        fig_l = px.line(
            train_df,
            x="episode",
            y="mean_loss",
            title="Mean Huber Loss per Episode",
        )
        st.plotly_chart(fig_l, use_container_width=True)
        st.caption(
            "**How to understand this chart:** Measures how closely the Q-network's predictions match Bellman targets. "
            "As the curriculum introduces harder traffic, loss rises briefly and then stabilizes."
        )

    c3, c4 = st.columns(2)
    with c3:
        fig_e = px.line(
            train_df,
            x="episode",
            y="epsilon",
            title="Exploration Rate (Epsilon: 1.0 → 0.05)",
        )
        st.plotly_chart(fig_e, use_container_width=True)
        st.caption(
            "**How to understand this chart:** Epsilon starts at `1.0` (random exploration) and decays linearly "
            "to exactly `0.05` on episode 100 (mostly greedy learned policy)."
        )
    with c4:
        fig_o = px.bar(
            train_df,
            x="episode",
            y="safety_overrides",
            title="Safety Layer Overrides per Episode",
        )
        st.plotly_chart(fig_o, use_container_width=True)
        st.caption(
            "**How to understand this chart:** Counts how often the hard `10 s` min-green safety rule blocked "
            "an early switch request. Overrides drop sharply as the agent learns to respect phase timing."
        )


def _page_statistical_results(stats_df: pd.DataFrame | None) -> None:
    st.header("Statistical Results — Paired Tests & Holm Correction")
    if stats_df is None or stats_df.empty:
        st.warning("Statistical results (`statistical_tests.csv`) not available yet.")
        return

    st.dataframe(stats_df, use_container_width=True)
    st.caption(
        "**How to understand this table:** Each row compares two controllers on the exact same 10 held-out seeds (`1001–1010`). "
        "`p_ttest_raw` and `p_perm_raw` are raw p-values from the paired t-test and exact `2^10 = 1024` sign-flip permutation test. "
        "`p_ttest_holm` and `p_perm_holm` apply Holm-Bonferroni correction for multiple comparisons. "
        "`Direction` tells you whether the first controller had lower (`Better`) or higher (`Worse`) waiting time."
    )


def main() -> None:
    st.set_page_config(
        page_title="Intelligent Traffic RL v2 — Dashboard",
        layout="wide",
    )
    st.sidebar.title("Traffic RL v2")
    page = st.sidebar.radio("Navigation", PAGES)

    summary_df = _read_csv_if_exists(config.FINAL_RESULTS_DIR / "summary_table.csv")
    stats_df = _read_csv_if_exists(config.FINAL_RESULTS_DIR / "statistical_tests.csv")
    train_df = _read_csv_if_exists(config.TRAINING_RESULTS_DIR / "dqn_training_log.csv")

    if page == "Overview":
        _page_overview(summary_df)
    elif page == "Traffic Simulation":
        _page_traffic_simulation()
    elif page == "Controller Comparison":
        _page_controller_comparison(summary_df)
    elif page == "Performance Analytics":
        _page_performance_analytics(summary_df)
    elif page == "Training Analysis":
        _page_training_analysis(train_df)
    elif page == "Statistical Results":
        _page_statistical_results(stats_df)


if __name__ == "__main__":
    main()
