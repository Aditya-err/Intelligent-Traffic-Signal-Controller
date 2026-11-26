"""Plotly 2D visualization of the Indian Left-Hand-Traffic 4-way intersection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from dashboard.live_sim import SimSnapshot


VTYPE_STYLE: dict[str, tuple[str, str, int, str]] = {
    "motorcycle": ("Motorcycle", "#fb923c", 12, "triangle-up"),
    "car": ("Car", "#38bdf8", 16, "diamond-tall"),
    "auto_rickshaw": ("Auto-Rickshaw", "#34d399", 15, "pentagon"),
    "bus": ("Bus", "#f43f5e", 22, "hexagon2"),
    "truck": ("Truck", "#cbd5e1", 19, "square"),
}


def _active_lamp(chars: str) -> str:
    """Return 'green', 'yellow', or 'red' from a SUMO TLS link-state slice."""
    if any(c in ("G", "g") for c in chars):
        return "green"
    if any(c == "y" for c in chars):
        return "yellow"
    return "red"


def _add_intersection_base(fig: go.Figure, view_range: int = 75) -> None:
    """Draw real-scale 4-way Indian LHT road geometry (center at 0,0)."""
    road_half = 7.2
    stop_dist = 11.0

    fig.add_shape(
        type="rect",
        x0=-road_half,
        y0=-view_range,
        x1=road_half,
        y1=view_range,
        fillcolor="#1e293b",
        line=dict(color="#64748b", width=1.5),
    )
    fig.add_shape(
        type="rect",
        x0=-view_range,
        y0=-road_half,
        x1=view_range,
        y1=road_half,
        fillcolor="#1e293b",
        line=dict(color="#64748b", width=1.5),
    )
    fig.add_shape(
        type="rect",
        x0=-road_half,
        y0=-road_half,
        x1=road_half,
        y1=road_half,
        fillcolor="#334155",
        line=dict(color="#94a3b8", width=1),
    )

    for offset in (-3.5, 3.5):
        fig.add_shape(type="line", x0=offset, y0=stop_dist, x1=offset, y1=view_range - 1, line=dict(color="#94a3b8", width=1.2, dash="dot"))
        fig.add_shape(type="line", x0=offset, y0=-view_range + 1, x1=offset, y1=-stop_dist, line=dict(color="#94a3b8", width=1.2, dash="dot"))
        fig.add_shape(type="line", x0=stop_dist, y0=offset, x1=view_range - 1, y1=offset, line=dict(color="#94a3b8", width=1.2, dash="dot"))
        fig.add_shape(type="line", x0=-view_range + 1, y0=offset, x1=-stop_dist, y1=offset, line=dict(color="#94a3b8", width=1.2, dash="dot"))

    for x0, y0, x1, y1 in [
        (0, stop_dist, 0, view_range - 1),
        (0, -view_range + 1, 0, -stop_dist),
        (stop_dist, 0, view_range - 1, 0),
        (-view_range + 1, 0, -stop_dist, 0),
    ]:
        fig.add_shape(
            type="line",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color="#f59e0b", width=2.5, dash="dash"),
        )

    for x0, y0, x1, y1 in [
        (0.4, stop_dist, road_half - 0.3, stop_dist),
        (-road_half + 0.3, -stop_dist, -0.4, -stop_dist),
        (stop_dist, -road_half + 0.3, stop_dist, -0.4),
        (-stop_dist, 0.4, -stop_dist, road_half - 0.3),
    ]:
        fig.add_shape(
            type="line",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color="#f8fafc", width=3.5),
        )


def _add_signal_heads(fig: go.Figure, tls_state: str) -> None:
    """Draw 4 physical signal heads (N, S, E, W), each with 3 stacked R/Y/G circular lamps."""
    tls = tls_state.ljust(12, "r")
    heads = [
        ("N", 14.5, 22.0, tls[6:9]),
        ("S", -14.5, -22.0, tls[0:3]),
        ("E", 22.0, -14.5, tls[3:6]),
        ("W", -22.0, 14.5, tls[9:12]),
    ]

    lamp_x: list[float] = []
    lamp_y: list[float] = []
    lamp_hover: list[str] = []

    for label, hx, hy, slice_str in heads:
        active = _active_lamp(slice_str)
        # Signal housing rectangle
        fig.add_shape(
            type="rect",
            x0=hx - 4.0,
            y0=hy - 10.0,
            x1=hx + 4.0,
            y1=hy + 10.0,
            fillcolor="#090d16",
            line=dict(color="#cbd5e1", width=2),
        )
        fig.add_annotation(
            x=hx,
            y=hy + 12.6,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color="#f8fafc", size=13),
        )

        # Three vertically stacked circular lamps: Red (top), Yellow (mid), Green (bottom)
        specs = [
            ("red", hy + 5.5, "#ff2e2e", "#450a0a"),
            ("yellow", hy, "#ffd60a", "#422006"),
            ("green", hy - 5.5, "#00e676", "#052e16"),
        ]
        r = 2.2
        for lamp_name, ly, bright, dim in specs:
            is_on = active == lamp_name
            fig.add_shape(
                type="circle",
                x0=hx - r,
                y0=ly - r,
                x1=hx + r,
                y1=ly + r,
                fillcolor=bright if is_on else dim,
                line=dict(color="#ffffff" if is_on else "#334155", width=2.2 if is_on else 1),
            )
            lamp_x.append(hx)
            lamp_y.append(ly)
            lamp_hover.append(
                f"{label} Signal — {lamp_name.upper()} ({'ON' if is_on else 'OFF'}) | TLS: {slice_str}"
            )

    fig.add_trace(
        go.Scatter(
            x=lamp_x,
            y=lamp_y,
            mode="markers",
            marker=dict(size=12, opacity=0.01),
            hovertext=lamp_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )


def create_indian_intersection_figure(
    scenario_label: str,
    active_phase: str,
    avg_queue_ns_ew: float | None = None,
) -> go.Figure:
    """Build a static schematic Plotly diagram of the Indian LHT 4-way intersection."""
    is_ns = active_phase.startswith("North-South")
    tls_state = "GGrrrrGGrrrr" if is_ns else "rrrGGrrrrGGr"

    fig = go.Figure()
    view_range = 75
    _add_intersection_base(fig, view_range=view_range)
    _add_signal_heads(fig, tls_state)

    annotations = [
        dict(x=3.5, y=48, text="N → S (Keep Left)", showarrow=True, ax=3.5, ay=66, arrowcolor="#38bdf8", font=dict(color="#e2e8f0", size=11)),
        dict(x=-3.5, y=-48, text="S → N (Keep Left)", showarrow=True, ax=-3.5, ay=-66, arrowcolor="#38bdf8", font=dict(color="#e2e8f0", size=11)),
        dict(x=48, y=-3.5, text="E → W (Keep Left)", showarrow=True, ax=66, ay=-3.5, arrowcolor="#fbbf24", font=dict(color="#e2e8f0", size=11)),
        dict(x=-48, y=3.5, text="W → E (Keep Left)", showarrow=True, ax=-66, ay=3.5, arrowcolor="#fbbf24", font=dict(color="#e2e8f0", size=11)),
    ]
    if avg_queue_ns_ew is not None:
        annotations.append(
            dict(
                x=0,
                y=0,
                text=f"Active: {'NS' if is_ns else 'EW'}<br>Avg Q: {avg_queue_ns_ew:.1f}",
                showarrow=False,
                font=dict(color="#f8fafc", size=10),
            )
        )

    fig.update_layout(
        title=f"Indian LHT 4-Way Intersection — {scenario_label} ({active_phase})",
        xaxis=dict(range=[-view_range, view_range], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-view_range, view_range], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0f172a",
        font=dict(color="#f1f5f9"),
        margin=dict(l=10, r=10, t=48, b=10),
        height=560,
        annotations=annotations,
    )
    return fig


def create_live_intersection_figure(snapshot: "SimSnapshot") -> go.Figure:
    """Render live SUMO oriented vehicles and 3-lamp N/S/E/W signal heads."""
    fig = go.Figure()
    view_range = 75
    _add_intersection_base(fig, view_range=view_range)
    _add_signal_heads(fig, snapshot.tls_state)

    for vtype_key, (label, color, size, symbol) in VTYPE_STYLE.items():
        subset = [
            v
            for v in snapshot.vehicles
            if v["vtype"] == vtype_key
            and abs(v["x"]) <= view_range
            and abs(v["y"]) <= view_range
        ]
        fig.add_trace(
            go.Scatter(
                x=[v["x"] for v in subset],
                y=[v["y"] for v in subset],
                mode="markers",
                name=f"{label} ({len(subset)})",
                marker=dict(
                    size=size,
                    color=color,
                    symbol=symbol,
                    angle=[v.get("angle", 0.0) for v in subset],
                    line=dict(color="#0b1120", width=1.2),
                ),
                hovertext=[
                    f"{v['id']} ({label}) | Speed: {v['speed']:.1f} m/s | Heading: {v.get('angle', 0.0):.0f}°"
                    for v in subset
                ],
                hoverinfo="text",
            )
        )

    annotations = [
        dict(x=3.5, y=68, text="<b>N → S</b> (Keep Left)", showarrow=False, font=dict(color="#cbd5e1", size=10)),
        dict(x=-3.5, y=-68, text="<b>S → N</b> (Keep Left)", showarrow=False, font=dict(color="#cbd5e1", size=10)),
        dict(x=56, y=-3.5, text="<b>E → W</b>", showarrow=False, font=dict(color="#cbd5e1", size=10)),
        dict(x=-56, y=3.5, text="<b>W → E</b>", showarrow=False, font=dict(color="#cbd5e1", size=10)),
    ]

    group_lbl = "North-South" if snapshot.current_group == 0 else "East-West"
    fig.update_layout(
        title=(
            f"Live Indian LHT Intersection — {snapshot.scenario.capitalize()} Demand | "
            f"t = {snapshot.sim_time} s | Signal: {group_lbl} ({snapshot.tls_state})"
        ),
        xaxis=dict(range=[-view_range, view_range], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(
            range=[-view_range, view_range],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0f172a",
        font=dict(color="#f1f5f9"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=650,
        uirevision="indian_lht_live",
        annotations=annotations,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(17, 24, 39, 0.85)",
        ),
    )
    return fig
