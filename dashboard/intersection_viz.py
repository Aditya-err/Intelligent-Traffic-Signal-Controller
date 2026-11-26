"""Plotly visualization of the Indian Left-Hand-Traffic 4-way intersection."""

from __future__ import annotations

import plotly.graph_objects as go


def create_indian_intersection_figure(
    scenario_label: str,
    active_phase: str,
    avg_queue_ns_ew: float | None = None,
) -> go.Figure:
    """Build a schematic Plotly diagram of the Indian LHT 4-way intersection."""
    is_ns = active_phase.startswith("North-South")
    ns_color = "#2a9d8f" if is_ns else "#d62828"
    ew_color = "#d62828" if is_ns else "#2a9d8f"

    fig = go.Figure()

    # Asphalt road arms
    fig.add_shape(type="rect", x0=-12, y0=-100, x1=12, y1=100, fillcolor="#2b2d42", line_width=0)
    fig.add_shape(type="rect", x0=-100, y0=-12, x1=100, y1=12, fillcolor="#2b2d42", line_width=0)

    # Central intersection box
    fig.add_shape(type="rect", x0=-12, y0=-12, x1=12, y1=12, fillcolor="#3d405b", line=dict(color="#edf2f4", width=1))

    # Median dividers (dashed yellow)
    for (x0, y0, x1, y1) in [
        (0, 14, 0, 96),
        (0, -96, 0, -14),
        (14, 0, 96, 0),
        (-96, 0, -14, 0),
    ]:
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="#fcbf49", width=2, dash="dash"))

    # Stop lines (white) on the 4 incoming LHT approaches
    for (x0, y0, x1, y1) in [
        (1, 13, 11, 13),      # North incoming (heading South, keeps left = +x side)
        (-11, -13, -1, -13),  # South incoming (heading North, keeps left = -x side)
        (13, -11, 13, -1),    # East incoming (heading West, keeps left = -y side)
        (-13, 1, -13, 11),    # West incoming (heading East, keeps left = +y side)
    ]:
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="#ffffff", width=3))

    # Signal heads (North, South, East, West)
    fig.add_trace(
        go.Scatter(
            x=[6, -6, 16, -16],
            y=[16, -16, -6, 6],
            mode="markers+text",
            marker=dict(
                size=22,
                color=[ns_color, ns_color, ew_color, ew_color],
                line=dict(color="#ffffff", width=2),
            ),
            text=["N", "S", "E", "W"],
            textposition="middle center",
            textfont=dict(color="white", size=11, family="Arial Black"),
            hovertext=[
                f"North Approach: {'GREEN' if is_ns else 'RED'}",
                f"South Approach: {'GREEN' if is_ns else 'RED'}",
                f"East Approach: {'RED' if is_ns else 'GREEN'}",
                f"West Approach: {'RED' if is_ns else 'GREEN'}",
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Annotations for Indian Left-Hand Traffic ("KEEP LEFT")
    annotations = [
        dict(x=6, y=58, text="N → S<br>(Keep Left)", showarrow=True, ax=6, ay=80, arrowcolor="#8ecae6", font=dict(color="#ffffff", size=11)),
        dict(x=-6, y=-58, text="S → N<br>(Keep Left)", showarrow=True, ax=-6, ay=-80, arrowcolor="#8ecae6", font=dict(color="#ffffff", size=11)),
        dict(x=58, y=-6, text="E → W (Keep Left)", showarrow=True, ax=84, ay=-6, arrowcolor="#ffd166", font=dict(color="#ffffff", size=11)),
        dict(x=-58, y=6, text="W → E (Keep Left)", showarrow=True, ax=-84, ay=6, arrowcolor="#ffd166", font=dict(color="#ffffff", size=11)),
    ]
    if avg_queue_ns_ew is not None:
        annotations.append(
            dict(
                x=0,
                y=0,
                text=f"Active:<br>{'NS' if is_ns else 'EW'}<br>Avg Q: {avg_queue_ns_ew:.1f}",
                showarrow=False,
                font=dict(color="#ffffff", size=10),
            )
        )

    fig.update_layout(
        title=f"Indian LHT 4-Way Intersection — {scenario_label} ({active_phase})",
        xaxis=dict(range=[-102, 102], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-102, 102], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#1d1e2c",
        margin=dict(l=10, r=10, t=45, b=10),
        height=460,
        annotations=annotations,
    )
    return fig
