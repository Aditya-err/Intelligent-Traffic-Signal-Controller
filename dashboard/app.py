"""AI Traffic Management Control Center — Streamlit Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path
import time

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.canvas_viz import (  # noqa: E402
    build_live_canvas_html,
    get_diagnostic_rows,
)
from dashboard.data_loader import (  # noqa: E402
    CONTROLLER_COLORS,
    CONTROLLER_DISPLAY_NAMES,
    CONTROLLER_ORDER,
    SCENARIO_DISPLAY_NAMES,
    SUMMARY_COLUMN_LABELS,
    format_p_value,
    load_dashboard_datasets,
)
from dashboard.intersection_viz import create_indian_intersection_figure  # noqa: E402
from dashboard.live_sim import LiveSimSession  # noqa: E402


st.set_page_config(
    page_title="AI Traffic Management Control Center",
    page_icon="🚦",
    layout="wide",
)

NAV_SECTIONS = [
    "Overview",
    "Traffic Simulation",
    "Controller Comparison",
    "Performance Analytics",
    "Training Analysis",
    "Statistical Results",
]


def _inject_control_center_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background-color: #111827;
            border: 1px solid rgba(59, 130, 246, 0.28);
            border-radius: 10px;
            padding: 0.75rem 1rem;
        }
        .cc-card {
            background-color: #111827;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.9rem;
        }
        .cc-badge {
            display: inline-block;
            background: rgba(59, 130, 246, 0.18);
            color: #93c5fd;
            border: 1px solid rgba(59, 130, 246, 0.4);
            border-radius: 999px;
            padding: 0.15rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _style_plotly(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0"),
        margin=dict(l=24, r=24, t=54, b=24),
    )
    return fig


def _format_controller_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "controller" in out.columns:
        out["Controller"] = out["controller"].map(
            lambda c: CONTROLLER_DISPLAY_NAMES.get(c, c)
        )
    if "scenario" in out.columns:
        out["Scenario"] = out["scenario"].str.capitalize()
    return out


def _go_to_simulation() -> None:
    st.session_state["nav_section"] = "Traffic Simulation"


def render_overview(summary_df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="cc-card">
            <span class="cc-badge">ECLIPSE SUMO 1.28.0 + TRACI + CPU PYTORCH</span>
            <h2 style="margin:0.45rem 0 0.3rem 0;">AI Traffic Management Control Center</h2>
            <p style="color:#cbd5e1; margin:0;">
                Adaptive Reinforcement Learning signal control for an <strong>Indian Left-Hand-Traffic (LHT)
                4-way urban intersection</strong> (<code>--lefthand true</code>) with mixed vehicles
                (motorcycles, cars, auto-rickshaws, buses, trucks) evaluated across <strong>10 held-out seeds (1001–1010)</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_btn, _ = st.columns([1.6, 3.4])
    c_btn.button(
        "▶ Launch Live 2D Simulation",
        type="primary",
        use_container_width=True,
        on_click=_go_to_simulation,
    )

    st.subheader("Best Controller by Traffic Demand (Phase 8 Benchmark)")
    cols = st.columns(3)
    for col, sc in zip(cols, ("low", "medium", "high")):
        sub = summary_df[summary_df["scenario"] == sc]
        best_row = sub.loc[sub["avg_waiting_time_s_mean"].idxmin()]
        ft_row = sub[sub["controller"] == "fixed_time"].iloc[0]
        best_name = CONTROLLER_DISPLAY_NAMES.get(
            best_row["controller"], best_row["controller"]
        )
        with col:
            st.metric(
                label=f"{sc.capitalize()} Demand — Best: {best_name}",
                value=f"{best_row['avg_waiting_time_s_mean']:.2f} s",
                delta=f"{best_row['avg_waiting_time_s_pct_vs_ft']:+.2f}% vs Fixed-Time ({ft_row['avg_waiting_time_s_mean']:.2f} s)",
            )
            st.caption(
                f"Avg Queue: `{best_row['avg_queue_length_mean']:.2f} veh` · "
                f"Throughput: `{best_row['throughput_veh_per_hr_mean']:.1f} veh/hr`"
            )

    st.subheader("Beginner-Friendly Guide to Controllers & Metrics")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            """
            **Controllers Compared**
            - **Fixed-Time:** Traditional timer (`25 s` NS / `20 s` EW) that ignores traffic queues.
            - **Q-Learning & SARSA:** Table-based RL agents (`96` states); **SARSA** leads Low demand (`10.50 s`).
            - **DQN (Deep Q-Network):** Neural-network controller (`18 → 128 → 64 → 2`) that learns signal decisions from live lane queues and waiting times.
            """
        )
    with g2:
        st.markdown(
            """
            **Key Metrics Explained**
            - **Waiting Time:** How long vehicles spend stopped at red lights.
            - **Queue Length:** Number of vehicles waiting at the signal across all 8 incoming lanes.
            - **Throughput:** Number of vehicles passing the intersection per hour.
            - **Episode Reward:** How well the RL controller performed according to its reward function.
            """
        )


def _get_or_create_live_session(scenario: str, controller_key: str) -> LiveSimSession:
    session: LiveSimSession | None = st.session_state.get("live_sim_session")
    if (
        session is None
        or session.scenario != scenario
        or session.controller_key != controller_key
    ):
        if session is not None:
            session.close()
        session = LiveSimSession(scenario=scenario, controller_key=controller_key)
        session.reset(warmup_steps=20)  # 20s warmup (1s/step) so vehicles are visible immediately
        st.session_state["live_sim_session"] = session
    return session


def render_simulation(summary_df: pd.DataFrame) -> None:
    st.subheader("Live 2D Indian 4-Way Traffic Simulation")
    c1, c2, c3 = st.columns(3)
    with c1:
        scenario = st.selectbox(
            "Traffic Demand",
            list(SCENARIO_DISPLAY_NAMES.keys()),
            index=1,
            format_func=lambda s: SCENARIO_DISPLAY_NAMES[s],
        )
    with c2:
        ctrl = st.selectbox(
            "Controller",
            list(CONTROLLER_ORDER),
            index=4,
            format_func=lambda c: CONTROLLER_DISPLAY_NAMES[c],
        )
    with c3:
        speed_label = st.selectbox("Simulation Speed", ["1x", "2x", "4x"], index=0)

    try:
        session = _get_or_create_live_session(scenario, ctrl)

        b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 2])
        if b1.button("▶ Start", type="primary", use_container_width=True):
            session.is_running = True
        if b2.button("⏸ Pause", use_container_width=True):
            session.is_running = False
        if b3.button("⏯ Resume", use_container_width=True):
            if not session.terminated:
                session.is_running = True
        if b4.button("↺ Reset", use_container_width=True):
            session.is_running = False
            session.reset(warmup_steps=20)

        status_text = (
            "COMPLETED"
            if session.terminated
            else ("RUNNING" if session.is_running else "PAUSED")
        )
        b5.markdown(f"**Status:** `{status_text}` · **Speed:** `{speed_label}`")

        run_interval = (
            {"1x": 0.5, "2x": 0.25, "4x": 0.15}[speed_label]
            if (session.is_running and not session.terminated)
            else None
        )
        steps_per_tick = 2 if speed_label == "4x" else 1

        @st.fragment(run_every=run_interval)
        def _live_viewport() -> None:
            if session.is_running and not session.terminated:
                for _ in range(steps_per_tick):
                    if session.terminated:
                        break
                    session.step_once()

            snap = session.snapshot
            assert snap is not None

            action_lbl = "0: North-South" if snap.selected_action == 0 else "1: East-West"
            phase_lbl = "North-South" if snap.current_group == 0 else "East-West"

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Sim Clock", f"{snap.sim_time} s")
            m2.metric("Active Controller", CONTROLLER_DISPLAY_NAMES[snap.controller_name])
            m3.metric("Total Spawned", f"{snap.total_spawned}")
            m4.metric("Active Vehicles", f"{snap.vehicle_count}")
            m5.metric("Completed", f"{snap.completed_vehicles}")
            m6.metric("Waiting / Queue", f"{snap.waiting_vehicles} / {snap.queue_length}")

            components.html(build_live_canvas_html(snap), height=640)

            st.caption(
                f"**Selected Action:** `{action_lbl}` · **Current Signal Phase:** `{phase_lbl}` "
                f"(Green elapsed: `{snap.green_elapsed} s`) · **Live SUMO TLS State:** `{snap.tls_state}`"
            )
            with st.expander("TraCI Vehicle & Viewport Diagnostics", expanded=False):
                diag_rows = get_diagnostic_rows(snap)
                rendered_cnt = sum(
                    1 for r in diag_rows if r["rendered_status"] == "RENDERED"
                )
                st.caption(
                    f"Active in SUMO: `{snap.vehicle_count}` · Rendered in Canvas Viewport: `{rendered_cnt}`"
                )
                if diag_rows:
                    st.dataframe(
                        pd.DataFrame(diag_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
            if snap.terminated:
                st.success("Simulation episode completed (1800 s). Click Reset to replay.")

        _live_viewport()

        st.info(
            "**How to understand this view?** Vehicles spawn from all four approaches (N, S, E, W) "
            "and drive on the **left** (Indian LHT). Each N/S/E/W signal head shows **Red (top), Yellow (middle), "
            "and Green (bottom)** directly from SUMO. Use **Start / Pause / Resume / Reset** to control simulation time."
        )

    except Exception as exc:  # noqa: BLE001
        st.warning(f"Live SUMO TraCI unavailable ({exc}). Showing static preview.")
        row = summary_df[
            (summary_df["scenario"] == scenario) & (summary_df["controller"] == ctrl)
        ].iloc[0]
        st.plotly_chart(
            create_indian_intersection_figure(
                SCENARIO_DISPLAY_NAMES[scenario],
                "North-South (Straight + Left)",
                avg_queue_ns_ew=float(row["avg_queue_length_mean"]),
            ),
            use_container_width=True,
        )


def render_comparison(summary_df: pd.DataFrame) -> None:
    st.subheader("Controller Comparison (10 Held-Out Seeds)")
    scenario = st.selectbox(
        "Select Demand Scenario",
        ("low", "medium", "high"),
        format_func=lambda s: SCENARIO_DISPLAY_NAMES[s],
        key="cmp_scenario",
    )
    sub = _format_controller_col(summary_df[summary_df["scenario"] == scenario])
    best_idx = sub["avg_waiting_time_s_mean"].idxmin()
    best_ctrl = sub.loc[best_idx, "Controller"]
    best_wait = sub.loc[best_idx, "avg_waiting_time_s_mean"]
    st.success(f"**Lowest Average Waiting Time ({scenario.capitalize()}):** `{best_ctrl}` (`{best_wait:.2f} s`)")

    fig = px.bar(
        sub,
        x="Controller",
        y="avg_waiting_time_s_mean",
        error_y="avg_waiting_time_s_ci95_half",
        color="Controller",
        color_discrete_map=CONTROLLER_COLORS,
        title=f"Average Waiting Time (Mean ± 95% CI) — {scenario.capitalize()} Demand",
        labels={"avg_waiting_time_s_mean": "Avg Waiting Time (s)"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(_style_plotly(fig), use_container_width=True)
    st.caption(
        "**How to understand this chart?** Shorter bars mean vehicles spend less time stopped at red lights. "
        "The thin vertical error bars show the 95% confidence interval across 10 independent random seeds."
    )

    cols_to_show = [
        "Controller",
        "avg_waiting_time_s_mean",
        "avg_waiting_time_s_std",
        "avg_waiting_time_s_pct_vs_ft",
        "avg_queue_length_mean",
        "avg_travel_time_s_mean",
        "throughput_veh_per_hr_mean",
        "vehicles_arrived_mean",
        "vehicles_remaining_mean",
        "episode_reward_mean",
    ]
    display_df = sub[cols_to_show].rename(columns=SUMMARY_COLUMN_LABELS)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_analytics(summary_df: pd.DataFrame, raw_df: pd.DataFrame | None) -> None:
    st.subheader("Performance Analytics Across Demand Regimes")
    df = _format_controller_col(summary_df)

    metric_map = {
        "Average Waiting Time (s)": "avg_waiting_time_s_mean",
        "Average Queue Length (veh)": "avg_queue_length_mean",
        "Average Travel Time (s)": "avg_travel_time_s_mean",
        "Throughput (veh/hr)": "throughput_veh_per_hr_mean",
        "% Waiting Improvement vs Fixed-Time": "avg_waiting_time_s_pct_vs_ft",
    }
    selected_label = st.selectbox("Metric to Inspect", list(metric_map.keys()))
    metric_col = metric_map[selected_label]

    fig = px.bar(
        df,
        x="Scenario",
        y=metric_col,
        color="Controller",
        color_discrete_map=CONTROLLER_COLORS,
        barmode="group",
        title=f"{selected_label} Across Scenarios",
        labels={metric_col: selected_label},
    )
    st.plotly_chart(_style_plotly(fig), use_container_width=True)
    st.caption(
        "**How to understand this chart?** Compare how each controller scales from Low to High congestion. "
        "For waiting time, queue length, and travel time, lower values are better; for throughput and % improvement, higher is better."
    )

    if raw_df is not None:
        raw_fmt = _format_controller_col(raw_df)
        sc_filter = st.selectbox(
            "Waiting-Time Distribution Scenario",
            ("low", "medium", "high"),
            format_func=lambda s: s.capitalize(),
            key="dist_sc",
        )
        fig_box = px.box(
            raw_fmt[raw_fmt["scenario"] == sc_filter],
            x="Controller",
            y="avg_waiting_time_s",
            color="Controller",
            color_discrete_map=CONTROLLER_COLORS,
            points="all",
            title=f"10-Seed Waiting Time Distribution ({sc_filter.capitalize()} Demand)",
            labels={"avg_waiting_time_s": "Avg Waiting Time (s)"},
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(_style_plotly(fig_box), use_container_width=True)
        st.caption(
            "**How to understand this chart?** Each dot is one 30-minute SUMO run with a different random seed. "
            "A tighter, lower box means the controller is both fast and consistent."
        )


def render_training(
    dqn_df: pd.DataFrame | None, curr_df: pd.DataFrame | None
) -> None:
    st.subheader("DQN Training Analysis (Phases 6 & 7)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Phase 6 Episodes", "15 (Medium)")
    c2.metric("Phase 7 Curriculum", "15 (L→M→H→Mix)")
    c3.metric("Controlled Hparam Runs", "6")
    c4.metric("Selected Curriculum LR", "5e-4 (exp1_lr_5e4)")

    if dqn_df is not None:
        st.markdown("**Phase 6: Medium-Only DQN (`lr=1e-3`, `batch=64`, `gamma=0.95`)**")
        col1, col2 = st.columns(2)
        with col1:
            fig_r = px.line(
                dqn_df,
                x="episode",
                y="episode_reward",
                markers=True,
                title="Episode Reward (Medium-Only DQN)",
                labels={"episode": "Episode", "episode_reward": "Total Shaped Reward"},
            )
            st.plotly_chart(_style_plotly(fig_r), use_container_width=True)
        with col2:
            fig_l = px.line(
                dqn_df,
                x="episode",
                y="avg_loss",
                markers=True,
                title="Mean Smooth L1 (Huber) Loss",
                labels={"episode": "Episode", "avg_loss": "Smooth L1 Loss"},
            )
            st.plotly_chart(_style_plotly(fig_l), use_container_width=True)

        fig_e = px.line(
            dqn_df,
            x="episode",
            y="epsilon",
            markers=True,
            title="Epsilon Exploration Decay (1.0 → 0.05)",
            labels={"episode": "Episode", "epsilon": "Epsilon (ε)"},
        )
        st.plotly_chart(_style_plotly(fig_e), use_container_width=True)
        st.caption(
            "**How to understand these charts?** As epsilon decays from `1.0` (random exploration) toward `0.05` (learned policy), "
            "Episode Reward becomes less negative, showing reduced intersection waiting and queue penalties."
        )
    else:
        st.warning("Phase 6 DQN training history CSV not found.")

    if curr_df is not None:
        st.markdown("**Phase 7: Curriculum Hyperparameter Experiments**")
        fig_c = px.line(
            curr_df,
            x="episode",
            y="episode_reward",
            color="experiment",
            markers=True,
            title="Curriculum Episode Rewards Across Hyperparameter Runs",
            labels={
                "episode": "Episode (1-4 Low, 5-8 Med, 9-12 High, 13-15 Mixed)",
                "episode_reward": "Total Shaped Reward",
                "experiment": "Experiment",
            },
        )
        st.plotly_chart(_style_plotly(fig_c), use_container_width=True)
        st.caption(
            "**How to understand this chart?** Episodes 9–12 test High congestion (naturally lower rewards due to higher volume), "
            "and Episodes 13–15 cycle Low → Medium → High. `exp1_lr_5e4` achieved the highest Mixed-stage training reward."
        )
    else:
        st.warning("Phase 7 curriculum training history CSV not found.")


def render_statistics(
    summary_df: pd.DataFrame, stats_df: pd.DataFrame | None
) -> None:
    st.subheader("Statistical Results (Phase 8 — 10 Held-Out Seeds)")
    st.markdown(
        """
        Evaluated across **10 matched seeds (`1001–1010`)** using both a **two-sided Paired $t$-test**
        and an **Exact Paired Sign-Flip Permutation Test** ($2^{10} = 1024$ permutations, $\\alpha = 0.05$).
        """
    )
    if stats_df is not None:
        sc = st.selectbox(
            "Filter by Scenario",
            ("all", "low", "medium", "high"),
            format_func=lambda s: s.capitalize(),
            key="stats_sc",
        )
        filtered = stats_df if sc == "all" else stats_df[stats_df["scenario"] == sc]
        disp = filtered.copy()
        disp["Scenario"] = disp["scenario"].str.capitalize()
        disp["Comparison"] = disp["comparison"].str.replace("_", " ").str.title()
        disp["Metric"] = disp["metric"].str.replace("_", " ").str.title()
        disp["Paired t p-value"] = disp["paired_t_p"].apply(format_p_value)
        disp["Exact Perm p-value"] = disp["perm_p"].apply(format_p_value)
        disp["Significant (p < 0.05)"] = disp["significant_005"].map(
            lambda x: "Yes" if bool(x) else "No"
        )
        cols = [
            "Scenario",
            "Comparison",
            "Metric",
            "mean_diff",
            "t_stat",
            "Paired t p-value",
            "Exact Perm p-value",
            "Significant (p < 0.05)",
        ]
        table_df = disp[cols].rename(
            columns={"mean_diff": "Mean Diff", "t_stat": "t-Statistic"}
        )

        def _highlight_sig(val: str) -> str:
            if val == "Yes":
                return "background-color: rgba(16, 185, 129, 0.22); color: #6ee7b7; font-weight: 600;"
            return "color: #94a3b8;"

        st.dataframe(
            table_df.style.map(_highlight_sig, subset=["Significant (p < 0.05)"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Statistical comparison CSV (`final_eval_stats.csv`) not found.")

    st.info(
        "**Statistical Limitation:** Paired $t$-tests and permutation tests are reported without "
        "Bonferroni multiple-comparison adjustment; borderline $p$-values (`0.03–0.04`) on High demand "
        "should be interpreted conservatively."
    )

    st.subheader("Full Summary Table (Mean, Std, 95% CI)")
    sum_fmt = _format_controller_col(summary_df)
    keep_cols = [c for c in SUMMARY_COLUMN_LABELS if c in sum_fmt.columns]
    st.dataframe(
        sum_fmt[keep_cols].rename(columns=SUMMARY_COLUMN_LABELS),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    _inject_control_center_css()
    datasets = load_dashboard_datasets()

    summary_res = datasets["final_summary"]
    if not summary_res.ok or summary_res.df is None:
        st.error(
            f"Cannot load required Phase 8 summary data: {summary_res.error}. "
            "Run `python -m evaluation.final_eval` first."
        )
        return

    if "nav_section" not in st.session_state:
        st.session_state["nav_section"] = "Overview"

    st.sidebar.markdown("### 🚦 Control Center")
    st.sidebar.caption("Indian LHT 4-Way Intersection RL")
    section = st.sidebar.radio(
        "Navigation",
        NAV_SECTIONS,
        key="nav_section",
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("SUMO 1.28.0 · CPU PyTorch · 10 Held-Out Seeds")

    summary_df = summary_res.df
    raw_df = datasets["final_raw"].df if datasets["final_raw"].ok else None
    stats_df = datasets["final_stats"].df if datasets["final_stats"].ok else None
    dqn_df = datasets["dqn_training"].df if datasets["dqn_training"].ok else None
    curr_df = (
        datasets["curriculum_training"].df
        if datasets["curriculum_training"].ok
        else None
    )

    if section != "Traffic Simulation" and "live_sim_session" in st.session_state:
        st.session_state["live_sim_session"].is_running = False

    if section == "Overview":
        render_overview(summary_df)
    elif section == "Traffic Simulation":
        render_simulation(summary_df)
    elif section == "Controller Comparison":
        render_comparison(summary_df)
    elif section == "Performance Analytics":
        render_analytics(summary_df, raw_df)
    elif section == "Training Analysis":
        render_training(dqn_df, curr_df)
    elif section == "Statistical Results":
        render_statistics(summary_df, stats_df)


if __name__ == "__main__":
    main()
