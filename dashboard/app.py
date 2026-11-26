"""Phase 9: Interactive Streamlit Dashboard for Intelligent-Traffic-RL."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure project root is importable when launched via `streamlit run dashboard/app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_loader import (  # noqa: E402
    CONTROLLER_DISPLAY_NAMES,
    CONTROLLER_ORDER,
    SCENARIO_DISPLAY_NAMES,
    load_dashboard_datasets,
)
from dashboard.intersection_viz import create_indian_intersection_figure  # noqa: E402


st.set_page_config(
    page_title="Intelligent Traffic Signal RL Dashboard",
    page_icon="🚦",
    layout="wide",
)


def _format_controller_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "controller" in out.columns:
        out["Controller"] = out["controller"].map(
            lambda c: CONTROLLER_DISPLAY_NAMES.get(c, c)
        )
    if "scenario" in out.columns:
        out["Scenario"] = out["scenario"].str.capitalize()
    return out


def render_overview(summary_df: pd.DataFrame) -> None:
    st.header("1. Project Overview")
    st.markdown(
        """
        This dashboard presents verified **Eclipse SUMO 1.28.0 + TraCI** simulation results
        for an **Indian Left-Hand-Traffic (LHT) 4-way urban intersection** (`--lefthand true`)
        with mixed vehicles (**motorcycles, cars, auto-rickshaws, buses, and trucks**).
        All five controllers were evaluated under identical demand, duration (`1800 s`),
        and `10` held-out seeds (`1001–1010`).
        """
    )

    cols = st.columns(3)
    for col, sc in zip(cols, ("low", "medium", "high")):
        sub = summary_df[summary_df["scenario"] == sc]
        best_row = sub.loc[sub["avg_waiting_time_s_mean"].idxmin()]
        ft_row = sub[sub["controller"] == "fixed_time"].iloc[0]
        with col:
            st.subheader(f"{sc.capitalize()} Demand")
            st.metric(
                label="Lowest Avg Waiting Time",
                value=f"{best_row['avg_waiting_time_s_mean']:.2f} s",
                delta=f"{best_row['avg_waiting_time_s_pct_vs_ft']:+.2f}% vs Fixed-Time",
            )
            st.caption(
                f"Best: **{CONTROLLER_DISPLAY_NAMES.get(best_row['controller'], best_row['controller'])}** "
                f"| Fixed-Time: `{ft_row['avg_waiting_time_s_mean']:.2f} s`"
            )

    st.subheader("Controller Highlights")
    st.markdown(
        """
        - **SARSA** achieved the lowest average waiting time in **Low** (`10.50 s`, `+23.83%` vs Fixed-Time).
        - **DQN Curriculum** achieved the lowest average waiting time in **Medium** (`16.04 s`, `+18.49%` vs Fixed-Time) and highest **High**-demand throughput (`1668.2 veh/hr`).
        - **DQN Medium-only** achieved the lowest average waiting time (`112.72 s`, `+5.44%`) and lowest queue (`62.41` veh) in **High** demand.
        """
    )


def render_simulation(summary_df: pd.DataFrame) -> None:
    st.header("2. Indian 4-Way Intersection Simulation")
    c1, c2, c3 = st.columns(3)
    with c1:
        scenario = st.selectbox(
            "Traffic Demand Scenario",
            list(SCENARIO_DISPLAY_NAMES.keys()),
            format_func=lambda s: SCENARIO_DISPLAY_NAMES[s],
        )
    with c2:
        active_phase = st.selectbox(
            "Signal Phase Preview",
            ["North-South (Straight + Left)", "East-West (Straight + Left)"],
        )
    with c3:
        ctrl = st.selectbox(
            "Controller Reference",
            list(CONTROLLER_ORDER),
            format_func=lambda c: CONTROLLER_DISPLAY_NAMES[c],
        )

    row = summary_df[
        (summary_df["scenario"] == scenario) & (summary_df["controller"] == ctrl)
    ].iloc[0]

    fig = create_indian_intersection_figure(
        SCENARIO_DISPLAY_NAMES[scenario],
        active_phase,
        avg_queue_ns_ew=float(row["avg_queue_length_mean"]),
    )
    st.plotly_chart(fig, use_container_width=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Waiting Time", f"{row['avg_waiting_time_s_mean']:.2f} s")
    k2.metric("Avg Queue Length", f"{row['avg_queue_length_mean']:.2f} veh")
    k3.metric("Throughput", f"{row['throughput_veh_per_hr_mean']:.1f} veh/hr")
    k4.metric(
        "Arrived / Remaining",
        f"{row['vehicles_arrived_mean']:.1f} / {row['vehicles_remaining_mean']:.1f}",
    )

    st.info(
        "**Indian LHT Safety Rules:** Vehicles keep left (`--lefthand true`). "
        "Lane `0` (curb/left) serves Left + Straight; Lane `1` (median/right) serves Protected Right turns. "
        "Every phase switch enforces `10 s` minimum green, `3 s` yellow, and `2 s` all-red clearance."
    )


def render_comparison(summary_df: pd.DataFrame) -> None:
    st.header("3. Controller Comparison")
    scenario = st.selectbox(
        "Select Demand Scenario",
        ("low", "medium", "high"),
        format_func=lambda s: SCENARIO_DISPLAY_NAMES[s],
        key="cmp_scenario",
    )
    sub = _format_controller_col(summary_df[summary_df["scenario"] == scenario])

    fig = px.bar(
        sub,
        x="Controller",
        y="avg_waiting_time_s_mean",
        error_y="avg_waiting_time_s_ci95_half",
        color="Controller",
        title=f"Average Waiting Time (Mean ± 95% CI) — {scenario.capitalize()} Demand",
        labels={"avg_waiting_time_s_mean": "Avg Waiting Time (s)"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    display_cols = [
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
    st.dataframe(sub[display_cols], use_container_width=True, hide_index=True)


def render_analytics(summary_df: pd.DataFrame, raw_df: pd.DataFrame | None) -> None:
    st.header("4. Performance Analytics")
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
        barmode="group",
        title=f"{selected_label} Across Scenarios",
        labels={metric_col: selected_label},
    )
    st.plotly_chart(fig, use_container_width=True)

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
            points="all",
            title=f"10-Seed Waiting Time Distribution ({sc_filter.capitalize()} Demand)",
            labels={"avg_waiting_time_s": "Avg Waiting Time (s)"},
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("Beginner-Friendly Metric Guide"):
        st.markdown(
            """
            - **Average Waiting Time (s):** Average time vehicles spend stopped (`speed < 0.1 m/s`) at the intersection. Lower is better.
            - **Average Queue Length (veh):** Average number of halted vehicles across all 8 incoming lanes. Lower is better.
            - **Average Travel Time (s):** Total time from entering to exiting the intersection network. Lower is better.
            - **Throughput (veh/hr):** Rate of vehicles completing their trip per hour. Higher is better.
            - **% Improvement vs Fixed-Time:** Relative waiting-time reduction compared to the traditional Fixed-Time signal.
            """
        )


def render_training(
    dqn_df: pd.DataFrame | None, curr_df: pd.DataFrame | None
) -> None:
    st.header("5. DQN Training Analysis")
    if dqn_df is not None:
        st.subheader("Phase 6: Medium-Only DQN Training (15 Episodes)")
        c1, c2 = st.columns(2)
        with c1:
            fig_r = px.line(
                dqn_df,
                x="episode",
                y="episode_reward",
                markers=True,
                title="Episode Reward (Medium-Only DQN)",
            )
            st.plotly_chart(fig_r, use_container_width=True)
        with c2:
            fig_l = px.line(
                dqn_df,
                x="episode",
                y="avg_loss",
                markers=True,
                title="Mean Smooth L1 Loss",
            )
            st.plotly_chart(fig_l, use_container_width=True)

        fig_e = px.line(
            dqn_df,
            x="episode",
            y="epsilon",
            markers=True,
            title="Epsilon Exploration Decay",
        )
        st.plotly_chart(fig_e, use_container_width=True)
    else:
        st.warning("Phase 6 DQN training history CSV not found.")

    if curr_df is not None:
        st.subheader("Phase 7: Curriculum Hyperparameter Experiments")
        fig_c = px.line(
            curr_df,
            x="episode",
            y="episode_reward",
            color="experiment",
            markers=True,
            title="Curriculum Episode Rewards Across Hyperparameter Runs",
        )
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.warning("Phase 7 curriculum training history CSV not found.")


def render_statistics(
    summary_df: pd.DataFrame, stats_df: pd.DataFrame | None
) -> None:
    st.header("6. Statistical Results (Phase 8)")
    st.markdown(
        """
        All controllers were evaluated across **10 held-out seeds (`1001–1010`)**.
        Significance is tested using both a **two-sided Paired $t$-test** and an
        **Exact Paired Sign-Flip Permutation Test** ($2^{10} = 1024$ permutations, $\\alpha = 0.05$).
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
        st.dataframe(filtered, use_container_width=True, hide_index=True)
    else:
        st.warning("Statistical comparison CSV (`final_eval_stats.csv`) not found.")

    st.subheader("Summary Mean, Std & 95% Confidence Intervals")
    st.dataframe(_format_controller_col(summary_df), use_container_width=True, hide_index=True)


def main() -> None:
    st.title("🚦 Intelligent Traffic Signal Controller — RL Dashboard")
    datasets = load_dashboard_datasets()

    summary_res = datasets["final_summary"]
    if not summary_res.ok or summary_res.df is None:
        st.error(
            f"Cannot load required Phase 8 summary data: {summary_res.error}. "
            "Run `python -m evaluation.final_eval` first."
        )
        return

    section = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Traffic Simulation",
            "Controller Comparison",
            "Performance Analytics",
            "Training Analysis",
            "Statistical Results",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Workspace: `Intelligent-Traffic-RL` (E: drive)")
    st.sidebar.caption("All metrics come from verified Phase 6–8 SUMO CSVs.")

    summary_df = summary_res.df
    raw_df = datasets["final_raw"].df if datasets["final_raw"].ok else None
    stats_df = datasets["final_stats"].df if datasets["final_stats"].ok else None
    dqn_df = datasets["dqn_training"].df if datasets["dqn_training"].ok else None
    curr_df = (
        datasets["curriculum_training"].df
        if datasets["curriculum_training"].ok
        else None
    )

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
