import sys
import os

# Ensure the viewer_app directory is on the path so `utils` can be imported
# whether the app is launched via `streamlit run app.py` from inside viewer_app/
# or from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.data_processing import process_fit, align_pair
from utils.calculations import (
    find_speed_intersections_same_time,
    filter_intersections_by_segment_duration,
    segment_by_intersections,
    segment_stats,
    power_split_stats,
    build_roles,
    compute_intersection_energy_table,
)
from utils.plotting import (
    plot_fit_interactive,
    compare_rides_power_speed,
    plot_power_speed_with_residuals,
    plot_speed_with_intersections,
    plot_opposed_segment_metric,
    plot_opposed_power_split_neutral_overlay,
    plot_opposed_power_split_active_overlay,
    plot_intersection_residuals,
    plot_cadence_optimization,
)

st.set_page_config(page_title="Parloebs Analyse Viewer", layout="wide")

# ---------- UI ----------
st.title("Parloebs Analyse – interactive viewer")
st.markdown("Upload two FIT files, align them on time, and explore speeds, power, intersections, and segments.")

with st.expander("1) Riders & files", expanded=True):
    col_r1, col_r2 = st.columns(2, gap="large")
    with col_r1:
        st.markdown("**Rider 1**")
        file1 = st.file_uploader("FIT file", type=["fit"], key="r1_file")
        rider1_name = st.text_input("Name", "Rider 1", key="r1_name")
        rider1_weight = st.number_input("Weight (kg)", 0.0, 200.0, 70.0, step=0.1, key="r1_weight")
        speed_mode1 = st.selectbox("Speed mode", ["sensor", "cadence", "gps", "gps_smooth"], index=1, key="r1_speedmode")
        meters_per_rev1 = st.number_input("Meters per rev (cadence)", 0.0, 20.0, 8.18, step=0.01, key="r1_mpr")

    with col_r2:
        st.markdown("**Rider 2**")
        file2 = st.file_uploader("FIT file", type=["fit"], key="r2_file")
        rider2_name = st.text_input("Name", "Rider 2", key="r2_name")
        rider2_weight = st.number_input("Weight (kg)", 0.0, 200.0, 70.0, step=0.1, key="r2_weight")
        speed_mode2 = st.selectbox("Speed mode", ["sensor", "cadence", "gps", "gps_smooth"], index=1, key="r2_speedmode")
        meters_per_rev2 = st.number_input("Meters per rev (cadence)", 0.0, 20.0, 8.18, step=0.01, key="r2_mpr")

    if speed_mode1 == "gps_smooth" or speed_mode2 == "gps_smooth":
        smooth_gps = st.slider("GPS speed smoothing points", 0, 20, 8)
    else:
        smooth_gps = 0

show_raw = st.checkbox("Show raw data tables", False)

if file1 and file2:
    exp_align = st.expander("2) Alignment & intersections", expanded=True)
    with exp_align:
        col_a1, col_a2 = st.columns(2, gap="large")
        with col_a1:
            start_time = st.text_input("Start time (optional, e.g. 2025-10-25 21:17:38)", "")
            end_time = st.text_input("End time (optional)", "")
            shift_r1_seconds = st.slider("Rider 1 time shift (seconds)", -60, 60, 0)
        with col_a2:
            min_speed = st.slider("Minimum speed for intersections (km/h)", 0, 60, 15)
            min_seg_len = st.slider("Minimum segment duration (s)", 5, 120, 10)
            window_points = st.slider("Intersection window (points each side)", 1, 5, 1)
    try:
        df1 = process_fit(file1.getvalue(), smooth_gps, meters_per_rev1)
        df2 = process_fit(file2.getvalue(), smooth_gps, meters_per_rev2)

        if shift_r1_seconds != 0:
            df1 = df1.copy()
            df1["timestamp"] = df1["timestamp"] + pd.to_timedelta(shift_r1_seconds, unit="s")

        df1, df2 = align_pair(df1, df2, start_time, end_time)

        if "power" in df1.columns and rider1_weight > 0:
            df1 = df1.copy()
            df1["power_wkg"] = df1["power"] / rider1_weight
        if "power" in df2.columns and rider2_weight > 0:
            df2 = df2.copy()
            df2["power_wkg"] = df2["power"] / rider2_weight
    except Exception as exc:
        st.error(f"Problem loading or aligning files: {exc}")
    else:
        st.success(f"Aligned {len(df1)} samples; timestamps match: {df1['timestamp'].equals(df2['timestamp'])}")

        plot_header_cols = st.columns([2, 1, 1], gap="large")
        with plot_header_cols[0]:
            st.subheader("Plots")
        with plot_header_cols[1]:
            use_wkg = st.toggle("Show power in W/kg", value=False)

        plot_options = [
            f"{rider1_name} signals",
            f"{rider2_name} signals",
            "Power and speed comparison",
            "Power/speed + residuals",
            "Opposed segment comparison",
            "Neutral-only power split overlay",
            "Active-only power split overlay",
            "Segment stats table",
            "Power split tables",
            "Intersection energy table",
            "Intersection residual plot",
            f"{rider1_name} cadence optimization",
            f"{rider2_name} cadence optimization",
        ]
        selected_plots = st.multiselect(
            "Choose plots to show",
            plot_options,
            default=["Power and speed comparison"],
        )
        tiles_per_row = st.slider("Tiles per row", 1, 3, 1)

        needs_power_split_window = any(
            name in selected_plots for name in [
                "Neutral-only power split overlay",
                "Active-only power split overlay",
                "Power split tables",
                f"{rider1_name} cadence optimization",
                f"{rider2_name} cadence optimization",
            ]
        )
        if needs_power_split_window:
            with plot_header_cols[2]:
                last_seconds = st.slider("Last X seconds / sling window", 3, 30, 6)
        else:
            last_seconds = 6

        plot_state = {"i": 0, "cols": st.columns(tiles_per_row, gap="large")}

        def place_plot(title, fig):
            i = plot_state["i"]
            if i % tiles_per_row == 0:
                plot_state["cols"] = st.columns(tiles_per_row, gap="large")
            col = plot_state["cols"][i % tiles_per_row]
            with col:
                st.markdown(f"**{title}**")
                st.plotly_chart(fig, use_container_width=True)
            plot_state["i"] = i + 1

        def place_table(title, render_fn):
            i = plot_state["i"]
            if i % tiles_per_row == 0:
                plot_state["cols"] = st.columns(tiles_per_row, gap="large")
            col = plot_state["cols"][i % tiles_per_row]
            with col:
                st.markdown(f"**{title}**")
                render_fn()
            plot_state["i"] = i + 1

        if f"{rider1_name} signals" in selected_plots:
            place_plot(f"{rider1_name} signals", plot_fit_interactive(df1, title=f"{rider1_name} signals"))
        if f"{rider2_name} signals" in selected_plots:
            place_plot(f"{rider2_name} signals", plot_fit_interactive(df2, title=f"{rider2_name} signals"))
        if "Power and speed comparison" in selected_plots:
            place_plot(
                "Power and speed comparison",
                compare_rides_power_speed(
                    df1, df2, name1=rider1_name, name2=rider2_name,
                    speed_mode1=speed_mode1, speed_mode2=speed_mode2, use_wkg=use_wkg,
                ),
            )

        intersections = find_speed_intersections_same_time(
            df1, df2, speed_mode1=speed_mode1, speed_mode2=speed_mode2, min_speed=min_speed
        )
        intersections_valid = filter_intersections_by_segment_duration(df1, intersections, min_seg_len)
        exp_align.plotly_chart(
            plot_speed_with_intersections(df1, df2, intersections_valid,
                                          name1=rider1_name, name2=rider2_name,
                                          speed_mode1=speed_mode1, speed_mode2=speed_mode2),
            use_container_width=True,
        )

        df1_seg = segment_by_intersections(df1, intersections, min_duration_s=min_seg_len)
        df2_seg = segment_by_intersections(df2, intersections, min_duration_s=min_seg_len)

        stats1 = segment_stats(df1_seg, weight_kg=rider1_weight)
        stats2 = segment_stats(df2_seg, weight_kg=rider2_weight)
        merged = stats1.merge(stats2, on="segment_id", suffixes=(f"_{rider1_name}", f"_{rider2_name}"))

        roles = build_roles(stats1, stats2)

        if "Opposed segment comparison" in selected_plots:
            st.subheader("Opposed segment comparison")
        available_metrics = [
            m for m in [
                "avg_power",
                "avg_power_wkg",
                "avg_speed_kmh",
                "normalized_power",
                "variability_index",
            ]
            if m in stats1.columns and m in stats2.columns
        ]
        if "Opposed segment comparison" in selected_plots:
            if available_metrics:
                metric_choice = st.selectbox("Metric", available_metrics, index=0)
                bar_scale = st.slider("Bar thickness scale", 0.5, 2.0, 1.2, 0.1)
                role_filter = st.selectbox("Show segments by role", ["all", "active", "neutral"], index=0)
                place_plot(
                    "Opposed segment comparison",
                    plot_opposed_segment_metric(
                        stats1, stats2, roles=roles, metric=metric_choice,
                        name1=rider1_name, name2=rider2_name, duration_scale=bar_scale, role_filter=role_filter,
                    ),
                )
            else:
                st.warning("No common metrics available to plot.")

        p1 = power_split_stats(df1_seg, last_seconds=last_seconds)
        p2 = power_split_stats(df2_seg, last_seconds=last_seconds)

        p1_disp = p1.copy()
        p2_disp = p2.copy()
        if use_wkg and rider1_weight > 0:
            p1_disp["avg_power_last"] = p1_disp["avg_power_last"] / rider1_weight
            p1_disp["avg_power_rest"] = p1_disp["avg_power_rest"] / rider1_weight
        if use_wkg and rider2_weight > 0:
            p2_disp["avg_power_last"] = p2_disp["avg_power_last"] / rider2_weight
            p2_disp["avg_power_rest"] = p2_disp["avg_power_rest"] / rider2_weight

        p1_plot = p1_disp
        p2_plot = p2_disp

        if use_wkg and rider1_weight <= 0:
            st.warning("Rider 1 weight must be > 0 to use W/kg.")
        if use_wkg and rider2_weight <= 0:
            st.warning("Rider 2 weight must be > 0 to use W/kg.")

        def render_power_split_tables():
            col_ps1, col_ps2 = st.columns(2, gap="large")
            with col_ps1:
                st.markdown(f"**{rider1_name}**")
                unit1 = "W/kg" if use_wkg else "W"
                st.dataframe(
                    p1_disp.rename(columns={
                        "avg_power_last": f"avg_power_last ({last_seconds}s, {unit1})",
                        "avg_power_rest": f"avg_power_rest ({unit1})",
                    })
                )
            with col_ps2:
                st.markdown(f"**{rider2_name}**")
                unit2 = "W/kg" if use_wkg else "W"
                st.dataframe(
                    p2_disp.rename(columns={
                        "avg_power_last": f"avg_power_last ({last_seconds}s, {unit2})",
                        "avg_power_rest": f"avg_power_rest ({unit2})",
                    })
                )

        if "Segment stats table" in selected_plots:
            place_table("Segment stats table", lambda: st.dataframe(merged))

        if "Power split tables" in selected_plots:
            place_table("Power split tables", render_power_split_tables)

        wants_energy = any(
            name in selected_plots
            for name in ["Intersection energy table", "Intersection residual plot", "Power/speed + residuals"]
        )
        energy_df = None
        if wants_energy:
            energy_df = compute_intersection_energy_table(
                df1, df2, intersections_valid,
                speed_mode1, speed_mode2,
                rider1_weight, rider2_weight,
                window_points=window_points,
            )
            if energy_df.empty:
                energy_df = None

        if "Intersection energy table" in selected_plots:
            if energy_df is None:
                place_table("Intersection energy table", lambda: st.info("Not enough data for energy table."))
            else:
                place_table("Intersection energy table", lambda: st.dataframe(energy_df))

        if "Power/speed + residuals" in selected_plots:
            if energy_df is None:
                place_plot("Power/speed + residuals", go.Figure())
                st.info("Not enough data for residual overlay.")
            else:
                place_plot(
                    "Power/speed + residuals",
                    plot_power_speed_with_residuals(
                        df1, df2, energy_df,
                        name1=rider1_name, name2=rider2_name,
                        speed_mode1=speed_mode1, speed_mode2=speed_mode2, use_wkg=use_wkg,
                    ),
                )

        if "Intersection residual plot" in selected_plots:
            if energy_df is None:
                place_plot("Intersection residual plot", go.Figure())
                st.info("Not enough data for residual plot.")
            else:
                place_plot("Intersection residual plot", plot_intersection_residuals(energy_df))

        if f"{rider1_name} cadence optimization" in selected_plots:
            place_plot(
                f"{rider1_name} cadence optimization",
                plot_cadence_optimization(
                    df_seg=df1_seg,
                    roles=roles,
                    p_split=p1,
                    rider_id="rider1",
                    name=rider1_name,
                    speed_mode=speed_mode1,
                    sling_seconds=last_seconds,
                    color="#636EFA",
                ),
            )

        if f"{rider2_name} cadence optimization" in selected_plots:
            place_plot(
                f"{rider2_name} cadence optimization",
                plot_cadence_optimization(
                    df_seg=df2_seg,
                    roles=roles,
                    p_split=p2,
                    rider_id="rider2",
                    name=rider2_name,
                    speed_mode=speed_mode2,
                    sling_seconds=last_seconds,
                    color="#EF553B",
                ),
            )

        if use_wkg:
            xaxis_title = "Average power [W/kg]"
        else:
            xaxis_title = "Average power [W]"
        if "Neutral-only power split overlay" in selected_plots:
            place_plot(
                "Neutral-only power split overlay",
                plot_opposed_power_split_neutral_overlay(
                    p1_plot, p2_plot, roles=roles, name1=rider1_name, name2=rider2_name,
                    title=f"Neutral segments: last {last_seconds}s vs rest",
                    xaxis_title=xaxis_title,
                ),
            )

        if "Active-only power split overlay" in selected_plots:
            place_plot(
                "Active-only power split overlay",
                plot_opposed_power_split_active_overlay(
                    p1_plot, p2_plot, roles=roles, name1=rider1_name, name2=rider2_name,
                    title=f"Active segments: last {last_seconds}s vs rest",
                    xaxis_title=xaxis_title,
                ),
            )

        if show_raw:
            st.subheader("Raw aligned data")
            st.dataframe(df1.head(500))
            st.dataframe(df2.head(500))
else:
    st.info("Upload two FIT files to begin.")
