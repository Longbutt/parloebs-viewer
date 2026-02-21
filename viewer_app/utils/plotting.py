"""
plotting.py
-----------
All Plotly chart generation utilities for the Parloebs Analyse app.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.calculations import get_speed_series

RIDER1_COLOR = "#636EFA"
RIDER2_COLOR = "#EF553B"


def plot_fit_interactive(df, title="FIT Data Overview"):
    df = df.sort_values("timestamp")
    numeric_cols = df.select_dtypes(include="number").columns
    fig = go.Figure()
    for col in numeric_cols:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[col], mode="lines", name=col, visible="legendonly"))
    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title="Value",
        hovermode="x unified",
        height=500,
        template="plotly_white",
    )
    return fig


def compare_rides_power_speed(
    df1,
    df2,
    name1="Ride 1",
    name2="Ride 2",
    speed_mode1="sensor",
    speed_mode2="sensor",
    use_wkg=False,
):
    df1 = df1.sort_values("timestamp")
    df2 = df2.sort_values("timestamp")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Power comparison", "Speed comparison"), vertical_spacing=0.1)

    if use_wkg and "power_wkg" in df1.columns:
        fig.add_trace(
            go.Scatter(x=df1["timestamp"], y=df1["power_wkg"], mode="lines", name=f"{name1} power (W/kg)", line=dict(color=RIDER1_COLOR)),
            row=1, col=1
        )
    elif "power" in df1.columns:
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=df1["power"], mode="lines", name=f"{name1} power", line=dict(color=RIDER1_COLOR)), row=1, col=1)
    if use_wkg and "power_wkg" in df2.columns:
        fig.add_trace(
            go.Scatter(x=df2["timestamp"], y=df2["power_wkg"], mode="lines", name=f"{name2} power (W/kg)", line=dict(color=RIDER2_COLOR)),
            row=1, col=1
        )
    elif "power" in df2.columns:
        fig.add_trace(go.Scatter(x=df2["timestamp"], y=df2["power"], mode="lines", name=f"{name2} power", line=dict(color=RIDER2_COLOR)), row=1, col=1)

    s1 = get_speed_series(df1, speed_mode1)
    s2 = get_speed_series(df2, speed_mode2)
    if s1 is not None:
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=s1, mode="lines", name=f"{name1} {speed_mode1} speed", line=dict(color=RIDER1_COLOR)), row=2, col=1)
    if s2 is not None:
        fig.add_trace(go.Scatter(x=df2["timestamp"], y=s2, mode="lines", name=f"{name2} {speed_mode2} speed", line=dict(color=RIDER2_COLOR)), row=2, col=1)

    fig.update_layout(title="Ride comparison: power and speed", hovermode="x unified", height=650, template="plotly_white")
    return fig


def plot_power_speed_with_residuals(
    df1,
    df2,
    energy_df,
    name1="Ride 1",
    name2="Ride 2",
    speed_mode1="sensor",
    speed_mode2="sensor",
    use_wkg=False,
):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Power comparison", "Speed comparison", "Intersection residuals"),
        vertical_spacing=0.08,
        row_heights=[0.45, 0.35, 0.35],
    )

    if use_wkg and "power_wkg" in df1.columns:
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=df1["power_wkg"], mode="lines", name=f"{name1} power (W/kg)", line=dict(color=RIDER1_COLOR)), row=1, col=1, secondary_y=False)
    elif "power" in df1.columns:
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=df1["power"], mode="lines", name=f"{name1} power", line=dict(color=RIDER1_COLOR)), row=1, col=1, secondary_y=False)

    if use_wkg and "power_wkg" in df2.columns:
        fig.add_trace(go.Scatter(x=df2["timestamp"], y=df2["power_wkg"], mode="lines", name=f"{name2} power (W/kg)", line=dict(color=RIDER2_COLOR)), row=1, col=1, secondary_y=False)
    elif "power" in df2.columns:
        fig.add_trace(go.Scatter(x=df2["timestamp"], y=df2["power"], mode="lines", name=f"{name2} power", line=dict(color=RIDER2_COLOR)), row=1, col=1, secondary_y=False)

    s1 = get_speed_series(df1, speed_mode1)
    s2 = get_speed_series(df2, speed_mode2)
    if s1 is not None:
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=s1, mode="lines", name=f"{name1} {speed_mode1} speed", line=dict(color=RIDER1_COLOR)), row=2, col=1)
    if s2 is not None:
        fig.add_trace(go.Scatter(x=df2["timestamp"], y=s2, mode="lines", name=f"{name2} {speed_mode2} speed", line=dict(color=RIDER2_COLOR)), row=2, col=1)

    avg_residual = (energy_df["r1_residual_j"] + energy_df["r2_residual_j"]) / 2.0
    net_residual = energy_df["residual_net_j"].abs() / 2.0
    fig.add_trace(
        go.Bar(
            x=energy_df["timestamp"],
            y=avg_residual,
            error_y=dict(type="data", array=net_residual, visible=True),
            name="Avg residual",
            width=10000,
            marker=dict(color=avg_residual, colorscale="RdYlGn_r", showscale=False),
        ),
        row=3,
        col=1,
    )

    power_unit = "W/kg" if use_wkg and "power_wkg" in df1.columns and "power_wkg" in df2.columns else "W"
    fig.update_yaxes(title_text=f"Power [{power_unit}]", row=1, col=1)
    fig.update_yaxes(title_text="Speed [km/h]", row=2, col=1)
    fig.update_yaxes(title_text="Residual energy [J]", row=3, col=1, rangemode="tozero", zeroline=True)
    fig.update_layout(title="Power, speed, and intersection residuals", hovermode="x unified", height=750, template="plotly_white")
    return fig


def plot_speed_with_intersections(df1, df2, intersections, name1="Ride 1", name2="Ride 2", speed_mode1="sensor", speed_mode2="cadence"):
    s1 = get_speed_series(df1, speed_mode1)
    s2 = get_speed_series(df2, speed_mode2)
    fig = go.Figure()
    if s1 is not None:
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=s1, mode="lines", name=f"{name1} ({speed_mode1})", line=dict(color=RIDER1_COLOR)))
    if s2 is not None:
        fig.add_trace(go.Scatter(x=df2["timestamp"], y=s2, mode="lines", name=f"{name2} ({speed_mode2})", line=dict(color=RIDER2_COLOR)))
    if not intersections.empty:
        fig.add_trace(go.Scatter(
            x=intersections["timestamp"], y=intersections["speed1_kmh"], mode="markers", name="Intersections",
            marker=dict(size=9, symbol="x", color="#333333")
        ))
    fig.update_layout(title="Speed comparison with intersections", xaxis_title="Timestamp",
                      yaxis_title="Speed [km/h]", hovermode="x unified", height=600, template="plotly_white")
    return fig


def plot_opposed_segment_metric(
    stats1,
    stats2,
    roles=None,
    metric="avg_power",
    name1="Rider 1",
    name2="Rider 2",
    title=None,
    duration_scale=1.0,
    role_filter="all",
):
    if metric not in stats1.columns or metric not in stats2.columns:
        raise ValueError(f"Metric {metric} not available in stats.")

    merged = stats1[["segment_id", metric, "duration_s"]].merge(
        stats2[["segment_id", metric, "duration_s"]],
        on="segment_id",
        suffixes=("_r1", "_r2"),
    ).sort_values("segment_id")

    if roles is not None:
        merged = merged.merge(roles, on="segment_id", how="left")
        merged["role_r1"] = np.where(merged["active"] == "rider1", "active", "neutral")
        merged["role_r2"] = np.where(merged["active"] == "rider2", "active", "neutral")
    else:
        merged["role_r1"] = "active"
        merged["role_r2"] = "active"

    if role_filter == "active":
        merged = merged[(merged["role_r1"] == "active") | (merged["role_r2"] == "active")]
    elif role_filter == "neutral":
        merged = merged[(merged["role_r1"] == "neutral") | (merged["role_r2"] == "neutral")]

    x1 = merged[f"{metric}_r1"].values
    x2 = merged[f"{metric}_r2"].values

    if role_filter in ("active", "neutral"):
        x1 = np.where(merged["role_r1"] == role_filter, x1, np.nan)
        x2 = np.where(merged["role_r2"] == role_filter, x2, np.nan)

    x1 = -x1
    y = merged["segment_id"].astype(str)

    durations = merged["duration_s_r1"].values
    max_dur = durations.max() if durations.size and durations.max() > 0 else 1
    widths = (durations / max_dur) * duration_scale

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x1, y=y, width=widths, orientation="h", name=name1, marker=dict(color=RIDER1_COLOR)))
    fig.add_trace(go.Bar(x=x2, y=y, width=widths, orientation="h", name=name2, marker=dict(color=RIDER2_COLOR)))

    if title is None:
        title = f"Segment comparison: {metric}"

    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title=metric,
        yaxis_title="segment_id",
        template="plotly_white",
        height=500,
    )
    fig.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=len(y) - 0.5, line=dict(width=1, color="black"))
    return fig


def plot_opposed_power_split_neutral_overlay(
    p1,
    p2,
    roles,
    name1="Rider 1",
    name2="Rider 2",
    title="Neutral segments: power split last 6s vs rest",
    xaxis_title="Average power [W] (neutral segments only)",
):
    merged = (
        p1.merge(p2, on="segment_id", suffixes=("_r1", "_r2"))
          .merge(roles, on="segment_id", how="left")
          .sort_values("segment_id")
    )

    merged["role_r1"] = np.where(merged["neutral"] == "rider1", "neutral", "active")
    merged["role_r2"] = np.where(merged["neutral"] == "rider2", "neutral", "active")

    y = merged["segment_id"].astype(str)
    r1_neutral = merged["role_r1"] == "neutral"
    r2_neutral = merged["role_r2"] == "neutral"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=-merged["avg_power_last_r1"].where(r1_neutral),
        y=y, orientation="h", name=f"{name1} neutral last 6s", opacity=0.7, marker=dict(color="#9467BD"),
    ))
    fig.add_trace(go.Bar(
        x=-merged["avg_power_rest_r1"].where(r1_neutral),
        y=y, orientation="h", name=f"{name1} neutral remainder", opacity=1.0, marker=dict(color=RIDER1_COLOR),
    ))
    fig.add_trace(go.Bar(
        x=merged["avg_power_last_r2"].where(r2_neutral),
        y=y, orientation="h", name=f"{name2} neutral last 6s", opacity=0.7, marker=dict(color="#FF9900"),
    ))
    fig.add_trace(go.Bar(
        x=merged["avg_power_rest_r2"].where(r2_neutral),
        y=y, orientation="h", name=f"{name2} neutral remainder", opacity=1.0, marker=dict(color=RIDER2_COLOR),
    ))

    fig.update_layout(title=title, barmode="overlay", xaxis_title=xaxis_title, yaxis_title="segment_id",
                      template="plotly_white", height=550)
    fig.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=len(y) - 0.5, line=dict(width=1, color="black"))
    return fig


def plot_opposed_power_split_active_overlay(
    p1,
    p2,
    roles,
    name1="Rider 1",
    name2="Rider 2",
    title="Active segments: power split last 6s vs rest",
    xaxis_title="Average power [W] (active segments only)",
):
    merged = (
        p1.merge(p2, on="segment_id", suffixes=("_r1", "_r2"))
          .merge(roles, on="segment_id", how="left")
          .sort_values("segment_id")
    )

    merged["role_r1"] = np.where(merged["active"] == "rider1", "active", "neutral")
    merged["role_r2"] = np.where(merged["active"] == "rider2", "active", "neutral")

    y = merged["segment_id"].astype(str)
    r1_active = merged["role_r1"] == "active"
    r2_active = merged["role_r2"] == "active"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=-merged["avg_power_last_r1"].where(r1_active),
        y=y, orientation="h", name=f"{name1} active last 6s", opacity=0.7, marker=dict(color="#9467BD"),
    ))
    fig.add_trace(go.Bar(
        x=-merged["avg_power_rest_r1"].where(r1_active),
        y=y, orientation="h", name=f"{name1} active remainder", opacity=1.0, marker=dict(color=RIDER1_COLOR),
    ))
    fig.add_trace(go.Bar(
        x=merged["avg_power_last_r2"].where(r2_active),
        y=y, orientation="h", name=f"{name2} active last 6s", opacity=0.7, marker=dict(color="#FF9900"),
    ))
    fig.add_trace(go.Bar(
        x=merged["avg_power_rest_r2"].where(r2_active),
        y=y, orientation="h", name=f"{name2} active remainder", opacity=1.0, marker=dict(color=RIDER2_COLOR),
    ))

    fig.update_layout(title=title, barmode="overlay", xaxis_title=xaxis_title, yaxis_title="segment_id",
                      template="plotly_white", height=550)
    fig.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=len(y) - 0.5, line=dict(width=1, color="black"))
    return fig


def plot_intersection_residuals(energy_df):
    avg_residual = (energy_df["r1_residual_j"] + energy_df["r2_residual_j"]) / 2.0
    net_residual = energy_df["residual_net_j"].abs() / 2.0
    x = energy_df["timestamp"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=avg_residual,
            error_y=dict(type="data", array=net_residual, visible=True),
            name="Avg residual",
            width=10000,
            marker=dict(color=avg_residual, colorscale="RdYlGn_r", showscale=False),
        )
    )
    fig.update_layout(
        title="Intersection residuals (avg with net residual brackets)",
        xaxis_title="Intersection time",
        yaxis_title="Residual energy [J]",
        template="plotly_white",
        height=500,
    )
    return fig
