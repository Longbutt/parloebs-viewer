"""
plotting.py
-----------
All Plotly chart generation utilities for the Parloebs Analyse app.
"""

import numpy as np
import pandas as pd
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


def plot_cadence_optimization(
    df_seg: pd.DataFrame,
    roles: pd.DataFrame,
    p_split: pd.DataFrame,
    rider_id: str,
    name: str,
    speed_mode: str,
    sling_seconds: int = 6,
    color: str = RIDER1_COLOR,
    use_wkg: bool = False,
    weight_kg: float = 0.0,
) -> go.Figure:
    """Interactive Power vs. Cadence powerband chart for fixed-gear track cycling.

    Parameters
    ----------
    df_seg     : segmented DataFrame from segment_by_intersections (has segment_id).
    roles      : DataFrame from build_roles with columns [segment_id, active, neutral].
    p_split    : DataFrame from power_split_stats; used to identify sling windows.
    rider_id   : "rider1" or "rider2" — which rider's active segments to show.
    name       : Display name for the rider.
    speed_mode : Which speed column to use for marker coloring (sensor/cadence/gps/gps_smooth).
    sling_seconds : Number of seconds before segment end to highlight as sling moments.
    color      : Primary highlight colour for this rider.
    """
    # ── 1. Filter to this rider's active segments ──────────────────────────────
    active_seg_ids = roles.loc[roles["active"] == rider_id, "segment_id"].values
    df_active = df_seg[df_seg["segment_id"].isin(active_seg_ids)].copy()

    # ── 2. Remove noise / coasting ─────────────────────────────────────────────
    if "cadence" not in df_active.columns or "power" not in df_active.columns:
        fig = go.Figure()
        fig.update_layout(
            title=f"{name} — Cadence Optimization (no cadence/power data)",
            template="plotly_white",
            height=500,
        )
        return fig

    df_active = df_active[(df_active["cadence"] >= 40) & (df_active["power"] >= 50)].copy()

    if df_active.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{name} — Cadence Optimization (insufficient data after filtering)",
            template="plotly_white",
            height=500,
        )
        return fig

    # ── 3. Torque enrichment ────────────────────────────────────────────────────
    df_active["torque_nm"] = (df_active["power"] * 60.0) / (
        df_active["cadence"] * 2.0 * np.pi
    )

    # ── 4. Speed column for colour scale ───────────────────────────────────────
    speed_col_map = {
        "sensor": "speed",
        "cadence": "speed_cadence",
        "gps": "speed_gps",
        "gps_smooth": "speed_gps_smooth",
    }
    raw_speed_col = speed_col_map.get(speed_mode, "speed_cadence")
    if raw_speed_col in df_active.columns:
        speed_kmh = df_active[raw_speed_col] * 3.6
    elif "speed_cadence" in df_active.columns:
        speed_kmh = df_active["speed_cadence"] * 3.6
    else:
        speed_kmh = pd.Series(np.zeros(len(df_active)), index=df_active.index)

    cadence_vals = df_active["cadence"].values
    power_vals = df_active["power"].values
    torque_vals = df_active["torque_nm"].values
    speed_vals = speed_kmh.values

    # Handle W/kg conversion if requested
    power_unit = "W"
    if use_wkg and weight_kg > 0:
        power_vals = power_vals / weight_kg
        power_unit = "W/kg"

    # ── 5. Scatter trace (colored by speed) ────────────────────────────────────
    hover_text = [
        f"Cadence: {c:.0f} rpm<br>Power: {p:.1f} {power_unit}<br>Torque: {t:.1f} Nm<br>Speed: {s:.1f} km/h"
        for c, p, t, s in zip(cadence_vals, power_vals, torque_vals, speed_vals)
    ]

    scatter_trace = go.Scatter(
        x=cadence_vals,
        y=power_vals,
        mode="markers",
        name=f"{name} data",
        text=hover_text,
        hoverinfo="text",
        marker=dict(
            size=6,
            color=speed_vals,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title="Speed (km/h)",
                thickness=14,
                len=0.7,
            ),
            opacity=0.65,
            line=dict(width=0),
        ),
    )

    # ── 6. Polynomial trendline (degree-2) ─────────────────────────────────────
    sort_idx = np.argsort(cadence_vals)
    x_sorted = cadence_vals[sort_idx]
    y_sorted = power_vals[sort_idx]

    poly_degree = 2
    coeffs = np.polyfit(x_sorted, y_sorted, poly_degree)
    poly_fn = np.poly1d(coeffs)
    x_line = np.linspace(x_sorted[0], x_sorted[-1], 200)
    y_line = poly_fn(x_line)

    trendline_trace = go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name=f"{name} trendline",
        line=dict(color=color, width=3, dash="solid"),
        hoverinfo="skip",
    )

    # ── 7. Reference lines ─────────────────────────────────────────────────────
    avg_power = float(np.nanmean(power_vals))
    avg_cadence = float(np.nanmean(cadence_vals))

    # ── 8. Sling moment overlay ────────────────────────────────────────────────
    # The rider receiving the sling is NEUTRAL just before a hand-off.
    # We identify the last `sling_seconds` of each segment where this rider is NEUTRAL.
    neutral_seg_ids = roles.loc[roles["neutral"] == rider_id, "segment_id"].values
    df_neutral = df_seg[df_seg["segment_id"].isin(neutral_seg_ids)].copy()

    sling_dfs = []
    for seg_id, grp in df_neutral.groupby("segment_id"):
        t_end = grp["timestamp"].max()
        t_cut = t_end - pd.Timedelta(seconds=sling_seconds)
        sling_window = grp[grp["timestamp"] >= t_cut]
        sling_dfs.append(sling_window)

    sling_traces = []
    if sling_dfs:
        df_sling = pd.concat(sling_dfs, ignore_index=True)
        # Apply the same noise filter
        if "cadence" in df_sling.columns and "power" in df_sling.columns:
            df_sling = df_sling[
                (df_sling["cadence"] >= 40) & (df_sling["power"] >= 50)
            ].copy()

        if not df_sling.empty:
            df_sling["torque_nm"] = (df_sling["power"] * 60.0) / (
                df_sling["cadence"] * 2.0 * np.pi
            )
            s_power_vals = df_sling["power"].values
            if use_wkg and weight_kg > 0:
                s_power_vals = s_power_vals / weight_kg

            sling_hover = [
                f"⚡ SLING — Cadence: {c:.0f} rpm<br>Power: {p:.1f} {power_unit}<br>"
                f"Torque: {t:.1f} Nm"
                for c, p, t in zip(
                    df_sling["cadence"], s_power_vals, df_sling["torque_nm"]
                )
            ]
            sling_traces.append(
                go.Scatter(
                    x=df_sling["cadence"].values,
                    y=s_power_vals,
                    mode="markers",
                    name=f"{name} sling ({sling_seconds}s)",
                    text=sling_hover,
                    hoverinfo="text",
                    marker=dict(
                        size=10,
                        symbol="x",
                        color="black",
                        opacity=0.8,
                    ),
                )
            )

    # ── 9. Assemble figure ─────────────────────────────────────────────────────
    fig = go.Figure(data=[scatter_trace, trendline_trace] + sling_traces)

    fig.add_hline(
        y=avg_power,
        line=dict(color="#FF7F0E", width=1.5, dash="dash"),
        annotation_text=f"Avg active power: {avg_power:.1f} {power_unit}",
        annotation_position="top right",
        annotation=dict(font=dict(size=11, color="#FF7F0E")),
    )
    fig.add_vline(
        x=avg_cadence,
        line=dict(color="#2CA02C", width=1.5, dash="dash"),
        annotation_text=f"Avg cadence: {avg_cadence:.0f} rpm",
        annotation_position="top left",
        annotation=dict(font=dict(size=11, color="#2CA02C")),
    )

    fig.update_layout(
        title=dict(
            text=f"{name} — Power Cadence Analysis",
            font=dict(size=15),
        ),
        xaxis_title="Cadence [rpm]",
        yaxis_title=f"Power [{power_unit}]",
        hovermode="closest",
        template="plotly_white",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
