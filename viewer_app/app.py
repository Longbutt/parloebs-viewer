import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from fitparse import FitFile
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Parloebs Analyse Viewer", layout="wide")

RIDER1_COLOR = "#636EFA"
RIDER2_COLOR = "#EF553B"


# ---------- Data utilities ----------
@st.cache_data(show_spinner=False)
def fit_to_dataframe(file_bytes: bytes) -> pd.DataFrame:
    """Parse a FIT file (bytes) into a cleaned DataFrame."""
    fitfile = FitFile(BytesIO(file_bytes))
    records = []
    for record in fitfile.get_messages("record"):
        row = {f.name: f.value for f in record}
        records.append(row)
    df = pd.DataFrame(records).dropna(axis=1, how="all")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def trim_and_align_on_timestamp(df1, df2, start_time=None, end_time=None):
    df1 = df1.sort_values("timestamp").reset_index(drop=True)
    df2 = df2.sort_values("timestamp").reset_index(drop=True)

    natural_start = max(df1["timestamp"].iloc[0], df2["timestamp"].iloc[0])
    natural_end = min(df1["timestamp"].iloc[-1], df2["timestamp"].iloc[-1])

    if start_time:
        start_time = pd.to_datetime(start_time)
    if end_time:
        end_time = pd.to_datetime(end_time)

    t_start = natural_start if start_time is None else max(natural_start, start_time)
    t_end = natural_end if end_time is None else min(natural_end, end_time)
    if t_start >= t_end:
        raise ValueError("Selected time window has no overlap between files.")

    df1 = df1[(df1["timestamp"] >= t_start) & (df1["timestamp"] <= t_end)]
    df2 = df2[(df2["timestamp"] >= t_start) & (df2["timestamp"] <= t_end)]

    common_ts = np.intersect1d(df1["timestamp"].values, df2["timestamp"].values)
    df1 = df1[df1["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    df2 = df2[df2["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    return df1, df2


@st.cache_data(show_spinner=False)
def process_fit(file_bytes: bytes, smooth_gps: int, meters_per_rev: float):
    """End-to-end rider preprocessing (parse + speeds)."""
    df = fit_to_dataframe(file_bytes)
    df = compute_gps_speed(df, smooth_points=smooth_gps)
    df = compute_cadence_speed(df, smooth_points=0, meters_per_rev=meters_per_rev)
    return df


@st.cache_data(show_spinner=False)
def align_pair(df1: pd.DataFrame, df2: pd.DataFrame, start_time, end_time):
    return trim_and_align_on_timestamp(df1, df2, start_time=start_time or None, end_time=end_time or None)


def semicircles_to_degrees(x):
    return x * (180 / 2 ** 31)


@st.cache_data(show_spinner=False)
def compute_gps_speed(df, smooth_points=None):
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "position_lat" not in df or "position_long" not in df:
        return df

    df["lat_deg"] = semicircles_to_degrees(df["position_lat"])
    df["lon_deg"] = semicircles_to_degrees(df["position_long"])

    R = 6371000
    lat1 = np.radians(df["lat_deg"].shift())
    lon1 = np.radians(df["lon_deg"].shift())
    lat2 = np.radians(df["lat_deg"])
    lon2 = np.radians(df["lon_deg"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["gps_distance"] = R * c
    df["dt"] = df["timestamp"].diff().dt.total_seconds()
    df["speed_gps"] = df["gps_distance"] / df["dt"]

    if smooth_points and smooth_points > 1:
        df["speed_gps_smooth"] = df["speed_gps"].rolling(
            window=smooth_points, center=True, min_periods=1
        ).mean()
    else:
        df["speed_gps_smooth"] = df["speed_gps"]
    return df


@st.cache_data(show_spinner=False)
def compute_cadence_speed(df, smooth_points=None, meters_per_rev=8.18):
    if "cadence" not in df:
        return df
    cadence_clean = df["cadence"].fillna(0)
    df["speed_cadence"] = cadence_clean * meters_per_rev / 60.0
    if smooth_points and smooth_points > 1:
        df["speed_cadence_smooth"] = df["speed_cadence"].rolling(
            window=smooth_points, center=True, min_periods=1
        ).mean()
    else:
        df["speed_cadence_smooth"] = df["speed_cadence"]
    return df


# ---------- Plot helpers ----------
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


def get_speed_series(df, mode):
    if mode == "sensor" and "speed" in df:
        return df["speed"] * 3.6
    if mode == "cadence" and "speed_cadence" in df:
        return df["speed_cadence"] * 3.6
    if mode == "gps" and "speed_gps" in df:
        return df["speed_gps"] * 3.6
    if mode == "gps_smooth" and "speed_gps_smooth" in df:
        return df["speed_gps_smooth"] * 3.6
    return None


def get_speed_series_ms(df, mode):
    if mode == "sensor" and "speed" in df:
        return df["speed"]
    if mode == "cadence" and "speed_cadence" in df:
        return df["speed_cadence"]
    if mode == "gps" and "speed_gps" in df:
        return df["speed_gps"]
    if mode == "gps_smooth" and "speed_gps_smooth" in df:
        return df["speed_gps_smooth"]
    return None


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

    # Power traces
    if use_wkg and "power_wkg" in df1.columns:
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=df1["power_wkg"], mode="lines", name=f"{name1} power (W/kg)", line=dict(color=RIDER1_COLOR)), row=1, col=1, secondary_y=False)
    elif "power" in df1.columns:
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=df1["power"], mode="lines", name=f"{name1} power", line=dict(color=RIDER1_COLOR)), row=1, col=1, secondary_y=False)

    if use_wkg and "power_wkg" in df2.columns:
        fig.add_trace(go.Scatter(x=df2["timestamp"], y=df2["power_wkg"], mode="lines", name=f"{name2} power (W/kg)", line=dict(color=RIDER2_COLOR)), row=1, col=1, secondary_y=False)
    elif "power" in df2.columns:
        fig.add_trace(go.Scatter(x=df2["timestamp"], y=df2["power"], mode="lines", name=f"{name2} power", line=dict(color=RIDER2_COLOR)), row=1, col=1, secondary_y=False)

    # Speed traces
    s1 = get_speed_series(df1, speed_mode1)
    s2 = get_speed_series(df2, speed_mode2)
    if s1 is not None:
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=s1, mode="lines", name=f"{name1} {speed_mode1} speed", line=dict(color=RIDER1_COLOR)), row=2, col=1)
    if s2 is not None:
        fig.add_trace(go.Scatter(x=df2["timestamp"], y=s2, mode="lines", name=f"{name2} {speed_mode2} speed", line=dict(color=RIDER2_COLOR)), row=2, col=1)

    # Residual bars on third row (last in legend)
    avg_residual = (energy_df["r1_residual_j"] + energy_df["r2_residual_j"]) / 2.0
    net_residual = energy_df["residual_net_j"].abs() / 2.0
    fig.add_trace(
        go.Bar(
            x=energy_df["timestamp"],
            y=avg_residual,
            error_y=dict(type="data", array=net_residual, visible=True),
            name="Avg residual",
            width=10000,
            marker=dict(
                color=avg_residual,
                colorscale="RdYlGn_r",
                showscale=False,
            ),
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


def find_speed_intersections_same_time(df1, df2, speed_mode1="sensor", speed_mode2="cadence", diff_tol=0.05, min_speed=0.0):
    t = df1["timestamp"]
    s1 = get_speed_series(df1, speed_mode1)
    s2 = get_speed_series(df2, speed_mode2)
    if s1 is None or s2 is None:
        return pd.DataFrame(columns=["timestamp", "speed1_kmh", "speed2_kmh"])

    diff = s1 - s2
    crossings = []

    def sign_with_tol(x):
        if np.isnan(x) or abs(x) < diff_tol:
            return 0
        return 1 if x > 0 else -1

    for i in range(1, len(diff)):
        d0 = diff.iloc[i - 1]
        d1 = diff.iloc[i]
        if np.isnan(d0) or np.isnan(d1):
            continue
        t0 = t.iloc[i - 1]
        t1 = t.iloc[i]
        v1_0 = s1.iloc[i - 1]
        v1_1 = s1.iloc[i]
        v2_0 = s2.iloc[i - 1]
        v2_1 = s2.iloc[i]

        if v1_0 < min_speed or v2_0 < min_speed:
            continue

        sgn0 = sign_with_tol(d0)
        sgn1 = sign_with_tol(d1)

        if sgn1 == 0:
            crossings.append((t1, v1_1, v2_1))
            continue
        if sgn0 * sgn1 < 0:
            frac = abs(d0) / (abs(d0) + abs(d1))
            t_cross = t0 + (t1 - t0) * frac
            v1_cross = v1_0 + (v1_1 - v1_0) * frac
            v2_cross = v2_0 + (v2_1 - v2_0) * frac
            crossings.append((t_cross, v1_cross, v2_cross))

    return pd.DataFrame(crossings, columns=["timestamp", "speed1_kmh", "speed2_kmh"])


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


@st.cache_data(show_spinner=False)
def segment_by_intersections(df, intersections, min_duration_s=10):
    df = df.sort_values("timestamp").reset_index(drop=True)
    if intersections.empty:
        df["segment_id"] = 0
        return df
    cuts = np.sort(intersections["timestamp"].values)
    seg_ids = np.searchsorted(cuts, df["timestamp"].values)
    df["segment_id"] = seg_ids
    durations = df.groupby("segment_id")["timestamp"].apply(lambda x: (x.max() - x.min()).total_seconds())
    keep_segments = durations[durations >= min_duration_s].index
    df = df[df["segment_id"].isin(keep_segments)].copy()
    new_ids = {old_id: new_id for new_id, old_id in enumerate(sorted(keep_segments))}
    df["segment_id"] = df["segment_id"].map(new_ids)
    return df


@st.cache_data(show_spinner=False)
def segment_stats(df, speed_col="speed_cadence", weight_kg=None):
    agg = {
        "start_time": ("timestamp", "min"),
        "end_time": ("timestamp", "max"),
        "avg_power": ("power", "mean"),
        "normalized_power": ("power", lambda p: (np.mean(p ** 4)) ** 0.25 if len(p) else np.nan),
    }
    if speed_col in df:
        agg["avg_speed_kmh"] = (speed_col, lambda s: s.mean() * 3.6)

    out = df.groupby("segment_id").agg(**agg).reset_index()
    out["duration_s"] = (out["end_time"] - out["start_time"]).dt.total_seconds()
    out["variability_index"] = out["normalized_power"] / out["avg_power"]
    if weight_kg is not None and weight_kg > 0:
        out["avg_power_wkg"] = out["avg_power"] / weight_kg
    return out


@st.cache_data(show_spinner=False)
def power_split_stats(df, last_seconds=6):
    df = df.sort_values(["segment_id", "timestamp"]).copy()

    def _one_segment(g):
        t_end = g["timestamp"].max()
        t_cut = t_end - pd.Timedelta(seconds=last_seconds)
        last_mask = g["timestamp"] >= t_cut
        rest_mask = g["timestamp"] < t_cut
        return pd.Series({
            "duration_s": (g["timestamp"].max() - g["timestamp"].min()).total_seconds(),
            "avg_power_last": g.loc[last_mask, "power"].mean(),
            "avg_power_rest": g.loc[rest_mask, "power"].mean() if rest_mask.any() else np.nan,
        })

    return df.groupby("segment_id", as_index=False).apply(_one_segment).reset_index(drop=True)


def build_roles(stats1, stats2):
    """Classify active/neutral per segment by higher avg_power."""
    merged = stats1[["segment_id", "avg_power"]].merge(
        stats2[["segment_id", "avg_power"]],
        on="segment_id",
        suffixes=("_r1", "_r2"),
    )
    active = np.where(merged["avg_power_r1"] >= merged["avg_power_r2"], "rider1", "rider2")
    neutral = np.where(active == "rider1", "rider2", "rider1")
    return pd.DataFrame({"segment_id": merged["segment_id"], "active": active, "neutral": neutral})


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

    # Filter segments by role across both riders (not just Rider 1)
    if role_filter == "active":
        merged = merged[(merged["role_r1"] == "active") | (merged["role_r2"] == "active")]
    elif role_filter == "neutral":
        merged = merged[(merged["role_r1"] == "neutral") | (merged["role_r2"] == "neutral")]

    x1 = merged[f"{metric}_r1"].values
    x2 = merged[f"{metric}_r2"].values

    # If filtering to a specific role, hide the other rider's bar on that segment
    if role_filter in ("active", "neutral"):
        x1 = np.where(merged["role_r1"] == role_filter, x1, np.nan)
        x2 = np.where(merged["role_r2"] == role_filter, x2, np.nan)

    x1 = -x1  # left side
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
        y=y,
        orientation="h",
        name=f"{name1} neutral last 6s",
        opacity=0.7,
        marker=dict(color="#9467BD"),
    ))
    fig.add_trace(go.Bar(
        x=-merged["avg_power_rest_r1"].where(r1_neutral),
        y=y,
        orientation="h",
        name=f"{name1} neutral remainder",
        opacity=1.0,
        marker=dict(color=RIDER1_COLOR),
    ))
    fig.add_trace(go.Bar(
        x=merged["avg_power_last_r2"].where(r2_neutral),
        y=y,
        orientation="h",
        name=f"{name2} neutral last 6s",
        opacity=0.7,
        marker=dict(color="#FF9900"),
    ))
    fig.add_trace(go.Bar(
        x=merged["avg_power_rest_r2"].where(r2_neutral),
        y=y,
        orientation="h",
        name=f"{name2} neutral remainder",
        opacity=1.0,
        marker=dict(color=RIDER2_COLOR),
    ))

    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title=xaxis_title,
        yaxis_title="segment_id",
        template="plotly_white",
        height=550,
    )
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
        y=y,
        orientation="h",
        name=f"{name1} active last 6s",
        opacity=0.7,
        marker=dict(color="#9467BD"),
    ))
    fig.add_trace(go.Bar(
        x=-merged["avg_power_rest_r1"].where(r1_active),
        y=y,
        orientation="h",
        name=f"{name1} active remainder",
        opacity=1.0,
        marker=dict(color=RIDER1_COLOR),
    ))
    fig.add_trace(go.Bar(
        x=merged["avg_power_last_r2"].where(r2_active),
        y=y,
        orientation="h",
        name=f"{name2} active last 6s",
        opacity=0.7,
        marker=dict(color="#FF9900"),
    ))
    fig.add_trace(go.Bar(
        x=merged["avg_power_rest_r2"].where(r2_active),
        y=y,
        orientation="h",
        name=f"{name2} active remainder",
        opacity=1.0,
        marker=dict(color=RIDER2_COLOR),
    ))

    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title=xaxis_title,
        yaxis_title="segment_id",
        template="plotly_white",
        height=550,
    )
    fig.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=len(y) - 0.5, line=dict(width=1, color="black"))
    return fig


def compute_intersection_energy_table(
    df1,
    df2,
    intersections,
    speed_mode1,
    speed_mode2,
    mass1,
    mass2,
    window_points=3,
):
    if intersections.empty:
        return pd.DataFrame()

    v1 = get_speed_series_ms(df1, speed_mode1)
    v2 = get_speed_series_ms(df2, speed_mode2)
    if v1 is None or v2 is None:
        return pd.DataFrame()

    if "altitude" in df1.columns:
        h1 = df1["altitude"]
    else:
        h1 = None
    if "altitude" in df2.columns:
        h2 = df2["altitude"]
    else:
        h2 = None

    power1 = df1["power"] if "power" in df1.columns else None
    power2 = df2["power"] if "power" in df2.columns else None
    dt1 = df1["timestamp"].diff().dt.total_seconds().fillna(0)
    dt2 = df2["timestamp"].diff().dt.total_seconds().fillna(0)

    ts = df1["timestamp"].values.astype("datetime64[ns]")
    out = []
    g = 9.81
    n = int(window_points)

    for t in intersections["timestamp"]:
        t64 = np.datetime64(t)
        idx = np.searchsorted(ts, t64)
        if idx >= len(ts):
            idx = len(ts) - 1
        if idx > 0 and abs(ts[idx - 1] - t64) < abs(ts[idx] - t64):
            idx = idx - 1

        i0 = max(0, idx - n)
        i1 = min(len(ts) - 1, idx + n)

        v1_start = v1.iloc[i0]
        v1_end = v1.iloc[i1]
        v2_start = v2.iloc[i0]
        v2_end = v2.iloc[i1]

        dke1 = 0.5 * mass1 * (v1_end ** 2 - v1_start ** 2) if mass1 > 0 else np.nan
        dke2 = 0.5 * mass2 * (v2_end ** 2 - v2_start ** 2) if mass2 > 0 else np.nan

        if h1 is not None and mass1 > 0:
            dpe1 = mass1 * g * (h1.iloc[i1] - h1.iloc[i0])
        else:
            dpe1 = np.nan

        if h2 is not None and mass2 > 0:
            dpe2 = mass2 * g * (h2.iloc[i1] - h2.iloc[i0])
        else:
            dpe2 = np.nan

        if power1 is not None:
            p1 = power1.iloc[i0:i1 + 1]
            work1 = (p1.fillna(0) * dt1.iloc[i0:i1 + 1]).sum()
        else:
            work1 = np.nan

        if power2 is not None:
            p2 = power2.iloc[i0:i1 + 1]
            work2 = (p2.fillna(0) * dt2.iloc[i0:i1 + 1]).sum()
        else:
            work2 = np.nan

        dpe1_val = 0.0 if np.isnan(dpe1) else dpe1
        dpe2_val = 0.0 if np.isnan(dpe2) else dpe2
        residual1 = np.abs(dke1) - dpe1_val - work1
        residual2 = np.abs(dke2) - dpe2_val - work2
        residual_net = residual1 - residual2

        out.append({
            "timestamp": t,
            "window_start": df1["timestamp"].iloc[i0],
            "window_end": df1["timestamp"].iloc[i1],
            "r1_dke_j": dke1,
            "r2_dke_j": dke2,
            "r1_dpe_j": dpe1,
            "r2_dpe_j": dpe2,
            "r1_work_j": work1,
            "r2_work_j": work2,
            "r1_residual_j": residual1,
            "r2_residual_j": residual2,
            "residual_net_j": residual_net,
        })

    return pd.DataFrame(out)


def filter_intersections_by_segment_duration(df, intersections, min_duration_s):
    if intersections.empty:
        return intersections
    cuts = np.sort(pd.to_datetime(intersections["timestamp"]).to_numpy(dtype="datetime64[ns]"))
    t_start = np.datetime64(pd.to_datetime(df["timestamp"].iloc[0]))
    t_end = np.datetime64(pd.to_datetime(df["timestamp"].iloc[-1]))
    segment_starts = np.concatenate((np.array([t_start], dtype="datetime64[ns]"), cuts))
    segment_ends = np.concatenate((cuts, np.array([t_end], dtype="datetime64[ns]")))
    durations = np.array(
        [(end - start) / np.timedelta64(1, "s") for start, end in zip(segment_starts, segment_ends)]
    )

    valid_idx = []
    for i, t in enumerate(cuts):
        before_ok = durations[i] >= min_duration_s
        after_ok = durations[i + 1] >= min_duration_s
        if before_ok and after_ok:
            valid_idx.append(i)

    if not valid_idx:
        return intersections.iloc[0:0]

    valid_times = cuts[valid_idx]
    return intersections[intersections["timestamp"].isin(valid_times)].reset_index(drop=True)


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
            marker=dict(
                color=avg_residual,
                colorscale="RdYlGn_r",
                showscale=False,
            ),
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
            ]
        )
        if needs_power_split_window:
            with plot_header_cols[2]:
                last_seconds = st.slider("Last X seconds", 3, 30, 6)
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
                    p1_disp.rename(
                        columns={
                            "avg_power_last": f"avg_power_last ({last_seconds}s, {unit1})",
                            "avg_power_rest": f"avg_power_rest ({unit1})",
                        }
                    )
                )
            with col_ps2:
                st.markdown(f"**{rider2_name}**")
                unit2 = "W/kg" if use_wkg else "W"
                st.dataframe(
                    p2_disp.rename(
                        columns={
                            "avg_power_last": f"avg_power_last ({last_seconds}s, {unit2})",
                            "avg_power_rest": f"avg_power_rest ({unit2})",
                        }
                    )
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
                df1,
                df2,
                intersections_valid,
                speed_mode1,
                speed_mode2,
                rider1_weight,
                rider2_weight,
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
