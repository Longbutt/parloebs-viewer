import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from fitparse import FitFile
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Parloebs Analyse Viewer", layout="wide")


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
def process_fit(file_bytes: bytes, smooth_gps: int, cadence_rev_m: float):
    """End-to-end rider preprocessing (parse + speeds)."""
    df = fit_to_dataframe(file_bytes)
    df = compute_gps_speed(df, smooth_points=smooth_gps)
    df = compute_cadence_speed(df, smooth_points=0, meters_per_rev=cadence_rev_m)
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


def compare_rides_power_speed(df1, df2, name1="Ride 1", name2="Ride 2", speed_mode1="sensor", speed_mode2="sensor"):
    df1 = df1.sort_values("timestamp")
    df2 = df2.sort_values("timestamp")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Power comparison", "Speed comparison"), vertical_spacing=0.1)

    if "power" in df1.columns:
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=df1["power"], mode="lines", name=f"{name1} power"), row=1, col=1)
    if "power" in df2.columns:
        fig.add_trace(go.Scatter(x=df2["timestamp"], y=df2["power"], mode="lines", name=f"{name2} power"), row=1, col=1)

    s1 = get_speed_series(df1, speed_mode1)
    s2 = get_speed_series(df2, speed_mode2)
    if s1 is not None:
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=s1, mode="lines", name=f"{name1} {speed_mode1} speed"), row=2, col=1)
    if s2 is not None:
        fig.add_trace(go.Scatter(x=df2["timestamp"], y=s2, mode="lines", name=f"{name2} {speed_mode2} speed"), row=2, col=1)

    fig.update_layout(title="Ride comparison: power and speed", hovermode="x unified", height=650, template="plotly_white")
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
        fig.add_trace(go.Scatter(x=df1["timestamp"], y=s1, mode="lines", name=f"{name1} ({speed_mode1})"))
    if s2 is not None:
        fig.add_trace(go.Scatter(x=df2["timestamp"], y=s2, mode="lines", name=f"{name2} ({speed_mode2})"))
    if not intersections.empty:
        fig.add_trace(go.Scatter(
            x=intersections["timestamp"], y=intersections["speed1_kmh"], mode="markers", name="Intersections",
            marker=dict(size=9, symbol="x")
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
def segment_stats(df, speed_col="speed_cadence"):
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
    fig.add_trace(go.Bar(x=x1, y=y, width=widths, orientation="h", name=name1))
    fig.add_trace(go.Bar(x=x2, y=y, width=widths, orientation="h", name=name2))

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
    ))
    fig.add_trace(go.Bar(
        x=-merged["avg_power_rest_r1"].where(r1_neutral),
        y=y,
        orientation="h",
        name=f"{name1} neutral remainder",
        opacity=0.4,
    ))
    fig.add_trace(go.Bar(
        x=merged["avg_power_last_r2"].where(r2_neutral),
        y=y,
        orientation="h",
        name=f"{name2} neutral last 6s",
        opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=merged["avg_power_rest_r2"].where(r2_neutral),
        y=y,
        orientation="h",
        name=f"{name2} neutral remainder",
        opacity=0.4,
    ))

    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title="Average power [W] (neutral segments only)",
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
    ))
    fig.add_trace(go.Bar(
        x=-merged["avg_power_rest_r1"].where(r1_active),
        y=y,
        orientation="h",
        name=f"{name1} active remainder",
        opacity=0.4,
    ))
    fig.add_trace(go.Bar(
        x=merged["avg_power_last_r2"].where(r2_active),
        y=y,
        orientation="h",
        name=f"{name2} active last 6s",
        opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=merged["avg_power_rest_r2"].where(r2_active),
        y=y,
        orientation="h",
        name=f"{name2} active remainder",
        opacity=0.4,
    ))

    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title="Average power [W] (active segments only)",
        yaxis_title="segment_id",
        template="plotly_white",
        height=550,
    )
    fig.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=len(y) - 0.5, line=dict(width=1, color="black"))
    return fig


# ---------- UI ----------
st.title("Parloebs Analyse – interactive viewer")
st.markdown("Upload two FIT files, align them on time, and explore speeds, power, intersections, and segments.")

with st.sidebar:
    st.header("Inputs")
    file1 = st.file_uploader("Rider 1 FIT", type=["fit"])
    file2 = st.file_uploader("Rider 2 FIT", type=["fit"])
    rider1_name = st.text_input("Rider 1 name", "Rider 1")
    rider2_name = st.text_input("Rider 2 name", "Rider 2")
    start_time = st.text_input("Start time (optional, e.g. 2025-10-25 21:17:38)", "")
    end_time = st.text_input("End time (optional)", "")
    smooth_gps = st.slider("GPS speed smoothing points", 0, 20, 8)
    cadence_rev_m = st.number_input("Meters per rev (cadence)", 0.0, 20.0, 8.18, step=0.01)
    speed_mode1 = st.selectbox("Speed mode rider 1", ["sensor", "cadence", "gps", "gps_smooth"])
    speed_mode2 = st.selectbox("Speed mode rider 2", ["sensor", "cadence", "gps", "gps_smooth"], index=1)
    min_speed = st.slider("Minimum speed for intersections (km/h)", 0, 60, 15)
    min_seg_len = st.slider("Minimum segment duration (s)", 5, 120, 10)
    last_seconds = st.slider("Neutral overlay: last X seconds window", 3, 30, 6)
    show_raw = st.checkbox("Show raw data tables", False)

if file1 and file2:
    try:
        df1 = process_fit(file1.getvalue(), smooth_gps, cadence_rev_m)
        df2 = process_fit(file2.getvalue(), smooth_gps, cadence_rev_m)
        df1, df2 = align_pair(df1, df2, start_time, end_time)
    except Exception as exc:
        st.error(f"Problem loading or aligning files: {exc}")
    else:
        st.success(f"Aligned {len(df1)} samples; timestamps match: {df1['timestamp'].equals(df2['timestamp'])}")

        st.subheader("Single-ride views")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.plotly_chart(plot_fit_interactive(df1, title=f"{rider1_name} signals"), use_container_width=True)
        with col2:
            st.plotly_chart(plot_fit_interactive(df2, title=f"{rider2_name} signals"), use_container_width=True)

        st.subheader("Power and speed comparison")
        st.plotly_chart(
            compare_rides_power_speed(df1, df2, name1=rider1_name, name2=rider2_name,
                                      speed_mode1=speed_mode1, speed_mode2=speed_mode2),
            use_container_width=True,
        )

        st.subheader("Speed intersections")
        intersections = find_speed_intersections_same_time(
            df1, df2, speed_mode1=speed_mode1, speed_mode2=speed_mode2, min_speed=min_speed
        )
        st.plotly_chart(
            plot_speed_with_intersections(df1, df2, intersections,
                                          name1=rider1_name, name2=rider2_name,
                                          speed_mode1=speed_mode1, speed_mode2=speed_mode2),
            use_container_width=True,
        )
        # st.dataframe(intersections)

        st.subheader("Segment analysis")
        df1_seg = segment_by_intersections(df1, intersections, min_duration_s=min_seg_len)
        df2_seg = segment_by_intersections(df2, intersections, min_duration_s=min_seg_len)

        stats1 = segment_stats(df1_seg)
        stats2 = segment_stats(df2_seg)
        merged = stats1.merge(stats2, on="segment_id", suffixes=(f"_{rider1_name}", f"_{rider2_name}"))
        st.dataframe(merged)

        roles = build_roles(stats1, stats2)

        st.subheader("Opposed segment comparison")
        available_metrics = [m for m in ["avg_power", "avg_speed_kmh", "normalized_power", "variability_index"] if m in stats1.columns and m in stats2.columns]
        if available_metrics:
            metric_choice = st.selectbox("Metric", available_metrics, index=0)
            bar_scale = st.slider("Bar thickness scale", 0.5, 2.0, 1.2, 0.1)
            role_filter = st.selectbox("Show segments by role", ["all", "active", "neutral"], index=0)
            st.plotly_chart(
                plot_opposed_segment_metric(
                    stats1, stats2, roles=roles, metric=metric_choice,
                    name1=rider1_name, name2=rider2_name, duration_scale=bar_scale, role_filter=role_filter,
                ),
                use_container_width=True,
            )
        else:
            st.warning("No common metrics available to plot.")

        st.subheader(f"Power split: last {last_seconds}s vs rest (per segment)")
        p1 = power_split_stats(df1_seg, last_seconds=last_seconds)
        p2 = power_split_stats(df2_seg, last_seconds=last_seconds)
        col_ps1, col_ps2 = st.columns(2, gap="large")
        with col_ps1:
            st.markdown(f"**{rider1_name}**")
            st.dataframe(
                p1.rename(
                    columns={
                        "avg_power_last": f"avg_power_last ({last_seconds}s, W)",
                        "avg_power_rest": "avg_power_rest (W)",
                    }
                )
            )
        with col_ps2:
            st.markdown(f"**{rider2_name}**")
            st.dataframe(
                p2.rename(
                    columns={
                        "avg_power_last": f"avg_power_last ({last_seconds}s, W)",
                        "avg_power_rest": "avg_power_rest (W)",
                    }
                )
            )

        st.subheader("Neutral-only power split overlay")
        st.plotly_chart(
            plot_opposed_power_split_neutral_overlay(
                p1, p2, roles=roles, name1=rider1_name, name2=rider2_name,
                title=f"Neutral segments: last {last_seconds}s vs rest"
            ),
            use_container_width=True,
        )

        st.subheader("Active-only power split overlay")
        st.plotly_chart(
            plot_opposed_power_split_active_overlay(
                p1, p2, roles=roles, name1=rider1_name, name2=rider2_name,
                title=f"Active segments: last {last_seconds}s vs rest"
            ),
            use_container_width=True,
        )

        if show_raw:
            st.subheader("Raw aligned data")
            st.dataframe(df1.head(500))
            st.dataframe(df2.head(500))
else:
    st.info("Upload two FIT files to begin.")
