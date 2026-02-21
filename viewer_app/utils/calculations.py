"""
calculations.py
---------------
All complex mathematical logic: intersection finding,
segmentation, power/speed stats, energy and residual calculations.
"""

import numpy as np
import pandas as pd
import streamlit as st


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

    h1 = df1["altitude"] if "altitude" in df1.columns else None
    h2 = df2["altitude"] if "altitude" in df2.columns else None

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

        dpe1 = mass1 * g * (h1.iloc[i1] - h1.iloc[i0]) if h1 is not None and mass1 > 0 else np.nan
        dpe2 = mass2 * g * (h2.iloc[i1] - h2.iloc[i0]) if h2 is not None and mass2 > 0 else np.nan

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
