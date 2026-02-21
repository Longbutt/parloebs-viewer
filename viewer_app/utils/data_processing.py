"""
data_processing.py
------------------
FIT file parsing, data loading, timestamp alignment,
and sensor/GPS speed computation utilities.
"""

import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from fitparse import FitFile


def semicircles_to_degrees(x):
    return x * (180 / 2 ** 31)


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
