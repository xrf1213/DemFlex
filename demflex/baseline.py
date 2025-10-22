from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Union, Sequence


MaskLike = Optional[Union[pd.Series, Sequence[bool]]]


def _to_mask(mask: MaskLike, length: int) -> pd.Series:
    if mask is None:
        return pd.Series([True] * length)
    if isinstance(mask, pd.Series):
        if len(mask) != length:
            raise ValueError("restrict_mask length does not match data length")
        return mask.astype(bool)
    arr = np.asarray(mask).astype(bool)
    if arr.shape[0] != length:
        raise ValueError("restrict_mask length does not match data length")
    return pd.Series(arr)


def percentile_by_hour(load_df: pd.DataFrame, percent: float = 0.9, restrict_mask: MaskLike = None) -> pd.DataFrame:
    """
    Compute a baseline as the percentile by hour-of-day across the dataset (optionally
    restricted to a mask such as season/weekday).

    Returns a DataFrame with columns: ts, load_MW, baseline_MW.
    """
    df = load_df.copy()
    df["hour"] = df["ts"].dt.hour

    mask = _to_mask(restrict_mask, len(df))
    pool = df.loc[mask.values]
    # group by hour-of-day and compute percentile
    basemap = pool.groupby("hour")["load_MW"].quantile(percent).to_dict()
    df["baseline_MW"] = df["hour"].map(basemap)
    return df.drop(columns=["hour"]).sort_values("ts").reset_index(drop=True)


def recent_percentile_by_hour(load_df: pd.DataFrame, window_days: int = 30, percent: float = 0.9,
                              restrict_mask: MaskLike = None) -> pd.DataFrame:
    """
    Rolling version: for each timestamp, use the prior N days of the same hour
    to compute the percentile. If insufficient history, fallback to global percentile.
    """
    df = load_df.copy().sort_values("ts").reset_index(drop=True)
    df["hour"] = df["ts"].dt.hour
    df["_mask"] = _to_mask(restrict_mask, len(df)).values

    # Precompute global percentile by hour for fallback
    global_map = df.loc[df["_mask"]].groupby("hour")["load_MW"].quantile(percent).to_dict()

    baselines = []
    for idx, row in df.iterrows():
        t = row["ts"]
        hour = row["hour"]
        t0 = t - pd.Timedelta(days=window_days)
        window = df[(df["ts"] >= t0) & (df["ts"] < t) & (df["hour"] == hour) & (df["_mask"])]
        if len(window) >= 5:  # require a few points
            b = window["load_MW"].quantile(percent)
        else:
            b = global_map.get(hour, np.nan)
        baselines.append(b)

    df["baseline_MW"] = baselines
    return df.drop(columns=["hour", "_mask"]).sort_values("ts").reset_index(drop=True)
