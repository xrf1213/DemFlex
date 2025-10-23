from __future__ import annotations

from typing import List, Tuple

import pandas as pd


# Removed older helpers (select_peaks, select_top_days_and_hours) as unused.


def select_top_days_and_windows(load_df: pd.DataFrame, season_months: List[int],
                                top_days: int, window_h: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summer-only selection with continuous windows per day.
    Steps:
      1) Filter to season months.
      2) Rank days by daily max (top `top_days`).
      3) For each selected day, find the continuous `window_h`-hour window with the highest average load.

    Returns:
      days_df: [date, daily_max_MW]
      windows_df: [date, start_ts, end_ts, duration_h, avg_load_MW, peak_within_window_MW]
    """
    df = load_df.copy().sort_values("ts").reset_index(drop=True)
    dt = pd.DatetimeIndex(df["ts"])  # tz-aware
    summer = df.loc[dt.month.isin(season_months)].copy()
    summer["date"] = summer["ts"].dt.tz_localize(None).dt.normalize()

    daily = summer.groupby("date")["load_MW"].max().reset_index(name="daily_max_MW")
    top_days_df = daily.sort_values("daily_max_MW", ascending=False).head(top_days).copy()
    top_dates = set(top_days_df["date"].tolist())

    rows = []
    for d, day_df in summer.groupby("date", sort=True):
        if d not in top_dates:
            continue
        day_df = day_df.sort_values("ts").reset_index(drop=True)
        if len(day_df) < window_h:
            # Use whole day if fewer hours
            start_idx = 0
            end_idx = len(day_df)
        else:
            # rolling sum over window_h
            s = day_df["load_MW"].rolling(window=window_h, min_periods=window_h).sum()
            end_idx = int(s.idxmax()) + 1  # rolling is right-aligned
            start_idx = end_idx - window_h
        win = day_df.iloc[start_idx:end_idx].copy()
        start_ts = win["ts"].iloc[0]
        end_ts = win["ts"].iloc[-1] + pd.Timedelta(hours=1)
        avg_load = win["load_MW"].mean()
        peak_in_win = win["load_MW"].max()
        rows.append({
            "date": d,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_h": int(window_h),
            "avg_load_MW": float(avg_load),
            "peak_within_window_MW": float(peak_in_win),
        })

    windows_df = pd.DataFrame(rows).sort_values("start_ts").reset_index(drop=True)
    top_days_df = top_days_df.sort_values("date").reset_index(drop=True)
    return top_days_df, windows_df
