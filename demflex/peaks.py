from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class PeakSelectionConfig:
    max_events: int
    min_gap_days: int


def select_peaks(baseline_df: pd.DataFrame, program_mask: pd.Series,
                 cfg: PeakSelectionConfig) -> pd.DataFrame:
    """
    Greedy selection of peak hours within the program mask based on baseline_MW.
    Enforces a minimum gap in days between selected events and caps total events.

    Returns DataFrame with columns: event_start (ts), baseline_MW, load_MW.
    """
    df = baseline_df.copy()
    cols_needed = {"ts", "baseline_MW", "load_MW"}
    missing = cols_needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for peak selection: {missing}")

    cand = df.loc[program_mask.values, ["ts", "baseline_MW", "load_MW"]].copy()
    cand = cand.sort_values(["baseline_MW", "ts"], ascending=[False, True]).reset_index(drop=True)

    selected_rows: List[dict] = []
    selected_dates: List[pd.Timestamp] = []
    for _, row in cand.iterrows():
        if len(selected_rows) >= cfg.max_events:
            break
        ts = row["ts"]
        # Enforce day-level spacing
        ok = True
        for prev in selected_dates:
            if abs((ts.normalize() - prev.normalize()).days) < cfg.min_gap_days:
                ok = False
                break
        if not ok:
            continue
        selected_rows.append({
            "event_start": ts,
            "baseline_MW": row["baseline_MW"],
            "load_MW": row["load_MW"],
        })
        selected_dates.append(ts)

    return pd.DataFrame(selected_rows)


def select_top_days_and_hours(load_df: pd.DataFrame, season_months: List[int],
                              top_days: int, top_hours_per_day: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simplified peak selection:
    - Filter to season months only (no weekday or time-window filtering).
    - Find the top `top_days` calendar days by their daily max load.
    - Within each selected day, pick the top `top_hours_per_day` hourly points by load.

    Returns:
      days_df: columns [date, daily_max_MW]
      hours_df: columns [ts, date, load_MW]
    """
    df = load_df.copy().sort_values("ts").reset_index(drop=True)
    dt = pd.DatetimeIndex(df["ts"])  # tz-aware
    summer = df.loc[dt.month.isin(season_months)].copy()
    # Use tz-naive local dates for display and grouping (avoid showing time zone suffix)
    summer["date"] = summer["ts"].dt.tz_localize(None).dt.normalize()

    daily = summer.groupby("date")["load_MW"].max().reset_index(name="daily_max_MW")
    daily_sorted = daily.sort_values("daily_max_MW", ascending=False).head(top_days).reset_index(drop=True)

    top_dates = set(daily_sorted["date"].tolist())
    in_top = summer["date"].isin(top_dates)
    hours = summer.loc[in_top].copy()
    hours = hours.sort_values(["date", "load_MW"], ascending=[True, False])
    # For each day, take top K hours
    hours["rank_in_day"] = hours.groupby("date")["load_MW"].rank(method="first", ascending=False)
    hours_topk = hours.loc[hours["rank_in_day"] <= top_hours_per_day].copy().drop(columns=["rank_in_day"]).reset_index(drop=True)
    # Add a tz-naive timestamp column for display convenience
    hours_topk["ts_local"] = hours_topk["ts"].dt.tz_localize(None)

    return daily_sorted, hours_topk


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
