from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, IO, List

import pandas as pd

from .config import IOConfig, ProgramWindow


def _ensure_tz(s: pd.DatetimeIndex | pd.Series, tz: str) -> pd.DatetimeIndex:
    dt = pd.DatetimeIndex(s)
    if dt.tz is None:
        # Handle DST transitions robustly with fallback
        try:
            return dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            # Fallback: choose standard-time mapping for ambiguous entries
            return dt.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
    return dt.tz_convert(tz)


def _parse_hour_ending(series: pd.Series, *, fmt: str = "%m/%d/%Y %H:%M", tz: str | None = None) -> pd.DatetimeIndex:
    """
    Parse hour-ending timestamps, including values like "24:00" which represent
    the midnight at the start of the next day. Strategy:
      - If already datetime-like, return localized/converted.
      - Else, convert to string; replace " 24:00" with " 00:00" and add 1 day to those rows.
      - Parse with the provided format; fallback to pandas parser if needed.
    """
    # Already datetime-like
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_datetime64tz_dtype(series):
        dt = pd.to_datetime(series, errors="coerce")
        return _ensure_tz(dt, tz) if tz else pd.DatetimeIndex(dt)

    s = series.astype(str).str.strip()
    # Extract only the date and HH:MM portion, drop trailing annotations like 'DST', 'PDT', etc.
    extracted = s.str.extract(r"^(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2})", expand=True)
    s2 = (extracted[0] + " " + extracted[1]).where(~extracted.isna().any(axis=1), other=s)

    mask_24 = s2.str.endswith("24:00", na=False)
    s2 = s2.mask(mask_24, s2.str.replace(" 24:00", " 00:00", regex=False))

    ts = pd.to_datetime(s2, format=fmt, errors="coerce")
    # Fallback to flexible parsing for any residuals
    if ts.isna().any():
        ts2 = pd.to_datetime(s2, errors="coerce")
        # Prefer parsed where available
        ts = ts.fillna(ts2)

    if ts.isna().any():
        bad = s2.loc[ts.isna()].head(3).tolist()
        raise ValueError(f"Failed to parse some timestamps, examples: {bad}")

    if mask_24.any():
        ts.loc[mask_24.values] = ts.loc[mask_24.values] + pd.Timedelta(days=1)

    return _ensure_tz(ts, tz) if tz else pd.DatetimeIndex(ts)


def read_load(io: IOConfig, zone: Optional[str] = None, file_obj: Optional[Union[IO[bytes], IO[str]]] = None) -> pd.DataFrame:
    path = Path(io.load_data_path) if file_obj is None else None
    zone_col = zone or io.load_zone_column

    if file_obj is not None:
        try:
            df = pd.read_excel(file_obj, sheet_name=0)
        except Exception:
            file_obj.seek(0)
            df = pd.read_csv(file_obj)
    else:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path, sheet_name=0)
        elif path.suffix.lower() in {".csv"}:
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported load data format: {path.suffix}")

    time_col = io.load_time_column
    if time_col not in df.columns:
        raise KeyError(f"Load time column '{time_col}' not found. Available: {list(df.columns)}")
    if zone_col not in df.columns:
        raise KeyError(f"Load zone column '{zone_col}' not found. Available: {list(df.columns)}")

    # Parse hour-ending timestamps
    ts = _parse_hour_ending(df[time_col], fmt="%m/%d/%Y %H:%M", tz=io.timezone)

    out = pd.DataFrame({
        "ts": ts,
        "load_MW": pd.to_numeric(df[zone_col], errors="coerce"),
    }).dropna()
    out = out.sort_values("ts").reset_index(drop=True)
    return out


def read_prices(io: IOConfig, lz: Optional[str] = None, file_obj: Optional[Union[IO[bytes], IO[str]]] = None) -> pd.DataFrame:
    path = Path(io.price_data_path) if file_obj is None else None
    lz_col = lz or io.price_zone_column
    if file_obj is not None:
        df = pd.read_csv(file_obj)
    else:
        df = pd.read_csv(path)
    time_col = io.price_time_column
    if time_col not in df.columns:
        raise KeyError(f"Price time column '{time_col}' not found. Available: {list(df.columns)}")
    if lz_col not in df.columns:
        raise KeyError(f"Price zone column '{lz_col}' not found. Available: {list(df.columns)}")

    ts = _parse_hour_ending(df[time_col], fmt="%m/%d/%Y %H:%M", tz=io.timezone)

    out = pd.DataFrame({
        "ts": ts,
        "price_per_kWh": pd.to_numeric(df[lz_col], errors="coerce") / 100.0 if df[lz_col].max() > 10 else pd.to_numeric(df[lz_col], errors="coerce"),
    }).dropna()
    out = out.sort_values("ts").reset_index(drop=True)
    return out


def program_mask(ts: pd.Series, program: ProgramWindow) -> pd.Series:
    dt = pd.DatetimeIndex(ts)
    in_month = pd.Series(dt.month.isin(program.season_months), index=dt)
    if program.weekdays_only:
        # Monday=0 .. Sunday=6
        is_weekday = pd.Series(dt.weekday <= 4, index=dt)
    else:
        is_weekday = pd.Series([True] * len(dt), index=dt)

    # Time window inclusive of start, exclusive of end
    start_h, start_m = map(int, program.start_time.split(":"))
    end_h, end_m = map(int, program.end_time.split(":"))
    hm = dt.hour * 60 + dt.minute
    in_window = pd.Series((hm >= start_h * 60 + start_m) & (hm < end_h * 60 + end_m), index=dt)

    # blackout dates (YYYY-MM-DD)
    blk = set(program.blackout_dates)
    day_str_idx = dt.normalize().strftime("%Y-%m-%d")
    not_blackout = pd.Series(~pd.Index(day_str_idx).isin(blk), index=dt)

    mask = (in_month & is_weekday & in_window & not_blackout).astype(bool)
    # Reindex mask to positional index aligned with input series
    return mask.reset_index(drop=True)


def season_weekday_mask(ts: pd.Series, program: ProgramWindow) -> pd.Series:
    """Mask for season months, weekday rule, and blackout dates, but NOT the daily time window.
    Useful for computing baselines continuously across the day.
    """
    dt = pd.DatetimeIndex(ts)
    in_month = pd.Series(dt.month.isin(program.season_months), index=dt)
    if program.weekdays_only:
        is_weekday = pd.Series(dt.weekday <= 4, index=dt)
    else:
        is_weekday = pd.Series([True] * len(dt), index=dt)
    blk = set(program.blackout_dates)
    day_str_idx = dt.normalize().strftime("%Y-%m-%d")
    not_blackout = pd.Series(~pd.Index(day_str_idx).isin(blk), index=dt)
    mask = (in_month & is_weekday & not_blackout).astype(bool)
    return mask.reset_index(drop=True)


def list_load_zone_columns(io: IOConfig, file_obj: Optional[Union[IO[bytes], IO[str]]] = None) -> List[str]:
    """Return candidate load zone columns by inspecting the source file header.
    Prefers numeric columns excluding the time column; falls back to known ERCOT names.
    """
    path = Path(io.load_data_path) if file_obj is None else None
    if file_obj is not None:
        try:
            df = pd.read_excel(file_obj, sheet_name=0, nrows=5)
        except Exception:
            file_obj.seek(0)
            df = pd.read_csv(file_obj, nrows=5)
    else:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path, sheet_name=0, nrows=5)
        else:
            df = pd.read_csv(path, nrows=5)

    time_col = io.load_time_column
    candidates = [c for c in df.columns if c != time_col]
    # Prefer numeric-looking columns
    numeric_cols: List[str] = []
    for c in candidates:
        ser = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series([], dtype=float)
        if getattr(ser, "notna", lambda: pd.Series([]))().any():
            numeric_cols.append(c)
    if numeric_cols:
        candidates = numeric_cols
    known = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST", "ERCOT"]
    known_present = [c for c in candidates if c in known]
    return known_present or candidates


def list_price_zone_columns(io: IOConfig, file_obj: Optional[Union[IO[bytes], IO[str]]] = None) -> List[str]:
    """Return candidate price zone columns by inspecting the CSV header, preferring LZ_* columns."""
    path = Path(io.price_data_path) if file_obj is None else None
    if file_obj is not None:
        df = pd.read_csv(file_obj, nrows=5)
    else:
        df = pd.read_csv(path, nrows=5)
    time_col = io.price_time_column
    candidates = [c for c in df.columns if c != time_col]
    lz = [c for c in candidates if str(c).startswith("LZ_")]
    return lz or candidates
