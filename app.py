from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from demflex.config import load_yaml_config, parse_config
from demflex.ingest import (
    read_load,
    read_prices,
    list_load_zone_columns,
    list_price_zone_columns,
)
from demflex import peaks as peaks_mod
from demflex.st_model import STParams, CohortParams, summarize_event


st.set_page_config(page_title="DEMFlex — Summer Peaks (MVP)", layout="wide")
st.title("DEMFlex — Smart Thermostat MVP")
st.caption("Summer peak identification with selectable zones. Next: impact & economics.")

with st.sidebar:
    st.header("Configuration")
    cfg_path = st.text_input("Config path", value="config/defaults.yaml")
    page = st.radio("View", ["Peaks", "Impact"], index=0)
    uploaded_load = st.file_uploader("Load data (Excel/CSV)", type=["xlsx", "xls", "csv"])
    uploaded_price = st.file_uploader("Price data (CSV)", type=["csv"])
    top_days = st.number_input("Top days (D)", min_value=1, max_value=100, value=10, step=1)
    event_length_h = st.number_input("Event length (hours, continuous)", min_value=1, max_value=24, value=3, step=1)
    deltaT_F = st.number_input("Thermostat setpoint change ΔT (°F)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    target_kW = st.number_input("Target capacity (kW)", min_value=0.0, value=float(50), step=10.0)
    rebound_hours = st.number_input("Rebound hours (post-event)", min_value=1, max_value=12, value=2, step=1)
    if page == "Peaks":
        run_btn_peaks = st.button("Find Summer Peaks")
        run_btn_impact = False
    else:
        run_btn_impact = st.button("Compute Impact")
        run_btn_peaks = False


@st.cache_data(show_spinner=False)
def _load_config(path: str, _mtime: float | None = None):
    """Load and parse YAML config; include file mtime in cache key so edits take effect."""
    return parse_config(load_yaml_config(path))


def _load_data(cfg):
    # Infer candidate zones from headers (supports uploads)
    with st.sidebar:
        st.subheader("Zones")
        # Load zone options
        if uploaded_load is not None:
            options_load = list_load_zone_columns(cfg.io, file_obj=uploaded_load)
            uploaded_load.seek(0)
        else:
            options_load = list_load_zone_columns(cfg.io)
        default_load_zone = cfg.io.load_zone_column if cfg.io.load_zone_column in options_load else (options_load[0] if options_load else None)
        load_zone = st.selectbox("Load zone", options=options_load, index=(options_load.index(default_load_zone) if default_load_zone in options_load else 0))

        # Price zone options
        if uploaded_price is not None:
            options_price = list_price_zone_columns(cfg.io, file_obj=uploaded_price)
            uploaded_price.seek(0)
        else:
            try:
                options_price = list_price_zone_columns(cfg.io)
            except Exception:
                options_price = []
        if options_price:
            default_price_zone = cfg.io.price_zone_column if cfg.io.price_zone_column in options_price else options_price[0]
            price_zone = st.selectbox("Price zone", options=options_price, index=(options_price.index(default_price_zone) if default_price_zone in options_price else 0))
        else:
            price_zone = None

    # Read data using selected zones
    if uploaded_load is not None:
        load_df = read_load(cfg.io, zone=load_zone, file_obj=uploaded_load)
    else:
        load_df = read_load(cfg.io, zone=load_zone)

    if uploaded_price is not None:
        price_df = read_prices(cfg.io, lz=price_zone, file_obj=uploaded_price) if price_zone else None
    else:
        try:
            price_df = read_prices(cfg.io, lz=price_zone) if price_zone else read_prices(cfg.io)
        except Exception:
            price_df = None

    return load_df, price_df, load_zone, price_zone


try:
    cfg_mtime = Path(cfg_path).stat().st_mtime
except Exception:
    cfg_mtime = None
cfg = _load_config(cfg_path, _mtime=cfg_mtime)
load_df, price_df, load_zone_selected, price_zone_selected = _load_data(cfg)

if run_btn_peaks or run_btn_impact:
    st.subheader("Inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("Load rows:", len(load_df))
    with c2:
        st.write("Price rows:", len(price_df) if price_df is not None else "(none)")
    with c3:
        st.write("Season months:", cfg.program.season_months)
    st.write(f"Load zone: {load_zone_selected}")
    if price_zone_selected:
        st.write(f"Price zone: {price_zone_selected}")

    # Simple summer-only peak selection with continuous windows per day
    days_df, windows_df = peaks_mod.select_top_days_and_windows(
        load_df,
        season_months=cfg.program.season_months,
        top_days=int(top_days),
        window_h=int(event_length_h),
    )

    # Sort outputs by time for readability
    days_df = days_df.sort_values("date").reset_index(drop=True)

    st.subheader("Summer Load (time series)")
    # Build common summer series (tz-naive)
    import altair as alt
    summer_mask = load_df["ts"].dt.month.isin(cfg.program.season_months)
    summer_plot = load_df.loc[summer_mask, ["ts", "load_MW"]].copy()
    summer_plot["ts"] = pd.to_datetime(summer_plot["ts"]).dt.tz_localize(None)

    # Build markers for event hours (for both pages)
    win_hours = []
    for _, row in windows_df.iterrows():
        start = pd.to_datetime(row["start_ts"])  # tz-aware
        for i in range(int(event_length_h)):
            t = (start + pd.Timedelta(hours=i)).tz_localize(None)
            match = summer_plot.loc[summer_plot["ts"] == t, "load_MW"]
            val = float(match.iloc[0]) if not match.empty else None
            win_hours.append({"ts": t, "load_MW": val})
    hours_plot = pd.DataFrame(win_hours)

    if page == "Peaks" and run_btn_peaks:
        st.subheader("Summer Load (time series)")
        line = alt.Chart(summer_plot).mark_line(color="#1f77b4").encode(
            x=alt.X("ts:T", title="Time"),
            y=alt.Y("load_MW:Q", title="Load (MW)"),
        )
        points = alt.Chart(hours_plot).mark_point(color="red", size=60).encode(
            x="ts:T",
            y="load_MW:Q",
            tooltip=["ts:T", "load_MW:Q"],
        )
        st.altair_chart(alt.layer(line, points).resolve_scale(y="shared"), use_container_width=True)

        st.subheader("Top Days by Daily Max (summer)")
        # Display date as YYYY-MM-DD without timezone
        days_display = days_df.copy()
        days_display["date"] = pd.to_datetime(days_display["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(days_display)

        st.subheader("Selected Event Windows (continuous)")
        win_display = windows_df.copy()
        win_display["date"] = pd.to_datetime(win_display["date"]).dt.strftime("%Y-%m-%d")
        win_display["start"] = pd.to_datetime(win_display["start_ts"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M")
        win_display["end"] = pd.to_datetime(win_display["end_ts"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(win_display[["date", "start", "end", "duration_h", "avg_load_MW", "peak_within_window_MW"]])

        # Downloads for Peaks page
        def to_csv_bytes(df: pd.DataFrame) -> bytes:
            return df.to_csv(index=False).encode("utf-8")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download Top Days CSV", data=to_csv_bytes(days_display), file_name="top_days.csv", mime="text/csv")
        with c2:
            st.download_button("Download Event Windows CSV", data=to_csv_bytes(win_display), file_name="event_windows.csv", mime="text/csv")

    elif page == "Impact" and run_btn_impact:
        # Smart Thermostat impact model (per event)
        if cfg.st is not None and cfg.cohort is not None:
            st_params = STParams(
                alpha_kW_per_deg=cfg.st.alpha_kW_per_deg,
                duration_breakpoints_h=cfg.st.duration_breakpoints_h,
                duration_multipliers=cfg.st.duration_multipliers,
                cap_kW_per_home=cfg.st.cap_kW_per_home,
                rebound_fraction=cfg.st.rebound_fraction,
            )
            cohort = CohortParams(
                N_eligible=cfg.cohort.N_eligible,
                penetration=cfg.cohort.penetration,
                enrollment_rate=cfg.cohort.enrollment_rate,
                enablement_rate=cfg.cohort.enablement_rate,
                diversity_factor=cfg.cohort.diversity_factor,
            )
            res = summarize_event(deltaT_F=float(deltaT_F), L_h=float(event_length_h), target_kW=float(target_kW), st=st_params, cohort=cohort)
            st.subheader("Smart Thermostat Impact (per event)")
            st.write({
                "r_home_kW": round(res.get("r_home_kW", 0.0), 4),
                "participants_needed": res.get("participants_needed"),
                "participants_used": res.get("participants_used"),
                "feasible": res.get("feasible", True),
                "event_kW": round(res.get("event_kW", 0.0), 2),
                "kWh_curtailed": round(res.get("kWh_curtailed", 0.0), 2),
                "kWh_rebound": round(res.get("kWh_rebound", 0.0), 2),
                "kWh_net": round(res.get("kWh_net", 0.0), 2),
            })

        # Build original vs after-event adjusted load series for summer months
        # Initialize adjusted series
        summer_plot["load_MW_after"] = summer_plot["load_MW"].astype(float)

        if cfg.st is not None and cfg.cohort is not None:
            event_kw = float(res.get("event_kW", 0.0))
            # Convert kW to MW for consistency with load_MW
            event_kw_mw = event_kw / 1000.0
            for _, row in windows_df.iterrows():
                start_tz = pd.to_datetime(row["start_ts"])  # tz-aware
                end_tz = pd.to_datetime(row["end_ts"])      # tz-aware

                # Event hours
                for i in range(int(event_length_h)):
                    t = (start_tz + pd.Timedelta(hours=i)).tz_localize(None)
                    sel = summer_plot["ts"] == t
                    summer_plot.loc[sel, "load_MW_after"] = (summer_plot.loc[sel, "load_MW_after"] - event_kw_mw).clip(lower=0.0)

                # Rebound hours (distribute evenly)
                # energy in MWh = (event_kw in MW) * hours
                rebound_energy_mwh = event_kw_mw * float(event_length_h) * float(st_params.rebound_fraction)
                if rebound_hours > 0 and rebound_energy_mwh > 0:
                    rebound_power_mw = rebound_energy_mwh / float(rebound_hours)
                    for j in range(int(rebound_hours)):
                        t = (end_tz + pd.Timedelta(hours=j)).tz_localize(None)
                        sel = summer_plot["ts"] == t
                        summer_plot.loc[sel, "load_MW_after"] = summer_plot.loc[sel, "load_MW_after"] + rebound_power_mw

            st.subheader("Summer Load: original vs after-event")
            line_orig = alt.Chart(summer_plot).mark_line(color="#1f77b4").encode(
                x=alt.X("ts:T", title="Time"),
                y=alt.Y("load_MW:Q", title="Load (MW)"),
            )
            line_after = alt.Chart(summer_plot).mark_line(color="green", strokeDash=[4,4]).encode(
                x="ts:T",
                y="load_MW_after:Q",
            )
            points = alt.Chart(hours_plot).mark_point(color="red", size=60).encode(
                x="ts:T",
                y="load_MW:Q",
                tooltip=["ts:T", "load_MW:Q"],
            )
            st.altair_chart(alt.layer(line_orig, line_after, points).resolve_scale(y="shared"), use_container_width=True)
        else:
            st.info("Cohort/ST model parameters not found in config; impact model summary skipped.")
            # Show original-only chart on Impact page as fallback
            st.subheader("Summer Load (time series)")
            st.altair_chart(
                alt.Chart(summer_plot).mark_line(color="#1f77b4").encode(
                    x=alt.X("ts:T", title="Time"),
                    y=alt.Y("load_MW:Q", title="Load (MW)"),
                ),
                use_container_width=True,
            )
