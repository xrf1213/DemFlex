from __future__ import annotations
from pathlib import Path
import math

import pandas as pd
import streamlit as st
import altair as alt

from demflex.config import load_yaml_config, parse_config
from demflex.ingest import (
    read_load,
    read_prices,
    list_load_zone_columns,
    list_price_zone_columns,
)
from demflex import peaks as peaks_mod
import importlib
import demflex.economics as economics_mod

from demflex import optimization

st.set_page_config(page_title="DemFlex", layout="wide")
st.title("DemFlex")

# Top-level view switch (outside the sidebar)
page = st.radio("View", ["Peaks", "Impact", "Economics", "Optimization"], index=0, horizontal=True)

with st.sidebar:
    st.header("Configuration")
    cfg_path = st.text_input("Config path", value="config/defaults.yaml")
    uploaded_load = st.file_uploader("Load data (Excel/CSV)", type=["xlsx", "xls", "csv"])
    uploaded_price = st.file_uploader("Price data (CSV)", type=["csv"])


@st.cache_data(show_spinner=False)
def _load_config(path: str, _mtime: float | None = None, _code_mtime: float | None = None):
    """Load and parse YAML config; cache key includes YAML mtime and config.py mtime."""
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

    # Align price series to the same time range as load (e.g., 2024 for default data)
    if price_df is not None and not load_df.empty:
        tmin, tmax = load_df["ts"].min(), load_df["ts"].max()
        price_df = price_df[(price_df["ts"] >= tmin) & (price_df["ts"] <= tmax)].reset_index(drop=True)

    return load_df, price_df, load_zone, price_zone


try:
    cfg_mtime = Path(cfg_path).stat().st_mtime
except Exception:
    cfg_mtime = None
try:
    from demflex import config as _cfg_mod
    code_mtime = Path(_cfg_mod.__file__).stat().st_mtime
except Exception:
    code_mtime = None
cfg = _load_config(cfg_path, _mtime=cfg_mtime, _code_mtime=code_mtime)
load_df, price_df, load_zone_selected, price_zone_selected = _load_data(cfg)

if page == "Peaks":
    top_days = st.number_input("Event Numbers per Season", min_value=1, max_value=100, value=50, step=1)
    event_length_h = st.number_input("Event length (hours, continuous)", min_value=1, max_value=24, value=4, step=1)
    run_btn_peaks = st.button("Find Summer Peaks")
    if run_btn_peaks:
        days_df, windows_df = peaks_mod.select_top_days_and_windows(
            load_df,
            season_months=cfg.program.season_months,
            top_days=int(top_days),
            window_h=int(event_length_h),
        )
        days_df = days_df.sort_values("date").reset_index(drop=True)
        st.session_state["windows_df"] = windows_df
        st.session_state["days_df"] = days_df
        st.session_state["peaks_event_length_h"] = int(event_length_h)
        st.session_state["peaks_top_days"] = int(top_days)

        import altair as alt
        summer_mask = load_df["ts"].dt.month.isin(cfg.program.season_months)
        summer_plot = load_df.loc[summer_mask, ["ts", "load_MW"]].copy()
        summer_plot["ts"] = pd.to_datetime(summer_plot["ts"]).dt.tz_localize(None)
        # Apply the same seasonal filtering to prices for consistent row counts
        price_season_rows = "N/A"
        if price_df is not None:
            price_summer_mask = price_df["ts"].dt.month.isin(cfg.program.season_months)
            price_summer = price_df.loc[price_summer_mask, ["ts", "price_per_MWh"]].copy()
            price_summer["ts"] = pd.to_datetime(price_summer["ts"]).dt.tz_localize(None)
            price_season_rows = f"{len(price_summer):,}"
        win_hours = []
        for _, row in windows_df.iterrows():
            start = pd.to_datetime(row["start_ts"]).tz_localize(None)
            for i in range(int(event_length_h)):
                t = start + pd.Timedelta(hours=i)
                match = summer_plot.loc[summer_plot["ts"] == t, "load_MW"]
                val = float(match.iloc[0]) if not match.empty else None
                win_hours.append({"ts": t, "load_MW": val})
        hours_plot = pd.DataFrame(win_hours)

        st.subheader("Inputs")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(label="Load rows (season)", value=f"{len(summer_plot):,}")
        with c2:
            st.metric(label="Price rows (season)", value=price_season_rows)
        with c3:
            months_str = ", ".join(str(m) for m in cfg.program.season_months)
            st.metric(label="Season months", value=months_str)
        with c4:
            st.metric(label="Event length (h)", value=str(event_length_h))

        z1, z2 = st.columns(2)
        with z1:
            st.caption(f"Load zone: `{load_zone_selected}`")
        with z2:
            pz = price_zone_selected if price_zone_selected else "N/A"
            st.caption(f"Price zone: `{pz}`")

        st.subheader("Summer Load (time series)")
        line = alt.Chart(summer_plot).mark_line(color="#1f77b4").encode(
            x=alt.X("ts:T", title="Time"),
            y=alt.Y("load_MW:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
        )
        points = alt.Chart(hours_plot).mark_point(color="red", size=60).encode(
            x="ts:T",
            y="load_MW:Q",
            tooltip=["ts:T", "load_MW:Q"],
        )
        st.altair_chart(alt.layer(line, points).resolve_scale(y="shared"), use_container_width=True)

        st.subheader("Top Days by Daily Max (summer)")
        days_display = days_df.copy()
        days_display["date"] = pd.to_datetime(days_display["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(days_display)

        st.subheader("Selected Event Windows (continuous)")
        win_display = windows_df.copy()
        win_display["date"] = pd.to_datetime(win_display["date"]).dt.strftime("%Y-%m-%d")
        win_display["start"] = pd.to_datetime(win_display["start_ts"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M")
        win_display["end"] = pd.to_datetime(win_display["end_ts"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(win_display[["date", "start", "end", "duration_h", "avg_load_MW", "peak_within_window_MW"]])

        def to_csv_bytes(df: pd.DataFrame) -> bytes:
            return df.to_csv(index=False).encode("utf-8")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download Top Days CSV", data=to_csv_bytes(days_display), file_name="top_days.csv", mime="text/csv")
        with c2:
            st.download_button("Download Event Windows CSV", data=to_csv_bytes(win_display), file_name="event_windows.csv", mime="text/csv")

elif page == "Impact":
    total_households = st.number_input("Total households (region)", min_value=0, value=3_000_000, step=1000)
    penetration = st.number_input("Penetration rate", min_value=0.0, max_value=1.0, value=0.08, step=0.01)
    target_mw = st.number_input("Target capacity (MW)", min_value=0.0, value=float(50), step=1.0)
    tech_choice = st.selectbox("Technology", options=["Thermostats", "Solar PV", "Battery Storage"], index=0)
    run_btn_impact = st.button("Compute Impact")
    if run_btn_impact:
        participants = int(total_households * penetration)
        if participants <= 0:
            st.warning("Participants computed as 0. Increase households or penetration.")
            st.stop()

        target_kW = target_mw * 1000.0

        if tech_choice == "Thermostats":
            windows_df = st.session_state.get("windows_df")
            event_length_h = st.session_state.get("peaks_event_length_h")
            if windows_df is None or event_length_h is None:
                st.warning("Please run Peaks first (Find Summer Peaks) to select event windows, then return to Impact.")
                st.stop()

            base_deltaT = 4.0
            base_per_device_kw = 0.512
            deltaT_raw = (target_kW / (participants * base_per_device_kw)) * base_deltaT
            deltaT_required = max(2, min(6, int(math.ceil(deltaT_raw))))
            per_device_kw = base_per_device_kw * (deltaT_required / base_deltaT)
            aggregate_kW = participants * per_device_kw
            aggregate_MW = aggregate_kW / 1000.0

            # Build original vs after-event adjusted load (no rebound)
            summer_mask = load_df["ts"].dt.month.isin(cfg.program.season_months)
            summer_plot = load_df.loc[summer_mask, ["ts", "load_MW"]].copy()
            summer_plot["ts"] = pd.to_datetime(summer_plot["ts"]).dt.tz_localize(None)
            summer_plot["load_MW_after"] = summer_plot["load_MW"].astype(float)
            for _, row in windows_df.iterrows():
                start_tz = pd.to_datetime(row["start_ts"])  # tz-aware
                for i in range(int(event_length_h)):
                    t = (start_tz + pd.Timedelta(hours=i)).tz_localize(None)
                    sel = summer_plot["ts"] == t
                    summer_plot.loc[sel, "load_MW_after"] = (summer_plot.loc[sel, "load_MW_after"] - aggregate_MW).clip(lower=0.0)

            # Build red-point markers for event hours
            win_hours = []
            for _, row in windows_df.iterrows():
                start = pd.to_datetime(row["start_ts"]).tz_localize(None)
                for i in range(int(event_length_h)):
                    t = start + pd.Timedelta(hours=i)
                    match = summer_plot.loc[summer_plot["ts"] == t, "load_MW"]
                    val = float(match.iloc[0]) if not match.empty else None
                    win_hours.append({"ts": t, "load_MW": val})
            hours_plot = pd.DataFrame(win_hours)

            # Persist for downstream (Economics page still reads ΔT/target and plots)
            st.session_state["impact_tech"] = "thermostats"
            st.session_state["impact_deltaT_F"] = float(deltaT_required)
            st.session_state["impact_target_kW"] = float(target_kW)
            st.session_state["impact_summer_plot"] = summer_plot
            st.session_state["impact_hours_plot"] = hours_plot
            st.session_state["impact_participants"] = participants
            st.session_state["impact_per_device_kw"] = float(per_device_kw)
            st.session_state["impact_aggregate_MW"] = float(aggregate_MW)

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Participants", f"{participants:,}")
            with m2:
                st.metric("ΔT required (°F)", f"{deltaT_required}")
            with m3:
                st.metric("Per-device reduction (kW)", f"{per_device_kw:.3f}")
            with m4:
                st.metric("Aggregate per hour (MW)", f"{aggregate_MW:.3f}")

            st.subheader("Summer Load: original vs after-event")
            base_orig = alt.Chart(summer_plot).mark_line(color="#1f77b4").encode(
                x=alt.X("ts:T", title="Time"),
                y=alt.Y("load_MW:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
            )
            base_after = alt.Chart(summer_plot).mark_line(color="green", strokeDash=[4,4]).encode(
                x="ts:T",
                y=alt.Y("load_MW_after:Q", scale=alt.Scale(zero=False)),
            )
            base_points = alt.Chart(hours_plot).mark_point(color="red", size=60).encode(
                x="ts:T",
                y=alt.Y("load_MW:Q", scale=alt.Scale(zero=False)),
                tooltip=["ts:T", "load_MW:Q"],
            )

            # Overview + detail with brush selection for zooming
            brush = alt.selection_interval(encodings=["x"])
            overview = alt.layer(base_orig, base_after).add_selection(brush).properties(height=120)
            detail = alt.layer(
                base_orig.transform_filter(brush),
                base_after.transform_filter(brush),
                base_points.transform_filter(brush),
            ).properties(height=300)

            st.altair_chart(alt.vconcat(detail, overview).resolve_scale(y="shared"), use_container_width=True)

            # Focus view: single-day before vs after
            st.subheader("Focused Day: before vs after")
            day_options = sorted(pd.to_datetime(windows_df["date"]).dt.strftime("%Y-%m-%d").unique().tolist())
            # Persist and update focused day via a form to avoid heavy reruns on every change
            default_focus = st.session_state.get("impact_focus_day", day_options[0] if day_options else None)
            with st.form("focus_day_form_compute"):
                focus_day_sel = st.selectbox("Select a day", options=day_options, index=(day_options.index(default_focus) if default_focus in day_options else 0), key="impact_focus_day_select_compute")
                submit_day = st.form_submit_button("Update")
            if submit_day:
                st.session_state["impact_focus_day"] = focus_day_sel
            focus_day = st.session_state.get("impact_focus_day", day_options[0] if day_options else None)
            if focus_day is None:
                st.info("No day available to display.")
                focus_day = day_options[0] if day_options else None
            day_dt = pd.to_datetime(focus_day) if focus_day else None
            day_mask = summer_plot["ts"].dt.normalize() == day_dt
            day_df = summer_plot.loc[day_mask, ["ts", "load_MW", "load_MW_after"]]
            # Event hours for the selected day
            hours_day = hours_plot.loc[hours_plot["ts"].dt.normalize() == day_dt] if not hours_plot.empty else hours_plot

            day_orig = alt.Chart(day_df).mark_line(color="#1f77b4").encode(
                x=alt.X("ts:T", title="Time"),
                y=alt.Y("load_MW:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
            )
            day_after = alt.Chart(day_df).mark_line(color="green", strokeDash=[4,4]).encode(
                x="ts:T",
                y=alt.Y("load_MW_after:Q", scale=alt.Scale(zero=False)),
            )
            day_points = alt.Chart(hours_day).mark_point(color="red", size=80).encode(
                x="ts:T",
                y=alt.Y("load_MW:Q", scale=alt.Scale(zero=False)),
                tooltip=["ts:T", "load_MW:Q"],
            ) if not hours_day.empty else None

            if day_points is not None:
                st.altair_chart(alt.layer(day_orig, day_after, day_points).resolve_scale(y="shared"), use_container_width=True)
            else:
                st.altair_chart(alt.layer(day_orig, day_after).resolve_scale(y="shared"), use_container_width=True)

        elif tech_choice == "Battery Storage":
            # Require Peaks for event windows and length
            windows_df = st.session_state.get("windows_df")
            event_length_h = st.session_state.get("peaks_event_length_h")
            if windows_df is None or event_length_h is None:
                st.warning("Please run Peaks first (Find Summer Peaks) to select event windows, then return to Impact.")
                st.stop()

            # Minimal per-device requirements to meet target capacity for full event length
            per_device_power_kW = target_kW / participants if participants > 0 else 0.0
            per_device_energy_kWh = per_device_power_kW * float(event_length_h)
            aggregate_kW = participants * per_device_power_kW
            aggregate_MW = aggregate_kW / 1000.0

            # Build original vs after-event adjusted load with discharge during events and charge prior night hours
            summer_mask = load_df["ts"].dt.month.isin(cfg.program.season_months)
            summer_plot = load_df.loc[summer_mask, ["ts", "load_MW"]].copy()
            summer_plot["ts"] = pd.to_datetime(summer_plot["ts"]).dt.tz_localize(None)
            summer_plot["load_MW_after"] = summer_plot["load_MW"].astype(float)

            discharge_marks = []
            charge_marks = []

            # Helper to choose charging hours before each event (default night 00:00–06:00, nearest first)
            def choose_charge_hours(start_naive: pd.Timestamp, hours_needed: int) -> list[pd.Timestamp]:
                chosen: list[pd.Timestamp] = []
                start_hour = int(start_naive.hour)
                day0 = start_naive.normalize()
                # Build candidate hours in descending recency before event
                candidates: list[pd.Timestamp] = []
                # same day early morning before event (00..06 and < start_hour)
                for h in range(6, -1, -1):
                    ts = day0 + pd.Timedelta(hours=h)
                    if ts < start_naive:
                        candidates.append(ts)
                # previous days early morning (up to 2 days back)
                for d in [1, 2]:
                    day = day0 - pd.Timedelta(days=d)
                    for h in range(6, -1, -1):
                        candidates.append(day + pd.Timedelta(hours=h))
                # Select nearest hours first
                for ts in candidates:
                    if len(chosen) >= hours_needed:
                        break
                    chosen.append(ts)
                return chosen[:hours_needed]

            # Apply discharge during event windows and schedule charging before them
            for _, row in windows_df.iterrows():
                start_tz = pd.to_datetime(row["start_ts"])  # tz-aware
                start = start_tz.tz_localize(None)
                for i in range(int(event_length_h)):
                    t = start + pd.Timedelta(hours=i)
                    sel = summer_plot["ts"] == t
                    if sel.any():
                        summer_plot.loc[sel, "load_MW_after"] = (
                            summer_plot.loc[sel, "load_MW_after"] - aggregate_MW
                        ).clip(lower=0.0)
                        # mark discharge hour
                        match = summer_plot.loc[sel, "load_MW"]
                        val = float(match.iloc[0]) if not match.empty else None
                        discharge_marks.append({"ts": t, "load_MW": val})

                # charging hours equal to event length (symmetrical power), default RTE = 1 for display
                charge_hours = choose_charge_hours(start, int(math.ceil(float(event_length_h))))
                for t in charge_hours:
                    sel = summer_plot["ts"] == t
                    if sel.any():
                        summer_plot.loc[sel, "load_MW_after"] = summer_plot.loc[sel, "load_MW_after"] + aggregate_MW
                        match = summer_plot.loc[sel, "load_MW"]
                        val = float(match.iloc[0]) if not match.empty else None
                        charge_marks.append({"ts": t, "load_MW": val})

            discharge_plot = pd.DataFrame(discharge_marks)
            charge_plot = pd.DataFrame(charge_marks)

            # Persist results
            st.session_state["impact_tech"] = "battery"
            st.session_state["impact_target_kW"] = float(target_kW)
            st.session_state["impact_summer_plot"] = summer_plot
            st.session_state["impact_hours_plot_discharge"] = discharge_plot
            st.session_state["impact_hours_plot_charge"] = charge_plot
            st.session_state["impact_participants"] = participants
            st.session_state["impact_batt_power_kW"] = float(per_device_power_kW)
            st.session_state["impact_batt_energy_kWh"] = float(per_device_energy_kWh)
            st.session_state["impact_aggregate_MW"] = float(aggregate_MW)

            # Summary metrics (Battery)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Participants", f"{participants:,}")
            with m2:
                st.metric("Power/device (kW)", f"{per_device_power_kW:.3f}")
            with m3:
                st.metric("Energy/device (kWh)", f"{per_device_energy_kWh:.2f}")
            with m4:
                st.metric("Aggregate (MW)", f"{aggregate_MW:.3f}")

            st.subheader("Summer Load: charge (+) and discharge (−)")
            base_orig = alt.Chart(summer_plot).mark_line(color="#1f77b4").encode(
                x=alt.X("ts:T", title="Time"),
                y=alt.Y("load_MW:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
            )
            base_after = alt.Chart(summer_plot).mark_line(color="green", strokeDash=[4,4]).encode(
                x="ts:T",
                y=alt.Y("load_MW_after:Q", scale=alt.Scale(zero=False)),
            )
            pts_dis = alt.Chart(discharge_plot).mark_point(color="red", size=60).encode(
                x="ts:T",
                y=alt.Y("load_MW:Q", scale=alt.Scale(zero=False)),
                tooltip=["ts:T", "load_MW:Q"],
            ) if not discharge_plot.empty else None
            pts_chg = alt.Chart(charge_plot).mark_point(color="orange", size=60).encode(
                x="ts:T",
                y=alt.Y("load_MW:Q", scale=alt.Scale(zero=False)),
                tooltip=["ts:T", "load_MW:Q"],
            ) if not charge_plot.empty else None

            brush = alt.selection_interval(encodings=["x"])
            overview = alt.layer(base_orig, base_after).add_selection(brush).properties(height=120)
            layers = [base_orig.transform_filter(brush), base_after.transform_filter(brush)]
            if pts_dis is not None:
                layers.append(pts_dis.transform_filter(brush))
            if pts_chg is not None:
                layers.append(pts_chg.transform_filter(brush))
            detail = alt.layer(*layers).properties(height=300)
            st.altair_chart(alt.vconcat(detail, overview).resolve_scale(y="shared"), use_container_width=True)

            # Focus day view
            st.subheader("Focused Day: before vs after")
            day_options = sorted(pd.to_datetime(windows_df["date"]).dt.strftime("%Y-%m-%d").unique().tolist())
            default_focus = st.session_state.get("impact_focus_day", day_options[0] if day_options else None)
            with st.form("focus_day_form_batt"):
                focus_day_sel = st.selectbox("Select a day", options=day_options, index=(day_options.index(default_focus) if default_focus in day_options else 0), key="impact_focus_day_select_batt")
                submit_day = st.form_submit_button("Update")
            if submit_day:
                st.session_state["impact_focus_day"] = focus_day_sel
            focus_day = st.session_state.get("impact_focus_day", day_options[0] if day_options else None)
            day_dt = pd.to_datetime(focus_day) if focus_day else None
            day_mask = summer_plot["ts"].dt.normalize() == day_dt
            day_df = summer_plot.loc[day_mask, ["ts", "load_MW", "load_MW_after"]].copy()
            dis_day = discharge_plot.loc[discharge_plot["ts"].dt.normalize() == day_dt] if not discharge_plot.empty else discharge_plot
            chg_day = charge_plot.loc[charge_plot["ts"].dt.normalize() == day_dt] if not charge_plot.empty else charge_plot

            day_orig = alt.Chart(day_df).mark_line(color="#1f77b4").encode(
                x=alt.X("ts:T", title="Time"),
                y=alt.Y("load_MW:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
            )
            day_after = alt.Chart(day_df).mark_line(color="green", strokeDash=[4,4]).encode(
                x="ts:T",
                y=alt.Y("load_MW_after:Q", scale=alt.Scale(zero=False)),
            )
            layers_day = [day_orig, day_after]
            if not dis_day.empty:
                layers_day.append(
                    alt.Chart(dis_day).mark_point(color="red", size=80).encode(
                        x="ts:T", y=alt.Y("load_MW:Q", scale=alt.Scale(zero=False)), tooltip=["ts:T", "load_MW:Q"]
                    )
                )
            if not chg_day.empty:
                layers_day.append(
                    alt.Chart(chg_day).mark_point(color="orange", size=80).encode(
                        x="ts:T", y=alt.Y("load_MW:Q", scale=alt.Scale(zero=False)), tooltip=["ts:T", "load_MW:Q"]
                    )
                )
            st.altair_chart(alt.layer(*layers_day).resolve_scale(y="shared"), use_container_width=True)

        else:  # Solar PV
            PER_M2_KW = 0.21
            if target_kW <= 0:
                st.warning("Target capacity must be greater than 0 for Solar PV.")
                st.stop()

            # Area per household required to meet aggregate target
            area_m2_per_household = target_kW / (participants * PER_M2_KW)
            per_household_kw = PER_M2_KW * area_m2_per_household  # equals target_kW / participants
            aggregate_kW = participants * per_household_kw
            aggregate_MW = aggregate_kW / 1000.0

            # Build original vs after adjustment for solar active hours (10:00–16:00)
            summer_mask = load_df["ts"].dt.month.isin(cfg.program.season_months)
            summer_plot = load_df.loc[summer_mask, ["ts", "load_MW"]].copy()
            summer_plot["ts"] = pd.to_datetime(summer_plot["ts"]).dt.tz_localize(None)
            summer_plot["load_MW_after"] = summer_plot["load_MW"].astype(float)
            hours_series = summer_plot["ts"].dt.hour
            solar_mask = (hours_series >= 10) & (hours_series <= 16)
            summer_plot.loc[solar_mask, "load_MW_after"] = (
                summer_plot.loc[solar_mask, "load_MW_after"] - aggregate_MW
            ).clip(lower=0.0)

            # Markers for solar hours
            hours_plot = summer_plot.loc[solar_mask, ["ts", "load_MW"]].copy()

            # Persist results for potential downstream use
            st.session_state["impact_tech"] = "solar_pv"
            st.session_state["impact_target_kW"] = float(target_kW)
            st.session_state["impact_summer_plot"] = summer_plot
            st.session_state["impact_hours_plot"] = hours_plot
            st.session_state["impact_participants"] = participants
            st.session_state["impact_per_device_kw"] = float(per_household_kw)
            st.session_state["impact_aggregate_MW"] = float(aggregate_MW)
            st.session_state["impact_solar_area_m2_per_household"] = float(area_m2_per_household)

            # Summary metrics (Solar PV)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Participants", f"{participants:,}")
            with m2:
                st.metric("Avg PV area/household (m²)", f"{area_m2_per_household:.2f}")
            with m3:
                st.metric("Per-household capacity (kW)", f"{per_household_kw:.3f}")
            with m4:
                st.metric("Aggregate per hour (MW)", f"{aggregate_MW:.3f}")

            st.subheader("Summer Load: original vs after (solar hours)")
            base_orig = alt.Chart(summer_plot).mark_line(color="#1f77b4").encode(
                x=alt.X("ts:T", title="Time"),
                y=alt.Y("load_MW:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
            )
            base_after = alt.Chart(summer_plot).mark_line(color="green", strokeDash=[4,4]).encode(
                x="ts:T",
                y=alt.Y("load_MW_after:Q", scale=alt.Scale(zero=False)),
            )
            base_points = alt.Chart(hours_plot).mark_point(color="red", size=60).encode(
                x="ts:T",
                y=alt.Y("load_MW:Q", scale=alt.Scale(zero=False)),
                tooltip=["ts:T", "load_MW:Q"],
            )

            brush = alt.selection_interval(encodings=["x"])
            overview = alt.layer(base_orig, base_after).add_selection(brush).properties(height=120)
            detail = alt.layer(
                base_orig.transform_filter(brush),
                base_after.transform_filter(brush),
                base_points.transform_filter(brush),
            ).properties(height=300)
            st.altair_chart(alt.vconcat(detail, overview).resolve_scale(y="shared"), use_container_width=True)

            # Focused day view based on solar hours (not event windows)
            st.subheader("Focused Day: before vs after")
            day_options = sorted(pd.to_datetime(summer_plot["ts"]).dt.strftime("%Y-%m-%d").unique().tolist())
            default_focus = st.session_state.get("impact_focus_day", day_options[0] if day_options else None)
            with st.form("focus_day_form_solar"):
                focus_day_sel = st.selectbox("Select a day", options=day_options, index=(day_options.index(default_focus) if default_focus in day_options else 0), key="impact_focus_day_select_solar")
                submit_day = st.form_submit_button("Update")
            if submit_day:
                st.session_state["impact_focus_day"] = focus_day_sel
            focus_day = st.session_state.get("impact_focus_day", day_options[0] if day_options else None)
            day_dt = pd.to_datetime(focus_day) if focus_day else None
            day_mask = summer_plot["ts"].dt.normalize() == day_dt
            day_df = summer_plot.loc[day_mask, ["ts", "load_MW", "load_MW_after"]].copy()
            hours_day = hours_plot.loc[hours_plot["ts"].dt.normalize() == day_dt] if not hours_plot.empty else hours_plot

            day_orig = alt.Chart(day_df).mark_line(color="#1f77b4").encode(
                x=alt.X("ts:T", title="Time"),
                y=alt.Y("load_MW:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
            )
            day_after = alt.Chart(day_df).mark_line(color="green", strokeDash=[4,4]).encode(
                x="ts:T",
                y=alt.Y("load_MW_after:Q", scale=alt.Scale(zero=False)),
            )
            if not hours_day.empty:
                day_points = alt.Chart(hours_day).mark_point(color="red", size=80).encode(
                    x="ts:T",
                    y=alt.Y("load_MW:Q", scale=alt.Scale(zero=False)),
                    tooltip=["ts:T", "load_MW:Q"],
                )
                st.altair_chart(alt.layer(day_orig, day_after, day_points).resolve_scale(y="shared"), use_container_width=True)
            else:
                st.altair_chart(alt.layer(day_orig, day_after).resolve_scale(y="shared"), use_container_width=True)


elif page == "Economics":
    # Inputs for economics values (capacity only; energy uses hourly prices)
    # Cost inputs per new cost model
    default_cap_value_per_MWyr = None
    try:
        default_cap_value_per_MWyr = float(cfg.economics.capacity_value_per_kWyr) * 1000.0
    except Exception:
        default_cap_value_per_MWyr = 0.0

    per_capacity_value_per_MWyr = st.number_input(
        "Capacity value ($/MW-yr)", min_value=0.0, value=float(default_cap_value_per_MWyr), step=1000.0
    )

    # Cost inputs
    # Derive sensible defaults from config if present
    econ_cfg = getattr(cfg, "economics", None)
    default_cost_per_device = 200.0
    default_enroll_credit = 0.0
    default_retention_credit = 0.0
    default_operational_per_year = 5.0
    # Override requested defaults regardless of config where specified
    try:
        if econ_cfg is not None:
            # Keep ability to derive device cost if not explicitly overridden
            # but per request, default to 200 $/device
            _derived_device = float(getattr(econ_cfg, "device_cost", 0.0)) + float(getattr(econ_cfg, "install_cost", 0.0))
            if _derived_device > 0:
                default_cost_per_device = 200.0
            # Operational cost per household per year defaults to $5/household-yr
            default_operational_per_year = 5.0
    except Exception:
        pass
    # Override requested defaults for credits regardless of config
    default_enroll_credit = 85.0
    default_retention_credit = 30.0

    c1, c2 = st.columns(2)
    with c1:
        cost_per_device = st.number_input("Cost per device ($/device)", min_value=0.0, value=float(default_cost_per_device), step=10.0)
        enroll_credit_per_household = st.number_input("Enroll credit per household ($)", min_value=0.0, value=float(default_enroll_credit), step=10.0)
    with c2:
        retention_credit_per_household = st.number_input("Retention credit per household ($/yr)", min_value=0.0, value=float(default_retention_credit), step=10.0)
        operational_cost_per_year = st.number_input("Operational cost per year ($/household-yr)", min_value=0.0, value=float(default_operational_per_year), step=1.0)

    run_btn_econ = st.button("Compute Economics")
    if run_btn_econ:
        if econ_cfg is None:
            st.warning("Missing economics config; cannot compute economics.")
        else:
            # Determine event hours/windows based on selected technology from Impact
            impact_tech = st.session_state.get("impact_tech", "thermostats")

            if impact_tech == "solar_pv":
                # Construct daily 10:00–16:00 windows for all summer days in the load data
                summer_mask = load_df["ts"].dt.month.isin(cfg.program.season_months)
                summer_ts = load_df.loc[summer_mask, ["ts"]].copy()
                if summer_ts.empty:
                    st.warning("No summer data to build Solar PV hours.")
                    st.stop()
                summer_ts["date"] = summer_ts["ts"].dt.tz_localize(None).dt.normalize()
                summer_ts["hour"] = summer_ts["ts"].dt.hour
                starts = summer_ts.loc[summer_ts["hour"] == 10].copy()
                if starts.empty:
                    st.warning("No 10:00 timestamps found in summer data to build Solar PV hours.")
                    st.stop()
                # Preserve timezone-aware timestamps by avoiding .values
                windows_df_use = pd.DataFrame({
                    "date": starts["date"],
                    "start_ts": starts["ts"],  # tz-aware
                    "end_ts": starts["ts"] + pd.Timedelta(hours=7),
                    "duration_h": 7,
                })
                events_per_year = len(windows_df_use)
                event_length_h = 7
            else:
                # Thermostat/Battery path continues to use Peaks-selected event windows
                windows_df_use = st.session_state.get("windows_df")
                event_length_h = st.session_state.get("peaks_event_length_h")
                if windows_df_use is None or event_length_h is None:
                    st.warning("Please complete Peaks first (Find Summer Peaks) to select event windows.")
                    st.stop()
                events_per_year = len(windows_df_use)

            # Build optional battery charging hours for cost deduction
            charge_hours_df = None
            charge_power_MW = None
            if impact_tech == "battery":
                aggregate_MW_val = st.session_state.get("impact_aggregate_MW")
                if aggregate_MW_val is None or float(aggregate_MW_val) <= 0:
                    st.warning("Battery aggregate MW missing from Impact; please re-run Impact for Battery.")
                    st.stop()
                charge_power_MW = float(aggregate_MW_val)

                def _choose_charge_hours_tzaware(start_tz: pd.Timestamp, hours_needed: int) -> list[pd.Timestamp]:
                    chosen: list[pd.Timestamp] = []
                    midnight = start_tz.normalize()  # tz-aware midnight
                    # Same-day early morning before start
                    for h in range(6, -1, -1):
                        ts = midnight + pd.Timedelta(hours=h)
                        if ts < start_tz:
                            chosen.append(ts)
                            if len(chosen) >= hours_needed:
                                return chosen
                    # Previous days early morning until filled (up to 7 days back)
                    d = 1
                    while len(chosen) < hours_needed and d <= 7:
                        day = midnight - pd.Timedelta(days=d)
                        for h in range(6, -1, -1):
                            chosen.append(day + pd.Timedelta(hours=h))
                            if len(chosen) >= hours_needed:
                                break
                        d += 1
                    return chosen[:hours_needed]

                charge_ts: list[pd.Timestamp] = []
                hours_needed_each = int(math.ceil(float(event_length_h)))
                for _, row in windows_df_use.iterrows():
                    start_tz = pd.to_datetime(row["start_ts"])  # tz-aware
                    charge_ts.extend(_choose_charge_hours_tzaware(start_tz, hours_needed_each))
                if charge_ts:
                    charge_hours_df = pd.DataFrame({"ts": pd.to_datetime(charge_ts)})

            # Require Impact completed to get reduced peak and participants
            reduced_peak_MW = st.session_state.get("impact_aggregate_MW")
            participants = st.session_state.get("impact_participants")
            if reduced_peak_MW is None:
                st.warning("Please compute Impact first to determine reduced peak (MW), then return to Economics.")
                st.stop()
            if participants is None or int(participants) <= 0:
                st.warning("Please compute Impact first to determine participants, then return to Economics.")
                st.stop()

            # Ensure latest economics module (handles Streamlit reloads)
            economics_mod = importlib.reload(economics_mod)
            econ_res = economics_mod.economics_from_peak(
                reduced_peak_MW=float(reduced_peak_MW),
                events_per_year=int(events_per_year),
                event_length_h=float(event_length_h),
                per_capacity_value_per_MWyr=float(per_capacity_value_per_MWyr),
                per_energy_value_per_MWh=None,  # use hourly prices instead
                program_life_years=int(econ_cfg.program_life_years),
                participants=int(participants),
                cost_per_device=float(cost_per_device),
                enroll_credit_per_household=float(enroll_credit_per_household),
                retention_credit_per_household=float(retention_credit_per_household),
                operational_cost_per_year=float(operational_cost_per_year),
                price_df=price_df,
                windows_df=windows_df_use,
                charge_hours_df=charge_hours_df,
                charge_power_MW=charge_power_MW,
                round_trip_efficiency=1.0,
            )

            st.subheader("Benefits (annualized)")
            st.dataframe(econ_res["benefits_table"])

            st.subheader("Costs")
            st.dataframe(econ_res["costs_table"])

            st.subheader("Net Profit Over Time (undiscounted)")
            cf = econ_res["cashflow"]
            import altair as alt
            chart = alt.Chart(cf).mark_line().encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("cumulative_net:Q", title="Cumulative net ($)"),
            )
            st.altair_chart(chart, use_container_width=True)

            with st.expander("Intermediates"):
                st.json(econ_res["intermediates"], expanded=False)

# ... (Existing code above) ...
# ... (Existing code above) ...

elif page == "Optimization":
    st.header("AI Optimization")
    st.markdown("Find the best **combination of technologies** to maximize Net Benefit.")

    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        min_target_mw = st.number_input("Min Target Peak Reduction (MW)", value=50.0, step=5.0)
        total_households = st.number_input("Total households", min_value=0, value=3_000_000, step=1000,
                                           key="opt_households")
    with col2:
        # Add Budget Limit Input
        max_budget = st.number_input("Budget Limit ($)", min_value=0.0, value=5_000_000.0, step=100_000.0, format="%f")
        n_trials = st.number_input("Optimization Trials", value=50, step=10, min_value=10, max_value=500)
    with col3:
        st.write("")  # Spacer

    # Tech specific penetration rates
    st.markdown("##### Penetration Rates per Technology")
    st.caption("If a technology is selected in a combination, it will use these participation assumptions.")

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        pen_batt = st.number_input("Battery Penetration", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
    with p_col2:
        pen_therm = st.number_input("Thermostat Penetration", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    with p_col3:
        pen_solar = st.number_input("Solar Penetration", min_value=0.0, max_value=1.0, value=0.10, step=0.01)

    participants_map = {
        "Battery Storage": int(total_households * pen_batt),
        "Thermostats": int(total_households * pen_therm),
        "Solar PV": int(total_households * pen_solar)
    }

    # Display participant counts
    st.info(
        f"**Potential Participants:** "
        f"{participants_map['Battery Storage']:,} | "
        f"{participants_map['Thermostats']:,} | "
        f"{participants_map['Solar PV']:,}"
    )

    # Unit Costs Configuration (User Input)
    with st.expander("Unit Costs Configuration", expanded=False):
        st.caption("Set unit costs for calculation. These override defaults.")
        uc1, uc2, uc3 = st.columns(3)
        with uc1:
            cost_solar = st.number_input("Solar Cost ($/kW)", value=3000.0, step=100.0)
        with uc2:
            cost_batt = st.number_input("Battery Cost ($/kWh)", value=500.0, step=50.0)
        with uc3:
            cost_therm = st.number_input("Thermostat Cost ($/device)", value=200.0, step=10.0)

    # Pack into dictionary for optimization
    unit_costs = {
        "solar_per_kw": cost_solar,
        "battery_per_kwh": cost_batt,
        "thermostat_per_device": cost_therm
    }

    # Advanced Configuration for Search Ranges
    with st.expander("Advanced Search Space Configuration", expanded=False):
        st.caption("Define the ranges for the AI to explore.")

        st.markdown("**Event Settings (Shared)**")
        ac1, ac2 = st.columns(2)
        with ac1:
            top_days_min = st.number_input("Min Event Days", value=10, min_value=1, max_value=90)
            event_len_min = st.number_input("Min Duration (h)", value=2, min_value=1, max_value=12)
        with ac2:
            top_days_max = st.number_input("Max Event Days", value=60, min_value=top_days_min, max_value=100)
            event_len_max = st.number_input("Max Duration (h)", value=6, min_value=event_len_min, max_value=24)

        st.markdown("**Battery Storage Ranges**")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            batt_cap_min = st.number_input("Min Capacity (kWh)", value=10.0, step=0.5)
            batt_cap_max = st.number_input("Max Capacity (kWh)", value=20.0, step=0.5)
        with bc2:
            batt_kw_min = st.number_input("Min Power (kW)", value=3.0, step=0.5)
            batt_kw_max = st.number_input("Max Power (kW)", value=7.0, step=0.5)
        with bc3:
            batt_dod_min = st.number_input("Min DoD", value=0.6, min_value=0.1, max_value=1.0, step=0.05)
            batt_dod_max = st.number_input("Max DoD", value=0.95, min_value=batt_dod_min, max_value=1.0, step=0.05)

        st.markdown("**Other Technologies**")
        oc1, oc2 = st.columns(2)
        with oc1:
            t_min = st.number_input("Min Delta T (F)", value=1.0, step=0.5)
            t_max = st.number_input("Max Delta T (F)", value=5.0, step=0.5)
        with oc2:
            s_min = st.number_input("Min Solar kW/home", value=3.0, step=0.5)
            s_max = st.number_input("Max Solar kW/home", value=10.0, step=0.5)

    if st.button("Run Optimization"):
        if load_df.empty or price_df is None:
            st.error("Please load data first.")
        elif all(p == 0 for p in participants_map.values()):
            st.error("All technologies have 0 participants. Please check inputs.")
        else:
            constraints = {
                "top_days_min": top_days_min, "top_days_max": top_days_max,
                "event_length_min": event_len_min, "event_length_max": event_len_max,
                "batt_kwh_min": batt_cap_min, "batt_kwh_max": batt_cap_max,
                "batt_kw_min": batt_kw_min, "batt_kw_max": batt_kw_max,
                "batt_dod_min": batt_dod_min, "batt_dod_max": batt_dod_max,
                "thermostat_delta_t_min": t_min, "thermostat_delta_t_max": t_max,
                "solar_kw_min": s_min, "solar_kw_max": s_max
            }

            with st.spinner(f"Running {n_trials} iterations for combinations..."):
                try:
                    best_trial = optimization.run_optimization(
                        load_df=load_df,
                        price_df=price_df,
                        config=cfg,
                        min_target_mw=min_target_mw,
                        max_budget=max_budget,  # Pass budget constraint
                        participants_map=participants_map,
                        unit_costs=unit_costs,  # Pass unit costs
                        constraints=constraints,
                        n_trials=n_trials
                    )

                    st.success("Optimization Complete!")

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Max Total Net Benefit", f"${best_trial.value:,.2f}")
                    with m2:
                        st.metric("Best Combination", best_trial.params.get("combination", "N/A"))
                    with m3:
                        reduced_mw = best_trial.user_attrs.get('reduced_peak_MW', 0)
                        st.metric("Total Reduced Peak MW", f"{reduced_mw:.2f} MW")
                    with m4:
                        # Show Investment Cost
                        inv_cost = best_trial.user_attrs.get('total_investment_cost', 0)
                        st.metric("Total Investment", f"${inv_cost:,.0f}")

                    st.subheader("Optimal Configuration Details")

                    # --- Formatting Results Table ---
                    display_params = {}
                    for k, v in best_trial.params.items():
                        clean_key = k.replace("_", " ").title()
                        if isinstance(v, float):
                            clean_val = round(v, 2)
                        else:
                            clean_val = v
                        display_params[clean_key] = clean_val

                    st.table(pd.DataFrame.from_dict(display_params, orient='index', columns=['Optimal Value']))
                    # --------------------------------

                    if best_trial.value == optimization.PENALTY_SCORE:
                        st.error("No feasible solution found. Please relax constraints (e.g., Budget or MW Target).")
                    elif "violation" in best_trial.user_attrs:
                        st.warning(f"Note: Violation recorded: {best_trial.user_attrs['violation']}")

                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")