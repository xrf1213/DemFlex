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
from demflex.economics import economics_from_peak


st.set_page_config(page_title="DemFlex", layout="wide")
st.title("DemFlex")

# Top-level view switch (outside the sidebar)
page = st.radio("View", ["Peaks", "Impact", "Economics"], index=0, horizontal=True)

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
            st.metric(label="Load rows", value=f"{len(load_df):,}")
        with c2:
            pr = f"{len(price_df):,}" if price_df is not None else "N/A"
            st.metric(label="Price rows", value=pr)
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
    total_households = st.number_input("Total households (region)", min_value=0, value=100000, step=1000)
    penetration = st.number_input("Penetration rate", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    target_mw = st.number_input("Target capacity (MW)", min_value=0.0, value=float(50), step=1.0)
    run_btn_impact = st.button("Compute Impact")
    if run_btn_impact:
        windows_df = st.session_state.get("windows_df")
        event_length_h = st.session_state.get("peaks_event_length_h")
        if windows_df is None or event_length_h is None:
            st.warning("Please run Peaks first (Find Summer Peaks) to select event windows, then return to Impact.")
            st.stop()

        participants = int(total_households * penetration)
        if participants <= 0:
            st.warning("Participants computed as 0. Increase households or penetration.")
            st.stop()

        base_deltaT = 4.0
        base_per_device_kw = 0.512
        target_kW = target_mw * 1000.0
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

# Render persisted Impact charts when changing selections (avoid recompute)
elif page == "Impact":
    if 'impact_summer_plot' in st.session_state and 'impact_hours_plot' in st.session_state and 'windows_df' in st.session_state:
        summer_plot = st.session_state['impact_summer_plot']
        hours_plot = st.session_state['impact_hours_plot']
        windows_df = st.session_state['windows_df']
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
        brush = alt.selection_interval(encodings=["x"])
        overview = alt.layer(base_orig, base_after).add_selection(brush).properties(height=120)
        detail = alt.layer(
            base_orig.transform_filter(brush),
            base_after.transform_filter(brush),
            base_points.transform_filter(brush),
        ).properties(height=300)
        st.altair_chart(alt.vconcat(detail, overview).resolve_scale(y="shared"), use_container_width=True)

        st.subheader("Focused Day: before vs after")
        day_options = sorted(pd.to_datetime(windows_df["date"]).dt.strftime("%Y-%m-%d").unique().tolist())
        default_focus = st.session_state.get("impact_focus_day", day_options[0] if day_options else None)
        with st.form("focus_day_form_persist"):
            focus_day_sel = st.selectbox("Select a day", options=day_options, index=(day_options.index(default_focus) if default_focus in day_options else 0), key="impact_focus_day_select_persist")
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
    # Inputs for economics values (keep layout simple; two user inputs as requested)
    # Defaults come from config where possible (converted to MW and MWh units)
    default_cap_value_per_MWyr = None
    default_energy_value_per_MWh = None
    try:
        default_cap_value_per_MWyr = float(cfg.economics.capacity_value_per_kWyr) * 1000.0
    except Exception:
        default_cap_value_per_MWyr = 0.0
    try:
        default_energy_value_per_MWh = float(cfg.economics.flat_energy_price_per_kWh) * 1000.0
    except Exception:
        default_energy_value_per_MWh = 0.0

    c1, c2 = st.columns(2)
    with c1:
        per_capacity_value_per_MWyr = st.number_input(
            "Capacity value ($/MW-yr)", min_value=0.0, value=float(default_cap_value_per_MWyr), step=1000.0
        )
    with c2:
        per_energy_value_per_MWh = st.number_input(
            "Energy value ($/MWh)", min_value=0.0, value=float(default_energy_value_per_MWh), step=1.0
        )

    run_btn_econ = st.button("Compute Economics")
    if run_btn_econ:
        econ_cfg = getattr(cfg, "economics", None)
        if econ_cfg is None:
            st.warning("Missing economics config; cannot compute economics.")
        else:
            # Require Peaks completed to get events and length
            windows_df = st.session_state.get("windows_df")
            event_length_h = st.session_state.get("peaks_event_length_h")
            if windows_df is None or event_length_h is None:
                st.warning("Please complete Peaks first (Find Summer Peaks) to select event windows.")
                st.stop()
            events_per_year = len(windows_df)

            # Require Impact completed to get reduced peak
            reduced_peak_MW = st.session_state.get("impact_aggregate_MW")
            if reduced_peak_MW is None:
                st.warning("Please compute Impact first to determine reduced peak (MW), then return to Economics.")
                st.stop()

            econ_res = economics_from_peak(
                reduced_peak_MW=float(reduced_peak_MW),
                events_per_year=int(events_per_year),
                event_length_h=float(event_length_h),
                per_capacity_value_per_MWyr=float(per_capacity_value_per_MWyr),
                per_energy_value_per_MWh=float(per_energy_value_per_MWh),
                program_life_years=int(econ_cfg.program_life_years),
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

# (removed duplicate persisted Impact renderer)
