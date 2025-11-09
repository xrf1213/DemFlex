from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd

"""Economics helpers for the Streamlit app."""


def economics_from_peak(
    reduced_peak_MW: float,
    events_per_year: int,
    event_length_h: float,
    per_capacity_value_per_MWyr: float,
    per_energy_value_per_MWh: Optional[float] = None,
    program_life_years: int = 10,
    # Participant-based cost model inputs
    participants: Optional[int] = None,
    cost_per_device: float = 0.0,
    enroll_credit_per_household: float = 0.0,
    retention_credit_per_household: float = 0.0,
    operational_cost_per_year: float = 0.0,
    price_df: Optional[pd.DataFrame] = None,
    windows_df: Optional[pd.DataFrame] = None,
    # Optional battery charging inputs
    charge_hours_df: Optional[pd.DataFrame] = None,
    charge_power_MW: Optional[float] = None,
    round_trip_efficiency: float = 1.0,
) -> Dict[str, Any]:
    """Custom economics based on Impact results (peak reduction in MW).

    - Benefits (annualized):
        1) Reduced peak (MW)
        2) Per capacity value ($/MW-yr) [input]
        3) Capacity benefit = 1) * 2)
        4) Energy saved (MWh) = reduced_peak_MW * total_event_hours
        5) Per energy value ($/MWh) [input]
        6) Energy benefit = 4) * 5)
        7) Total benefit = capacity + energy
    - Costs:
        - First-year cost = fixed_cost_per_MW * reduced_peak_MW
        - Operational cost (annual) = operational_cost_per_MWyr * reduced_peak_MW
    - Cashflow (undiscounted): Year 0 = -First-year cost; Years 1..N = Total benefit - Operational cost
    """

    reduced_peak_MW = max(0.0, float(reduced_peak_MW))
    events_per_year = int(max(0, events_per_year))
    event_length_h = max(0.0, float(event_length_h))
    total_event_hours = events_per_year * event_length_h

    # Benefits
    capacity_benefit = reduced_peak_MW * float(per_capacity_value_per_MWyr)
    energy_saved_MWh = reduced_peak_MW * total_event_hours

    # Energy benefit: prefer hourly price series during event hours; fallback to flat rate if provided
    energy_benefit_gross = 0.0
    charge_cost = 0.0
    matched_hours = 0
    avg_energy_price_per_MWh = None
    if price_df is not None and windows_df is not None and reduced_peak_MW > 0 and total_event_hours > 0:
        # Build list of event hours
        hours = []
        for _, row in windows_df.iterrows():
            start_ts = pd.to_datetime(row["start_ts"])  # tz-aware
            dur = int(row.get("duration_h", event_length_h))
            for i in range(dur):
                hours.append(start_ts + pd.Timedelta(hours=i))
        hours_df = pd.DataFrame({"ts": pd.to_datetime(hours)})
        # Join on ts and compute $/MWh from $/kWh
        prices = price_df.copy()[["ts", "price_per_MWh"]]
        merged = hours_df.merge(prices, on="ts", how="left")
        merged = merged.dropna(subset=["price_per_MWh"])  # skip missing price hours
        if not merged.empty:
            avg_energy_price_per_MWh = float(merged["price_per_MWh"].astype(float).mean())
            matched_hours = len(merged)
            # Sum benefit across event hours: MW * $/MWh
            energy_benefit_gross = float((reduced_peak_MW * merged["price_per_MWh"]).sum())
        else:
            energy_benefit_gross = 0.0
    elif per_energy_value_per_MWh is not None:
        # Flat rate fallback
        energy_benefit_gross = energy_saved_MWh * float(per_energy_value_per_MWh)
        avg_energy_price_per_MWh = float(per_energy_value_per_MWh)
        matched_hours = int(total_event_hours)

    # Charging cost (optional): sum over charge hours of (charge_power_MW × price_per_MWh)
    matched_charge_hours = 0
    if price_df is not None and charge_hours_df is not None and (charge_power_MW or 0.0) > 0:
        prices = price_df.copy()[["ts", "price_per_MWh"]]
        ch = charge_hours_df.copy()[["ts"]]
        ch["ts"] = pd.to_datetime(ch["ts"])  # keep tz as-is
        merged_ch = ch.merge(prices, on="ts", how="left").dropna(subset=["price_per_MWh"])  # skip missing
        if not merged_ch.empty:
            matched_charge_hours = len(merged_ch)
            # If RTE != 1, the grid energy required increases; model simply by dividing by RTE
            rte = float(round_trip_efficiency) if round_trip_efficiency and round_trip_efficiency > 0 else 1.0
            effective_charge_MW = float(charge_power_MW) / rte
            charge_cost = float((effective_charge_MW * merged_ch["price_per_MWh"]).sum())
        else:
            charge_cost = 0.0

    energy_benefit_net = energy_benefit_gross - charge_cost
    total_benefit = capacity_benefit + energy_benefit_net

    benefits_table = pd.DataFrame([
        {"component": "Reduced peak (MW)", "value": round(reduced_peak_MW, 6)},
        {"component": "Capacity value ($/MW-yr)", "value": round(float(per_capacity_value_per_MWyr), 2)},
        {"component": "Capacity benefit ($/yr)", "value": round(capacity_benefit, 2)},
        {"component": "Energy saved (MWh/yr)", "value": round(energy_saved_MWh, 6)},
        {"component": "Avg energy price ($/MWh)", "value": round(avg_energy_price_per_MWh or 0.0, 4)},
        {"component": "Energy benefit (gross $/yr)", "value": round(energy_benefit_gross, 2)},
        {"component": "Charging cost ($/yr)", "value": round(charge_cost, 2)},
        {"component": "Energy benefit (net $/yr)", "value": round(energy_benefit_net, 2)},
        {"component": "Total benefit ($/yr)", "value": round(total_benefit, 2)},
    ])

    # Costs (participant-based per requirements)
    P = int(participants or 0)
    cost_device = P * float(cost_per_device)
    cost_enroll = P * float(enroll_credit_per_household)
    cost_retention = P * float(retention_credit_per_household)
    # Operational cost is per-household per year per request
    cost_op = P * float(operational_cost_per_year)
    # Totals by period
    total_year0 = cost_device + cost_enroll
    total_annual = cost_retention + cost_op
    costs_table = pd.DataFrame([
        {"component": "Cost_device", "value": round(cost_device, 2), "period": "Year 0"},
        {"component": "Cost_enroll", "value": round(cost_enroll, 2), "period": "Year 0"},
        {"component": "Cost_retention", "value": round(cost_retention, 2), "period": "Annual"},
        {"component": "Cost_op", "value": round(cost_op, 2), "period": "Annual"},
        {"component": "Cost_total", "value": round(total_annual, 2), "period": "Annual"},
    ])

    # Cashflow
    years = list(range(0, int(program_life_years) + 1))
    rows = []
    for y in years:
        if y == 0:
            benefit = 0.0
            cost = total_year0
        else:
            benefit = total_benefit
            cost = total_annual
        net = benefit - cost
        rows.append({"year": y, "benefit": benefit, "cost": cost, "net": net})
    cf = pd.DataFrame(rows)
    cf["cumulative_net"] = cf["net"].cumsum()

    intermediates = {
        "events_per_year": events_per_year,
        "event_length_h": event_length_h,
        "total_event_hours": total_event_hours,
        "reduced_peak_MW": reduced_peak_MW,
        "per_capacity_value_per_MWyr": float(per_capacity_value_per_MWyr),
        "per_energy_value_per_MWh": (None if per_energy_value_per_MWh is None else float(per_energy_value_per_MWh)),
        "participants": P,
        "charge_power_MW": (None if charge_power_MW is None else float(charge_power_MW)),
        "cost_device_year0": float(cost_device),
        "cost_enroll_year0": float(cost_enroll),
        "cost_retention_annual": float(cost_retention),
        "cost_op_annual": float(cost_op),
        "total_cost_year0": float(total_year0),
        "total_cost_annual": float(total_annual),
        "matched_event_hours_with_price": int(matched_hours),
        "matched_charge_hours_with_price": int(matched_charge_hours),
        "avg_energy_price_per_MWh": avg_energy_price_per_MWh,
    }

    return {
        "benefits_table": benefits_table,
        "costs_table": costs_table,
        "cashflow": cf,
        "intermediates": intermediates,
    }
