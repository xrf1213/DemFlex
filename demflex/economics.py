from __future__ import annotations

from typing import Dict, Any
import pandas as pd

"""Economics helpers for the Streamlit app."""


def economics_from_peak(
    reduced_peak_MW: float,
    events_per_year: int,
    event_length_h: float,
    per_capacity_value_per_MWyr: float,
    per_energy_value_per_MWh: float,
    program_life_years: int,
    fixed_cost_per_MW: float = 163_000.0,
    operational_cost_per_MWyr: float = 10_000.0,
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
    energy_benefit = energy_saved_MWh * float(per_energy_value_per_MWh)
    total_benefit = capacity_benefit + energy_benefit

    benefits_table = pd.DataFrame([
        {"component": "Reduced peak (MW)", "value": round(reduced_peak_MW, 6)},
        {"component": "Capacity value ($/MW-yr)", "value": round(float(per_capacity_value_per_MWyr), 2)},
        {"component": "Capacity benefit ($/yr)", "value": round(capacity_benefit, 2)},
        {"component": "Energy saved (MWh/yr)", "value": round(energy_saved_MWh, 6)},
        {"component": "Energy value ($/MWh)", "value": round(float(per_energy_value_per_MWh), 2)},
        {"component": "Energy benefit ($/yr)", "value": round(energy_benefit, 2)},
        {"component": "Total benefit ($/yr)", "value": round(total_benefit, 2)},
    ])

    # Costs
    first_year_cost = float(fixed_cost_per_MW) * reduced_peak_MW
    operational_cost_annual = float(operational_cost_per_MWyr) * reduced_peak_MW
    costs_table = pd.DataFrame([
        {"component": "First-year cost ($)", "value": round(first_year_cost, 2)},
        {"component": "Operational cost annual ($/yr)", "value": round(operational_cost_annual, 2)},
    ])

    # Cashflow
    years = list(range(0, int(program_life_years) + 1))
    rows = []
    for y in years:
        if y == 0:
            benefit = 0.0
            cost = first_year_cost
        else:
            benefit = total_benefit
            cost = operational_cost_annual
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
        "per_energy_value_per_MWh": float(per_energy_value_per_MWh),
        "fixed_cost_per_MW": float(fixed_cost_per_MW),
        "operational_cost_per_MWyr": float(operational_cost_per_MWyr),
    }

    return {
        "benefits_table": benefits_table,
        "costs_table": costs_table,
        "cashflow": cf,
        "intermediates": intermediates,
    }
