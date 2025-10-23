from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd


@dataclass
class EconInputs:
    capacity_value_per_kWyr: float
    flat_energy_price_per_kWh: float
    device_cost: float
    install_cost: float
    upfront_incentive: float
    annual_incentive: float
    per_event_incentive: float
    platform_fee_annual: float
    admin_annual: float
    mv_annual: float
    per_event_cost: float
    program_life_years: int
    discount_rate: float


def simple_economics(events_per_year: int,
                     event_kW_kw: float,
                     event_length_h: float,
                     rebound_fraction: float,
                     participants: int,
                     econ: EconInputs) -> Dict[str, Any]:
    """Compute simple annual benefits and costs and build a cashflow over program life.

    MVP assumptions:
      - Capacity benefit = event_kW_kw * capacity_value_per_kWyr.
      - Energy benefit/event = (event_kW_kw * event_length_h) * (1 - rebound_fraction) [kWh] * flat_price.
      - Energy benefit (annual) = per-event energy benefit * events_per_year.
      - CapEx at year 0 = participants * (device_cost + install_cost + upfront_incentive).
      - OpEx (annual) = platform + admin + M&V + participants * annual_incentive + events_per_year * (per_event_incentive + per_event_cost).
      - Net cashflow: Year0 = -CapEx; Years 1..N = Benefits - OpEx (no CapEx).
    """
    # Benefits
    capacity_benefit = event_kW_kw * econ.capacity_value_per_kWyr
    kWh_net_per_event = max(event_kW_kw, 0.0) * max(event_length_h, 0.0) * max(1.0 - rebound_fraction, 0.0)
    energy_benefit_per_event = kWh_net_per_event * econ.flat_energy_price_per_kWh
    energy_benefit_annual = events_per_year * energy_benefit_per_event
    total_benefit_annual = capacity_benefit + energy_benefit_annual

    benefits_table = pd.DataFrame([
        {"component": "Capacity benefit ($/yr)", "value": round(capacity_benefit, 2)},
        {"component": "Energy benefit per event ($)", "value": round(energy_benefit_per_event, 2)},
        {"component": "Energy benefit annual ($/yr)", "value": round(energy_benefit_annual, 2)},
        {"component": "Total benefit annual ($/yr)", "value": round(total_benefit_annual, 2)},
    ])

    # Costs
    capex = participants * (econ.device_cost + econ.install_cost + econ.upfront_incentive)
    opex_annual = (
        econ.platform_fee_annual
        + econ.admin_annual
        + econ.mv_annual
        + participants * econ.annual_incentive
        + events_per_year * (econ.per_event_incentive + econ.per_event_cost)
    )
    costs_table = pd.DataFrame([
        {"component": "CapEx (year 0) ($)", "value": round(capex, 2)},
        {"component": "OpEx annual ($/yr)", "value": round(opex_annual, 2)},
    ])

    # Cashflow over life (undiscounted for MVP; can add NPV later)
    years = list(range(0, int(econ.program_life_years) + 1))
    cash = []
    for y in years:
        if y == 0:
            benefit = 0.0
            cost = capex
        else:
            benefit = total_benefit_annual
            cost = opex_annual
        net = benefit - cost
        cash.append({"year": y, "benefit": benefit, "cost": cost, "net": net})
    cf = pd.DataFrame(cash)
    cf["cumulative_net"] = cf["net"].cumsum()

    intermediates = {
        "events_per_year": events_per_year,
        "event_kW_kw": event_kW_kw,
        "event_length_h": event_length_h,
        "rebound_fraction": rebound_fraction,
        "participants": participants,
        "kWh_net_per_event": kWh_net_per_event,
    }

    return {
        "benefits_table": benefits_table,
        "costs_table": costs_table,
        "cashflow": cf,
        "intermediates": intermediates,
    }

