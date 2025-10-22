from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class STParams:
    """Smart Thermostat response model parameters."""
    alpha_kW_per_deg: float = 0.2
    duration_breakpoints_h: List[float] = None
    duration_multipliers: List[float] = None
    cap_kW_per_home: float = 1.2
    rebound_fraction: float = 0.4

    def __post_init__(self):
        if self.duration_breakpoints_h is None:
            self.duration_breakpoints_h = [1, 2, 3, 4]
        if self.duration_multipliers is None:
            self.duration_multipliers = [1.0, 0.9, 0.8, 0.7]
        if len(self.duration_breakpoints_h) != len(self.duration_multipliers):
            raise ValueError("duration_breakpoints_h and duration_multipliers must have same length")


@dataclass
class CohortParams:
    N_eligible: int
    penetration: float
    enrollment_rate: float
    enablement_rate: float
    diversity_factor: float = 1.0

    @property
    def participants_max(self) -> int:
        return int(math.floor(self.N_eligible * self.penetration * self.enrollment_rate))


def duration_multiplier(L_h: float, breakpoints: List[float], multipliers: List[float]) -> float:
    """Piecewise-linear multiplier g(L) for duration decay.

    For L below the first breakpoint, use the first multiplier. For L above the last, use the last.
    Between breakpoints, linearly interpolate.
    """
    if L_h is None or L_h <= 0:
        return 0.0
    if L_h <= breakpoints[0]:
        return multipliers[0]
    for i in range(1, len(breakpoints)):
        if L_h <= breakpoints[i]:
            # linear interpolate between (x0,y0) and (x1,y1)
            x0, y0 = breakpoints[i-1], multipliers[i-1]
            x1, y1 = breakpoints[i], multipliers[i]
            t = (L_h - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + t * (y1 - y0)
    return multipliers[-1]


def r_home_kW(deltaT_F: float, L_h: float, params: STParams) -> float:
    """Per-home average kW reduction during an event with setpoint change deltaT_F and duration L_h."""
    mult = duration_multiplier(L_h, params.duration_breakpoints_h, params.duration_multipliers)
    r = params.alpha_kW_per_deg * max(deltaT_F, 0.0) * mult
    return min(r, params.cap_kW_per_home)


def aggregate_event_kW(r_home: float, participants: int, enablement_rate: float, diversity_factor: float) -> float:
    """Aggregate event-level kW reduction."""
    per_participant_effective = r_home * max(enablement_rate, 0.0) * max(diversity_factor, 0.0)
    return participants * per_participant_effective


def participants_for_target(target_kW: float, r_home: float, cohort: CohortParams) -> Dict[str, Any]:
    """Compute participants required to meet a target kW, capped by available participants.

    Returns dict with: participants_needed, participants_used, feasible, max_achievable_kW
    """
    per_participant_effective = r_home * cohort.enablement_rate * cohort.diversity_factor
    if per_participant_effective <= 0:
        return {
            "participants_needed": math.inf,
            "participants_used": 0,
            "feasible": False,
            "max_achievable_kW": 0.0,
        }

    needed = int(math.ceil(target_kW / per_participant_effective))
    used = min(needed, cohort.participants_max)
    feasible = needed <= cohort.participants_max
    max_achievable = cohort.participants_max * per_participant_effective
    return {
        "participants_needed": needed,
        "participants_used": used,
        "feasible": feasible,
        "max_achievable_kW": max_achievable,
    }


def event_energy_kwh(event_kW: float, L_h: float, rebound_fraction: float) -> Dict[str, float]:
    """Compute curtailed kWh and rebound kWh for an event of duration L_h."""
    curtailed = max(event_kW, 0.0) * max(L_h, 0.0)
    rebound = curtailed * max(min(rebound_fraction, 1.0), 0.0)
    net = curtailed - rebound
    return {"kWh_curtailed": curtailed, "kWh_rebound": rebound, "kWh_net": net}


def summarize_event(deltaT_F: float, L_h: float, target_kW: float | None, st: STParams, cohort: CohortParams) -> Dict[str, Any]:
    """High-level wrapper to compute event metrics and feasibility.

    If target_kW is provided, compute required participants and feasibility; otherwise compute event kW with max participants.
    Returns a dict of key metrics.
    """
    r = r_home_kW(deltaT_F, L_h, st)

    # Using max available participants when no target is specified
    if target_kW is None or target_kW <= 0:
        used = cohort.participants_max
        event_kw = aggregate_event_kW(r, used, cohort.enablement_rate, cohort.diversity_factor)
        e = event_energy_kwh(event_kw, L_h, st.rebound_fraction)
        return {
            "r_home_kW": r,
            "participants_used": used,
            "event_kW": event_kw,
            **e,
        }

    # Solve for participants to meet target
    need_info = participants_for_target(target_kW, r, cohort)
    used = need_info["participants_used"]
    feasible = need_info["feasible"]
    event_kw = aggregate_event_kW(r, used, cohort.enablement_rate, cohort.diversity_factor)
    e = event_energy_kwh(event_kw, L_h, st.rebound_fraction)

    return {
        "r_home_kW": r,
        "participants_needed": need_info["participants_needed"],
        "participants_used": used,
        "feasible": feasible,
        "max_achievable_kW": need_info["max_achievable_kW"],
        "event_kW": event_kw,
        **e,
    }

