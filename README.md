# DEMFlex — MVP Design (Smart Thermostat Only)

This document outlines the initial design and data flow for DEMFlex, a planning tool for utility planners to meet capacity reduction targets using demand response via Smart Thermostats. The MVP supports a single technology (Smart Thermostat) and produces program/event parameters plus cost–benefit results.

## Goal & Scope
- Objective: Given a capacity reduction target, derive Smart Thermostat demand response parameters (number of events, event length, setpoint change), estimate participants, and compute program cost/benefit.
- MVP Scope: Smart Thermostat only; heuristic event planning; simple baseline; annualized economics; report best feasible plan.

## Key Assumptions (MVP)
- Load data is provided (hourly or 15‑minute) and tagged with calendar info (weekday/season) or can be inferred.
- Program window (e.g., summer weekdays 2–7 pm) and constraints (max events, min spacing, bounds on event length and setpoint change) are configurable.
- Participation rates and thermostat penetration cap the number of participants.
- Per‑home kW reduction depends primarily on setpoint change and event duration; optional weather sensitivity.

## Inputs
- Capacity target: `target_kW` (system or feeder scale).
- Load data: time series `load[t]` (baseline or system load), optional temperatures.
- Program constraints: season window, blackout days/holidays, `max_events_per_season`, `min_gap_days`, bounds for `event_length` and `ΔT_setpoint`.
- Population: `N_eligible`, `penetration`, `enrollment_rate`, `enablement_rate`.
- Economics: energy prices by interval, capacity value ($/kW‑yr), optional T&D deferral value, incentives, device/install/admin/O&M costs, discount rate and program life.

## Core Data Models
- Config: program window, constraints, defaults, price series.
- Cohort: `{N_eligible, penetration, enrollment_rate, enablement_rate}`.
- Event: `{start, duration_h, ΔT_setpoint_F, expected_kW_reduction, kWh_shifted, rebound_kWh}`.
- ST Response Params: `{alpha_kW_per_deg, duration_decay, saturation_caps}`.
- Results: `{num_events, event_length_h, ΔT_F, participants, total_cost, total_benefit, NPV, payback_years, BCR, capacity_met_pct}`.

## Processing Pipeline
1. Data ingest → cleaning → baseline build.
2. Peak identification within the program window.
3. Heuristic event scheduling and parameter derivation to meet `target_kW`.
4. Smart Thermostat impact modeling (kW, kWh, rebound).
5. Cost/benefit calculation and selection of best plan.
6. Reporting (console + CSV/JSON).

## Baseline & Peaks (MVP)
- Baseline: simple high‑percentile by hour (e.g., 90–95th percentile of non‑event days) or recent‑window average.
- Peak candidates: top intervals in the program window by baseline load (or by price if energy value is prioritized).
- Constraints: respect blackout days, holidays, and minimum spacing between events.

## Smart Thermostat Impact Model
- Per‑home kW during event: `r_home(ΔT, L) = alpha * ΔT * g(L)` with caps; `g(L)` applies duration decay (e.g., diminishing returns after 2–3 h).
- Aggregation: `r_event_kW = participants * enablement * r_home(...) * diversity_factor`.
- Rebound: portion `β_rebound` of curtailed kWh returns post‑event; optional pre‑cool credit.

## From Target to Parameters (Heuristic)
- Feasibility bound: `max_kW_feasible ≈ participants_max * r_home(ΔT_max, L_max)`; flag infeasible if `target_kW` exceeds this.
- Derivation steps:
  - Choose top‑N peak days/hours within season.
  - Start with defaults: `ΔT_default`, `L_default`.
  - Compute `r_event_kW`; if short, increase `ΔT` up to bound, then increase `L` within bounds; if still short, add events (respecting constraints) until peak capacity achieved.
  - Participants can be adjusted within penetration/enrollment caps to meet target in a minimal, practical plan.

## Cost Model (Annualized)
- One‑time: device + install per participant (or BYOT upfront incentive).
- Recurring: per‑participant incentives (enrollment + per‑event), platform/admin/M&V/O&M, and per‑event costs.
- Annualized cost: `OpEx + CRF * CapEx` where CRF is the capital recovery factor over program life and discount rate.

## Benefit Model
- Avoided capacity: `target_kW_met * capacity_value_per_kWyr` (adjust for coincidence with system peak when applicable).
- Energy market: sum of `kWh_saved * price[t]` during events minus rebound kWh costs.
- Optional: T&D deferral value, environmental credits.
- Financial metrics: NPV, payback, benefit‑cost ratio (BCR).

## Selection & Outputs
- Selection: choose feasible parameter set with highest NPV, then highest BCR, then lowest participants for practicality.
- Outputs:
  - Event plan: `num_events`, `event_length_h`, `ΔT_setpoint_F`, candidate dates/windows.
  - Program size: participants required/used.
  - Economics: total cost, total benefit, NPV, payback, BCR, capacity met %.

## Data Flow Summary
`target_kW` → load ingest/clean → baseline → peak pick → heuristic derive (`num_events`, `L`, `ΔT`, participants) → ST model (kW/kWh/rebound) → cost/benefit → select best → report.

## Default Parameters (to calibrate later)
- `ΔT_setpoint_F ∈ [1, 4]` (default 2°F).
- `event_length_h ∈ [1, 4]` (default 3 h).
- `max_events_per_season = 20`, `min_gap_days = 1`.
- `alpha_kW_per_deg ≈ 0.1–0.3 kW/home/°F` (climate‑dependent).
- `enablement_rate ≈ 0.7–0.9`; `enrollment_rate` per program design.
- `β_rebound ≈ 0.3–0.6` of event kWh.

## MVP Implementation Plan
- CLI flow:
  1. Read config and load series; accept `target_kW` input.
  2. Build baseline and pick candidate peaks.
  3. Derive `num_events`, `event_length`, `ΔT` heuristically to meet `target_kW`.
  4. Compute participants (respecting penetration/enrollment caps).
  5. Run cost/benefit; write a concise report and JSON/CSV outputs.
- Suggested modules:
  - `ingest.py`, `baseline.py`, `peaks.py`, `st_model.py`, `planner.py`, `economics.py`, `report.py`.
- Validation: check kW vs target; events within window and constraints; quick sensitivity toggles.

## Edge Cases
- Infeasible target under constraints → return max achievable and suggest relaxing constraints or increasing participants.
- Sparse peaks or low variance → broaden window or shorten events.
- Price vs load prioritization → user option to optimize for capacity or energy value.

## Future Extensions
- Add other technologies (heat pumps, EVs, batteries) and portfolio optimization.
- Weather‑adjusted baselines and predictive scheduling.
- Locational targeting and T&D deferral valuation.
- M&V integration using telemetry for ex‑post tuning.

---
If you’d like, I can scaffold a minimal CLI in this repo with editable defaults and a small sample load series to start iterating quickly.

## Roadmap & Work Plan

This MVP will be delivered in clear, incremental steps. We will keep this section up to date as work progresses.

1) Initialize config and sample data
- Define editable defaults (program window, constraints, economics) in a simple config file.
- Add a small sample load series for quick iteration.

Deliverables (Step 1):
- Config defaults at `config/defaults.yaml` describing program window, constraints, cohort, ST model, economics, and I/O.
- Output folder stub at `outputs/`.
- Using your data files in `data/` (no bundled samples).

Quick edits to try first:
- Change default target in `config/defaults.yaml` near `run_defaults.target_kW`.
- `io.load_data_path`: set to `data/Native_Load_2023.xlsx`, `data/Native_Load_2024.xlsx`, or `data/Native_Load_2025.xlsx`.
- `io.load_time_column`: defaults to `Hour Ending` (Excel); `io.load_zone_column`: defaults to `ERCOT`.
- `io.price_data_path`: set to `data/ercot_lz_prices_hourly.csv` (default already set).
- `io.price_time_column`: `hour_ending_str`; `io.price_zone_column`: choose one of `LZ_*` columns (e.g., `LZ_HOUSTON`, `LZ_NORTH`).

2) Implement baseline and peak logic
- Build simple percentile/rolling baseline and pick peak intervals within the program window.
- Respect blackout days and spacing constraints.

Deliverables (Step 2):
- Code: `demflex/ingest.py` (data loading), `demflex/peaks.py` (summer peak selection), and Streamlit `app.py`.
- Selectable zones: pick load zone (e.g., `ERCOT`, `COAST`) and price zone (e.g., `LZ_NORTH`) in the UI; config provides defaults.
- Current mode (no baseline): filter only by season months (default `[6,7,8,9]`), then:
  - Select top `D` days by daily maximum load, and
  - Within each selected day, select top `K` hours by load.
  - Results are sorted chronologically (Top Days by date asc; Top Hours by timestamp asc).

3) Build thermostat impact model
- Implement `r_home(ΔT, L)` with duration decay and caps; aggregate to event kW with enablement/diversity.
- Model kWh shifted and rebound factor.

Deliverables (Step 3):
- Module: `demflex/st_model.py` with:
  - `STParams`, `CohortParams` dataclasses.
  - `duration_multiplier(L_h, breakpoints, multipliers)` piecewise-linear decay.
  - `r_home_kW(ΔT, L, params)` per-home kW with caps.
  - `aggregate_event_kW(r_home, participants, enablement, diversity)` aggregate kW.
  - `participants_for_target(target_kW, r_home, cohort)` feasibility + required participants.
  - `event_energy_kwh(event_kW, L, rebound_fraction)` curtailed/rebound/net kWh.
  - `summarize_event(ΔT, L, target_kW, st, cohort)` convenience wrapper.
- Config parsing: `demflex/config.py` now parses `cohort` and `st_model` blocks into dataclasses and attaches them to `RootConfig`.

Example usage (pseudo):
```
from demflex.config import load_yaml_config, parse_config
from demflex.st_model import STParams, CohortParams, summarize_event

cfg = parse_config(load_yaml_config('config/defaults.yaml'))
st = STParams(
  alpha_kW_per_deg=cfg.st.alpha_kW_per_deg,
  duration_breakpoints_h=cfg.st.duration_breakpoints_h,
  duration_multipliers=cfg.st.duration_multipliers,
  cap_kW_per_home=cfg.st.cap_kW_per_home,
  rebound_fraction=cfg.st.rebound_fraction,
)
cohort = CohortParams(**cfg.cohort.__dict__)
res = summarize_event(deltaT_F=2.0, L_h=3.0, target_kW=50, st=st, cohort=cohort)
```

4) Add heuristic planner from target
- Given `target_kW`, derive `num_events`, `event_length`, `ΔT`, and participants within bounds.
- Handle infeasible targets and suggest relaxations.

5) Implement economics (cost/benefit)
- Annualized cost (CapEx via CRF + OpEx) and benefits (capacity + energy minus rebound).
- Compute NPV, payback, and BCR.

6) Wire CLI and reporting
- CLI inputs (config path, `target_kW`) and outputs (console summary + JSON/CSV).
- Minimal, readable report of parameters and economics.

9) Add Streamlit frontend
- Build a Streamlit UI to load config/data, select zones, and identify summer peaks (top `D` days and top `K` hours per day).
- Provide an interactive chart with top hours highlighted and CSV downloads for review.

7) Validate with tests and sensitivities
- Unit checks for kW vs target, constraints, and simple sensitivity toggles for key params.

8) Document and refine README
- Keep assumptions, decisions log, and examples current; add usage examples as CLI stabilizes.

## Streamlit App
- Install deps: `pip install -r requirements.txt`
- Run: `streamlit run app.py`
- Sidebar controls:
  - Config path (defaults to `config/defaults.yaml`).
  - Optional uploads for load (Excel/CSV) and price (CSV) files; otherwise uses paths from config.
  - Zones: pick Load zone and Price zone columns.
  - Top days (D) and Top hours per day (K).
- Output:
  - Summer load time series (months from config `program.season_months`) with red markers for selected top hours (Altair overlay).
  - Table: Top Days by daily maximum (sorted by date ascending; dates shown without timezone).
  - Table: Top Hours within the selected days (sorted by timestamp ascending; timestamps shown without timezone).
  - Download buttons for Top Days/Top Hours CSVs.
