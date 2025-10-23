# DemFlex — Smart Thermostat MVP

This document outlines the initial design and data flow for DEMFlex, a planning tool for utility planners to meet capacity reduction targets using demand response via Smart Thermostats. The MVP supports a single technology (Smart Thermostat) and produces program/event parameters plus cost–benefit results.

DemFlex helps planners identify summer demand response events using smart thermostats, estimate per‑event reduction, and review simple benefits/costs. The app is a Streamlit UI with three views: Peaks, Impact, and Economics.

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

## Install & Run
- Install: `pip install -r requirements.txt`
- Launch: `streamlit run app.py`

## Data & Config
- Load data (required): hourly Excel (e.g., `data/Native_Load_2024.xlsx`) with `Hour Ending` and zone columns (e.g., `ERCOT`).
- Price data (optional for MVP): `data/ercot_lz_prices_hourly.csv` for the Economics view.
- Config path: `config/defaults.yaml` provides program window, IO paths/zones, cohort, st_model (for Economics), and economics parameters.
- Sidebar only contains file paths and zone pickers; all other inputs are inside views.

## Views

### Peaks
- Purpose: Pick summer peak events with minimal inputs.
- Inputs: `Top days (D)`, `Event length (hours, continuous)`.
- Method: Summer months only (config `program.season_months`, default `[6,7,8,9]`). Select top `D` days by daily max, then per day choose the best continuous `K`‑hour window (highest average load).
- Outputs:
  - Chart: summer load with red markers on selected windows (y‑axis does not start at zero).
  - Tables: Top Days and Event Windows; CSV downloads.
- Persistence: Selected windows are saved and reused by Impact/Economics.

### Impact
- Purpose: Derive thermostat setpoint (ΔT) and visualize before/after load.
- Inputs: `Total households (region)`, `Penetration rate` (0..1), `Target capacity (MW)`.
- Assumption: Each device reduces 0.512 kW at ΔT=4°F; linear scaling with ΔT.
- Computation:
  - Participants = households × penetration (integer).
  - `ΔT_required = ceil(target_kW / (participants × 0.512) × 4)`; clamp to integer range [2, 6] °F.
  - Per‑device reduction = 0.512 × (ΔT_required / 4) kW; aggregate per hour = participants × per‑device / 1000 (MW).
  - Apply aggregate reduction to all hours inside the selected event windows (no rebound in MVP).
- Charts:
  - Overview + detail (brush‑zoom) of summer `original vs after‑event` with event‑hour markers.
  - Focused Day: pick a day (via small form) to view `before vs after` for that day; does not trigger recomputation.
- Persistence: Impact results (ΔT, target, plotted data) are stored so switching days is fast and the overview stays visible.

### Economics (MVP)
- Prerequisites: Run Peaks then Impact (the app enforces order).
- Inputs: Uses `config.economics` and Impact results (participants/ΔT/target). Uses `st_model` and `cohort` from config to estimate per‑event kW if needed.
- Benefits (annualized): Capacity + Energy (flat price from config if no series).
- Costs: CapEx (year 0) + OpEx (annual).
- Output: benefits/costs tables and an undiscounted cumulative net cashflow chart; intermediates for transparency.

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
`target_kW` → load ingest/clean → summer peak pick (top days + continuous windows) → heuristic derive (`num_events`, `L`, `ΔT`, participants) → ST model (kW/kWh/rebound) → cost/benefit → select best → report.

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
  2. Identify summer peaks (top `D` days; per-day best continuous `K`-hour window).
  3. Derive `num_events`, `event_length`, `ΔT` heuristically to meet `target_kW`.
  4. Compute participants (respecting penetration/enrollment caps).
  5. Run cost/benefit; write a concise report and JSON/CSV outputs.
- Suggested modules:
  - `ingest.py`, `peaks.py`, `st_model.py`, `planner.py`, `economics.py`, `report.py`.
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

2) Implement summer peak logic
- Identify top `D` days by daily max; for each, pick best continuous `K`-hour window.
- Visualize summer load and highlight selected windows; provide CSV exports.

Deliverables (Step 2):
- Code: `demflex/ingest.py` (data loading), `demflex/peaks.py` (summer peak selection), and Streamlit `app.py`.
- Selectable zones: pick load zone (e.g., `ERCOT`, `COAST`) and price zone (e.g., `LZ_NORTH`) in the UI; config provides defaults.
- Current mode (no baseline): filter only by season months (default `[6,7,8,9]`), then:
  - Select top `D` days by daily maximum load, and
  - For each selected day, choose the best continuous `K`-hour window.
  - Results are sorted chronologically (Top Days by date asc; Event Windows by start time asc).

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

## Utilities
- Top‑level buttons:
  - `Reset Session (keep Peaks)`: clears Impact/Economics context, preserves Peaks windows.
  - `Full Reset`: clears all session state and cached data.

## File Map
- `app.py`: Streamlit app (Peaks / Impact / Economics).
- `demflex/peaks.py`: summer peak selection (top days + continuous windows).
- `demflex/ingest.py`: data loading and helpers.
- `demflex/st_model.py`: thermostat model (kept for Economics and future extensions).
- `demflex/economics.py`: simple annualized benefits/costs and cashflow.
- `config/defaults.yaml`: program, IO paths, cohort, st_model, economics.

## Notes & Limitations
- Impact view uses a simple linear model (no rebound); Economics uses a simplified annualized model. Both are placeholders for calibration/expansion.
- Price data is optional in the MVP; flat price from config is used in Economics when no series is provided.
