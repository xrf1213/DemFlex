# DemFlex — DR Planning MVP

DemFlex is a Streamlit tool to identify summer demand response events (Peaks), estimate aggregate impact from multiple technologies (Impact), and review benefits/costs (Economics). Supported technologies: Thermostats, Solar PV, and Battery Storage.

## Key Assumptions
- Hourly load data is provided with a load zone column.
- Peaks filters by summer months only (config `program.season_months`). No weekday, time‑of‑day, or blackout filtering.
- Impact uses a simple linear rule to size ΔT and compute aggregate reduction; no rebound modeling.
- Economics uses reduced peak from Impact with user‑entered capacity value (`$/MW‑yr`), hourly price series (`$/MWh`) during event hours, and participant‑based costs (device/enroll/retention/operational).

## Inputs
- Peaks: Top days `D`, event length `K` (hours, continuous).
- Impact: Total households, penetration (0..1), target capacity (MW).
- Economics: Capacity value (`$/MW‑yr`), Costs: cost per device, enroll credit per household, retention credit per household (`$/yr`), operational cost per household per year (`$/household‑yr`). Energy benefit is computed from the hourly price CSV.
- Config used: `program.season_months`, `io.*` (paths/columns). `economics.capacity_value_per_kWyr` seeds the capacity default; `economics.program_life_years` sets cashflow years. Note: a flat energy price is no longer used.

## Core Data Models
- Config (used): `program.season_months`、`io.*`、`economics.capacity_value_per_kWyr`、`economics.program_life_years`（不再使用 `flat_energy_price_per_kWh`）。
- EventWindow: `{date, start_ts, end_ts, duration_h, avg_load_MW, peak_within_window_MW}`（Peaks 输出）。
- ImpactSummary: `{participants, ΔT_required_F, per_device_kW, aggregate_MW}`。
- EconomicsOutput: tables for benefits/costs and undiscounted cumulative cashflow。

## Install & Run
- Install: `pip install -r requirements.txt`
- Launch: `streamlit run app.py`

## Data & Config
- Load data: hourly Excel/CSV with time column `Hour Ending` and a load zone (e.g., `ERCOT`).
- Price data (optional): `data/ercot_lz_prices_hourly.csv` for default price columns; prices are expected in `$/MWh`. The app aligns price time range to the loaded load series and applies the same summer‑month filter when displaying seasonal counts.
- Config: `config/defaults.yaml` defines `program.season_months`, `io.*`, and Economics defaults.

## Views

### Peaks
- Select top `D` summer days by daily max; for each, choose the best continuous `K`‑hour window (highest average load).
- Outputs: summer chart with event markers; tables for top days and windows (CSV download).
- Results persist for Impact/Economics.

### Impact
- Inputs: households, penetration, target MW.
- Assumption: 0.512 kW/device at ΔT=4°F; linear scaling with ΔT; no rebound.
- Computes ΔT, per‑device kW, aggregate MW; applies a flat reduction during event hours; plots before/after.
- Results persist for Economics.

### Economics
- Inputs: capacity value (`$/MW‑yr`) and costs — cost per device, enroll credit per household, retention credit per household (`$/yr`), operational cost per household per year (`$/household‑yr`).
- Uses Impact reduced peak and Peaks’ event windows; energy benefit is computed by summing hourly prices (`$/MWh`) during event hours.
- Outputs: benefits/costs tables and cumulative undiscounted cashflow over program life.

## Technology Details

- Thermostats
  - Impact: sizes ΔT to meet target; applies a flat reduction during Peaks windows; no rebound.
  - Economics: uses Peaks windows for event hours; capacity + hourly energy benefits.

- Solar PV
  - Impact: computes average PV area per household required to meet target using 0.21 kW/m²; applies reduction only during 10:00–16:00 across summer months (does not use Peaks windows).
  - Economics: reuses capacity + energy logic over daily 10:00–16:00 summer windows (independent of Peaks).

- Battery Storage (MVP)
  - Impact: computes minimal per‑device power/energy to meet target over the event length; discharges during Peaks windows and shows charging in nearby 00:00–06:00 hours before each event (visualization).
  - Economics: capacity + energy over Peaks windows; net energy = discharge benefit − charging cost from pre‑event 00:00–06:00 hours (assumes round‑trip efficiency = 1.0 by default).

## Current Computation Logic

Only the following formulas are used.

### Peaks
- Inputs: `D` (top days), `K` (event length), `season_months`.
- Filter: keep rows with `month(ts) ∈ season_months`.
- Daily max per day `d`: `daily_max_MW(d) = max_t load_MW(d, t)`.
- Select `TopD` days by descending `daily_max_MW` (display sorted by date).
- Best window per selected day `d`:
  - Rolling K‑hour sum `S(t) = sum_{i=0..K-1} load_MW(t+i)`; if <K points, use whole day.
  - Pick `t* = argmax S(t)`; window `[t*, t*+K)`.
  - Store `start_ts`, `end_ts`, `duration_h = K`, `avg_load_MW = S(t*)/K`, `peak_within_window_MW`.

### Impact
- Inputs: households `H`, penetration `p`, target capacity `T_MW`.
- Constants: `base_per_device_kw = 0.512` at `ΔT = 4°F`.
- Participants: `participants = int(H × p)`.
- Target in kW: `T_kW = T_MW × 1000`.
- Required ΔT: `ΔT_raw = (T_kW / (participants × 0.512)) × 4`; `ΔT_required = clamp(ceil(ΔT_raw), 2, 6)` (°F).
- Per‑device kW: `r_device_kW = 0.512 × (ΔT_required / 4)`.
- Aggregate: `aggregate_kW = participants × r_device_kW`; `aggregate_MW = aggregate_kW / 1000`.
- Apply reduction for event hours: `load_after = max(load_before − aggregate_MW, 0)`; no rebound.

### Economics
- From Peaks: `events_per_year`, `event_length_h`, and each event's start time.
- From Impact: `reduced_peak_MW` and `participants = int(H × p)`.
- Totals: `total_event_hours = events_per_year × event_length_h`.
- Benefits ($/yr):
  - Capacity: `B_cap = reduced_peak_MW × C` where `C` is `$/MW‑yr`.
  - Energy: sum over event hours `h` of `reduced_peak_MW × price_per_MWh(h)` using the price CSV.
  - `B_tot = B_cap + B_en`.
- Costs:
  - Year 0: `Cost_device = participants × cost_per_device`; `Cost_enroll = participants × enroll_credit_per_household`.
  - Annual (Years 1..Y): `Cost_retention = participants × retention_credit_per_household`; `Cost_op = participants × operational_cost_per_year` (per household per year); `Cost_total = Cost_retention + Cost_op`.
- Cashflow (undiscounted): Year 0 net `= −(Cost_device + Cost_enroll)`; Years 1..Y net `= B_tot − (Cost_retention + Cost_op)`; cumulative net is the running sum.

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

## Benefit Model (current)
- Capacity benefit ($/yr) = `Reduced peak (MW) × Capacity value ($/MW‑yr)`。
- Energy benefit ($/yr) = `Reduced peak (MW) × Total event hours (h/yr) × Energy value ($/MWh)`。
- Total benefit ($/yr) = Capacity + Energy。
- Costs：First‑year cost = `$163,000/MW × Reduced peak`；Operational cost annual = `$10,000/MW‑yr × Reduced peak`。
- Cashflow：Year 0 = `−First‑year cost`；Years 1..N = `Total benefit − Operational cost`（未折现）。

## Selection & Outputs
- Selection: choose feasible parameter set with highest NPV, then highest BCR, then lowest participants for practicality.
- Outputs:
  - Event plan: `num_events`, `event_length_h`, `ΔT_setpoint_F`, candidate dates/windows.
  - Program size: participants required/used.
  - Economics: total cost, total benefit, NPV, payback, BCR, capacity met %.

## Data Flow Summary
`target_kW` → load ingest/clean → summer peak pick (top days + continuous windows) → heuristic derive (`num_events`, `L`, `ΔT`, participants) → impact sizing (aggregate MW) → cost/benefit → select best → report.

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
- Suggested modules (current):
  - `ingest.py`, `peaks.py`, `economics.py`.
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

## Quick Edits
- `program.season_months`: update summer months for Peaks filtering.
- `io.load_data_path`: choose your load file.
- `io.price_data_path`: choose your price series (CSV, `$/MWh`).
- `economics.capacity_value_per_kWyr`: seeds the capacity value default (`$/MW‑yr`).

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

3) Impact sizing (current MVP)
- Impact view uses a simple linear rule to size ΔT and compute per‑hour aggregate reduction（no rebound）。

4) Add heuristic planner from target
- Given `target_kW`, derive `num_events`, `event_length`, `ΔT`, and participants within bounds.
- Handle infeasible targets and suggest relaxations.

5) Implement economics (cost/benefit)
- Annualized benefits and costs per updated Economics page (capacity + energy) with simple cashflow.

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

## File Map
- `app.py`: Streamlit app（Peaks / Impact / Economics）。
- `demflex/peaks.py`: summer peak selection（top days + continuous windows）。
- `demflex/ingest.py`: data loading and helpers。
- `demflex/economics.py`: benefits (capacity + hourly energy) and participant‑based cost model with cashflow。
- `config/defaults.yaml`: program, IO paths, economics defaults。
 
## Notes
- Impact uses a simple linear model (no rebound). Economics uses capacity value with hourly energy pricing (`$/MWh`) and participant‑based costs (device/enroll/retention/operational). Price data is optional but required for energy benefits.
