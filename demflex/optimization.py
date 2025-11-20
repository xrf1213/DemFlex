from __future__ import annotations

import math
import itertools
from typing import Any, Dict, List, Tuple
import pandas as pd

# Import existing project modules
from demflex import peaks as peaks_mod
from demflex import economics as economics_mod
from demflex.config import RootConfig

import optuna
import logging
import sys

# Define a large penalty score for constraint violations
PENALTY_SCORE = -1e9

# Configure logging to see optimization progress
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_logger = logging.getLogger(__name__)


def run_optimization(
        load_df: pd.DataFrame,
        price_df: pd.DataFrame,
        config: RootConfig,
        min_target_mw: float,
        participants_map: Dict[str, int],
        constraints: Dict[str, Any],
        n_trials: int = 100
):
    """
    Run parameter optimization using Optuna, supporting combinations of technologies.
    Effect is calculated via simple superposition.
    """

    # Define available technologies
    available_techs = ["Battery Storage", "Thermostats", "Solar PV"]

    # Generate all possible non-empty combinations (size 1 to len(techs))
    # e.g. [('Battery',), ('Battery', 'Solar'), ...]
    all_combinations_tuples = []
    for r in range(1, len(available_techs) + 1):
        all_combinations_tuples.extend(itertools.combinations(available_techs, r))

    # Create string labels for Optuna categorical choice
    # e.g. "Battery Storage + Solar PV"
    combo_labels = [" + ".join(c) for c in all_combinations_tuples]

    # 1. Define Objective Function
    def objective(trial: optuna.Trial) -> float:

        # === A. Suggest Combination ===
        selected_combo_label = trial.suggest_categorical("combination", combo_labels)
        selected_techs = selected_combo_label.split(" + ")

        # === B. Define Search Space (Shared & Specific) ===

        # 1. Shared Event Parameters (for Battery & Thermostats)
        # If any tech in the combination needs Peak Windows, we generate these params once.
        needs_peak_windows = any(t in ["Battery Storage", "Thermostats"] for t in selected_techs)

        shared_top_days = 0
        shared_event_len = 0.0

        if needs_peak_windows:
            d_min = int(constraints.get("top_days_min", 10))
            d_max = int(constraints.get("top_days_max", 60))
            h_min = int(constraints.get("event_length_min", 2))
            h_max = int(constraints.get("event_length_max", 6))

            shared_top_days = trial.suggest_int("shared_top_days", d_min, d_max)
            shared_event_len = trial.suggest_int("shared_event_length_h", h_min, h_max)

        # Initialize Aggregators
        total_net_benefit = 0.0
        total_reduced_mw = 0.0
        violation_reasons = []
        is_global_feasible = True

        # === C. Evaluate Each Technology in Combination ===
        for tech in selected_techs:
            current_participants = participants_map.get(tech, 0)

            # Skip if no participants (should be caught by validation, but safe to keep)
            if current_participants <= 0:
                continue

            # Prepare params for this specific tech
            tech_params = {
                "tech": tech,
                "participants": current_participants,
                "top_days": shared_top_days,
                "event_length_h": shared_event_len
            }

            # Suggest tech-specific parameters
            if tech == "Solar PV":
                s_min = constraints.get("solar_kw_min", 3.0)
                s_max = constraints.get("solar_kw_max", 10.0)
                tech_params["solar_kw_per_home"] = trial.suggest_float(f"{tech}_kw_per_home", s_min, s_max)
                # Solar doesn't use the shared peak windows usually, but we pass 0 to be safe/ignored
                # depending on implementation. In our evaluate_scenario, Solar forces its own windows.

            elif tech == "Battery Storage":
                cap_min = constraints.get("batt_kwh_min", 10.0)
                cap_max = constraints.get("batt_kwh_max", 20.0)
                tech_params["batt_kwh"] = trial.suggest_float(f"{tech}_kwh", cap_min, cap_max)

                p_min = constraints.get("batt_kw_min", 3.0)
                p_max = constraints.get("batt_kw_max", 7.0)
                tech_params["batt_kw"] = trial.suggest_float(f"{tech}_kw", p_min, p_max)

                dod_min = constraints.get("batt_dod_min", 0.60)
                dod_max = constraints.get("batt_dod_max", 0.95)
                tech_params["batt_dod"] = trial.suggest_float(f"{tech}_dod", dod_min, dod_max)

            elif tech == "Thermostats":
                t_min = constraints.get("thermostat_delta_t_min", 1.0)
                t_max = constraints.get("thermostat_delta_t_max", 5.0)
                tech_params["thermostat_delta_t"] = trial.suggest_float(f"{tech}_delta_t", t_min, t_max)

            # Run Evaluation for this tech
            # Note: We bypass the "min_target_mw" check inside evaluate_scenario for individual techs
            # because the target applies to the *Aggregate* of the combination.
            # We pass min_target_mw=0 here to avoid premature failure, and check sum later.
            result = evaluate_scenario(
                params=tech_params,
                load_df=load_df,
                price_df=price_df,
                config=config,
                min_target_mw=0.0
            )

            if not result["is_feasible"]:
                is_global_feasible = False
                violation_reasons.append(f"{tech}: {result.get('violation_reason')}")
                break  # Stop evaluating other techs if one fails physically

            # Superposition (Add up results)
            total_net_benefit += result["net_benefit"]
            total_reduced_mw += result["reduced_peak_MW"]

        # === D. Global Constraints Check ===

        # 1. Check if any individual tech failed physical constraints
        if not is_global_feasible:
            trial.set_user_attr("violation", "; ".join(violation_reasons))
            return PENALTY_SCORE

        # 2. Check Aggregate Target MW
        if total_reduced_mw < min_target_mw:
            trial.set_user_attr("violation", f"Total Reduced MW ({total_reduced_mw:.2f}) < Target ({min_target_mw})")
            return PENALTY_SCORE

        # Record success metrics
        trial.set_user_attr("reduced_peak_MW", total_reduced_mw)

        return total_net_benefit

    # 2. Create Study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    print(f"Starting optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    # 3. Output results
    print("\n" + "=" * 40)
    print("OPTIMIZATION RESULTS")
    print("=" * 40)

    best_trial = study.best_trial
    print(f"Best Value (Total Net Benefit): ${best_trial.value:,.2f}")
    print("Best Params:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    return study.best_trial


def evaluate_scenario(
        params: Dict[str, Any],
        load_df: pd.DataFrame,
        price_df: pd.DataFrame,
        config: RootConfig,
        min_target_mw: float
) -> Dict[str, Any]:
    """
    Evaluate a single technology scenario.
    Used internally by the optimization loop.
    """

    # 1. Unpack general parameters
    tech_choice = params.get("tech")
    top_days = int(params.get("top_days", 20))
    event_length_h = float(params.get("event_length_h", 4.0))
    participants = int(params.get("participants", 0))

    # ---------------------------------------------------------
    # Step 1: Run Peaks module
    # ---------------------------------------------------------
    if tech_choice == "Solar PV":
        windows_df = _get_solar_windows(load_df, config.program.season_months)
        calc_event_length_h = 7.0
    else:
        _, windows_df = peaks_mod.select_top_days_and_windows(
            load_df,
            season_months=config.program.season_months,
            top_days=top_days,
            window_h=int(event_length_h)
        )
        calc_event_length_h = event_length_h

    events_per_year = len(windows_df)
    if events_per_year == 0:
        return _infeasible_result("No event windows found.")

    # ---------------------------------------------------------
    # Step 2: Run Impact logic
    # ---------------------------------------------------------
    if tech_choice == "Battery Storage":
        impact_res = _calculate_impact_battery(params, calc_event_length_h, participants)
    elif tech_choice == "Thermostats":
        impact_res = _calculate_impact_thermostats(params, participants)
    elif tech_choice == "Solar PV":
        impact_res = _calculate_impact_solar(params, participants)
    else:
        return _infeasible_result(f"Unknown technology: {tech_choice}")

    if not impact_res["valid"]:
        return _infeasible_result(impact_res["reason"])

    reduced_peak_MW = impact_res["reduced_peak_MW"]

    # ---------------------------------------------------------
    # Step 3: Check MW Target (Individual or part of aggregate)
    # ---------------------------------------------------------
    # If min_target_mw is > 0, we check it.
    # In combination mode, we pass 0 here and check sum later.
    if min_target_mw > 0 and reduced_peak_MW < min_target_mw:
        return _infeasible_result(
            f"Reduced MW ({reduced_peak_MW:.2f}) < Target ({min_target_mw:.2f})"
        )

    # ---------------------------------------------------------
    # Step 4: Run Economics
    # ---------------------------------------------------------
    charge_hours_df = None
    charge_power_MW = None
    if tech_choice == "Battery Storage":
        charge_power_MW = reduced_peak_MW
        charge_hours_df = _get_battery_charge_hours(windows_df, calc_event_length_h)

    try:
        econ_defaults = config.economics
        cost_per_device = float(params.get("cost_per_device", econ_defaults.device_cost))

        econ_res = economics_mod.economics_from_peak(
            reduced_peak_MW=reduced_peak_MW,
            events_per_year=events_per_year,
            event_length_h=calc_event_length_h,
            per_capacity_value_per_MWyr=econ_defaults.capacity_value_per_kWyr * 1000,
            per_energy_value_per_MWh=None,
            program_life_years=econ_defaults.program_life_years,
            participants=participants,
            cost_per_device=cost_per_device,
            enroll_credit_per_household=econ_defaults.upfront_incentive,
            retention_credit_per_household=econ_defaults.annual_incentive,
            operational_cost_per_year=econ_defaults.admin_annual,
            price_df=price_df,
            windows_df=windows_df,
            charge_hours_df=charge_hours_df,
            charge_power_MW=charge_power_MW,
            round_trip_efficiency=0.9
        )

        cashflow = econ_res["cashflow"]
        final_net_benefit = cashflow.iloc[-1]["cumulative_net"]

        return {
            "net_benefit": final_net_benefit,
            "is_feasible": True,
            "violation_reason": None,
            "reduced_peak_MW": reduced_peak_MW,
            "details": econ_res
        }

    except Exception as e:
        return _infeasible_result(f"Economics calculation failed: {str(e)}")


# ==============================================================================
# Helper Functions
# ==============================================================================

def _infeasible_result(reason: str) -> Dict[str, Any]:
    return {
        "net_benefit": PENALTY_SCORE,
        "is_feasible": False,
        "violation_reason": reason,
        "reduced_peak_MW": 0.0,
        "details": None
    }


def _calculate_impact_battery(params: Dict, duration_h: float, participants: int) -> Dict:
    batt_kwh = float(params.get("batt_kwh", 13.5))
    batt_kw = float(params.get("batt_kw", 5.0))
    batt_dod = float(params.get("batt_dod", 0.8))

    energy_needed_kwh = batt_kw * duration_h
    energy_allowed_kwh = batt_kwh * batt_dod

    if energy_needed_kwh > energy_allowed_kwh:
        return {
            "valid": False,
            "reason": f"Battery violation: Need {energy_needed_kwh:.1f}kWh, Limit {energy_allowed_kwh:.1f}kWh"
        }

    reduced_peak_MW = (participants * batt_kw) / 1000.0
    return {"valid": True, "reduced_peak_MW": reduced_peak_MW}


def _calculate_impact_thermostats(params: Dict, participants: int) -> Dict:
    delta_t = float(params.get("thermostat_delta_t", 2.0))
    if not (0 < delta_t <= 6):
        return {"valid": False, "reason": f"Thermostat violation: Delta T {delta_t} out of bounds"}

    base_deltaT = 4.0
    base_per_device_kw = 0.512
    per_device_kw = base_per_device_kw * (delta_t / base_deltaT)
    reduced_peak_MW = (participants * per_device_kw) / 1000.0
    return {"valid": True, "reduced_peak_MW": reduced_peak_MW}


def _calculate_impact_solar(params: Dict, participants: int) -> Dict:
    per_household_kw = float(params.get("solar_kw_per_home", 5.0))
    reduced_peak_MW = (participants * per_household_kw) / 1000.0
    return {"valid": True, "reduced_peak_MW": reduced_peak_MW}


def _get_solar_windows(load_df: pd.DataFrame, season_months: list) -> pd.DataFrame:
    summer_mask = load_df["ts"].dt.month.isin(season_months)
    summer_ts = load_df.loc[summer_mask, ["ts"]].copy()
    if summer_ts.empty: return pd.DataFrame()
    summer_ts["date"] = summer_ts["ts"].dt.tz_localize(None).dt.normalize()
    summer_ts["hour"] = summer_ts["ts"].dt.hour
    starts = summer_ts.loc[summer_ts["hour"] == 10].copy()
    return pd.DataFrame({
        "date": starts["date"],
        "start_ts": starts["ts"],
        "end_ts": starts["ts"] + pd.Timedelta(hours=7),
        "duration_h": 7,
    })


def _get_battery_charge_hours(windows_df: pd.DataFrame, event_length_h: float) -> pd.DataFrame:
    charge_ts = []
    hours_needed = int(math.ceil(event_length_h))

    def _find_charge_slots(start_tz):
        chosen = []
        midnight = start_tz.normalize()
        for h in range(6, -1, -1):
            ts = midnight + pd.Timedelta(hours=h)
            if ts < start_tz:
                chosen.append(ts)
                if len(chosen) >= hours_needed: return chosen
        prev_day = midnight - pd.Timedelta(days=1)
        for h in range(6, -1, -1):
            chosen.append(prev_day + pd.Timedelta(hours=h))
            if len(chosen) >= hours_needed: return chosen
        return chosen

    for _, row in windows_df.iterrows():
        charge_ts.extend(_find_charge_slots(pd.to_datetime(row["start_ts"])))
    return pd.DataFrame({"ts": pd.to_datetime(charge_ts)})