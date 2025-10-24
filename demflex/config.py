from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class ProgramWindow:
    season_months: List[int]
    start_time: str  # "HH:MM" local time
    end_time: str    # "HH:MM" local time
    weekdays_only: bool
    blackout_dates: List[str]
    max_events_per_season: int
    min_gap_days: int


@dataclass
class IOConfig:
    timezone: str
    load_data_path: str
    load_time_column: str
    load_zone_column: str
    price_data_path: str
    price_time_column: str
    price_zone_column: str
    output_dir: str


@dataclass
class RootConfig:
    program: ProgramWindow
    io: IOConfig
    baseline_method: str
    economics: "EconomicsConfig | None" = None


# (Removed legacy CohortConfig and STModelConfig; not used in current MVP.)


@dataclass
class EconomicsConfig:
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


def load_yaml_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_config(cfg: dict) -> RootConfig:
    program = cfg.get("program", {})
    window = program.get("window_local", {})
    io = cfg.get("io", {})

    program_window = ProgramWindow(
        season_months=program.get("season_months", [6, 7, 8, 9]),
        start_time=window.get("start_time", "14:00"),
        end_time=window.get("end_time", "19:00"),
        weekdays_only=window.get("weekdays_only", True),
        blackout_dates=program.get("blackout_dates", []),
        max_events_per_season=program.get("max_events_per_season", 20),
        min_gap_days=program.get("min_gap_days", 1),
    )

    io_cfg = IOConfig(
        timezone=io.get("timezone", "America/Los_Angeles"),
        load_data_path=io.get("load_data_path", "data/Native_Load_2024.xlsx"),
        load_time_column=io.get("load_time_column", "Hour Ending"),
        load_zone_column=io.get("load_zone_column", "ERCOT"),
        price_data_path=io.get("price_data_path", "data/ercot_lz_prices_hourly.csv"),
        price_time_column=io.get("price_time_column", "hour_ending_str"),
        price_zone_column=io.get("price_zone_column", "LZ_NORTH"),
        output_dir=io.get("output_dir", "outputs"),
    )

    baseline_method = cfg.get("run_defaults", {}).get("baseline_method", cfg.get("baseline", {}).get("method", "percentile90"))

    # (Cohort and ST model blocks are ignored in current MVP)

    # Optional economics config
    econ_cfg = None
    if "economics" in cfg:
        e = cfg["economics"]
        econ_cfg = EconomicsConfig(
            capacity_value_per_kWyr=float(e.get("capacity_value_per_kWyr", 0.0)),
            flat_energy_price_per_kWh=float(e.get("flat_energy_price_per_kWh", 0.0)),
            device_cost=float(e.get("device_cost", 0.0)),
            install_cost=float(e.get("install_cost", 0.0)),
            upfront_incentive=float(e.get("upfront_incentive", 0.0)),
            annual_incentive=float(e.get("annual_incentive", 0.0)),
            per_event_incentive=float(e.get("per_event_incentive", 0.0)),
            platform_fee_annual=float(e.get("platform_fee_annual", 0.0)),
            admin_annual=float(e.get("admin_annual", 0.0)),
            mv_annual=float(e.get("mv_annual", 0.0)),
            per_event_cost=float(e.get("per_event_cost", 0.0)),
            program_life_years=int(e.get("program_life_years", 10)),
            discount_rate=float(e.get("discount_rate", 0.07)),
        )

    return RootConfig(program=program_window, io=io_cfg, baseline_method=baseline_method, economics=econ_cfg)
