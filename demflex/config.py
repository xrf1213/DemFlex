from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
    # Optional extended configs parsed from YAML for modeling
    cohort: "CohortConfig | None" = None
    st: "STModelConfig | None" = None


@dataclass
class CohortConfig:
    N_eligible: int
    penetration: float
    enrollment_rate: float
    enablement_rate: float
    diversity_factor: float


@dataclass
class STModelConfig:
    alpha_kW_per_deg: float
    duration_breakpoints_h: List[float]
    duration_multipliers: List[float]
    cap_kW_per_home: float
    rebound_fraction: float


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

    # Optional cohort config
    cohort_cfg = None
    if "cohort" in cfg:
        c = cfg["cohort"]
        cohort_cfg = CohortConfig(
            N_eligible=int(c.get("N_eligible", 0)),
            penetration=float(c.get("penetration", 0.0)),
            enrollment_rate=float(c.get("enrollment_rate", 0.0)),
            enablement_rate=float(c.get("enablement_rate", 0.0)),
            diversity_factor=float(c.get("diversity_factor", 1.0)),
        )

    # Optional ST model config
    st_cfg = None
    if "st_model" in cfg:
        s = cfg["st_model"]
        st_cfg = STModelConfig(
            alpha_kW_per_deg=float(s.get("alpha_kW_per_deg", 0.2)),
            duration_breakpoints_h=list(s.get("duration_decay", {}).get("breakpoints_h", [1, 2, 3, 4])),
            duration_multipliers=list(s.get("duration_decay", {}).get("multipliers", [1.0, 0.9, 0.8, 0.7])),
            cap_kW_per_home=float(s.get("cap_kW_per_home", 1.2)),
            rebound_fraction=float(s.get("rebound_fraction", 0.4)),
        )

    return RootConfig(program=program_window, io=io_cfg, baseline_method=baseline_method, cohort=cohort_cfg, st=st_cfg)
