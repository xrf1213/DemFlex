from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_yaml_config, parse_config
from .ingest import read_load, read_prices, program_mask
from .baseline import percentile_by_hour
from .peaks import PeakSelectionConfig, select_peaks


def run(config_path: str, load_zone: str | None = None, price_zone: str | None = None,
        baseline_method: str | None = None) -> dict:
    cfg_dict = load_yaml_config(config_path)
    cfg = parse_config(cfg_dict)
    if baseline_method is None:
        baseline_method = cfg.baseline_method

    # Read data
    load_df = read_load(cfg.io, zone=load_zone)
    # Price reading is not used in baseline/peaks but loaded here for future use
    try:
        _ = read_prices(cfg.io, lz=price_zone)
    except Exception:
        _ = None

    # Build program mask
    pmask = program_mask(load_df["ts"], cfg.program)

    # Baseline
    if baseline_method == "percentile90":
        bl_df = percentile_by_hour(load_df, percent=0.9, restrict_mask=pmask)
    elif baseline_method == "percentile95":
        bl_df = percentile_by_hour(load_df, percent=0.95, restrict_mask=pmask)
    else:
        # default fallback
        bl_df = percentile_by_hour(load_df, percent=0.9, restrict_mask=pmask)

    # Peaks
    peak_cfg = PeakSelectionConfig(
        max_events=cfg.program.max_events_per_season,
        min_gap_days=cfg.program.min_gap_days,
    )
    peaks_df = select_peaks(bl_df, pmask, peak_cfg)

    return {
        "baseline": bl_df,
        "peaks": peaks_df,
        "program_mask": pmask,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="DEMFlex baseline and peak identification")
    ap.add_argument("--config", default="config/defaults.yaml", help="Path to YAML config")
    ap.add_argument("--load-zone", default=None, help="Load zone column (e.g., ERCOT, COAST)")
    ap.add_argument("--price-zone", default=None, help="Price zone column (e.g., LZ_HOUSTON)")
    ap.add_argument("--baseline-method", default=None, help="percentile90 or percentile95")
    args = ap.parse_args()

    res = run(args.config, load_zone=args.load_zone, price_zone=args.price_zone,
              baseline_method=args.baseline_method)
    bl = res["baseline"]
    peaks = res["peaks"]
    print("Baseline computed for:", len(bl), "intervals")
    print("Selected peaks:")
    print(peaks.head(20).to_string(index=False))

