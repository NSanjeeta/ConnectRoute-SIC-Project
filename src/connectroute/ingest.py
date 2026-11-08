"""Validate, clean, and persist the Salzburg drive-test measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"lat", "long", "rsrp", "technology"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def prepare_measurements(
    measurements: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return normalized LTE measurements and an auditable row-count summary."""
    missing_columns = REQUIRED_COLUMNS.difference(measurements.columns)
    if missing_columns:
        raise ValueError(
            "Measurement data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    summary: dict[str, int] = {"input_rows": len(measurements)}
    lte_mask = measurements["technology"].astype("string").str.upper().eq("LTE")
    lte = measurements.loc[lte_mask].copy()
    summary["non_lte_rows"] = len(measurements) - len(lte)

    latitude = pd.to_numeric(lte["lat"], errors="coerce")
    longitude = pd.to_numeric(lte["long"], errors="coerce")
    valid_gps = (
        latitude.notna()
        & longitude.notna()
        & np.isfinite(latitude)
        & np.isfinite(longitude)
    )
    summary["missing_or_invalid_gps_rows"] = int((~valid_gps).sum())
    lte = lte.loc[valid_gps].copy()
    lte["lat"] = latitude.loc[valid_gps]
    lte["long"] = longitude.loc[valid_gps]

    in_range = lte["lat"].between(-90, 90) & lte["long"].between(-180, 180)
    if not in_range.all():
        raise ValueError("GPS coordinates contain values outside valid latitude/longitude ranges.")

    rsrp = pd.to_numeric(lte["rsrp"], errors="coerce")
    valid_rsrp = rsrp.notna() & np.isfinite(rsrp)
    summary["missing_or_invalid_rsrp_rows"] = int((~valid_rsrp).sum())
    lte = lte.loc[valid_rsrp].copy()
    lte["rsrp"] = rsrp.loc[valid_rsrp]

    summary["exact_duplicate_rows_preserved"] = int(lte.duplicated().sum())
    lte = lte.rename(
        columns={
            "lat": "latitude",
            "long": "longitude",
            "rsrp": "rsrp_dbm",
            "time": "timestamp",
        }
    )
    if "timestamp" in lte.columns:
        lte["timestamp"] = pd.to_datetime(lte["timestamp"], errors="raise")

    summary["output_rows"] = len(lte)
    return lte.reset_index(drop=True), summary


def ingest(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Read the raw CSV, write cleaned Parquet, and persist filter counts."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Measurement CSV not found: {input_path}")

    raw = pd.read_csv(input_path)
    cleaned, summary = prepare_measurements(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)
    summary["input_path"] = str(input_path)
    summary["output_path"] = str(output_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "SRFG-v1.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "salzburg_measurements.parquet",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "data_ingest_summary.json",
    )
    args = parser.parse_args(argv)
    summary = ingest(args.input, args.output, args.summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
