"""Extract and validate LTE tower subsets from an OpenCelliD snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "radio",
    "mcc",
    "net",
    "area",
    "cell",
    "unit",
    "lon",
    "lat",
    "range",
    "samples",
    "created",
    "updated",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOWER_COLUMNS = sorted(REQUIRED_COLUMNS)


def extract_country_lte_towers(
    snapshot_path: Path,
    country_code: int | tuple[int, ...] = 232,
    chunk_size: int = 500_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read an xz CSV snapshot and return valid LTE towers for one or more MCCs."""
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"OpenCelliD snapshot not found: {snapshot_path}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")
    country_codes = (
        (country_code,)
        if isinstance(country_code, int)
        else tuple(int(code) for code in country_code)
    )
    if not country_codes or any(code < 1 or code > 999 for code in country_codes):
        raise ValueError("At least one valid MCC country code is required.")

    input_rows = 0
    country_rows = 0
    non_lte_rows = 0
    invalid_gps_rows = 0
    selected_chunks: list[pd.DataFrame] = []

    chunks = pd.read_csv(
        snapshot_path,
        compression="xz",
        usecols=TOWER_COLUMNS,
        chunksize=chunk_size,
    )
    for chunk in chunks:
        input_rows += len(chunk)
        mcc = pd.to_numeric(chunk["mcc"], errors="coerce")
        country = chunk.loc[mcc.isin(country_codes)].copy()
        country_rows += len(country)
        lte = country.loc[country["radio"].astype("string").str.upper().eq("LTE")].copy()
        non_lte_rows += len(country) - len(lte)

        latitude = pd.to_numeric(lte["lat"], errors="coerce")
        longitude = pd.to_numeric(lte["lon"], errors="coerce")
        valid_gps = (
            latitude.notna()
            & longitude.notna()
            & np.isfinite(latitude)
            & np.isfinite(longitude)
        )
        valid_gps &= latitude.between(-90, 90) & longitude.between(-180, 180)
        invalid_gps_rows += int((~valid_gps).sum())
        lte = lte.loc[valid_gps].copy()
        lte["lat"] = latitude.loc[valid_gps]
        lte["lon"] = longitude.loc[valid_gps]
        selected_chunks.append(lte)

    towers = pd.concat(selected_chunks, ignore_index=True)
    towers = towers.rename(columns={"lat": "latitude", "lon": "longitude"})
    summary = {
        "input_rows": input_rows,
        "country_rows": country_rows,
        "non_lte_country_rows": non_lte_rows,
        "invalid_gps_lte_rows": invalid_gps_rows,
        "output_rows": len(towers),
    }
    if len(country_codes) == 1:
        summary["country_code"] = country_codes[0]
    else:
        summary["country_codes"] = list(country_codes)
    return towers, summary


def ingest_towers(
    snapshot_path: Path,
    output_path: Path,
    summary_path: Path,
    country_code: int | tuple[int, ...] = 232,
) -> dict[str, Any]:
    """Write a filtered country LTE Parquet file and its extraction summary."""
    towers, summary = extract_country_lte_towers(snapshot_path, country_code)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    towers.to_parquet(output_path, index=False)
    summary["input_path"] = str(snapshot_path)
    summary["output_path"] = str(output_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "cell_towers_2021.csv.xz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "opencellid_austria_lte_2021.parquet",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "tower_ingest_summary.json",
    )
    parser.add_argument("--country-code", type=int, nargs="+", default=[232])
    args = parser.parse_args(argv)
    summary = ingest_towers(
        args.input,
        args.output,
        args.summary,
        country_code=(
            args.country_code[0]
            if len(args.country_code) == 1
            else tuple(args.country_code)
        ),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
