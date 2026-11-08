"""Match projected measurements to nearby LTE tower records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
MEASUREMENT_COLUMNS = {"easting_m", "northing_m"}
TOWER_COLUMNS = {
    "mcc",
    "net",
    "area",
    "cell",
    "unit",
    "latitude",
    "longitude",
    "range",
    "samples",
    "created",
    "updated",
}


def match_nearest_towers(
    measurements: pd.DataFrame,
    towers: pd.DataFrame,
    projected_crs: str = "EPSG:32633",
    k: int = 3,
    minimum_samples: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add K nearest tower attributes, distances, and bearings to each point."""
    missing_measurements = MEASUREMENT_COLUMNS.difference(measurements.columns)
    missing_towers = TOWER_COLUMNS.difference(towers.columns)
    if missing_measurements:
        raise ValueError(
            "Measurements are missing projected columns: "
            + ", ".join(sorted(missing_measurements))
        )
    if missing_towers:
        raise ValueError(
            "Tower data is missing columns: " + ", ".join(sorted(missing_towers))
        )
    if not np.isfinite(measurements[["easting_m", "northing_m"]].to_numpy(dtype=float)).all():
        raise ValueError("Measurement projected coordinates must be finite.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    if minimum_samples < 0:
        raise ValueError("minimum_samples cannot be negative.")

    sample_count = pd.to_numeric(towers["samples"], errors="coerce")
    eligible = towers.loc[sample_count.ge(minimum_samples)].copy()
    if eligible.empty:
        raise ValueError("No towers satisfy the minimum sample-count threshold.")

    tower_x, tower_y = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True,
    ).transform(
        pd.to_numeric(eligible["longitude"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(eligible["latitude"], errors="coerce").to_numpy(dtype=float),
    )
    tower_xy = np.column_stack((tower_x, tower_y))
    valid_towers = np.isfinite(tower_xy).all(axis=1)
    eligible = eligible.loc[valid_towers].reset_index(drop=True)
    tower_xy = tower_xy[valid_towers]
    if len(eligible) < k:
        raise ValueError(
            f"Need at least {k} eligible towers; found {len(eligible)}."
        )

    measurement_xy = measurements[["easting_m", "northing_m"]].to_numpy(dtype=float)
    distances, positions = cKDTree(tower_xy).query(measurement_xy, k=k)
    if k == 1:
        distances = distances[:, np.newaxis]
        positions = positions[:, np.newaxis]

    features = pd.DataFrame(index=np.arange(len(measurements)))
    feature_columns = [
        "mcc",
        "net",
        "area",
        "cell",
        "unit",
        "latitude",
        "longitude",
        "samples",
        "range",
    ]
    tower_times = {
        column: pd.to_datetime(eligible[column], errors="coerce")
        for column in ("created", "updated")
    }
    measurement_times = (
        pd.to_datetime(measurements["timestamp"], errors="coerce")
        if "timestamp" in measurements.columns
        else None
    )

    for rank in range(k):
        selected = eligible.iloc[positions[:, rank]].reset_index(drop=True)
        dx = tower_xy[positions[:, rank], 0] - measurement_xy[:, 0]
        dy = tower_xy[positions[:, rank], 1] - measurement_xy[:, 1]
        prefix = f"tower_{rank + 1}"
        features[f"{prefix}_distance_m"] = distances[:, rank]
        features[f"{prefix}_bearing_deg"] = np.mod(np.degrees(np.arctan2(dx, dy)), 360)
        for column in feature_columns:
            features[f"{prefix}_{column}"] = selected[column].to_numpy()
        for column, tower_time in tower_times.items():
            selected_time = tower_time.iloc[positions[:, rank]].reset_index(drop=True)
            features[f"{prefix}_{column}"] = selected_time
            if measurement_times is not None:
                # Signed ages make records first seen after the drive observable.
                feature_name = "created" if column == "created" else "updated"
                features[f"{prefix}_{feature_name}_age_at_measurement_days"] = (
                    measurement_times.reset_index(drop=True) - selected_time
                ).dt.total_seconds() / 86400

    result = features.reset_index(drop=True)
    result.insert(0, "measurement_index", np.arange(len(measurements), dtype=np.int64))
    summary: dict[str, Any] = {
        "measurement_rows": int(len(measurements)),
        "input_tower_rows": int(len(towers)),
        "eligible_tower_rows": int(len(eligible)),
        "minimum_samples": int(minimum_samples),
        "k": int(k),
        "projected_crs": projected_crs,
        "nearest_distance_m": {
            "minimum": float(np.min(distances[:, 0])),
            "median": float(np.median(distances[:, 0])),
            "p95": float(np.quantile(distances[:, 0], 0.95)),
            "maximum": float(np.max(distances[:, 0])),
        },
        "negative_created_age_matches": (
            int((result["tower_1_created_age_at_measurement_days"] < 0).sum())
            if "tower_1_created_age_at_measurement_days" in result
            else None
        ),
    }
    return result, summary


def _read_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Project config must contain a YAML mapping: {config_path}")
    return config


def process_tower_matches(
    measurements_path: Path,
    towers_path: Path,
    output_path: Path,
    summary_path: Path,
    projected_crs: str = "EPSG:32633",
    k: int = 3,
    minimum_samples: int = 5,
) -> dict[str, Any]:
    """Read processed inputs and persist tower features plus an audit summary."""
    if not measurements_path.is_file():
        raise FileNotFoundError(f"Projected measurements not found: {measurements_path}")
    if not towers_path.is_file():
        raise FileNotFoundError(f"Processed tower data not found: {towers_path}")
    measurements = pd.read_parquet(measurements_path)
    towers = pd.read_parquet(towers_path)
    matches, summary = match_nearest_towers(
        measurements,
        towers,
        projected_crs=projected_crs,
        k=k,
        minimum_samples=minimum_samples,
    )
    spatial_features = measurements.reset_index(drop=True).copy()
    for column in matches.columns:
        if column != "measurement_index":
            spatial_features[column] = matches[column].to_numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    spatial_features.to_parquet(output_path, index=False)
    summary["measurements_path"] = str(measurements_path)
    summary["towers_path"] = str(towers_path)
    summary["output_path"] = str(output_path)
    summary["output_rows"] = int(len(spatial_features))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config = _read_config(args.config)
    validation = config.get("validation", {})
    prediction = config.get("prediction", {})
    summary = process_tower_matches(
        PROJECT_ROOT / "data" / "processed" / "salzburg_geometry.parquet",
        PROJECT_ROOT
        / "data"
        / "processed"
        / "opencellid_austria_lte_2021.parquet",
        PROJECT_ROOT / "data" / "processed" / "salzburg_spatial_features.parquet",
        PROJECT_ROOT / "results" / "tower_match_summary.json",
        projected_crs=validation.get("projected_crs", "EPSG:32633"),
        k=prediction.get("nearest_towers", 3),
        minimum_samples=prediction.get("tower_sample_count_minimum", 5),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
